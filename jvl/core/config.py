from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

from jvl.utils.errors import ConfigError


class OllamaConfig(BaseModel):
    model: str
    base_url: str = "http://localhost:11434"


class OpenAIConfig(BaseModel):
    model: str
    api_key: str | None = None


class AnthropicConfig(BaseModel):
    model: str
    api_key: str | None = None


class AzureConfig(BaseModel):
    model: str
    api_key: str | None = None
    api_base: str | None = None
    api_version: str = "2024-02-01"


class GeminiConfig(BaseModel):
    model: str
    api_key: str | None = None


class BackendsConfig(BaseModel):
    ollama: OllamaConfig | None = None
    openai: OpenAIConfig | None = None
    anthropic: AnthropicConfig | None = None
    azure: AzureConfig | None = None
    gemini: GeminiConfig | None = None


class Config(BaseModel):
    default_backend: str
    backends: BackendsConfig

    @field_validator("default_backend")
    @classmethod
    def backend_must_be_known(cls, v: str) -> str:
        allowed = {"ollama", "openai", "anthropic", "azure", "gemini"}
        if v not in allowed:
            raise ValueError(
                f"default_backend '{v}' inconnu. Valeurs possibles : {allowed}"
            )
        return v


def _resolve_env_vars(data: dict) -> dict:
    if isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_env_vars(i) for i in data]
    if isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        var_name = data[2:-1]
        value = os.getenv(var_name)
        if value is None:
            return None
        return value
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(
    config_path: Path = Path("config/config.yaml"),
    local_path: Path = Path("config/config.local.yaml"),
) -> Config:
    load_dotenv()
    with open(config_path) as f:
        data = yaml.safe_load(f)

    if local_path.exists():
        with open(local_path) as f:
            local_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, local_data)

    data = _resolve_env_vars(data)
    return Config(**data)

# ── User Config (étapes 1-2) ─────────────────────────────────────────────────
USER_CONFIG_DIR = Path.home() / ".jvl"
USER_CONFIG_PATH = USER_CONFIG_DIR / "config.yaml"


class ProviderConfig(BaseModel):
    """Configuration générique d'un provider LLM."""

    api_key: str | None = None
    base_url: str | None = None
    api_base: str | None = None  # Azure spécifique
    api_version: str | None = None  # Azure spécifique
    default_model: str | None = None


class UserConfig(BaseModel):
    """Configuration utilisateur persistée dans ~/.jvl/config.yaml."""

    active_provider: str = "ollama"
    active_model: str | None = None  # None = utilise default_model du provider
    providers: dict[str, ProviderConfig] = {}


def load_user_config(
    config_path: Path = USER_CONFIG_PATH,
) -> UserConfig:
    """Charge la configuration utilisateur depuis ~/.jvl/config.yaml."""
    if not config_path.exists():
        return UserConfig()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    data = _resolve_env_vars(data)
    return UserConfig(**data)


def save_user_config(
    config: UserConfig,
    config_path: Path = USER_CONFIG_PATH,
) -> None:
    """Écrit la configuration utilisateur dans ~/.jvl/config.yaml."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(exclude_none=True)
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def resolve_active_config(
    user_config: UserConfig,
    repo_config: Config | None = None,
) -> tuple[str, str, ProviderConfig]:
    """Résout le provider, modèle et config actifs.

    Chaîne de résolution :
        1. user_config.active_model (si défini)
        2. provider.default_model (du provider actif dans user_config)
        3. Fallback vers repo_config (si fourni)

    Returns:
        (provider_name, model_name, provider_config)

    Raises:
        ConfigError: si aucun modèle ne peut être résolu.
    """
    provider_name = user_config.active_provider
    provider_cfg = user_config.providers.get(provider_name)

    # Résolution du modèle
    model = user_config.active_model

    if not model and provider_cfg:
        model = provider_cfg.default_model

    # Fallback vers la config repo
    if not model and repo_config:
        repo_backend_cfg = getattr(repo_config.backends, provider_name, None)
        if repo_backend_cfg:
            model = getattr(repo_backend_cfg, "model", None)

    # Fallback provider config depuis repo
    if not provider_cfg and repo_config:
        repo_backend_cfg = getattr(repo_config.backends, provider_name, None)
        if repo_backend_cfg:
            provider_cfg = ProviderConfig(
                api_key=getattr(repo_backend_cfg, "api_key", None),
                base_url=getattr(repo_backend_cfg, "base_url", None),
                api_base=getattr(repo_backend_cfg, "api_base", None),
                api_version=getattr(repo_backend_cfg, "api_version", None),
                default_model=getattr(repo_backend_cfg, "model", None),
            )

    if not model:
        raise ConfigError(
            f"Aucun modèle résolu pour le provider '{provider_name}'. "
            "Configurez-en un via 'jvl config model set'."
        )

    if not provider_cfg:
        provider_cfg = ProviderConfig()

    return provider_name, model, provider_cfg
