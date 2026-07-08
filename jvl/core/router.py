from __future__ import annotations

from typing import AsyncIterator

from jvl.backends.anthropic import AnthropicClient
from jvl.backends.azure import AzureOpenAIClient
from jvl.backends.base import BaseClient
from jvl.backends.gemini import GeminiClient
from jvl.backends.ollama import OllamaClient
from jvl.backends.openai import OpenAIClient
from jvl.core.session import get_session_backend, get_session_info
from jvl.core.config import Config, ProviderConfig, UserConfig
from jvl.utils.errors import BackendNotAvailable, ConfigError


# ── Provider Registry (étape 3) ──────────────────────────────────────────────

PROVIDER_REGISTRY: dict[str, type[BaseClient]] = {
    "ollama": OllamaClient,
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "azure": AzureOpenAIClient,
    "gemini": GeminiClient,
}


def build_client_from_provider_config(
    name: str,
    provider_config: ProviderConfig,
    model: str,
) -> BaseClient:
    """Instancie un client backend à partir d'un ProviderConfig générique.

    Args:
        name: Nom du provider (ex: 'ollama', 'openai').
        provider_config: Configuration générique du provider.
        model: Nom du modèle à utiliser.

    Returns:
        Instance de BaseClient prête à l'emploi.

    Raises:
        ConfigError: si le provider est inconnu ou mal configuré.
    """
    if name not in PROVIDER_REGISTRY:
        raise ConfigError(f"Provider '{name}' inconnu. Providers disponibles : {', '.join(sorted(PROVIDER_REGISTRY))}")

    match name:
        case "ollama":
            base_url = provider_config.base_url or "http://localhost:11434"
            return OllamaClient(base_url, model)

        case "openai":
            if not provider_config.api_key:
                raise ConfigError("Provider 'openai' : clé API manquante.")
            return OpenAIClient(provider_config.api_key, model)

        case "anthropic":
            if not provider_config.api_key:
                raise ConfigError("Provider 'anthropic' : clé API manquante.")
            return AnthropicClient(provider_config.api_key, model)

        case "azure":
            if not provider_config.api_key or not provider_config.api_base:
                raise ConfigError(
                    "Provider 'azure' : clé API ou endpoint manquant."
                )
            api_version = provider_config.api_version or "2024-02-01"
            return AzureOpenAIClient(
                provider_config.api_key,
                provider_config.api_base,
                model,
                api_version,
            )

        case "gemini":
            if not provider_config.api_key:
                raise ConfigError("Provider 'gemini' : clé API manquante.")
            return GeminiClient(provider_config.api_key, model)

        case _:
            raise ConfigError(f"Provider '{name}' inconnu.")


class BackendRouter:

    def __init__(self, config: Config | UserConfig, repo_config: Config | None = None) -> None:
        self.config = config
        self.repo_config = repo_config
        self._clients: dict[str, BaseClient] = {}
        
        session_info = get_session_info()
        if isinstance(config, UserConfig):
            # Nouveau modèle
            if session_info:
                self._active_backend = session_info["provider"]
                self._active_model = session_info.get("model")
            else:
                self._active_backend = config.active_provider
                self._active_model = config.active_model
        else:
            # Ancien modèle
            session_backend = get_session_backend()
            self._active_backend = session_backend or config.default_backend
            self._active_model = None
            
        self._last_usage: dict | None = None

    def _get_client(self, backend: str | None = None) -> BaseClient:
        name = backend or self._active_backend
        if name not in self._clients:
            self._clients[name] = self._build_client_with_fallback(name)
        return self._clients[name]

    def _build_client_with_fallback(self, name: str) -> BaseClient:
        if isinstance(self.config, UserConfig):
            # Déterminer le modèle pour ce backend
            model = None
            if name == self._active_backend:
                model = self._active_model
            
            if not model:
                pcfg = self.config.providers.get(name)
                if pcfg:
                    model = pcfg.default_model
            
            if not model and self.repo_config:
                repo_backend = getattr(self.repo_config.backends, name, None)
                if repo_backend:
                    model = repo_backend.model
            
            if not model:
                raise ConfigError(f"Aucun modèle défini pour le backend '{name}'.")
                
            provider_cfg = self.config.providers.get(name)
            if not provider_cfg and self.repo_config:
                repo_backend = getattr(self.repo_config.backends, name, None)
                if repo_backend:
                    provider_cfg = ProviderConfig(
                        api_key=getattr(repo_backend, "api_key", None),
                        base_url=getattr(repo_backend, "base_url", None),
                        api_base=getattr(repo_backend, "api_base", None),
                        api_version=getattr(repo_backend, "api_version", None),
                        default_model=getattr(repo_backend, "model", None),
                    )
            if not provider_cfg:
                provider_cfg = ProviderConfig()
                
            return build_client_from_provider_config(name, provider_cfg, model)
        else:
            return self._build_client(name)

    def _build_client(self, name: str) -> BaseClient:
        cfg = self.config.backends
        match name:
            case "ollama":
                if cfg.ollama is None:
                    raise ConfigError("Backend 'ollama' non configuré dans config.yaml")
                return OllamaClient(cfg.ollama.base_url, cfg.ollama.model)

            case "openai":
                if cfg.openai is None or not cfg.openai.api_key:
                    raise ConfigError(
                        "Backend 'openai' non configuré dans config.yaml (clé API OPENAI_API_KEY manquante)"
                    )
                return OpenAIClient(cfg.openai.api_key, cfg.openai.model)

            case "anthropic":
                if cfg.anthropic is None or not cfg.anthropic.api_key:
                    raise ConfigError(
                        "Backend 'anthropic' non configuré dans config.yaml (clé API ANTHROPIC_API_KEY manquante)"
                    )
                return AnthropicClient(cfg.anthropic.api_key, cfg.anthropic.model)

            case "azure":
                if cfg.azure is None or not cfg.azure.api_key or not cfg.azure.api_base:
                    raise ConfigError(
                        "Backend 'azure' non configuré dans config.yaml (clé API ou endpoint AZURE_OPENAI_KEY/ENDPOINT manquant)"
                    )
                return AzureOpenAIClient(
                    cfg.azure.api_key,
                    cfg.azure.api_base,
                    cfg.azure.model,
                    cfg.azure.api_version,
                )

            case "gemini":
                if cfg.gemini is None or not cfg.gemini.api_key:
                    raise ConfigError(
                        "Backend 'gemini' non configuré dans config.yaml (clé API GEMINI_API_KEY manquante)"
                    )
                return GeminiClient(cfg.gemini.api_key, cfg.gemini.model)

            case _:
                raise ConfigError(f"Backend '{name}' inconnu ou non encore implémenté")

    async def validate(self, backend: str | None = None) -> bool:
        """Vérifie si le backend spécifié (ou actif) est joignable."""
        client = self._get_client(backend)
        return await client.validate()

    async def stream(
        self,
        messages: list[dict],
        backend: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        client = self._get_client(backend)
        async for chunk in client.stream(messages, **kwargs):
            yield chunk
        self._last_usage = client.last_usage

    def switch(self, backend: str, model: str | None = None) -> None:
        self._active_backend = backend
        if model:
            self._active_model = model

    @property
    def active_backend(self) -> str:
        return self._active_backend

    @property
    def active_model(self) -> str | None:
        return self._active_model

    @property
    def last_usage(self) -> dict | None:
        return self._last_usage
