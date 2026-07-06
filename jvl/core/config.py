from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator


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
