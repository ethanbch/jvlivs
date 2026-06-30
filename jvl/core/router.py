from __future__ import annotations

from typing import AsyncIterator

from jvl.backends.anthropic import AnthropicClient
from jvl.backends.azure import AzureOpenAIClient
from jvl.backends.base import BaseClient
from jvl.backends.gemini import GeminiClient
from jvl.backends.ollama import OllamaClient
from jvl.backends.openai import OpenAIClient
from jvl.cli.model import get_session_backend  # noqa: E402
from jvl.core.config import Config
from jvl.utils.errors import BackendNotAvailable, ConfigError


class BackendRouter:

    def __init__(self, config: Config) -> None:
        self.config = config
        self._clients: dict[str, BaseClient] = {}
        # Priorité : session TTY > config.default_backend
        session_backend = get_session_backend()
        self._active_backend = session_backend or config.default_backend
        self._last_usage: dict | None = None

    def _get_client(self, backend: str | None = None) -> BaseClient:
        name = backend or self._active_backend
        if name not in self._clients:
            self._clients[name] = self._build_client(name)
        return self._clients[name]

    def _build_client(self, name: str) -> BaseClient:
        cfg = self.config.backends
        match name:
            case "ollama":
                if cfg.ollama is None:
                    raise ConfigError("Backend 'ollama' non configuré dans config.yaml")
                return OllamaClient(cfg.ollama.base_url, cfg.ollama.model)

            case "openai":
                if cfg.openai is None:
                    raise ConfigError("Backend 'openai' non configuré dans config.yaml")
                return OpenAIClient(cfg.openai.api_key, cfg.openai.model)

            case "anthropic":
                if cfg.anthropic is None:
                    raise ConfigError(
                        "Backend 'anthropic' non configuré dans config.yaml"
                    )
                return AnthropicClient(cfg.anthropic.api_key, cfg.anthropic.model)

            case "azure":
                if cfg.azure is None:
                    raise ConfigError("Backend 'azure' non configuré dans config.yaml")
                return AzureOpenAIClient(
                    cfg.azure.api_key,
                    cfg.azure.api_base,
                    cfg.azure.model,
                    cfg.azure.api_version,
                )

            case "gemini":
                if cfg.gemini is None:
                    raise ConfigError("Backend 'gemini' non configuré dans config.yaml")
                return GeminiClient(cfg.gemini.api_key, cfg.gemini.model)

            case _:
                raise ConfigError(f"Backend '{name}' inconnu ou non encore implémenté")

    async def stream(
        self,
        messages: list[dict],
        backend: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        client = self._get_client(backend)
        available = await client.validate()
        if not available:
            raise BackendNotAvailable(
                f"Backend '{backend or self._active_backend}' non joignable."
            )
        async for chunk in client.stream(messages, **kwargs):
            yield chunk
        self._last_usage = client.last_usage

    def switch(self, backend: str) -> None:
        self._active_backend = backend

    @property
    def active_backend(self) -> str:
        return self._active_backend

    @property
    def last_usage(self) -> dict | None:
        return self._last_usage
