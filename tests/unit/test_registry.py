"""Tests pour PROVIDER_REGISTRY et build_client_from_provider_config."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from jvl.backends.anthropic import AnthropicClient
from jvl.backends.azure import AzureOpenAIClient
from jvl.backends.gemini import GeminiClient
from jvl.backends.ollama import OllamaClient
from jvl.backends.openai import OpenAIClient
from jvl.core.config import ProviderConfig
from jvl.core.router import PROVIDER_REGISTRY, build_client_from_provider_config
from jvl.utils.errors import ConfigError


# ── PROVIDER_REGISTRY ─────────────────────────────────────────────────────────


def test_registry_contains_all_providers():
    expected = {"ollama", "openai", "anthropic", "azure", "gemini"}
    assert set(PROVIDER_REGISTRY.keys()) == expected


def test_registry_maps_to_correct_classes():
    assert PROVIDER_REGISTRY["ollama"] is OllamaClient
    assert PROVIDER_REGISTRY["openai"] is OpenAIClient
    assert PROVIDER_REGISTRY["anthropic"] is AnthropicClient
    assert PROVIDER_REGISTRY["azure"] is AzureOpenAIClient
    assert PROVIDER_REGISTRY["gemini"] is GeminiClient


# ── build_client_from_provider_config ─────────────────────────────────────────


def test_build_ollama_client():
    cfg = ProviderConfig(base_url="http://localhost:11434")
    client = build_client_from_provider_config("ollama", cfg, "llama3.2:3b")
    assert isinstance(client, OllamaClient)
    assert client.model == "llama3.2:3b"
    assert client.base_url == "http://localhost:11434"


def test_build_ollama_client_default_url():
    cfg = ProviderConfig()
    client = build_client_from_provider_config("ollama", cfg, "phi3")
    assert isinstance(client, OllamaClient)
    assert client.base_url == "http://localhost:11434"


def test_build_openai_client():
    cfg = ProviderConfig(api_key="sk-test")
    client = build_client_from_provider_config("openai", cfg, "gpt-4o")
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o"


def test_build_openai_missing_key():
    cfg = ProviderConfig()
    with pytest.raises(ConfigError, match="clé API manquante"):
        build_client_from_provider_config("openai", cfg, "gpt-4o")


def test_build_anthropic_client():
    cfg = ProviderConfig(api_key="ant-test")
    client = build_client_from_provider_config("anthropic", cfg, "claude-sonnet-4-5")
    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-sonnet-4-5"


def test_build_anthropic_missing_key():
    cfg = ProviderConfig()
    with pytest.raises(ConfigError, match="clé API manquante"):
        build_client_from_provider_config("anthropic", cfg, "claude-sonnet-4-5")


def test_build_azure_client():
    cfg = ProviderConfig(
        api_key="az-test",
        api_base="https://test.openai.azure.com",
        api_version="2024-02-01",
    )
    client = build_client_from_provider_config("azure", cfg, "gpt-4o")
    assert isinstance(client, AzureOpenAIClient)
    assert client.model == "gpt-4o"


def test_build_azure_missing_key():
    cfg = ProviderConfig(api_base="https://test.openai.azure.com")
    with pytest.raises(ConfigError, match="clé API ou endpoint manquant"):
        build_client_from_provider_config("azure", cfg, "gpt-4o")


def test_build_azure_missing_endpoint():
    cfg = ProviderConfig(api_key="az-test")
    with pytest.raises(ConfigError, match="clé API ou endpoint manquant"):
        build_client_from_provider_config("azure", cfg, "gpt-4o")


def test_build_gemini_client():
    with patch("jvl.backends.gemini.genai.Client"):
        cfg = ProviderConfig(api_key="gem-test")
        client = build_client_from_provider_config("gemini", cfg, "gemini-2.5-flash")
        assert isinstance(client, GeminiClient)
        assert client.model == "gemini-2.5-flash"


def test_build_gemini_missing_key():
    cfg = ProviderConfig()
    with pytest.raises(ConfigError, match="clé API manquante"):
        build_client_from_provider_config("gemini", cfg, "gemini-2.5-flash")


def test_build_unknown_provider():
    cfg = ProviderConfig()
    with pytest.raises(ConfigError, match="inconnu"):
        build_client_from_provider_config("unknown_xyz", cfg, "some-model")
