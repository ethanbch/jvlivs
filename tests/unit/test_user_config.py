"""Tests pour UserConfig, ProviderConfig, load/save_user_config, resolve_active_config."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jvl.core.config import (
    Config,
    ProviderConfig,
    UserConfig,
    _resolve_env_vars,
    load_user_config,
    resolve_active_config,
    save_user_config,
)
from jvl.utils.errors import ConfigError


# ── ProviderConfig ────────────────────────────────────────────────────────────


def test_provider_config_defaults():
    cfg = ProviderConfig()
    assert cfg.api_key is None
    assert cfg.base_url is None
    assert cfg.api_base is None
    assert cfg.api_version is None
    assert cfg.default_model is None


def test_provider_config_with_values():
    cfg = ProviderConfig(
        api_key="sk-test",
        base_url="http://localhost:11434",
        default_model="llama3.2:3b",
    )
    assert cfg.api_key == "sk-test"
    assert cfg.base_url == "http://localhost:11434"
    assert cfg.default_model == "llama3.2:3b"


# ── UserConfig ────────────────────────────────────────────────────────────────


def test_user_config_defaults():
    cfg = UserConfig()
    assert cfg.active_provider == "ollama"
    assert cfg.active_model is None
    assert cfg.providers == {}


def test_user_config_with_providers():
    cfg = UserConfig(
        active_provider="openai",
        active_model="gpt-4o",
        providers={
            "openai": ProviderConfig(api_key="sk-test", default_model="gpt-4o"),
            "ollama": ProviderConfig(base_url="http://localhost:11434"),
        },
    )
    assert cfg.active_provider == "openai"
    assert cfg.active_model == "gpt-4o"
    assert len(cfg.providers) == 2
    assert cfg.providers["openai"].api_key == "sk-test"


# ── load_user_config ─────────────────────────────────────────────────────────


def test_load_user_config_file_not_exists(tmp_path):
    cfg = load_user_config(config_path=tmp_path / "nonexistent.yaml")
    assert cfg == UserConfig()


def test_load_user_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "active_provider: openai\n"
        "active_model: gpt-4o\n"
        "providers:\n"
        "  openai:\n"
        "    api_key: sk-test\n"
        "    default_model: gpt-4o\n"
    )
    cfg = load_user_config(config_path=config_file)
    assert cfg.active_provider == "openai"
    assert cfg.active_model == "gpt-4o"
    assert cfg.providers["openai"].api_key == "sk-test"


def test_load_user_config_resolves_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "sk-resolved")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "active_provider: openai\n"
        "providers:\n"
        "  openai:\n"
        "    api_key: ${MY_API_KEY}\n"
        "    default_model: gpt-4o\n"
    )
    cfg = load_user_config(config_path=config_file)
    assert cfg.providers["openai"].api_key == "sk-resolved"


def test_load_user_config_empty_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")
    cfg = load_user_config(config_path=config_file)
    assert cfg == UserConfig()


# ── save_user_config ─────────────────────────────────────────────────────────


def test_save_user_config_creates_file(tmp_path):
    config_file = tmp_path / "subdir" / "config.yaml"
    cfg = UserConfig(
        active_provider="anthropic",
        active_model="claude-sonnet-4-5",
        providers={
            "anthropic": ProviderConfig(
                api_key="ant-test", default_model="claude-sonnet-4-5"
            ),
        },
    )
    save_user_config(cfg, config_path=config_file)
    assert config_file.exists()

    data = yaml.safe_load(config_file.read_text())
    assert data["active_provider"] == "anthropic"
    assert data["active_model"] == "claude-sonnet-4-5"
    assert data["providers"]["anthropic"]["api_key"] == "ant-test"


def test_save_then_load_roundtrip(tmp_path):
    config_file = tmp_path / "config.yaml"
    original = UserConfig(
        active_provider="ollama",
        active_model="mistral:7b",
        providers={
            "ollama": ProviderConfig(
                base_url="http://localhost:11434",
                default_model="llama3.2:3b",
            ),
        },
    )
    save_user_config(original, config_path=config_file)
    loaded = load_user_config(config_path=config_file)
    assert loaded.active_provider == original.active_provider
    assert loaded.active_model == original.active_model
    assert loaded.providers["ollama"].base_url == "http://localhost:11434"
    assert loaded.providers["ollama"].default_model == "llama3.2:3b"


# ── resolve_active_config ────────────────────────────────────────────────────


def test_resolve_uses_active_model():
    user_cfg = UserConfig(
        active_provider="openai",
        active_model="gpt-4-turbo",
        providers={"openai": ProviderConfig(api_key="sk-test", default_model="gpt-4o")},
    )
    name, model, cfg = resolve_active_config(user_cfg)
    assert name == "openai"
    assert model == "gpt-4-turbo"  # active_model a priorité sur default_model


def test_resolve_falls_back_to_default_model():
    user_cfg = UserConfig(
        active_provider="openai",
        providers={"openai": ProviderConfig(api_key="sk-test", default_model="gpt-4o")},
    )
    name, model, cfg = resolve_active_config(user_cfg)
    assert model == "gpt-4o"


def test_resolve_falls_back_to_repo_config():
    user_cfg = UserConfig(active_provider="ollama")
    repo_cfg = Config(
        default_backend="ollama",
        backends={"ollama": {"model": "phi3:3.8b", "base_url": "http://localhost:11434"}},
    )
    name, model, cfg = resolve_active_config(user_cfg, repo_config=repo_cfg)
    assert name == "ollama"
    assert model == "phi3:3.8b"


def test_resolve_raises_when_no_model():
    user_cfg = UserConfig(active_provider="ollama")
    with pytest.raises(ConfigError, match="Aucun modèle résolu"):
        resolve_active_config(user_cfg)


def test_resolve_builds_provider_config_from_repo():
    user_cfg = UserConfig(active_provider="openai")
    repo_cfg = Config(
        default_backend="openai",
        backends={
            "openai": {"model": "gpt-4o", "api_key": "sk-repo"},
        },
    )
    name, model, cfg = resolve_active_config(user_cfg, repo_config=repo_cfg)
    assert cfg.api_key == "sk-repo"
    assert cfg.default_model == "gpt-4o"
