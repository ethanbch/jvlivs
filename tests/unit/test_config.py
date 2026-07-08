import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml
from pydantic import ValidationError

from jvl.core.config import (
    Config,
    _deep_merge,
    _resolve_env_vars,
    load_config,
)

# --- _resolve_env_vars ---


def test_resolve_env_vars_string(monkeypatch):
    monkeypatch.setenv("MY_KEY", "hello")
    assert _resolve_env_vars("${MY_KEY}") == "hello"


def test_resolve_env_vars_missing_returns_none():
    assert _resolve_env_vars("${DOES_NOT_EXIST_XYZ}") is None


def test_resolve_env_vars_nested(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    data = {"backends": {"openai": {"api_key": "${API_KEY}"}}}
    result = _resolve_env_vars(data)
    assert result["backends"]["openai"]["api_key"] == "secret"


def test_resolve_env_vars_passthrough():
    assert _resolve_env_vars("plain_string") == "plain_string"
    assert _resolve_env_vars(42) == 42
    assert _resolve_env_vars(None) is None


# --- _deep_merge ---


def test_deep_merge_override_scalar():
    base = {"default_backend": "ollama", "x": 1}
    override = {"default_backend": "openai"}
    result = _deep_merge(base, override)
    assert result["default_backend"] == "openai"
    assert result["x"] == 1


def test_deep_merge_nested():
    base = {"backends": {"ollama": {"model": "phi3", "base_url": "http://localhost"}}}
    override = {"backends": {"ollama": {"model": "llama3"}}}
    result = _deep_merge(base, override)
    assert result["backends"]["ollama"]["model"] == "llama3"
    assert result["backends"]["ollama"]["base_url"] == "http://localhost"


def test_deep_merge_adds_new_key():
    base = {"a": 1}
    override = {"b": 2}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": 2}


# --- Config validation ---


def test_config_valid(full_config_data):
    config = Config(**full_config_data)
    assert config.default_backend == "ollama"
    assert config.backends.ollama.model == "phi3:3.8b"
    assert config.backends.openai.api_key == "sk-test"


def test_config_invalid_backend():
    with pytest.raises(ValidationError, match="inconnu"):
        Config(
            default_backend="unknown_backend",
            backends={"ollama": {"model": "phi3", "base_url": "http://localhost"}},
        )


def test_config_missing_backends_are_none(minimal_config_data):
    config = Config(**minimal_config_data)
    assert config.backends.openai is None
    assert config.backends.anthropic is None
    assert config.backends.gemini is None


# --- load_config ---


def test_load_config_no_local(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        "default_backend: openai\nbackends:\n  openai:\n    model: gpt-4o\n    api_key: ${OPENAI_API_KEY}\n"
    )
    config = load_config(config_path=config_yaml, local_path=tmp_path / "nope.yaml")
    assert config.default_backend == "openai"
    assert config.backends.openai.api_key == "sk-x"


def test_load_config_with_local_override(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        "default_backend: ollama\nbackends:\n  ollama:\n    model: phi3\n    base_url: http://localhost\n  openai:\n    model: gpt-4o\n    api_key: ${OPENAI_API_KEY}\n"
    )
    local_yaml = tmp_path / "config.local.yaml"
    local_yaml.write_text("default_backend: openai\n")

    config = load_config(config_path=config_yaml, local_path=local_yaml)
    assert config.default_backend == "openai"
