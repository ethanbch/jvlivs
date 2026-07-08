"""Tests pour les commandes `jvl config`."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from jvl.cli.main import app
from jvl.core.config import ProviderConfig, UserConfig, save_user_config

runner = CliRunner()


@pytest.fixture
def user_config_file(tmp_path, monkeypatch):
    """Redirige la user config vers un fichier temporaire."""
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("jvl.cli.config.load_user_config", lambda: _load(config_path))
    monkeypatch.setattr(
        "jvl.cli.config.save_user_config",
        lambda cfg: _save(cfg, config_path),
    )
    return config_path


def _load(path: Path) -> UserConfig:
    from jvl.core.config import load_user_config
    return load_user_config(config_path=path)


def _save(cfg: UserConfig, path: Path) -> None:
    save_user_config(cfg, config_path=path)


# ── jvl config provider ──────────────────────────────────────────────────────


def test_config_provider_add(user_config_file):
    result = runner.invoke(app, [
        "config", "provider", "add", "openai",
        "--api-key", "sk-test",
        "--default-model", "gpt-4o",
    ])
    assert result.exit_code == 0
    assert "configuré" in result.output

    data = yaml.safe_load(user_config_file.read_text())
    assert data["providers"]["openai"]["api_key"] == "sk-test"
    assert data["providers"]["openai"]["default_model"] == "gpt-4o"


def test_config_provider_add_unknown(user_config_file):
    result = runner.invoke(app, ["config", "provider", "add", "unknown_xyz"])
    assert result.exit_code == 1
    assert "inconnu" in result.output


def test_config_provider_list(user_config_file):
    # Seed some providers
    cfg = UserConfig(
        active_provider="openai",
        providers={
            "openai": ProviderConfig(api_key="sk-test", default_model="gpt-4o"),
            "ollama": ProviderConfig(base_url="http://localhost:11434", default_model="llama3.2:3b"),
        },
    )
    _save(cfg, user_config_file)

    result = runner.invoke(app, ["config", "provider", "list"])
    assert result.exit_code == 0
    assert "openai" in result.output
    assert "ollama" in result.output


def test_config_provider_remove(user_config_file):
    cfg = UserConfig(
        active_provider="ollama",
        providers={
            "openai": ProviderConfig(api_key="sk-test"),
        },
    )
    _save(cfg, user_config_file)

    result = runner.invoke(app, ["config", "provider", "remove", "openai"])
    assert result.exit_code == 0
    assert "supprimé" in result.output

    data = yaml.safe_load(user_config_file.read_text())
    assert "openai" not in data.get("providers", {})


def test_config_provider_remove_not_found(user_config_file):
    result = runner.invoke(app, ["config", "provider", "remove", "openai"])
    assert result.exit_code == 1


# ── jvl config model ─────────────────────────────────────────────────────────


def test_config_model_set(user_config_file):
    result = runner.invoke(app, ["config", "model", "set", "ollama", "mistral:7b"])
    assert result.exit_code == 0
    assert "mistral:7b" in result.output

    data = yaml.safe_load(user_config_file.read_text())
    assert data["providers"]["ollama"]["default_model"] == "mistral:7b"


def test_config_model_use(user_config_file):
    result = runner.invoke(app, ["config", "model", "use", "anthropic", "claude-sonnet-4-5"])
    assert result.exit_code == 0
    assert "anthropic" in result.output

    data = yaml.safe_load(user_config_file.read_text())
    assert data["active_provider"] == "anthropic"
    assert data["active_model"] == "claude-sonnet-4-5"


def test_config_model_use_unknown_provider(user_config_file):
    result = runner.invoke(app, ["config", "model", "use", "unknown_xyz"])
    assert result.exit_code == 1
    assert "inconnu" in result.output


def test_config_model_show(user_config_file):
    cfg = UserConfig(
        active_provider="openai",
        active_model="gpt-4-turbo",
        providers={"openai": ProviderConfig(api_key="sk-test")},
    )
    _save(cfg, user_config_file)

    result = runner.invoke(app, ["config", "model", "show"])
    assert result.exit_code == 0
    assert "openai" in result.output
    assert "gpt-4-turbo" in result.output


# ── jvl config key ────────────────────────────────────────────────────────────


def test_config_key_set_value(user_config_file):
    result = runner.invoke(app, ["config", "key", "set", "openai", "sk-xxx"])
    assert result.exit_code == 0
    assert "définie" in result.output

    data = yaml.safe_load(user_config_file.read_text())
    assert data["providers"]["openai"]["api_key"] == "sk-xxx"


def test_config_key_set_env(user_config_file):
    result = runner.invoke(app, ["config", "key", "set", "openai", "--env", "OPENAI_API_KEY"])
    assert result.exit_code == 0
    assert "OPENAI_API_KEY" in result.output

    data = yaml.safe_load(user_config_file.read_text())
    assert data["providers"]["openai"]["api_key"] == "${OPENAI_API_KEY}"


def test_config_key_set_missing_value(user_config_file):
    result = runner.invoke(app, ["config", "key", "set", "openai"])
    assert result.exit_code == 1


def test_config_key_remove(user_config_file):
    cfg = UserConfig(
        providers={"openai": ProviderConfig(api_key="sk-test")},
    )
    _save(cfg, user_config_file)

    result = runner.invoke(app, ["config", "key", "remove", "openai"])
    assert result.exit_code == 0
    assert "supprimée" in result.output


def test_config_key_remove_not_found(user_config_file):
    result = runner.invoke(app, ["config", "key", "remove", "openai"])
    assert result.exit_code == 1


# ── jvl config migrate ───────────────────────────────────────────────────────


def test_config_migrate(user_config_file, tmp_path, monkeypatch):
    # Create a repo config to migrate from
    repo_config_path = tmp_path / "repo_config.yaml"
    repo_config_path.write_text(
        "default_backend: ollama\n"
        "backends:\n"
        "  ollama:\n"
        "    model: llama3.2:3b\n"
        "    base_url: http://localhost:11434\n"
    )
    from jvl.core.config import Config, load_config

    # Monkey-patch load_config in the config CLI module
    repo_cfg = load_config(config_path=repo_config_path, local_path=tmp_path / "nope.yaml")
    monkeypatch.setattr("jvl.cli.config.load_config", lambda: repo_cfg)

    result = runner.invoke(app, ["config", "migrate"])
    assert result.exit_code == 0
    assert "ollama" in result.output
