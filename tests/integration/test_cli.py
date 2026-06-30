from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from jvl.cli.main import app

runner = CliRunner()


# ── jvl default ───────────────────────────────────────────────


def test_default_sets_backend(tmp_path, monkeypatch):
    local_config = tmp_path / "config.local.yaml"
    monkeypatch.setattr("jvl.cli.default.LOCAL_CONFIG_PATH", local_config)
    result = runner.invoke(app, ["default", "openai"])
    assert result.exit_code == 0
    data = yaml.safe_load(local_config.read_text())
    assert data["default_backend"] == "openai"


def test_default_invalid_backend(tmp_path, monkeypatch):
    monkeypatch.setattr("jvl.cli.default.LOCAL_CONFIG_PATH", tmp_path / "x.yaml")
    result = runner.invoke(app, ["default", "unknown_xyz"])
    assert result.exit_code == 1
    assert "inconnu" in result.output


def test_default_preserves_existing_keys(tmp_path, monkeypatch):
    local_config = tmp_path / "config.local.yaml"
    local_config.write_text("some_key: some_value\n")
    monkeypatch.setattr("jvl.cli.default.LOCAL_CONFIG_PATH", local_config)
    runner.invoke(app, ["default", "gemini"])
    data = yaml.safe_load(local_config.read_text())
    assert data["default_backend"] == "gemini"
    assert data["some_key"] == "some_value"


# ── jvl model ─────────────────────────────────────────────────


def test_model_sets_session(tmp_path, monkeypatch):
    session_file = tmp_path / "session_notty"
    monkeypatch.setattr("jvl.cli.model.SESSION_DIR", tmp_path)
    with patch("jvl.cli.model._session_file_for_tty", return_value=session_file):
        result = runner.invoke(app, ["model", "gemini"])
        assert result.exit_code == 0
        assert session_file.read_text() == "gemini"


def test_model_reset(tmp_path, monkeypatch):
    session_file = tmp_path / "session_notty"
    session_file.write_text("gemini")
    with patch("jvl.cli.model._session_file_for_tty", return_value=session_file):
        result = runner.invoke(app, ["model", "--reset"])
        assert result.exit_code == 0
        assert not session_file.exists()


def test_model_invalid_backend(tmp_path, monkeypatch):
    monkeypatch.setattr("jvl.cli.model.SESSION_DIR", tmp_path)
    result = runner.invoke(app, ["model", "unknown_xyz"])
    assert result.exit_code == 1
    assert "inconnu" in result.output


def test_model_sets_session_interactive(tmp_path, monkeypatch):
    session_file = tmp_path / "session_notty"
    monkeypatch.setattr("jvl.cli.model.SESSION_DIR", tmp_path)
    with patch("jvl.cli.model._session_file_for_tty", return_value=session_file):
        with patch("jvl.cli.model.pick_backend_interactive", return_value="openai"):
            result = runner.invoke(app, ["model"])
            assert result.exit_code == 0
            assert session_file.read_text() == "openai"

