from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
    import json
    session_file = tmp_path / "session_notty"
    monkeypatch.setattr("jvl.cli.model.SESSION_DIR", tmp_path)
    with patch("jvl.core.session.session_file_for_tty", return_value=session_file):
        with patch("jvl.cli.model.session_file_for_tty", return_value=session_file):
            result = runner.invoke(app, ["model", "gemini"])
            assert result.exit_code == 0
            data = json.loads(session_file.read_text())
            assert data["provider"] == "gemini"


def test_model_reset(tmp_path, monkeypatch):
    session_file = tmp_path / "session_notty"
    session_file.write_text("gemini")
    with patch("jvl.core.session.session_file_for_tty", return_value=session_file):
        with patch("jvl.cli.model.session_file_for_tty", return_value=session_file):
            result = runner.invoke(app, ["model", "--reset"])
            assert result.exit_code == 0
            assert not session_file.exists()


def test_model_invalid_backend(tmp_path, monkeypatch):
    monkeypatch.setattr("jvl.cli.model.SESSION_DIR", tmp_path)
    result = runner.invoke(app, ["model", "unknown_xyz"])
    assert result.exit_code == 1
    assert "inconnu" in result.output


def test_model_sets_session_interactive(tmp_path, monkeypatch):
    import json
    session_file = tmp_path / "session_notty"
    monkeypatch.setattr("jvl.cli.model.SESSION_DIR", tmp_path)
    with patch("jvl.core.session.session_file_for_tty", return_value=session_file):
        with patch("jvl.cli.model.session_file_for_tty", return_value=session_file):
            with patch("jvl.cli.model.pick_provider_and_model", return_value=("openai", None)):
                result = runner.invoke(app, ["model"])
                assert result.exit_code == 0
                data = json.loads(session_file.read_text())
                assert data["provider"] == "openai"


# ── jvl ask ───────────────────────────────────────────────────


def test_ask_no_stdin():
    from jvl.cli.ask import ask

    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    with (
        patch("sys.stdin", mock_stdin),
        patch("jvl.cli.ask._stream_response", new_callable=AsyncMock) as mock_stream,
    ):
        ask(prompt="my prompt", system=None)
        mock_stream.assert_called_once()
        messages = mock_stream.call_args[0][0]
        assert messages == [{"role": "user", "content": "my prompt"}]


def test_ask_with_stdin_empty(capsys):
    from jvl.cli.ask import ask

    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = False
    mock_stdin.read.side_effect = ["   ", ""]
    with (
        patch("sys.stdin", mock_stdin),
        patch("jvl.cli.ask._stream_response", new_callable=AsyncMock) as mock_stream,
    ):
        ask(prompt="my prompt", system=None)
        mock_stream.assert_called_once()
        messages = mock_stream.call_args[0][0]
        assert messages == [{"role": "user", "content": "my prompt"}]
        captured = capsys.readouterr()
        assert "Attention : l'entrée standard (stdin) est vide" in captured.out


def test_ask_with_stdin_completely_empty(capsys):
    from jvl.cli.ask import ask

    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = False
    mock_stdin.read.side_effect = ["", ""]
    with (
        patch("sys.stdin", mock_stdin),
        patch("jvl.cli.ask._stream_response", new_callable=AsyncMock) as mock_stream,
    ):
        ask(prompt="my prompt", system=None)
        mock_stream.assert_called_once()
        messages = mock_stream.call_args[0][0]
        assert messages == [{"role": "user", "content": "my prompt"}]
        captured = capsys.readouterr()
        assert "Attention : l'entrée standard (stdin) est vide" in captured.out


def test_ask_with_stdin_data():
    from jvl.cli.ask import ask

    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = False
    mock_stdin.read.side_effect = ["some file content\nline 2", ""]
    with (
        patch("sys.stdin", mock_stdin),
        patch("jvl.cli.ask._stream_response", new_callable=AsyncMock) as mock_stream,
    ):
        ask(prompt="my prompt", system=None)
        mock_stream.assert_called_once()
        messages = mock_stream.call_args[0][0]
        assert messages == [
            {
                "role": "user",
                "content": "my prompt\n\nContexte fourni via stdin :\nsome file content\nline 2\n",
            }
        ]


def test_ask_with_system_and_stdin():
    from jvl.cli.ask import ask

    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = False
    mock_stdin.read.side_effect = ["some code", ""]
    with (
        patch("sys.stdin", mock_stdin),
        patch("jvl.cli.ask._stream_response", new_callable=AsyncMock) as mock_stream,
    ):
        ask(prompt="my prompt", system="sys prompt")
        mock_stream.assert_called_once()
        messages = mock_stream.call_args[0][0]
        assert messages == [
            {"role": "system", "content": "sys prompt"},
            {
                "role": "user",
                "content": "my prompt\n\nContexte fourni via stdin :\nsome code\n",
            },
        ]


@pytest.mark.asyncio
async def test_stream_response_debug_prints_stdin(capsys):
    from jvl.cli.ask import _stream_response

    mock_config = MagicMock()
    mock_router = MagicMock()
    mock_router.validate = AsyncMock(return_value=True)
    mock_client = AsyncMock()
    mock_client.validate.return_value = True
    mock_router._get_client.return_value = mock_client

    async def mock_stream_with_thinking(*args, **kwargs):
        yield "content", "chunk 1"
        yield "content", "chunk 2"

    mock_router.stream_with_thinking = mock_stream_with_thinking
    mock_router.active_backend = "openai"
    mock_router.last_usage = {"prompt_tokens": 10, "completion_tokens": 20}

    with (
        patch("jvl.cli.ask.load_config", return_value=mock_config),
        patch("jvl.cli.ask.BackendRouter", return_value=mock_router),
    ):

        await _stream_response(
            messages=[{"role": "user", "content": "hello"}],
            backend="openai",
            temperature=0.7,
            max_tokens=100,
            no_markdown=True,
            debug=True,
            stdin_content="my debugged stdin content",
        )

    captured = capsys.readouterr()
    assert "stdin" in captured.out
    assert "my debugged stdin content" in captured.out
