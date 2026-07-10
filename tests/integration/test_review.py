"""Tests pour la commande `jvl review`."""
from __future__ import annotations

from unittest.mock import patch, AsyncMock

import pytest
from typer.testing import CliRunner

from jvl.cli.main import app

runner = CliRunner()


@pytest.fixture
def temp_code_file(tmp_path):
    """Crée un fichier temporaire contenant du code simple à reviewer."""
    f = tmp_path / "simple_code.py"
    f.write_text(
        "def add(a, b):\n"
        "    # Code simple\n"
        "    return a + b\n"
    )
    return f


@patch("jvl.cli.review.BackendRouter")
@patch("jvl.cli.review.get_db")
def test_review_file_argument(mock_get_db, mock_router_cls, temp_code_file, db):
    mock_get_db.return_value = db
    
    # Mocking BackendRouter
    mock_router = mock_router_cls.return_value
    mock_router.active_backend = "openai"
    mock_router.active_model = "gpt-4o"
    mock_router.validate = AsyncMock(return_value=True)
    
    # Mocking router.stream_with_thinking
    async def mock_stream_with_thinking(*args, **kwargs):
        yield "content", "### Revue de code\n"
        yield "content", "- Score : 9/10\n"
    mock_router.stream_with_thinking = mock_stream_with_thinking
    mock_router.last_usage = {"prompt_tokens": 10, "completion_tokens": 5}

    result = runner.invoke(app, ["review", str(temp_code_file)])
    assert result.exit_code == 0
    assert "Revue de code" in result.output
    assert "Score : 9/10" in result.output

    # Vérifier que la session a été enregistrée en base
    from jvl.db.models import ChatSession
    sessions = db.query(ChatSession).all()
    assert len(sessions) == 1
    assert sessions[0].backend == "openai"
    assert sessions[0].model == "gpt-4o"
    assert len(sessions[0].messages) == 3  # system prompt + user code + assistant review


def test_review_no_args_no_stdin():
    result = runner.invoke(app, ["review"], env={"COLUMNS": "120"})
    assert result.exit_code == 1
    assert "Vous devez spécifier un fichier" in result.output


def test_review_nonexistent_file():
    result = runner.invoke(app, ["review", "nonexistent_file.py"], env={"COLUMNS": "120"})
    assert result.exit_code == 1
    assert "n'existe pas" in result.output


@patch("jvl.cli.review.BackendRouter")
@patch("jvl.cli.review.get_db")
def test_review_stdin(mock_get_db, mock_router_cls, db):
    mock_get_db.return_value = db
    
    mock_router = mock_router_cls.return_value
    mock_router.active_backend = "ollama"
    mock_router.active_model = "llama3"
    mock_router.validate = AsyncMock(return_value=True)
    
    async def mock_stream_with_thinking(*args, **kwargs):
        yield "content", "Revue Stdin OK"
    mock_router.stream_with_thinking = mock_stream_with_thinking

    result = runner.invoke(app, ["review"], input="def sub(a, b): return a - b\n")
    assert result.exit_code == 0
    assert "Revue Stdin OK" in result.output


@patch("jvl.cli.review.BackendRouter")
@patch("jvl.cli.review.get_db")
def test_review_custom_prompt_file(mock_get_db, mock_router_cls, db, tmp_path, monkeypatch):
    mock_get_db.return_value = db
    
    # Path patch prompts/review.md
    custom_prompt_file = tmp_path / "custom_review.md"
    custom_prompt_file.write_text("CUSTOM REVIEW SYSTEM INSTRUCTION")
    monkeypatch.setattr("jvl.cli.review.DEFAULT_REVIEW_PROMPT_PATH", custom_prompt_file)
    
    mock_router = mock_router_cls.return_value
    mock_router.active_backend = "openai"
    mock_router.active_model = "gpt-4o"
    mock_router.validate = AsyncMock(return_value=True)
    
    async def mock_stream_with_thinking(*args, **kwargs):
        yield "content", "OK"
    mock_router.stream_with_thinking = mock_stream_with_thinking

    result = runner.invoke(app, ["review"], input="x = 1\n")
    assert result.exit_code == 0
    
    # Verify DB session prompt is custom
    from jvl.db.models import ChatSession
    sessions = db.query(ChatSession).all()
    assert len(sessions) == 1
    assert sessions[0].system_prompt == "CUSTOM REVIEW SYSTEM INSTRUCTION"


def test_review_missing_prompt_file(tmp_path, monkeypatch):
    missing_file = tmp_path / "nonexistent_review.md"
    monkeypatch.setattr("jvl.cli.review.DEFAULT_REVIEW_PROMPT_PATH", missing_file)
    
    result = runner.invoke(app, ["review"], input="x = 1\n", env={"COLUMNS": "120"})
    assert result.exit_code == 1
    assert "introuvable" in result.output


def test_review_empty_prompt_file(tmp_path, monkeypatch):
    empty_file = tmp_path / "empty_review.md"
    empty_file.write_text("")
    monkeypatch.setattr("jvl.cli.review.DEFAULT_REVIEW_PROMPT_PATH", empty_file)
    
    result = runner.invoke(app, ["review"], input="x = 1\n", env={"COLUMNS": "120"})
    assert result.exit_code == 1
    assert "vide" in result.output


