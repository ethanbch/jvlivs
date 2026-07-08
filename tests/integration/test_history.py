"""Tests pour la commande `jvl history`."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from jvl.cli.main import app
from jvl.db.session import create_chat_session, add_message

runner = CliRunner()


@pytest.fixture
def populated_db(db):
    """Prépare une base avec plusieurs sessions et messages pour les tests de history."""
    # Session 1: Ollama - Contient "RAG"
    s1 = create_chat_session(db, backend="ollama", model="llama3.2:3b")
    add_message(db, s1.id, "user", "Qu'est-ce que le RAG ?")
    add_message(db, s1.id, "assistant", "Le RAG est la génération augmentée de récupération.")

    # Session 2: OpenAI - Contient "FastAPI"
    s2 = create_chat_session(db, backend="openai", model="gpt-4o")
    add_message(db, s2.id, "user", "Comment utiliser FastAPI ?")
    add_message(db, s2.id, "assistant", "FastAPI est très rapide.")

    # Session 3: Ollama - Contient "Docker"
    s3 = create_chat_session(db, backend="ollama", model="phi3:3.8b")
    add_message(db, s3.id, "user", "Explique Docker.")
    add_message(db, s3.id, "assistant", "Docker isole les conteneurs.")

    return s1, s2, s3


@patch("jvl.cli.history.get_db")
def test_history_list_default(mock_get_db, populated_db, db):
    mock_get_db.return_value = db
    s1, s2, s3 = populated_db
    s1_id, s2_id, s3_id = s1.id, s2.id, s3.id

    result = runner.invoke(app, ["history"], env={"COLUMNS": "120"})
    assert result.exit_code == 0
    assert "Historique des 3 dernières sessions" in result.output
    assert s1_id in result.output
    assert s2_id in result.output
    assert s3_id in result.output


@patch("jvl.cli.history.get_db")
def test_history_list_limit(mock_get_db, populated_db, db):
    mock_get_db.return_value = db

    result = runner.invoke(app, ["history", "--limit", "2"], env={"COLUMNS": "120"})
    assert result.exit_code == 0
    assert "Historique des 2 dernières sessions" in result.output


@patch("jvl.cli.history.get_db")
def test_history_filter_backend(mock_get_db, populated_db, db):
    mock_get_db.return_value = db
    s1, s2, s3 = populated_db

    result = runner.invoke(app, ["history", "--backend", "openai"], env={"COLUMNS": "120"})
    assert result.exit_code == 0
    assert "openai / gpt-4o" in result.output
    assert "ollama" not in result.output


@patch("jvl.cli.history.get_db")
def test_history_search_content(mock_get_db, populated_db, db):
    mock_get_db.return_value = db
    s1, s2, s3 = populated_db
    s1_id, s2_id, s3_id = s1.id, s2.id, s3.id

    # Recherche "RAG"
    result = runner.invoke(app, ["history", "--search", "RAG"], env={"COLUMNS": "120"})
    assert result.exit_code == 0
    assert s1_id in result.output
    assert s2_id not in result.output
    assert s3_id not in result.output


@patch("jvl.cli.history.get_db")
def test_history_show_session_details(mock_get_db, populated_db, db):
    mock_get_db.return_value = db
    s1, s2, s3 = populated_db
    s1_id = s1.id

    result = runner.invoke(app, ["history", s1_id], env={"COLUMNS": "120"})
    assert result.exit_code == 0
    assert f"Détails de la Session : {s1_id}" in result.output
    assert "Qu'est-ce que le RAG ?" in result.output
    assert "Le RAG est la génération augmentée" in result.output


@patch("jvl.cli.history.get_db")
def test_history_show_nonexistent_session(mock_get_db, db):
    mock_get_db.return_value = db

    result = runner.invoke(app, ["history", "nonexistent-uuid"], env={"COLUMNS": "120"})
    assert result.exit_code == 1
    assert "introuvable" in result.output
