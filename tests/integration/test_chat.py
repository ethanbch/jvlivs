import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from jvl.cli.main import app
from jvl.db import get_db
from jvl.db.session import get_session_messages, get_session_usage

runner = CliRunner()


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.validate = AsyncMock(return_value=True)
    client = AsyncMock()
    client.validate.return_value = True
    router._get_client.return_value = client
    router.active_backend = "gemini"

    async def mock_stream_with_thinking(*args, **kwargs):
        yield "content", "Hello! "
        yield "content", "How can I help you today?"

    router.stream_with_thinking = mock_stream_with_thinking
    router.last_usage = {"prompt_tokens": 10, "completion_tokens": 15}
    return router


@pytest.fixture
def mock_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from jvl.db.models import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@patch("jvl.cli.chat.get_db")
@patch("jvl.cli.chat.BackendRouter")
def test_chat_starts_and_exits(mock_router_cls, mock_get_db, mock_router, mock_db):
    mock_router_cls.return_value = mock_router
    mock_get_db.return_value = mock_db

    # Simuler Ctrl+C (KeyboardInterrupt) lors du premier prompt
    with patch(
        "prompt_toolkit.PromptSession.prompt_async", new_callable=AsyncMock, side_effect=KeyboardInterrupt
    ):
        result = runner.invoke(app, ["chat", "--backend", "gemini"])
        assert result.exit_code == 0
        assert "JVLIVS Chat" in result.output
        assert "Au revoir !" in result.output


@patch("jvl.cli.chat.get_db")
@patch("jvl.cli.chat.BackendRouter")
def test_chat_slash_help(mock_router_cls, mock_get_db, mock_router, mock_db):
    mock_router_cls.return_value = mock_router
    mock_get_db.return_value = mock_db

    # Envoyer /help puis /exit
    with patch(
        "prompt_toolkit.PromptSession.prompt_async", new_callable=AsyncMock, side_effect=["/help", "/exit"]
    ):
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 0
        assert "/exit" in result.output
        assert "/clear" in result.output
        assert "/model" in result.output


@patch("jvl.cli.chat.get_db")
@patch("jvl.cli.chat.BackendRouter")
def test_chat_conversation_loop(mock_router_cls, mock_get_db, mock_router, mock_db):
    mock_router_cls.return_value = mock_router
    mock_get_db.return_value = mock_db

    # Envoyer un message utilisateur puis /exit
    with patch(
        "prompt_toolkit.PromptSession.prompt_async", new_callable=AsyncMock, side_effect=["hello", "/exit"]
    ):
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 0
        assert "How can I help you today?" in result.output

        # Vérifier que la session a été créée et les messages ont été enregistrés
        # Il y a 1 session dans la DB
        from jvl.db.models import ChatSession
        sessions = mock_db.query(ChatSession).all()
        assert len(sessions) == 1
        session_id = sessions[0].id

        # Récupérer les messages de la session
        msgs = get_session_messages(mock_db, session_id)
        assert len(msgs) == 3  # System, User ("hello"), Assistant ("Hello! How can I help you today?")
        assert msgs[1] == {"role": "user", "content": "hello"}
        assert msgs[2] == {"role": "assistant", "content": "Hello! How can I help you today?"}

        # Vérifier l'usage
        usage = get_session_usage(mock_db, session_id)
        assert usage["total_in"] == 10
        assert usage["total_out"] == 15
        assert usage["message_count"] == 3


@patch("jvl.db.session.uuid.uuid4")
@patch("jvl.cli.chat.get_db")
@patch("jvl.cli.chat.BackendRouter")
def test_chat_resume_session(mock_router_cls, mock_get_db, mock_uuid, mock_router, mock_db):
    mock_router_cls.return_value = mock_router
    mock_get_db.return_value = mock_db

    # Fixer l'UUID de session
    mock_uuid.return_value = "fixed-uuid-123"

    # Créer d'abord une session et lui ajouter des messages
    from jvl.db.session import create_chat_session, add_message
    create_chat_session(mock_db, backend="gemini", model="gemini-2.5-flash", system_prompt="Sys")
    add_message(mock_db, "fixed-uuid-123", "user", "Hello first time", tokens_in=5, tokens_out=0)
    add_message(mock_db, "fixed-uuid-123", "assistant", "Hi there!", tokens_in=0, tokens_out=8)

    # Reprendre la session via CLI et faire une commande /usage, puis /exit
    with patch(
        "prompt_toolkit.PromptSession.prompt_async", new_callable=AsyncMock, side_effect=["/usage", "/exit"]
    ):
        result = runner.invoke(app, ["chat", "--session", "fixed-uuid-123"])
        assert result.exit_code == 0
        assert "Session reprise : fixed-uuid-123" in result.output
        assert "prompt     : 5" in result.output
        assert "completion : 8" in result.output


@patch("jvl.cli.chat.get_db")
@patch("jvl.cli.chat.BackendRouter")
@patch("jvl.cli.chat.pick_provider_and_model_async", new_callable=AsyncMock)
def test_chat_slash_model(mock_pick, mock_router_cls, mock_get_db, mock_router, mock_db):
    mock_router_cls.return_value = mock_router
    mock_get_db.return_value = mock_db
    mock_pick.return_value = ("ollama", "phi3:3.8b")

    # Envoyer /model, /model ollama, /model non_existent, puis /exit
    with patch(
        "prompt_toolkit.PromptSession.prompt_async",
        new_callable=AsyncMock,
        side_effect=["/model", "/model ollama", "/model non_existent", "/exit"]
    ):
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 0
        assert "Backend changé → ollama / phi3:3.8b" in result.output
        assert "Backend 'non_existent' non configuré" in result.output
        mock_pick.assert_called_once()


@patch("jvl.cli.chat.get_db")
@patch("jvl.cli.chat.BackendRouter")
def test_chat_slash_think(mock_router_cls, mock_get_db, mock_router, mock_db):
    mock_router_cls.return_value = mock_router
    mock_get_db.return_value = mock_db

    # Envoyer les commandes /think et /nothink puis /exit
    with patch(
        "prompt_toolkit.PromptSession.prompt_async",
        new_callable=AsyncMock,
        side_effect=[
            "/think",
            "/think true",
            "/think low",
            "/nothink",
            "/think clear",
            "/exit"
        ]
    ):
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 0
        assert "Aucun override de thinking mode actif" in result.output
        assert "Thinking mode activé (True) pour cette session" in result.output
        assert "Thinking mode défini sur 'low' pour cette session" in result.output
        assert "Thinking mode désactivé pour cette session" in result.output
        assert "Thinking mode réinitialisé aux paramètres par défaut" in result.output
