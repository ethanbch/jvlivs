from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from jvl.db.models import Base, ChatMessage, ChatSession
from jvl.db.session import (
    add_message,
    create_chat_session,
    get_chat_session,
    get_session_messages,
    get_session_usage,
)


# ── ChatSession CRUD ──────────────────────────────────────────


def test_create_chat_session(db):
    session = create_chat_session(db, backend="ollama", model="phi3:3.8b")
    assert session.id is not None
    assert session.backend == "ollama"
    assert session.model == "phi3:3.8b"
    assert session.system_prompt is None


def test_create_chat_session_with_system_prompt(db):
    session = create_chat_session(
        db, backend="openai", model="gpt-4o", system_prompt="Tu es un assistant."
    )
    assert session.system_prompt == "Tu es un assistant."
    messages = get_session_messages(db, session.id)
    assert len(messages) == 1
    assert messages[0] == {"role": "system", "content": "Tu es un assistant."}


def test_get_chat_session(db):
    session = create_chat_session(db, backend="ollama", model="phi3:3.8b")
    found = get_chat_session(db, session.id)
    assert found is not None
    assert found.id == session.id


def test_get_chat_session_not_found(db):
    found = get_chat_session(db, "nonexistent-id")
    assert found is None


# ── Messages ──────────────────────────────────────────────────


def test_add_and_get_messages(db):
    session = create_chat_session(db, backend="ollama", model="phi3:3.8b")
    add_message(db, session.id, "user", "Bonjour")
    add_message(db, session.id, "assistant", "Salut !", tokens_in=5, tokens_out=3)
    messages = get_session_messages(db, session.id)
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "Bonjour"}
    assert messages[1] == {"role": "assistant", "content": "Salut !"}


def test_add_message_updates_session_timestamp(db):
    session = create_chat_session(db, backend="ollama", model="phi3:3.8b")
    original_updated = session.ended_at
    add_message(db, session.id, "user", "hello")
    db.refresh(session)
    assert session.ended_at >= original_updated


# ── Usage ─────────────────────────────────────────────────────


def test_get_session_usage_empty(db):
    session = create_chat_session(db, backend="ollama", model="phi3:3.8b")
    usage = get_session_usage(db, session.id)
    assert usage == {"total_in": 0, "total_out": 0, "message_count": 0}


def test_get_session_usage_with_messages(db):
    session = create_chat_session(db, backend="ollama", model="phi3:3.8b")
    add_message(db, session.id, "user", "hello")
    add_message(db, session.id, "assistant", "hi", tokens_in=10, tokens_out=5)
    add_message(db, session.id, "user", "how are you?")
    add_message(db, session.id, "assistant", "fine", tokens_in=20, tokens_out=8)
    usage = get_session_usage(db, session.id)
    assert usage == {"total_in": 30, "total_out": 13, "message_count": 4}
