import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from jvl.db.models import ChatMessage, ChatSession


def create_chat_session(
    db: Session,
    backend: str,
    model: str,
    system_prompt: str | None = None,
) -> ChatSession:
    """Crée une nouvelle session de chat et persiste le system prompt."""
    session_id = str(uuid.uuid4())
    chat_session = ChatSession(
        id=session_id,
        backend=backend,
        model=model,
        system_prompt=system_prompt,
    )
    db.add(chat_session)

    if system_prompt:
        msg = ChatMessage(
            session_id=session_id,
            role="system",
            content=system_prompt,
        )
        db.add(msg)

    db.commit()
    return chat_session


def get_chat_session(db: Session, session_id: str) -> ChatSession | None:
    """Récupère une session par son ID."""
    return db.execute(
        select(ChatSession).filter_by(id=session_id)
    ).scalar_one_or_none()


def add_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
) -> ChatMessage:
    """Ajoute un message à une session existante."""
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )
    db.add(msg)

    session = db.execute(
        select(ChatSession).filter_by(id=session_id)
    ).scalar_one_or_none()
    if session:
        session.ended_at = datetime.now(timezone.utc)

    db.commit()
    return msg


def get_session_messages(db: Session, session_id: str) -> list[dict]:
    """Retourne tous les messages d'une session sous forme de dicts."""
    messages = (
        db.execute(
            select(ChatMessage)
            .filter_by(session_id=session_id)
            .order_by(ChatMessage.timestamp)
        )
        .scalars()
        .all()
    )
    return [{"role": m.role, "content": m.content} for m in messages]


def get_session_usage(db: Session, session_id: str) -> dict:
    """Calcule l'usage total de tokens pour une session."""
    messages = db.execute(
        select(ChatMessage).filter_by(session_id=session_id)
    ).scalars().all()
    total_in = sum(m.tokens_in or 0 for m in messages)
    total_out = sum(m.tokens_out or 0 for m in messages)
    return {
        "total_in": total_in,
        "total_out": total_out,
        "message_count": len(messages),
    }
