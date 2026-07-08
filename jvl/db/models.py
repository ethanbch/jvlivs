from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    backend: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        "created_at",
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    ended_at: Mapped[datetime] = mapped_column(
        "updated_at",
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", order_by="ChatMessage.timestamp"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("chat_sessions.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        "created_at",
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")
