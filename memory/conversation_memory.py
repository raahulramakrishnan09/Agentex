"""Short-term memory: conversations and their messages, persisted in SQLite.

Every query is scoped by ``conversation_id`` so context never leaks between
conversations.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from database.db import DatabaseError
from utils.logger import get_logger

logger = get_logger("memory.conversation")

DEFAULT_TITLE = "New conversation"
VALID_ROLES = frozenset({"user", "assistant", "system", "tool"})
TITLE_MAX_LENGTH = 40


class ConversationNotFoundError(ValueError):
    """Raised when a conversation id does not exist."""


@dataclass(frozen=True)
class Conversation:
    id: int
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Message:
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: str


@contextmanager
def _db_errors(action: str) -> Iterator[None]:
    """Convert SQLAlchemy errors into DatabaseError with logging."""
    try:
        yield
    except SQLAlchemyError as exc:
        logger.exception("Database error while trying to %s", action)
        raise DatabaseError(f"Failed to {action}: {exc}") from exc


def _make_title(text_value: str) -> str:
    single_line = " ".join(text_value.split())
    if len(single_line) <= TITLE_MAX_LENGTH:
        return single_line
    return single_line[: TITLE_MAX_LENGTH - 1].rstrip() + "…"


class ConversationMemory:
    """CRUD operations for conversations and messages."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    # -- conversations -------------------------------------------------
    def create_conversation(self, title: str = DEFAULT_TITLE) -> int:
        """Create a conversation and return its id."""
        with _db_errors("create conversation"), self._engine.begin() as conn:
            result = conn.execute(
                text("INSERT INTO conversations (title) VALUES (:title)"),
                {"title": title.strip() or DEFAULT_TITLE},
            )
            return int(result.lastrowid)

    def list_conversations(self) -> list[Conversation]:
        """Return conversations, most recently active first."""
        with _db_errors("list conversations"), self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, title, created_at, updated_at FROM conversations "
                    "ORDER BY updated_at DESC, id DESC"
                )
            ).all()
        return [Conversation(*row) for row in rows]

    def delete_conversation(self, conversation_id: int) -> None:
        """Delete a conversation and (via ON DELETE CASCADE) all its messages."""
        with _db_errors("delete conversation"), self._engine.begin() as conn:
            conn.execute(
                text("DELETE FROM conversations WHERE id = :id"), {"id": conversation_id}
            )

    def clear_conversation(self, conversation_id: int) -> None:
        """Remove all messages but keep the conversation itself."""
        with _db_errors("clear conversation"), self._engine.begin() as conn:
            self._require_conversation(conn, conversation_id)
            conn.execute(
                text("DELETE FROM messages WHERE conversation_id = :id"),
                {"id": conversation_id},
            )
            conn.execute(
                text(
                    "UPDATE conversations SET title = :title, updated_at = datetime('now') "
                    "WHERE id = :id"
                ),
                {"title": DEFAULT_TITLE, "id": conversation_id},
            )

    # -- messages ------------------------------------------------------
    def add_message(self, conversation_id: int, role: str, content: str) -> Message:
        """Append a message. The first user message becomes the conversation title."""
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role '{role}'. Expected one of {sorted(VALID_ROLES)}.")
        if not content or not content.strip():
            raise ValueError("Message content must not be empty.")

        with _db_errors("save message"), self._engine.begin() as conn:
            self._require_conversation(conn, conversation_id)
            result = conn.execute(
                text(
                    "INSERT INTO messages (conversation_id, role, content) "
                    "VALUES (:cid, :role, :content)"
                ),
                {"cid": conversation_id, "role": role, "content": content},
            )
            message_id = int(result.lastrowid)
            if role == "user":
                conn.execute(
                    text(
                        "UPDATE conversations SET title = :title "
                        "WHERE id = :id AND title = :default"
                    ),
                    {"title": _make_title(content), "id": conversation_id, "default": DEFAULT_TITLE},
                )
            conn.execute(
                text("UPDATE conversations SET updated_at = datetime('now') WHERE id = :id"),
                {"id": conversation_id},
            )
            row = conn.execute(
                text(
                    "SELECT id, conversation_id, role, content, created_at "
                    "FROM messages WHERE id = :id"
                ),
                {"id": message_id},
            ).one()
        return Message(*row)

    def get_messages(self, conversation_id: int) -> list[Message]:
        """Return the full message history of one conversation, oldest first."""
        with _db_errors("load messages"), self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, conversation_id, role, content, created_at FROM messages "
                    "WHERE conversation_id = :cid ORDER BY id ASC"
                ),
                {"cid": conversation_id},
            ).all()
        return [Message(*row) for row in rows]

    def get_recent_messages(self, conversation_id: int, limit: int) -> list[Message]:
        """Return the last ``limit`` messages of a conversation, oldest first."""
        if limit <= 0:
            return []
        with _db_errors("load recent messages"), self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, conversation_id, role, content, created_at FROM ("
                    "  SELECT * FROM messages WHERE conversation_id = :cid "
                    "  ORDER BY id DESC LIMIT :limit"
                    ") ORDER BY id ASC"
                ),
                {"cid": conversation_id, "limit": limit},
            ).all()
        return [Message(*row) for row in rows]

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _require_conversation(conn: Connection, conversation_id: int) -> None:
        found = conn.execute(
            text("SELECT 1 FROM conversations WHERE id = :id"), {"id": conversation_id}
        ).first()
        if found is None:
            raise ConversationNotFoundError(f"Conversation {conversation_id} does not exist.")
