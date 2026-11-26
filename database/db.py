"""SQLite engine creation and automatic schema initialisation."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL, Engine
from sqlalchemy.exc import SQLAlchemyError

from utils.logger import get_logger

logger = get_logger("database")

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class DatabaseError(Exception):
    """Raised when a database operation fails."""


def create_db_engine(db_path: Path) -> Engine:
    """Create a SQLite engine, creating the parent directory if needed."""
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DatabaseError(f"Cannot create database directory {db_path.parent}: {exc}") from exc

    engine = create_engine(
        URL.create("sqlite", database=str(db_path)),
        connect_args={"check_same_thread": False, "timeout": 15},  # Streamlit uses several threads
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
        cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    """Create all tables from schema.sql if they do not exist yet."""
    try:
        script = SCHEMA_PATH.read_text(encoding="utf-8")
        raw = engine.raw_connection()
        try:
            raw.executescript(script)
            raw.commit()
        finally:
            raw.close()
    except (OSError, SQLAlchemyError) as exc:
        logger.exception("Database initialisation failed")
        raise DatabaseError(f"Could not initialise the database: {exc}") from exc
    logger.info("Database ready at %s", engine.url.database)
