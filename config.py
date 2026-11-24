"""Centralised configuration.

Every adjustable setting lives here and can be overridden with an environment
variable, so no machine-specific values are hard-coded anywhere else.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    ollama_base_url: str
    ollama_model: str
    ollama_think: bool          # Qwen3 "thinking" mode; slow on CPU, so off by default
    ollama_temperature: float
    ollama_num_ctx: int         # context window; keep modest for 16 GB RAM
    ollama_timeout_seconds: int
    ollama_no_think_suffix: bool  # append /no_think to Qwen3 prompts while thinking is off
    reasoning_hold_chars: int   # max chars held back to catch reasoning; default ~ a whole reply (0 = stream immediately)
    ollama_keep_alive: str      # how long Ollama keeps the model in RAM after a reply
    max_history_messages: int   # how many past messages are sent to the model
    max_history_chars: int      # character budget for history (keeps prompt inside the context window)
    db_path: Path
    log_level: str
    log_file: Path


def load_settings() -> Settings:
    """Build settings from environment variables, falling back to defaults."""
    data_dir = BASE_DIR / "data"
    return Settings(
        ollama_base_url=_get_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ollama_model=_get_str("OLLAMA_MODEL", "qwen3:4b"),
        ollama_think=_get_bool("OLLAMA_THINK", False),
        ollama_temperature=_get_float("OLLAMA_TEMPERATURE", 0.5),
        ollama_num_ctx=_get_int("OLLAMA_NUM_CTX", 4096),
        ollama_timeout_seconds=_get_int("OLLAMA_TIMEOUT_SECONDS", 300),
        ollama_no_think_suffix=_get_bool("OLLAMA_NO_THINK_SUFFIX", True),
        reasoning_hold_chars=_get_int("OLLAMA_REASONING_HOLD_CHARS", 20000),
        ollama_keep_alive=_get_str("OLLAMA_KEEP_ALIVE", "30m"),
        max_history_messages=_get_int("MAX_HISTORY_MESSAGES", 20),
        max_history_chars=_get_int("MAX_HISTORY_CHARS", 6000),
        db_path=Path(_get_str("ASSISTANT_DB_PATH", str(data_dir / "assistant.db"))),
        log_level=_get_str("LOG_LEVEL", "INFO").upper(),
        log_file=Path(_get_str("ASSISTANT_LOG_FILE", str(data_dir / "assistant.log"))),
    )


settings = load_settings()
