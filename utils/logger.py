"""Logging setup: console plus a rotating log file."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config import settings

_ROOT_NAME = "assistant"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def _configure_root() -> logging.Logger:
    root = logging.getLogger(_ROOT_NAME)
    if root.handlers:  # already configured (Streamlit reruns import modules often)
        return root
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))
    root.propagate = False
    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            settings.log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        root.warning("Could not open log file %s; logging to console only.", settings.log_file)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger such as ``assistant.memory``."""
    _configure_root()
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
