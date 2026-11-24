"""Command-line entry point: quick connectivity check and terminal chat.

    python main.py --check     # test Ollama connection only
    python main.py             # chat in the terminal (type 'exit' to quit)
"""
from __future__ import annotations

import argparse
import sys

from agent.agent import Assistant
from config import settings
from database.db import DatabaseError, create_db_engine, init_db
from llm.ollama_client import OllamaUnavailableError, check_connection
from memory.conversation_memory import ConversationMemory


def _use_utf8_console() -> None:
    """Avoid UnicodeEncodeError on Windows consoles that default to cp1252."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    _use_utf8_console()
    parser = argparse.ArgumentParser(description="Personal AI Assistant (terminal)")
    parser.add_argument("--check", action="store_true", help="only check the Ollama connection")
    args = parser.parse_args()

    status = check_connection()
    print(status.message)
    if args.check or not status.ok:
        return 0 if status.ok else 1

    try:
        engine = create_db_engine(settings.db_path)
        init_db(engine)
        memory = ConversationMemory(engine)
        conversation_id = memory.create_conversation()
    except DatabaseError as exc:
        print(f"Database error: {exc}")
        return 1

    assistant = Assistant(memory)
    print("Type 'exit' to quit.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if user_text.lower() in {"exit", "quit"}:
            return 0
        if not user_text:
            continue
        print("Assistant: ", end="", flush=True)
        try:
            for piece in assistant.respond_stream(conversation_id, user_text):
                print(piece, end="", flush=True)
            print("\n")
        except (OllamaUnavailableError, DatabaseError, ValueError) as exc:
            print(f"\n[error] {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
