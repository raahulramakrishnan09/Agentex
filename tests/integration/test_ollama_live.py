"""Needs a running Ollama server with the model pulled.

Run with:  pytest -m integration
"""
import pytest

from llm.ollama_client import build_chat_model, check_connection, stream_chat

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def status():
    s = check_connection()
    if not s.ok:
        pytest.skip(s.message)
    return s


def test_connection(status):
    assert status.ok


def test_basic_llm_response(status):
    from langchain_core.messages import HumanMessage

    reply = "".join(stream_chat(build_chat_model(), [HumanMessage(content="Reply with the single word: pong")]))
    assert reply.strip()


REASONING_MARKERS = ("<think>", "</think>", "Okay, the user", "Let me recall", "Wait,", "The safest answer")


def _assert_clean(reply: str) -> None:
    for marker in REASONING_MARKERS:
        assert marker not in reply, f"reasoning leaked ({marker!r}): {reply[:300]!r}"


def test_follow_up_question_shows_no_reasoning(status, tmp_path):
    from agent.agent import Assistant
    from database.db import create_db_engine, init_db
    from memory.conversation_memory import ConversationMemory

    engine = create_db_engine(tmp_path / "live.db")
    init_db(engine)
    memory = ConversationMemory(engine)
    cid = memory.create_conversation()
    assistant = Assistant(memory)

    first = "".join(assistant.respond_stream(cid, "Who is Cristiano Ronaldo?"))
    second = "".join(assistant.respond_stream(cid, "Which team does he currently play for?"))

    _assert_clean(first)
    _assert_clean(second)
    assert len(second) < 1500  # concise: no multi-paragraph deliberation
    assert "football" in (first + second).lower() or "soccer" in (first + second).lower()
