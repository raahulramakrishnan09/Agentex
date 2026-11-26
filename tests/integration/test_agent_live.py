"""Real-model behaviour of the agent. Needs Ollama + the model:  pytest -m integration

These check the model's own tool-selection decisions, which scripted tests cannot.
A 4B model can occasionally choose differently; rerun once before treating a failure as a bug.
"""
import pytest

from agent.agent import Assistant
from database.db import create_db_engine, init_db
from llm.ollama_client import check_connection
from memory.conversation_memory import ConversationMemory

pytestmark = pytest.mark.integration


@pytest.fixture()
def assistant_and_cid(tmp_path):
    status = check_connection()
    if not status.ok:
        pytest.skip(status.message)
    engine = create_db_engine(tmp_path / "live.db")
    init_db(engine)
    memory = ConversationMemory(engine)
    return Assistant(memory), memory.create_conversation()


def run(assistant, cid, text):
    events = []
    reply = "".join(assistant.respond_stream(cid, text, on_event=events.append))
    return reply, [e for e in events if e.kind == "tool_call"]


@pytest.mark.parametrize("question", ["Who is Cristiano Ronaldo?", "Explain what an AI agent is"])
def test_direct_questions_use_no_tool(assistant_and_cid, question):
    assistant, cid = assistant_and_cid
    reply, calls = run(assistant, cid, question)
    assert calls == [] and reply.strip()


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Calculate 15% of 8500", "1275"),
        ("What is 1250 + 3750?", "5000"),
        ("Calculate 20% of 5000 and add 750", "1750"),
    ],
)
def test_arithmetic_uses_the_calculator(assistant_and_cid, question, expected):
    assistant, cid = assistant_and_cid
    reply, calls = run(assistant, cid, question)
    assert [c.name for c in calls][:1] == ["calculator"]
    assert expected.replace(",", "") in reply.replace(",", "")


def test_previous_question_uses_context_not_a_tool(assistant_and_cid):
    assistant, cid = assistant_and_cid
    run(assistant, cid, "Who is Cristiano Ronaldo?")
    reply, calls = run(assistant, cid, "What was my previous question?")
    assert calls == [] and "ronaldo" in reply.lower()
