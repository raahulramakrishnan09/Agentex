from dataclasses import dataclass, replace

import pytest

from agent.agent import Assistant
from config import load_settings as _load_settings
from database.db import create_db_engine, init_db
from llm.ollama_client import OllamaUnavailableError
from memory.conversation_memory import ConversationMemory



def load_settings():
    """Phase 1 behaviour under test: plain chat, agent switched off."""
    return replace(_load_settings(), agent_enabled=False)


@dataclass
class _Chunk:
    content: str


class FakeLLM:
    def __init__(self, reply="Hello!", fail=False):
        self.reply, self.fail, self.seen = reply, fail, []

    def stream(self, messages):
        self.seen.append(messages)
        if self.fail:
            raise ConnectionError("connection refused")
        for word in self.reply.split(" "):
            yield _Chunk(word + " ")


@pytest.fixture()
def memory(tmp_path):
    engine = create_db_engine(tmp_path / "t.db")
    init_db(engine)
    return ConversationMemory(engine)


def test_reply_is_streamed_and_persisted(memory):
    cid = memory.create_conversation()
    assistant = Assistant(memory, llm=FakeLLM("Hi there"), cfg=load_settings())
    assert "".join(assistant.respond_stream(cid, "hey")).strip() == "Hi there"
    assert [m.role for m in memory.get_messages(cid)] == ["user", "assistant"]


def test_history_is_sent_to_model_but_not_across_conversations(memory):
    a, b = memory.create_conversation(), memory.create_conversation()
    llm = FakeLLM()
    assistant = Assistant(memory, llm=llm, cfg=load_settings())
    list(assistant.respond_stream(a, "my name is Sam"))
    list(assistant.respond_stream(a, "what is my name?"))
    list(assistant.respond_stream(b, "hello"))
    second_turn_text = " ".join(str(m.content) for m in llm.seen[1])
    third_turn_text = " ".join(str(m.content) for m in llm.seen[2])
    assert "my name is Sam" in second_turn_text
    assert "Sam" not in third_turn_text


def test_failure_saves_nothing(memory):
    cid = memory.create_conversation()
    assistant = Assistant(memory, llm=FakeLLM(fail=True), cfg=load_settings())
    with pytest.raises(OllamaUnavailableError):
        list(assistant.respond_stream(cid, "hello"))
    assert memory.get_messages(cid) == []


def test_empty_input_rejected(memory):
    cid = memory.create_conversation()
    assistant = Assistant(memory, llm=FakeLLM(), cfg=load_settings())
    with pytest.raises(ValueError):
        list(assistant.respond_stream(cid, "   "))


def test_history_is_trimmed_to_character_budget(memory):
    cid = memory.create_conversation()
    for i in range(4):
        memory.add_message(cid, "user", f"old question {i} " + "x" * 200)
        memory.add_message(cid, "assistant", f"old answer {i} " + "y" * 200)
    llm = FakeLLM()
    cfg = replace(load_settings(), max_history_chars=500)
    list(Assistant(memory, llm=llm, cfg=cfg).respond_stream(cid, "new question"))
    sent = llm.seen[0]
    assert "old question 0" not in " ".join(str(m.content) for m in sent)
    assert sent[1].type == "human"  # history never starts with an assistant turn
    assert sent[-1].content.startswith("new question")


def test_no_think_switch_is_sent_but_not_saved(memory):
    cid = memory.create_conversation()
    llm = FakeLLM()
    cfg = replace(load_settings(), ollama_model="qwen3:4b", ollama_think=False, ollama_no_think_suffix=True)
    list(Assistant(memory, llm=llm, cfg=cfg).respond_stream(cid, "Calculate 15% of 8500"))
    assert llm.seen[0][-1].content.endswith("/no_think")
    assert memory.get_messages(cid)[0].content == "Calculate 15% of 8500"


def test_reasoning_is_not_saved_to_history(memory):
    cid = memory.create_conversation()
    llm = FakeLLM("Okay, let me think about this.</think> 1275")
    cfg = replace(load_settings(), reasoning_hold_chars=20000)
    reply = "".join(Assistant(memory, llm=llm, cfg=cfg).respond_stream(cid, "15% of 8500?"))
    assert reply.strip() == "1275"
    assert memory.get_messages(cid)[1].content == "1275"


def test_agent_conversation_persists_in_sqlite(memory):
    from agent.agent import Assistant as AgentAssistant
    from config import load_settings as real_settings
    from tests.unit.helpers import ScriptedToolLLM, text_reply, tool_reply

    cid = memory.create_conversation()
    llm = ScriptedToolLLM([
        tool_reply("calculator", {"expression": "8500 * 15%"}),
        text_reply("15% of 8500 is 1275."),
        text_reply("You asked: Calculate 15% of 8500"),
    ])
    assistant = AgentAssistant(memory, llm=llm, cfg=real_settings())
    assert "1275" in "".join(assistant.respond_stream(cid, "Calculate 15% of 8500"))
    assert "".join(assistant.respond_stream(cid, "What was my previous question?")).startswith("You asked")
    assert [(m.role) for m in memory.get_messages(cid)] == ["user", "assistant", "user", "assistant"]
    assert memory.get_messages(cid)[1].content == "15% of 8500 is 1275."
