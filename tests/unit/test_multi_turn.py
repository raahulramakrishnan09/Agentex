"""Regression tests for multi-turn conversations (no SQLite or Ollama needed)."""
from dataclasses import dataclass, replace

from agent.agent import Assistant
from config import load_settings as _load_settings
from memory.conversation_memory import Message

LONG_REASONING = (
    "Okay, the user is asking which team he currently plays for. Let me recall the latest info. "
    "Wait, I should double-check. The safest answer is to mention my knowledge cutoff. "
) * 40  # ~7,000 characters



def load_settings():
    """Phase 1 behaviour under test: plain chat, agent switched off."""
    return replace(_load_settings(), agent_enabled=False)


@dataclass
class _Chunk:
    content: str


class ScriptedLLM:
    """Returns pre-written replies in order and records every prompt it receives."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts = []

    def stream(self, messages):
        self.prompts.append(list(messages))
        reply = self._replies.pop(0)
        for i in range(0, len(reply), 50):
            yield _Chunk(reply[i:i + 50])


class FakeMemory:
    def __init__(self):
        self.rows = []

    def add_message(self, conversation_id, role, content):
        self.rows.append(Message(len(self.rows) + 1, conversation_id, role, content, ""))

    def get_recent_messages(self, conversation_id, limit):
        rows = [m for m in self.rows if m.conversation_id == conversation_id]
        return rows[-limit:] if limit > 0 else []


def ask(assistant, text, cid=1):
    return "".join(assistant.respond_stream(cid, text)).strip()


def test_single_question_with_reasoning_shows_only_answer():
    llm = ScriptedLLM(["Okay, let me compute 15% of 8500.</think>\n\n1275"])
    memory = FakeMemory()
    assistant = Assistant(memory, llm=llm, cfg=load_settings())
    assert ask(assistant, "Calculate 15% of 8500") == "1275"
    assert [m.content for m in memory.rows] == ["Calculate 15% of 8500", "1275"]


def test_follow_up_with_very_long_reasoning_shows_only_final_answer():
    final = "As far as I know he plays for Al Nassr, but this may be out of date."
    llm = ScriptedLLM([
        "Cristiano Ronaldo is a Portuguese professional footballer.",
        LONG_REASONING + "</think>\n\n" + final,
    ])
    memory = FakeMemory()
    assistant = Assistant(memory, llm=llm, cfg=load_settings())

    assert ask(assistant, "Who is Cristiano Ronaldo?").startswith("Cristiano Ronaldo")
    reply = ask(assistant, "Which team does he currently play for?")

    assert reply == final
    assert "Okay, the user" not in reply and "Wait," not in reply
    assert memory.rows[-1].content == final  # saved history is clean too


def test_follow_up_prompt_carries_the_earlier_turn_for_pronouns():
    llm = ScriptedLLM(["Ronaldo is a footballer.", "He plays for Al Nassr."])
    assistant = Assistant(FakeMemory(), llm=llm, cfg=load_settings())
    ask(assistant, "Who is Cristiano Ronaldo?")
    ask(assistant, "Which team does he currently play for?")

    second = llm.prompts[1]
    assert [m.type for m in second] == ["system", "human", "ai", "human"]
    assert "Cristiano Ronaldo" in second[1].content
    assert second[2].content == "Ronaldo is a footballer."
    assert second[3].content.startswith("Which team does he currently play for?")


def test_every_user_turn_and_system_prompt_carry_no_think_for_qwen3():
    llm = ScriptedLLM(["a", "b"])
    cfg = replace(load_settings(), ollama_model="qwen3:4b", ollama_think=False)
    assistant = Assistant(FakeMemory(), llm=llm, cfg=cfg)
    ask(assistant, "first")
    ask(assistant, "second")
    second = llm.prompts[1]
    assert second[0].content.endswith("/no_think")
    assert second[1].content.endswith("/no_think")  # replayed earlier user turn
    assert second[3].content.endswith("/no_think")  # current user turn


def test_saved_history_never_contains_the_switch_or_reasoning():
    llm = ScriptedLLM(["thinking...</think>fine", "second answer"])
    memory = FakeMemory()
    assistant = Assistant(memory, llm=llm, cfg=load_settings())
    ask(assistant, "q1")
    ask(assistant, "q2")
    assert all("/no_think" not in m.content and "</think>" not in m.content for m in memory.rows)
    third_turn_history = [m.content for m in memory.rows]
    assert "thinking..." not in " ".join(third_turn_history)


def test_conversations_do_not_share_context():
    llm = ScriptedLLM(["one", "two"])
    assistant = Assistant(FakeMemory(), llm=llm, cfg=load_settings())
    ask(assistant, "secret in conversation 1", cid=1)
    ask(assistant, "hello", cid=2)
    assert "secret" not in " ".join(str(m.content) for m in llm.prompts[1])


def test_thinking_disabled_models_other_than_qwen3_get_no_switch():
    llm = ScriptedLLM(["ok"])
    cfg = replace(load_settings(), ollama_model="llama3.2:3b")
    assistant = Assistant(FakeMemory(), llm=llm, cfg=cfg)
    ask(assistant, "hello")
    assert all("/no_think" not in m.content for m in llm.prompts[0])
