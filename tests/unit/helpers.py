"""Shared test doubles (no Ollama, no SQLite, no LangGraph needed to import)."""
from dataclasses import dataclass

from langchain_core.messages import AIMessage

from memory.conversation_memory import Message


def text_reply(text: str) -> AIMessage:
    return AIMessage(content=text)


def tool_reply(name: str, args: dict, content: str = "", call_id: str = "call_1") -> AIMessage:
    return AIMessage(content=content, tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@dataclass
class _Chunk:
    content: str


class ScriptedToolLLM:
    """Stands in for ChatOllama. ``bind_tools`` returns itself; ``invoke`` pops scripted replies.

    A reply may be an AIMessage or an Exception (raised). With ``repeat_last`` the final
    reply is returned forever (used to provoke runaway loops).
    """

    def __init__(self, replies, repeat_last: bool = False):
        self.replies = list(replies)
        self.repeat_last = repeat_last
        self.calls: list[list] = []
        self.bound: list = []

    def bind_tools(self, tools):
        self.bound = tools
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        if not self.replies:
            raise AssertionError("The LLM was called more often than scripted.")
        reply = self.replies[0] if (self.repeat_last and len(self.replies) == 1) else self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def stream(self, messages):  # used only by the AGENT_ENABLED=false path
        self.calls.append(list(messages))
        reply = self.replies.pop(0)
        yield _Chunk(reply.content)


class FakeMemory:
    """In-memory stand-in for ConversationMemory."""

    def __init__(self):
        self.rows: list[Message] = []

    def add_message(self, conversation_id, role, content):
        self.rows.append(Message(len(self.rows) + 1, conversation_id, role, content, ""))

    def get_recent_messages(self, conversation_id, limit):
        rows = [m for m in self.rows if m.conversation_id == conversation_id]
        return rows[-limit:] if limit > 0 else []


def ask(assistant, text, cid=1, events=None):
    callback = events.append if events is not None else None
    return "".join(assistant.respond_stream(cid, text, on_event=callback)).strip()
