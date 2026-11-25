"""Phase 1 assistant: plain conversational chat backed by persisted history.

In Phase 2 this module is replaced by the LangGraph agent. The public method
``respond_stream`` is kept stable so the UI does not need to change.
"""
from __future__ import annotations

from typing import Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent.prompts import get_system_prompt
from config import Settings, settings
from llm.ollama_client import apply_no_think, build_chat_model, no_think_enabled, stream_chat
from memory.conversation_memory import ConversationMemory, Message
from utils.logger import get_logger

logger = get_logger("agent")


def _to_langchain(messages: list[Message], cfg: Settings = settings) -> list[BaseMessage]:
    """Convert stored messages to LangChain messages.

    Earlier user turns get the same ``/no_think`` switch as the current one, so the
    whole prompt consistently asks for answer-only replies (stored text stays clean).
    """
    converted: list[BaseMessage] = []
    for msg in messages:
        if msg.role == "user":
            converted.append(HumanMessage(content=apply_no_think(msg.content, cfg)))
        elif msg.role == "assistant":
            converted.append(AIMessage(content=msg.content))
        # system/tool rows are not replayed in Phase 1
    return converted


def _trim_history(messages: list[Message], max_chars: int) -> list[Message]:
    """Keep the newest messages that fit in ``max_chars``.

    Ollama silently truncates prompts that exceed the context window, which can
    cut off the system prompt. Trimming here keeps the prompt predictable.
    The result never starts with an assistant message.
    """
    kept: list[Message] = []
    total = 0
    for msg in reversed(messages):
        total += len(msg.content)
        if total > max_chars:
            break
        kept.append(msg)
    kept.reverse()
    while kept and kept[0].role != "user":
        kept.pop(0)
    return kept


class Assistant:
    """Chat assistant with per-conversation memory."""

    def __init__(self, memory: ConversationMemory, llm=None, cfg: Settings = settings) -> None:  # noqa: ANN001
        self._memory = memory
        self._cfg = cfg
        self._llm = llm if llm is not None else build_chat_model(cfg)

    def respond_stream(self, conversation_id: int, user_text: str) -> Iterator[str]:
        """Stream the reply to ``user_text`` and persist both turns on success.

        Nothing is saved if the model fails, so history never contains a user
        message without an answer.

        Raises:
            ValueError: for empty input.
            OllamaUnavailableError: if the model cannot answer.
            DatabaseError: if history cannot be read or saved.
        """
        text = user_text.strip()
        if not text:
            raise ValueError("Please enter a message.")

        history = _trim_history(
            self._memory.get_recent_messages(conversation_id, self._cfg.max_history_messages),
            self._cfg.max_history_chars,
        )
        prompt: list[BaseMessage] = [
            SystemMessage(content=get_system_prompt(no_think=no_think_enabled(self._cfg)))
        ]
        prompt += _to_langchain(history, self._cfg)
        # The /no_think switch goes only to the model; the saved message stays clean.
        prompt.append(HumanMessage(content=apply_no_think(text, self._cfg)))

        logger.info("Conversation %s: user message (%d chars)", conversation_id, len(text))
        parts: list[str] = []
        for piece in stream_chat(self._llm, prompt, self._cfg):
            parts.append(piece)
            yield piece

        reply = "".join(parts).strip()
        if not reply:
            reply = "(The model returned an empty response.)"
        self._memory.add_message(conversation_id, "user", text)
        self._memory.add_message(conversation_id, "assistant", reply)
        logger.info("Conversation %s: reply saved (%d chars)", conversation_id, len(reply))
