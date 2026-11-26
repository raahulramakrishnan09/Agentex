"""The assistant: persisted conversation + (Phase 2) a LangGraph tool-using agent.

``Assistant.respond_stream`` keeps the Phase 1 signature (it yields reply text and
saves both turns on success). With ``AGENT_ENABLED=false`` it behaves exactly like
Phase 1 (plain streaming chat, no tools, LangGraph not even imported).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent.prompts import LIMIT_MESSAGE, get_system_prompt
from config import Settings, settings
from llm.ollama_client import (
    OllamaUnavailableError,
    apply_no_think,
    build_chat_model,
    no_think_enabled,
    stream_chat,
)
from memory.conversation_memory import ConversationMemory, Message
from tools import ToolRegistry, default_registry
from utils.logger import get_logger

logger = get_logger("agent")


class AgentError(Exception):
    """Raised when the agent fails for a reason other than the model being unavailable."""


@dataclass(frozen=True)
class AgentEvent:
    """Progress notification for the UI. Never contains reasoning or prompts."""

    kind: str     # "tool_call" | "tool_result" | "tool_error"
    name: str
    detail: str


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
        # system/tool rows are never replayed
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


def _summarise_args(args: Any, limit: int = 120) -> str:
    if isinstance(args, dict):
        text = ", ".join(f"{key}={value}" for key, value in args.items())
    else:
        text = str(args)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Assistant:
    """Chat assistant with per-conversation memory and optional tool use."""

    def __init__(
        self,
        memory: ConversationMemory,
        llm: Any = None,
        cfg: Settings = settings,
        registry: ToolRegistry | None = None,
    ) -> None:
        self._memory = memory
        self._cfg = cfg
        self._llm = llm if llm is not None else build_chat_model(cfg)
        self._registry = registry if registry is not None else default_registry()
        self._graph: Any = None

    # -- public API ----------------------------------------------------
    def respond_stream(
        self,
        conversation_id: int,
        user_text: str,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> Iterator[str]:
        """Yield the reply to ``user_text`` and persist both turns on success.

        Nothing is saved if the model fails, so history never contains a user
        message without an answer. Tool traces are not stored, only the final text.

        Args:
            on_event: optional callback for tool progress (used by the UI).

        Raises:
            ValueError: for empty input.
            OllamaUnavailableError: if the model cannot answer.
            AgentError: if the agent fails unexpectedly.
            DatabaseError: if history cannot be read or saved.
        """
        text = user_text.strip()
        if not text:
            raise ValueError("Please enter a message.")

        use_agent = self._cfg.agent_enabled
        history = _trim_history(
            self._memory.get_recent_messages(conversation_id, self._cfg.max_history_messages),
            self._cfg.max_history_chars,
        )
        prompt: list[BaseMessage] = [
            SystemMessage(
                content=get_system_prompt(
                    no_think=no_think_enabled(self._cfg),
                    tool_hints=self._registry.usage_hints() if use_agent else (),
                )
            )
        ]
        prompt += _to_langchain(history, self._cfg)
        # The /no_think switch goes only to the model; the saved message stays clean.
        prompt.append(HumanMessage(content=apply_no_think(text, self._cfg)))

        logger.info("Conversation %s: user message (%d chars, agent=%s)", conversation_id, len(text), use_agent)
        source = self._run_agent(prompt, on_event) if use_agent else stream_chat(self._llm, prompt, self._cfg)
        parts: list[str] = []
        for piece in source:
            parts.append(piece)
            yield piece

        reply = "".join(parts).strip()
        if not reply:
            reply = "(The model returned an empty response.)"
        self._memory.add_message(conversation_id, "user", text)
        self._memory.add_message(conversation_id, "assistant", reply)
        logger.info("Conversation %s: reply saved (%d chars)", conversation_id, len(reply))

    # -- agent path ----------------------------------------------------
    def _get_graph(self) -> Any:
        if self._graph is None:
            from agent.graph import build_agent_graph  # lazy: Phase 1 chat works without LangGraph

            self._graph = build_agent_graph(self._llm, self._registry, self._cfg)
        return self._graph

    def _run_agent(
        self, prompt: list[BaseMessage], on_event: Callable[[AgentEvent], None] | None
    ) -> Iterator[str]:
        graph = self._get_graph()
        limit = self._cfg.agent_max_iterations
        config = {"recursion_limit": max(25, 4 * limit + 10)}
        try:
            for update in graph.stream({"messages": prompt, "iterations": 0}, config=config, stream_mode="updates"):
                for node, output in update.items():
                    yield from self._handle_update(node, output, on_event)
        except OllamaUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "GraphRecursionError":
                logger.warning("LangGraph recursion limit reached")
                yield LIMIT_MESSAGE.format(limit=limit)
                return
            logger.exception("Agent failed")
            raise AgentError(f"The agent failed unexpectedly: {exc}") from exc

    @staticmethod
    def _handle_update(
        node: str, output: Any, on_event: Callable[[AgentEvent], None] | None
    ) -> Iterator[str]:
        messages = output.get("messages", []) if isinstance(output, dict) else []
        if isinstance(messages, BaseMessage):
            messages = [messages]
        for message in messages:
            kind = getattr(message, "type", "")
            if kind == "ai":
                calls = getattr(message, "tool_calls", None) or []
                if calls:
                    # Any text written alongside a tool call is a preamble: dropped, never shown or saved.
                    for call in calls:
                        if on_event:
                            on_event(AgentEvent("tool_call", call["name"], _summarise_args(call.get("args"))))
                elif message.content:
                    yield message.content
            elif kind == "tool" and on_event:
                ok = (getattr(message, "additional_kwargs", None) or {}).get("tool_ok", True)
                on_event(AgentEvent("tool_result" if ok else "tool_error", message.name or "tool", message.content))
