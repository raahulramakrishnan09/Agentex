"""Ollama integration: connection checks, model construction and streaming."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import httpx
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

from config import Settings, settings
from llm.think_filter import ThinkTagFilter
from utils.logger import get_logger

logger = get_logger("llm")


class OllamaUnavailableError(Exception):
    """Raised when the local model cannot be reached or fails to answer."""


@dataclass(frozen=True)
class ConnectionStatus:
    """Result of probing the local Ollama server."""

    server_up: bool
    model_available: bool
    message: str
    models: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.server_up and self.model_available


def check_connection(cfg: Settings = settings, timeout: float = 5.0) -> ConnectionStatus:
    """Check that Ollama is running and the configured model is downloaded."""
    url = cfg.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except httpx.ConnectError:
        return ConnectionStatus(
            False, False,
            f"Cannot reach Ollama at {cfg.ollama_base_url}. Start it (open the Ollama app "
            "or run 'ollama serve').",
        )
    except (httpx.HTTPError, ValueError) as exc:
        return ConnectionStatus(False, False, f"Unexpected response from Ollama: {exc}")

    names = [m.get("name", "") for m in payload.get("models", []) if isinstance(m, dict)]
    wanted = cfg.ollama_model if ":" in cfg.ollama_model else f"{cfg.ollama_model}:latest"
    if wanted not in names:
        return ConnectionStatus(
            True, False,
            f"Ollama is running but model '{cfg.ollama_model}' is not installed. "
            f"Run: ollama pull {cfg.ollama_model}",
            names,
        )
    return ConnectionStatus(True, True, f"Connected. Model '{cfg.ollama_model}' is ready.", names)


NO_THINK_SUFFIX = " /no_think"


def no_think_enabled(cfg: Settings = settings) -> bool:
    """True when Qwen3's ``/no_think`` switch should be sent to the model."""
    return (
        not cfg.ollama_think
        and cfg.ollama_no_think_suffix
        and "qwen3" in cfg.ollama_model.lower()
    )


def apply_no_think(text: str, cfg: Settings = settings) -> str:
    """Append ``/no_think`` to a user prompt when thinking is off (Qwen3 only).

    Other models would see it as literal text. The caller must NOT store the
    returned text in the conversation history.
    """
    return text + NO_THINK_SUFFIX if no_think_enabled(cfg) else text


def build_chat_model(cfg: Settings = settings) -> ChatOllama:
    """Create the LangChain chat model. This does not contact the server yet."""
    return ChatOllama(
        model=cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=cfg.ollama_temperature,
        num_ctx=cfg.ollama_num_ctx,
        reasoning=cfg.ollama_think,
        keep_alive=cfg.ollama_keep_alive,
        client_kwargs={"timeout": cfg.ollama_timeout_seconds},
    )


def _explain_error(exc: Exception, cfg: Settings) -> str:
    message = str(exc).lower()
    if "not found" in message:
        return f"Model '{cfg.ollama_model}' was not found. Run: ollama pull {cfg.ollama_model}"
    if isinstance(exc, httpx.TimeoutException) or "timed out" in message:
        return (
            "The model took too long to answer. CPU inference can be slow; try a shorter "
            "prompt or increase OLLAMA_TIMEOUT_SECONDS."
        )
    if isinstance(exc, (httpx.ConnectError, ConnectionError)) or "refused" in message or "connect" in message:
        return (
            f"Cannot reach Ollama at {cfg.ollama_base_url}. Make sure Ollama is running."
        )
    return f"The model request failed: {exc}"


def stream_chat(
    llm: ChatOllama, messages: Sequence[BaseMessage], cfg: Settings = settings
) -> Iterator[str]:
    """Stream the model's answer as plain text chunks.

    Raises:
        OllamaUnavailableError: if the server is unreachable, the model is missing,
            or the request fails for any other reason.
    """
    think_filter = ThinkTagFilter(hold_chars=cfg.reasoning_hold_chars)
    try:
        for chunk in llm.stream(list(messages)):
            # Only ``content`` is read. Reasoning that langchain-ollama separates
            # into ``additional_kwargs['reasoning_content']`` is never shown or saved.
            content = getattr(chunk, "content", "")
            if not isinstance(content, str) or not content:
                continue
            visible = think_filter.feed(content)
            if visible:
                yield visible
        tail = think_filter.flush()
        if tail:
            yield tail
        if think_filter.stripped_chars and not cfg.ollama_think:
            logger.warning(
                "Removed %d characters of reasoning although thinking is disabled "
                "(model '%s'). See README: 'Reasoning leakage'.",
                think_filter.stripped_chars, cfg.ollama_model,
            )
    except OllamaUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalise every client failure
        logger.exception("LLM streaming failed")
        raise OllamaUnavailableError(_explain_error(exc, cfg)) from exc
