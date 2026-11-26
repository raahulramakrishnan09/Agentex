"""Validated fallback for models that write a tool call as text.

Qwen3 normally produces structured tool calls, which Ollama parses for us. If a
small model instead prints its native ``<tool_call>{...}</tool_call>`` block (or
a bare JSON object) as plain text, this module recovers the call, but only when
it is valid JSON naming a *registered* tool with an object of arguments. Argument
values are validated again by the tool registry before anything runs.
Anything else is never treated as a tool call.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Sequence

from utils.logger import get_logger

logger = get_logger("agent.parsing")

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"


def _to_call(obj: Any, known_tools: Sequence[str]) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    args = obj.get("arguments", obj.get("args", {}))
    if not isinstance(name, str) or name not in known_tools:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}


def _load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


def extract_fallback_tool_calls(content: str, known_tools: Sequence[str]) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(visible_text, tool_calls)`` recovered from plain-text output.

    ``<tool_call>`` blocks are always removed from the visible text (valid or not)
    so raw markup is never shown to the user.
    """
    calls: list[dict[str, Any]] = []
    kept: list[str] = []
    pos = 0
    while True:
        start = content.find(OPEN_TAG, pos)
        if start == -1:
            kept.append(content[pos:])
            break
        kept.append(content[pos:start])
        end = content.find(CLOSE_TAG, start + len(OPEN_TAG))
        payload = content[start + len(OPEN_TAG): end if end != -1 else len(content)]
        call = _to_call(_load_json(payload.strip()), known_tools)
        if call is None:
            logger.warning("Ignored a malformed or unknown <tool_call> block from the model.")
        else:
            calls.append(call)
        if end == -1:
            break
        pos = end + len(CLOSE_TAG)

    visible = "".join(kept).strip()
    if not calls and OPEN_TAG not in content and visible.startswith("{"):
        call = _to_call(_load_json(visible), known_tools)
        if call is not None:
            return "", [call]
    return visible, calls
