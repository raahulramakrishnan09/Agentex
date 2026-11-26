"""Tool registry: describes tools to the model and executes them safely.

Tools are plain Python functions plus a JSON-schema description. The registry
validates every call (name, argument names and types) before running anything,
and turns every failure into a short error string the model can react to.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from utils.logger import get_logger

logger = get_logger("tools")


class ToolInputError(ValueError):
    """Bad input to a tool. The message is written for the user/model and is safe to show."""


@dataclass(frozen=True)
class ToolSpec:
    """Everything needed to expose one tool."""

    name: str
    description: str
    parameters: dict[str, Any]       # JSON schema of the arguments
    func: Callable[..., str]
    usage_hint: str = ""             # one-line guidance added to the system prompt


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


class ToolRegistry:
    """Holds the available tools and executes calls against them."""

    def __init__(self, tools: tuple[ToolSpec, ...] | list[ToolSpec] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for spec in tools:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered.")
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        return list(self._tools)

    def usage_hints(self) -> list[str]:
        return [spec.usage_hint for spec in self._tools.values() if spec.usage_hint]

    def openai_schemas(self) -> list[dict[str, Any]]:
        """Tool descriptions in the OpenAI function format that ``bind_tools`` accepts."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._tools.values()
        ]

    def execute(self, name: str, args: Any) -> ToolResult:
        """Validate and run a tool call. Never raises."""
        spec = self._tools.get(name)
        if spec is None:
            available = ", ".join(self.names()) or "none"
            return ToolResult(False, f"Error: unknown tool '{name}'. Available tools: {available}.")
        try:
            clean_args = self._validate_args(spec, args)
            return ToolResult(True, str(spec.func(**clean_args)))
        except ToolInputError as exc:
            logger.info("Tool '%s' rejected its input: %s", name, exc)
            return ToolResult(False, f"Error: {exc}")
        except Exception:  # noqa: BLE001 - a tool must never crash the agent
            logger.exception("Tool '%s' failed unexpectedly", name)
            return ToolResult(
                False,
                f"Error: the tool '{name}' failed unexpectedly. Tell the user it could not be completed.",
            )

    @staticmethod
    def _validate_args(spec: ToolSpec, args: Any) -> dict[str, Any]:
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError as exc:
                raise ToolInputError("the tool arguments were not valid JSON.") from exc
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ToolInputError("the tool arguments must be an object of named values.")

        properties: dict[str, Any] = spec.parameters.get("properties", {})
        required: list[str] = spec.parameters.get("required", [])

        unexpected = sorted(set(args) - set(properties))
        if unexpected:
            raise ToolInputError(f"unexpected argument(s): {', '.join(unexpected)}.")
        missing = [key for key in required if key not in args]
        if missing:
            raise ToolInputError(f"missing required argument(s): {', '.join(missing)}.")

        for key, value in args.items():
            expected = _JSON_TYPES.get(properties[key].get("type", ""), ())
            is_bool_for_number = isinstance(value, bool) and bool not in expected
            if expected and (is_bool_for_number or not isinstance(value, expected)):
                raise ToolInputError(f"argument '{key}' must be of type {properties[key]['type']}.")
        return args
