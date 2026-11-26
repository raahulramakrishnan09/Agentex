"""Agent tools. Add new tools by registering a ``ToolSpec`` in ``default_registry``."""
from tools.registry import ToolInputError, ToolRegistry, ToolResult, ToolSpec
from tools.calculator_tool import CALCULATOR_TOOL


def default_registry() -> ToolRegistry:
    """Registry with every tool that is enabled in the current phase."""
    return ToolRegistry([CALCULATOR_TOOL])


__all__ = ["ToolInputError", "ToolRegistry", "ToolResult", "ToolSpec", "default_registry"]
