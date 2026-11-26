"""The LangGraph agent.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END
                 |
                 +--(iteration limit reached)--> limit -> END

* ``agent``  : one model turn (tools bound). Either a final answer or tool calls.
* ``tools``  : runs the requested tools through the registry; errors become tool messages.
* ``limit``  : fixed, safe final message when the model keeps asking for tools.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from agent.prompts import LIMIT_MESSAGE
from agent.state import AgentState
from agent.tool_parsing import extract_fallback_tool_calls
from config import Settings
from llm.ollama_client import invoke_chat
from tools.registry import ToolRegistry
from utils.logger import get_logger

logger = get_logger("agent.graph")

TOO_MANY_CALLS = "Error: too many tool calls were requested in one step; this one was skipped."


def build_agent_graph(llm: Any, registry: ToolRegistry, cfg: Settings):
    """Compile the agent graph for the given model, tools and settings."""
    model = llm.bind_tools(registry.openai_schemas()) if registry.names() else llm

    def agent_node(state: AgentState) -> dict[str, Any]:
        ai = invoke_chat(model, state["messages"], cfg)
        if not ai.tool_calls:
            text, calls = extract_fallback_tool_calls(ai.content, registry.names())
            if calls or text != ai.content:
                ai = AIMessage(content=text, tool_calls=calls)
        logger.info(
            "Model turn %d: %d tool call(s), %d chars of text",
            state["iterations"] + 1, len(ai.tool_calls), len(ai.content),
        )
        return {"messages": [ai], "iterations": state["iterations"] + 1}

    def tools_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        results: list[ToolMessage] = []
        for index, call in enumerate(last.tool_calls):
            if index >= cfg.agent_max_tool_calls_per_turn:
                ok, content = False, TOO_MANY_CALLS
            else:
                result = registry.execute(call["name"], call.get("args"))
                ok, content = result.ok, result.content
                logger.info("Tool '%s' -> %s", call["name"], "ok" if ok else "error")
            results.append(
                ToolMessage(
                    content=content,
                    tool_call_id=call["id"],
                    name=call["name"],
                    additional_kwargs={"tool_ok": ok},
                )
            )
        return {"messages": results}

    def limit_node(state: AgentState) -> dict[str, Any]:
        logger.warning("Iteration limit (%d) reached", cfg.agent_max_iterations)
        return {"messages": [AIMessage(content=LIMIT_MESSAGE.format(limit=cfg.agent_max_iterations))]}

    def route(state: AgentState) -> str:
        if not getattr(state["messages"][-1], "tool_calls", None):
            return "end"
        return "limit" if state["iterations"] >= cfg.agent_max_iterations else "tools"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("limit", limit_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", "limit": "limit", "end": END})
    builder.add_edge("tools", "agent")
    builder.add_edge("limit", END)
    return builder.compile()
