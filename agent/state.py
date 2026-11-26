"""State carried through the LangGraph agent."""
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """``messages`` accumulates (add_messages reducer); ``iterations`` counts model calls."""

    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
