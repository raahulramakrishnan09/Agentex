"""Streamlit UI (Phase 1: chat, conversation history, model status)."""
from __future__ import annotations

import streamlit as st

from agent.agent import Assistant
from config import settings
from database.db import DatabaseError, create_db_engine, init_db
from llm.ollama_client import OllamaUnavailableError, check_connection
from memory.conversation_memory import ConversationMemory

st.set_page_config(page_title="Personal AI Assistant", page_icon="🤖", layout="wide")


@st.cache_resource
def get_services() -> tuple[ConversationMemory, Assistant]:
    """Create the database, memory and assistant once per server process."""
    engine = create_db_engine(settings.db_path)
    init_db(engine)
    memory = ConversationMemory(engine)
    return memory, Assistant(memory)


def ensure_conversation(memory: ConversationMemory) -> int:
    """Return a valid current conversation id (session state only stores this pointer)."""
    conversations = memory.list_conversations()
    current = st.session_state.get("conversation_id")
    if current not in {c.id for c in conversations}:
        current = conversations[0].id if conversations else memory.create_conversation()
        st.session_state["conversation_id"] = current
        st.session_state.pop("pending_action", None)
    return current


def render_connection_status() -> None:
    if "conn_status" not in st.session_state:
        st.session_state["conn_status"] = check_connection()
    status = st.session_state["conn_status"]
    if status.ok:
        st.success(status.message)
    elif status.server_up:
        st.warning(status.message)
    else:
        st.error(status.message)
    if st.button("Re-check connection"):
        st.session_state["conn_status"] = check_connection()
        st.rerun()


def render_destructive_controls(memory: ConversationMemory, current_id: int) -> None:
    """Clear/Delete with an explicit confirmation step."""
    pending = st.session_state.get("pending_action")
    if pending and pending[1] == current_id:
        action = pending[0]
        st.warning(f"Really {action} this conversation? This cannot be undone.")
        confirm, cancel = st.columns(2)
        if confirm.button("Confirm", type="primary"):
            if action == "clear":
                memory.clear_conversation(current_id)
            else:
                memory.delete_conversation(current_id)
                st.session_state.pop("conversation_id", None)
            st.session_state.pop("pending_action", None)
            st.rerun()
        if cancel.button("Cancel"):
            st.session_state.pop("pending_action", None)
            st.rerun()
        return

    col_clear, col_delete = st.columns(2)
    if col_clear.button("Clear", help="Delete messages, keep the conversation"):
        st.session_state["pending_action"] = ("clear", current_id)
        st.rerun()
    if col_delete.button("Delete", help="Delete this conversation"):
        st.session_state["pending_action"] = ("delete", current_id)
        st.rerun()


def render_sidebar(memory: ConversationMemory, current_id: int) -> None:
    with st.sidebar:
        st.header("🤖 Assistant")
        render_connection_status()

        st.divider()
        if st.button("➕ New conversation", type="primary"):
            st.session_state["conversation_id"] = memory.create_conversation()
            st.session_state.pop("pending_action", None)
            st.rerun()
        render_destructive_controls(memory, current_id)

        st.subheader("Conversations")
        for conv in memory.list_conversations():
            label = ("▶ " if conv.id == current_id else "") + conv.title
            if st.button(label, key=f"conv_{conv.id}"):
                st.session_state["conversation_id"] = conv.id
                st.session_state.pop("pending_action", None)
                st.rerun()


def stream_reply(assistant: Assistant, conversation_id: int, prompt: str):
    """Yield reply chunks, showing a 'thinking' hint until the first one arrives."""
    indicator = st.empty()
    indicator.caption("⏳ Working on it… the reply appears when it is ready (CPU inference can take a while)")
    try:
        for piece in assistant.respond_stream(conversation_id, prompt):
            indicator.empty()
            yield piece
    finally:
        indicator.empty()


def main() -> None:
    st.title("Personal AI Assistant")
    try:
        memory, assistant = get_services()
        current_id = ensure_conversation(memory)
        render_sidebar(memory, current_id)

        for msg in memory.get_messages(current_id):
            with st.chat_message(msg.role):
                st.markdown(msg.content)

        prompt = st.chat_input("Message your assistant…")
        if not prompt:
            return

        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                st.write_stream(stream_reply(assistant, current_id, prompt))
            except ValueError as exc:
                st.warning(str(exc))
                return
            except OllamaUnavailableError as exc:
                st.session_state["conn_status"] = check_connection()
                st.error(f"{exc}\n\nYour message was not saved; fix the issue and send it again.")
                return
        st.rerun()  # reload from SQLite so the sidebar title updates
    except DatabaseError as exc:
        st.error(f"Database error: {exc}")


main()
