import pytest

from database.db import create_db_engine, init_db
from memory.conversation_memory import (
    DEFAULT_TITLE,
    ConversationMemory,
    ConversationNotFoundError,
)


@pytest.fixture()
def memory(tmp_path):
    engine = create_db_engine(tmp_path / "test.db")
    init_db(engine)
    return ConversationMemory(engine)


def test_database_is_created_automatically(tmp_path):
    path = tmp_path / "nested" / "x.db"
    init_db(create_db_engine(path))
    assert path.exists()


def test_messages_persist_in_order(memory):
    cid = memory.create_conversation()
    memory.add_message(cid, "user", "hello")
    memory.add_message(cid, "assistant", "hi there")
    assert [(m.role, m.content) for m in memory.get_messages(cid)] == [
        ("user", "hello"), ("assistant", "hi there"),
    ]


def test_conversations_are_isolated(memory):
    a, b = memory.create_conversation(), memory.create_conversation()
    memory.add_message(a, "user", "secret in A")
    assert memory.get_messages(b) == []


def test_recent_messages_limit(memory):
    cid = memory.create_conversation()
    for i in range(5):
        memory.add_message(cid, "user", f"m{i}")
    assert [m.content for m in memory.get_recent_messages(cid, 2)] == ["m3", "m4"]
    assert memory.get_recent_messages(cid, 0) == []


def test_first_user_message_sets_title(memory):
    cid = memory.create_conversation()
    memory.add_message(cid, "user", "Plan my week")
    assert memory.list_conversations()[0].title == "Plan my week"


def test_clear_keeps_conversation_and_resets_title(memory):
    cid = memory.create_conversation()
    memory.add_message(cid, "user", "hello")
    memory.clear_conversation(cid)
    assert memory.get_messages(cid) == []
    assert memory.list_conversations()[0].title == DEFAULT_TITLE


def test_delete_cascades_messages(memory):
    cid = memory.create_conversation()
    memory.add_message(cid, "user", "hello")
    memory.delete_conversation(cid)
    assert memory.list_conversations() == []
    assert memory.get_messages(cid) == []


def test_invalid_inputs(memory):
    cid = memory.create_conversation()
    with pytest.raises(ValueError):
        memory.add_message(cid, "robot", "x")
    with pytest.raises(ValueError):
        memory.add_message(cid, "user", "   ")
    with pytest.raises(ConversationNotFoundError):
        memory.add_message(9999, "user", "x")
