from dataclasses import dataclass, field, replace

import httpx
import pytest

from config import load_settings
from llm.ollama_client import (
    OllamaUnavailableError,
    apply_no_think,
    check_connection,
    stream_chat,
)


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_check_connection_ok(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp({"models": [{"name": "qwen3:4b"}]}))
    assert check_connection(load_settings()).ok


def test_check_connection_model_missing(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp({"models": []}))
    status = check_connection(load_settings())
    assert status.server_up and not status.model_available
    assert "ollama pull" in status.message


def test_check_connection_server_down(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", boom)
    assert not check_connection(load_settings()).server_up


@dataclass
class _Chunk:
    content: str
    additional_kwargs: dict = field(default_factory=dict)


class _FakeLLM:
    def __init__(self, chunks):
        self._chunks = chunks

    def stream(self, messages):
        yield from self._chunks


class _FailingLLM:
    def stream(self, messages):
        raise ConnectionError("connection refused")
        yield  # pragma: no cover


def _collect(chunks, **overrides):
    cfg = replace(load_settings(), **overrides)
    return "".join(stream_chat(_FakeLLM(chunks), [], cfg))


def test_stream_chat_wraps_errors():
    with pytest.raises(OllamaUnavailableError):
        list(stream_chat(_FailingLLM(), []))


def test_stream_chat_removes_untagged_reasoning():
    chunks = [_Chunk("Okay, the user wants 15% of 8500."), _Chunk("</think>"), _Chunk("\n\n1275")]
    assert _collect(chunks) == "1275"


def test_stream_chat_removes_tagged_reasoning():
    assert _collect([_Chunk("<think>hmm</think>1275")]) == "1275"


def test_stream_chat_never_shows_reasoning_content_field():
    chunks = [_Chunk("", {"reasoning_content": "secret chain of thought"}), _Chunk("1275")]
    assert "secret" not in _collect(chunks)


def test_stream_chat_ignores_non_string_and_empty_chunks():
    chunks = [_Chunk(""), _Chunk(["not", "a", "string"]), _Chunk("ok")]
    assert _collect(chunks) == "ok"


def test_apply_no_think_for_qwen3():
    cfg = replace(load_settings(), ollama_model="qwen3:4b", ollama_think=False, ollama_no_think_suffix=True)
    assert apply_no_think("hello", cfg) == "hello /no_think"


def test_apply_no_think_skipped_when_thinking_enabled():
    cfg = replace(load_settings(), ollama_model="qwen3:4b", ollama_think=True)
    assert apply_no_think("hello", cfg) == "hello"


def test_apply_no_think_skipped_for_other_models_or_when_disabled():
    other = replace(load_settings(), ollama_model="llama3.2:3b")
    off = replace(load_settings(), ollama_model="qwen3:4b", ollama_no_think_suffix=False)
    assert apply_no_think("hello", other) == "hello"
    assert apply_no_think("hello", off) == "hello"


def test_reasoning_is_off_by_default():
    assert load_settings().ollama_think is False


def test_default_hold_window_is_large_enough_for_a_whole_reply():
    # A reply cannot exceed the context window (~4 chars per token).
    cfg = load_settings()
    assert cfg.reasoning_hold_chars >= cfg.ollama_num_ctx * 3


def test_stream_chat_removes_very_long_follow_up_reasoning():
    reasoning = "Okay, the user is asking... Let me recall the latest info. Wait, I should double-check. " * 80
    chunks = [_Chunk(reasoning[i:i + 40]) for i in range(0, len(reasoning), 40)]
    chunks += [_Chunk("</think>"), _Chunk("\n\nAs far as I know, he plays for Al Nassr.")]
    assert _collect(chunks) == "As far as I know, he plays for Al Nassr."
