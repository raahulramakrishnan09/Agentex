"""Streaming filter that keeps model reasoning out of user-visible text.

Qwen3 can leak reasoning in three shapes:

1. ``<think>reasoning</think>answer``            (normal tagged block)
2. ``reasoning</think>answer``                   (the opening tag is missing; some
   Qwen3 variants have ``<think>`` pre-filled by the chat template, so it is
   never generated)
3. ``answer``                                    (no reasoning at all)

Case 2 cannot be recognised until ``</think>`` arrives, so the beginning of the
stream is *held back* for up to ``hold_chars`` characters. If ``</think>``
shows up inside that window, everything before it is discarded as reasoning.
If a complete reply ends (or the window is exceeded) without any tag, the held
text is released unchanged as the answer.

No pattern matching is used on the content itself: only the two literal tags
are ever interpreted.
"""
from __future__ import annotations

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"

_HOLD = "hold"      # nothing decided yet; text is buffered
_INSIDE = "inside"  # inside <think>...</think>; text is discarded
_STREAM = "stream"  # in the answer; text is emitted immediately


class ThinkTagFilter:
    """Incrementally removes reasoning from a stream of text chunks.

    Use one instance per response: call :meth:`feed` for every chunk and
    :meth:`flush` once at the end.

    Args:
        hold_chars: How many characters at the start of the stream may be held
            back while waiting for a possible ``</think>``. ``0`` disables the
            hold (text streams immediately; untagged reasoning is not detected).
    """

    def __init__(self, hold_chars: int = 20000) -> None:
        self._hold_chars = max(0, hold_chars)
        self._buffer = ""
        self._state = _HOLD if self._hold_chars > 0 else _STREAM
        self._started = False       # has any visible text been emitted yet?
        self.stripped_chars = 0     # how much reasoning was discarded (for logging)

    @staticmethod
    def _partial_tag_suffix_len(text: str) -> int:
        """Length of a trailing piece of ``text`` that could be the start of a tag."""
        longest = 0
        for tag in (OPEN_TAG, CLOSE_TAG):
            for size in range(min(len(tag) - 1, len(text)), 0, -1):
                if text.endswith(tag[:size]):
                    longest = max(longest, size)
                    break
        return longest

    def _emit(self, text: str, out: list[str]) -> None:
        if not self._started:
            text = text.lstrip()  # drop the blank lines that follow </think>
            if not text:
                return
            self._started = True
        out.append(text)

    def feed(self, chunk: str) -> str:
        """Consume a chunk; return the visible text that is safe to show now."""
        self._buffer += chunk
        out: list[str] = []
        while self._step(out):
            pass
        return "".join(out)

    def _step(self, out: list[str]) -> bool:
        """Process the buffer once. Returns True if the loop should run again."""
        buf = self._buffer

        if self._state == _INSIDE:
            end = buf.find(CLOSE_TAG)
            if end == -1:
                keep = buf[-(len(CLOSE_TAG) - 1):]
                self.stripped_chars += len(buf) - len(keep)
                self._buffer = keep
                return False
            self.stripped_chars += end
            self._buffer = buf[end + len(CLOSE_TAG):]
            self._state = _STREAM
            return True

        opened = buf.find(OPEN_TAG)
        closed = buf.find(CLOSE_TAG)

        if opened != -1 and (closed == -1 or opened < closed):
            self._emit(buf[:opened], out)
            self._buffer = buf[opened + len(OPEN_TAG):]
            self._state = _INSIDE
            return True

        if closed != -1:
            before = buf[:closed]
            if self._state == _HOLD:
                self.stripped_chars += len(before)  # untagged reasoning
            else:
                self._emit(before, out)  # answer already started: drop the tag only
            self._buffer = buf[closed + len(CLOSE_TAG):]
            self._state = _STREAM
            return True

        # No complete tag in the buffer.
        if self._state == _HOLD:
            if len(buf) > self._hold_chars:
                self._state = _STREAM  # hold window exceeded: treat as the answer
                return True
            return False

        keep = self._partial_tag_suffix_len(buf)
        cut = len(buf) - keep
        self._emit(buf[:cut], out)
        self._buffer = buf[cut:]
        return False

    def flush(self) -> str:
        """Call once when the stream ends; returns any remaining visible text.

        An unfinished ``<think>`` block yields nothing (it is reasoning with no
        answer). Held text that never met a tag is a normal reply and is released.
        """
        out: list[str] = []
        if self._state == _INSIDE:
            self.stripped_chars += len(self._buffer)
        else:
            self._emit(self._buffer, out)
        self._buffer = ""
        return "".join(out)
