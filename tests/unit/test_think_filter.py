from llm.think_filter import ThinkTagFilter


def run(chunks, hold_chars=3000):
    f = ThinkTagFilter(hold_chars=hold_chars)
    text = "".join(f.feed(c) for c in chunks) + f.flush()
    return text, f


def by_char(text, hold_chars=3000):
    return run(list(text), hold_chars)[0]


# 1. normal tagged output
def test_tagged_block_is_removed():
    assert run(["<think>reasoning</think>answer"])[0] == "answer"


def test_blank_lines_after_closing_tag_are_dropped():
    assert run(["<think>\n\n</think>\n\n1275"])[0] == "1275"


# 2. reasoning followed by </think> without an opening tag
def test_untagged_reasoning_is_removed():
    chunks = ["Okay, the user wants 15% of 8500. ", "Let me compute 8500 * 0.15.", "</think>", "\n\nThe answer is 1275."]
    text, f = run(chunks)
    assert text == "The answer is 1275."
    assert f.stripped_chars > 0


def test_untagged_reasoning_in_a_single_chunk():
    assert run(["Okay, thinking...</think>1275"])[0] == "1275"


# 3. tags split across chunks
def test_split_opening_and_closing_tags():
    # Surrounding text keeps its own whitespace, so the removed block leaves two spaces.
    assert run(["Hi <th", "ink>hidden</th", "ink> there"])[0] == "Hi  there"


def test_split_closing_tag_without_opening_tag():
    assert run(["Okay, let me think.</th", "ink>", "1275"])[0] == "1275"


def test_character_by_character_matches_whole_chunk():
    samples = [
        "<think>reasoning</think>answer",
        "Okay, let me think.</think>\n\n1275",
        "plain answer with a < b and <b>bold</b>",
        "Hi <think>x</think>there <think>y</think>friend",
    ]
    for sample in samples:
        assert by_char(sample) == run([sample])[0], sample


# 4. normal responses without tags
def test_plain_answer_is_untouched():
    assert run(["Hello ", "world"])[0] == "Hello world"


def test_angle_brackets_and_inner_newlines_are_preserved():
    text = "if a < b:\n    print('<ok>')\n\nDone"
    assert run([text])[0] == text


def test_answer_after_reasoning_is_never_discarded():
    text, _ = run(["<think>a</think>First part. ", "Second part."])
    assert text == "First part. Second part."


def test_multiple_think_blocks():
    assert run(["<think>a</think>One <think>b</think>Two"])[0] == "One Two"


# 5. empty or incomplete streams
def test_empty_stream():
    assert run([])[0] == ""
    assert run([""])[0] == ""


def test_unclosed_think_block_produces_nothing():
    text, f = run(["<think>reasoning that never ends"])
    assert text == ""
    assert f.stripped_chars > 0


def test_stream_ending_with_partial_tag_keeps_visible_text():
    assert run(["Answer <thi"])[0] == "Answer <thi"


def test_only_whitespace_stream():
    assert run(["  \n "])[0] == ""


# hold-window behaviour
def test_zero_hold_streams_immediately():
    f = ThinkTagFilter(hold_chars=0)
    assert f.feed("Hello") == "Hello"


def test_hold_delays_first_text_until_decision():
    f = ThinkTagFilter(hold_chars=100)
    assert f.feed("Okay, reasoning") == ""
    assert f.feed("</think>Answer") == "Answer"


def test_text_longer_than_hold_window_is_released():
    f = ThinkTagFilter(hold_chars=10)
    assert f.feed("a" * 30) == "a" * 30


def test_known_limitation_reasoning_longer_than_hold_window_leaks():
    # Documented: untagged reasoning longer than hold_chars cannot be recognised.
    f = ThinkTagFilter(hold_chars=10)
    out = f.feed("x" * 30) + f.feed("</think>answer") + f.flush()
    assert out.endswith("answer") and out.startswith("x")


# regression: long reasoning for follow-up questions ("which team does he currently play for?")
LONG_REASONING = (
    "Okay, the user is asking which team he currently plays for. Let me recall the latest info. "
    "Wait, I should double-check. The safest answer is to mention my knowledge cutoff. "
) * 60  # ~10,000 characters, far beyond the old 3000-character window


def test_default_hold_window_covers_very_long_reasoning():
    text, f = run([LONG_REASONING, "</think>", "As far as I know, he plays for Al Nassr."], hold_chars=20000)
    assert text == "As far as I know, he plays for Al Nassr."
    assert f.stripped_chars >= len(LONG_REASONING)


def test_default_constructor_hold_covers_long_reasoning():
    f = ThinkTagFilter()
    out = f.feed(LONG_REASONING) + f.feed("</think>Final.") + f.flush()
    assert out == "Final."


def test_long_untagged_reasoning_fed_character_by_character():
    sample = LONG_REASONING[:4000] + "</think>Final answer."
    assert by_char(sample, hold_chars=20000) == "Final answer."


def test_long_tagged_reasoning_followed_by_streamed_answer():
    f = ThinkTagFilter()
    assert f.feed("<think>" + LONG_REASONING) == ""
    assert f.feed("</think>Hello ") == "Hello "
    assert f.feed("world") == "world"
