from agent.tool_parsing import extract_fallback_tool_calls

KNOWN = ["calculator"]


def test_tagged_block_is_recovered():
    text = '<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>'
    visible, calls = extract_fallback_tool_calls(text, KNOWN)
    assert visible == "" and len(calls) == 1
    assert calls[0]["name"] == "calculator" and calls[0]["args"] == {"expression": "1+1"} and calls[0]["id"]


def test_bare_json_object_naming_a_known_tool_is_recovered():
    _, calls = extract_fallback_tool_calls('{"name": "calculator", "arguments": {"expression": "2*2"}}', KNOWN)
    assert [c["args"] for c in calls] == [{"expression": "2*2"}]


def test_arbitrary_json_is_not_a_tool_call():
    for text in ('{"answer": 42}', '{"name": "calculator"', '[1, 2, 3]', '{"name": "rm", "arguments": {}}'):
        visible, calls = extract_fallback_tool_calls(text, KNOWN)
        assert calls == [] and visible == text


def test_unknown_tool_in_tag_is_dropped_and_hidden():
    visible, calls = extract_fallback_tool_calls('<tool_call>{"name": "rm", "arguments": {}}</tool_call>', KNOWN)
    assert calls == [] and visible == ""


def test_malformed_block_is_hidden_and_ignored():
    visible, calls = extract_fallback_tool_calls("Sure. <tool_call>{broken", KNOWN)
    assert calls == [] and visible == "Sure."


def test_non_object_arguments_are_rejected():
    _, calls = extract_fallback_tool_calls('<tool_call>{"name": "calculator", "arguments": [1]}</tool_call>', KNOWN)
    assert calls == []


def test_arguments_given_as_json_string_are_decoded():
    text = '<tool_call>{"name": "calculator", "arguments": "{\\"expression\\": \\"3+4\\"}"}</tool_call>'
    assert extract_fallback_tool_calls(text, KNOWN)[1][0]["args"] == {"expression": "3+4"}


def test_text_around_a_block_is_kept():
    visible, calls = extract_fallback_tool_calls(
        'Checking. <tool_call>{"name": "calculator", "arguments": {"expression": "1"}}</tool_call> Done.', KNOWN
    )
    assert len(calls) == 1 and visible == "Checking.  Done."


def test_plain_text_is_untouched():
    assert extract_fallback_tool_calls("Paris is the capital of France.", KNOWN) == ("Paris is the capital of France.", [])
