from datetime import datetime

from agent.prompts import get_system_prompt

NOW = datetime(2026, 11, 23, 9, 0)


def test_prompt_forbids_showing_reasoning():
    prompt = get_system_prompt(NOW)
    assert "final answer only" in prompt
    assert "Okay, the user" in prompt  # named as a forbidden opening


def test_prompt_asks_for_concise_answers_and_pronoun_resolution():
    prompt = get_system_prompt(NOW)
    assert "concise" in prompt
    assert '"he"' in prompt and "earlier messages" in prompt


def test_prompt_handles_current_facts_without_deliberation():
    prompt = get_system_prompt(NOW)
    assert "may be out of date" in prompt
    assert "Do not deliberate" in prompt


def test_prompt_contains_date():
    assert "Monday, 23 November 2026" in get_system_prompt(NOW)


def test_no_think_switch_only_when_requested():
    assert get_system_prompt(NOW, no_think=True).endswith("/no_think")
    assert "/no_think" not in get_system_prompt(NOW)


def test_tool_rules_appear_only_when_tools_exist():
    assert "Tools:" not in get_system_prompt(NOW)
    prompt = get_system_prompt(NOW, tool_hints=["calculator: use it for arithmetic"])
    assert "Tools:" in prompt and "calculator: use it for arithmetic" in prompt
    assert "without any tool" in prompt and "write nothing else" in prompt


def test_no_think_switch_stays_last_with_tools():
    assert get_system_prompt(NOW, no_think=True, tool_hints=["x: y"]).endswith("/no_think")
