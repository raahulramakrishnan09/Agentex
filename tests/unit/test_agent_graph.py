"""Agent behaviour with a scripted model (no Ollama). LangGraph is required for these."""
from dataclasses import replace

from agent.agent import Assistant
from agent.prompts import LIMIT_MESSAGE
from config import load_settings
from tests.unit.helpers import FakeMemory, ScriptedToolLLM, ask, text_reply, tool_reply
from tools import ToolRegistry, ToolSpec, default_registry
from llm.ollama_client import OllamaUnavailableError

CFG = load_settings()


def build(replies, cfg=CFG, registry=None, memory=None, repeat_last=False):
    llm = ScriptedToolLLM(replies, repeat_last=repeat_last)
    memory = memory if memory is not None else FakeMemory()
    return Assistant(memory, llm=llm, cfg=cfg, registry=registry or default_registry()), llm, memory


def tool_messages(prompt):
    return [m for m in prompt if m.type == "tool"]


# --- direct answers: no tool
def test_direct_question_does_not_use_tools():
    assistant, llm, _ = build([text_reply("Cristiano Ronaldo is a Portuguese footballer.")])
    events = []
    assert ask(assistant, "Who is Cristiano Ronaldo?", events=events).startswith("Cristiano Ronaldo")
    assert events == [] and len(llm.calls) == 1
    assert tool_messages(llm.calls[0]) == []


def test_calculator_schema_is_bound_to_the_model():
    assistant, llm, _ = build([text_reply("An AI agent is software that acts toward goals.")])
    ask(assistant, "Explain what an AI agent is")
    assert [t["function"]["name"] for t in llm.bound] == ["calculator"]


def test_system_prompt_tells_the_model_when_to_use_the_calculator():
    assistant, llm, _ = build([text_reply("ok")])
    ask(assistant, "hello")
    system = llm.calls[0][0].content
    assert "calculator" in system and "general knowledge" in system and "never do such arithmetic yourself" in system


# --- calculator use
def test_percentage_uses_the_calculator_and_reports_the_result():
    assistant, llm, memory = build([
        tool_reply("calculator", {"expression": "8500 * 15%"}),
        text_reply("15% of 8500 is 1275."),
    ])
    events = []
    assert ask(assistant, "Calculate 15% of 8500", events=events) == "15% of 8500 is 1275."
    assert [e.kind for e in events] == ["tool_call", "tool_result"]
    assert events[0].name == "calculator" and "8500 * 15%" in events[0].detail
    result = tool_messages(llm.calls[1])[0]
    assert result.content == "8500 * 15% = 1275" and result.tool_call_id == "call_1"
    assert [m.role for m in memory.rows] == ["user", "assistant"]


def test_plain_addition_uses_the_calculator():
    assistant, llm, _ = build([
        tool_reply("calculator", {"expression": "1250 + 3750"}),
        text_reply("1250 + 3750 = 5000."),
    ])
    assert "5000" in ask(assistant, "What is 1250 + 3750?")
    assert tool_messages(llm.calls[1])[0].content.endswith("= 5000")


def test_multi_step_calculation_in_one_expression():
    assistant, llm, _ = build([
        tool_reply("calculator", {"expression": "5000 * 20% + 750"}),
        text_reply("The result is 1750."),
    ])
    assert "1750" in ask(assistant, "Calculate 20% of 5000 and add 750")
    assert tool_messages(llm.calls[1])[0].content.endswith("= 1750")


def test_multiple_sequential_tool_calls():
    assistant, llm, _ = build([
        tool_reply("calculator", {"expression": "5000 * 20%"}, call_id="a"),
        tool_reply("calculator", {"expression": "1000 + 750"}, call_id="b"),
        text_reply("The total is 1750."),
    ])
    events = []
    assert ask(assistant, "20% of 5000, then add 750", events=events) == "The total is 1750."
    assert [e.kind for e in events] == ["tool_call", "tool_result", "tool_call", "tool_result"]
    assert len(llm.calls) == 3
    assert [m.content for m in tool_messages(llm.calls[2])] == ["5000 * 20% = 1000", "1000 + 750 = 1750"]


def test_several_tool_calls_in_one_turn():
    two_calls = text_reply("")
    two_calls.tool_calls = [
        {"name": "calculator", "args": {"expression": "1+1"}, "id": "x", "type": "tool_call"},
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "y", "type": "tool_call"},
    ]
    assistant, llm, _ = build([two_calls, text_reply("2 and 4.")])
    ask(assistant, "compute 1+1 and 2+2")
    assert [m.tool_call_id for m in tool_messages(llm.calls[1])] == ["x", "y"]


# --- consecutive questions and context
def test_consecutive_questions_in_one_conversation():
    assistant, llm, memory = build([
        text_reply("Paris."),
        tool_reply("calculator", {"expression": "2 * 21"}),
        text_reply("42."),
    ])
    assert ask(assistant, "Capital of France?") == "Paris."
    assert ask(assistant, "What is 2 * 21?") == "42."
    assert [m.role for m in memory.rows] == ["user", "assistant", "user", "assistant"]
    # the calculator trace is never stored, only the final text
    assert all(m.type != "tool" for call in llm.calls[:1] for m in call)
    assert all("calculator" not in m.content and "=" not in m.content for m in memory.rows)


def test_follow_up_uses_conversation_context_without_a_tool():
    assistant, llm, _ = build([text_reply("Paris."), text_reply("Your previous question was: Capital of France?")])
    ask(assistant, "Capital of France?")
    events = []
    reply = ask(assistant, "What was my previous question?", events=events)
    assert "Capital of France" in reply and events == []
    prompt = llm.calls[1]
    assert [m.type for m in prompt] == ["system", "human", "ai", "human"]
    assert "Capital of France?" in prompt[1].content and prompt[2].content == "Paris."


def test_follow_up_can_build_on_a_previous_calculation_result():
    assistant, llm, _ = build([
        tool_reply("calculator", {"expression": "8500 * 15%"}),
        text_reply("15% of 8500 is 1275."),
        tool_reply("calculator", {"expression": "1275 * 2"}),
        text_reply("Double that is 2550."),
    ])
    ask(assistant, "15% of 8500?")
    assert "2550" in ask(assistant, "Double that.")
    assert "1275" in " ".join(m.content for m in llm.calls[2] if m.type == "ai")


def test_conversations_do_not_share_context():
    assistant, llm, _ = build([text_reply("one"), text_reply("two")])
    ask(assistant, "secret in conversation 1", cid=1)
    ask(assistant, "hello", cid=2)
    assert "secret" not in " ".join(str(m.content) for m in llm.calls[1])


# --- errors and robustness
def test_invalid_expression_is_explained_not_crashed():
    assistant, llm, _ = build([
        tool_reply("calculator", {"expression": "2 +"}),
        text_reply("That expression is incomplete."),
    ])
    events = []
    assert ask(assistant, "calculate 2 +", events=events) == "That expression is incomplete."
    assert events[-1].kind == "tool_error" and tool_messages(llm.calls[1])[0].content.startswith("Error:")


def test_division_by_zero_is_reported():
    assistant, llm, _ = build([
        tool_reply("calculator", {"expression": "5 / 0"}),
        text_reply("Division by zero is undefined."),
    ])
    events = []
    ask(assistant, "What is 5 divided by 0?", events=events)
    assert "division by zero" in events[-1].detail and events[-1].kind == "tool_error"


def test_tool_execution_failure_is_contained():
    def boom(expression):
        raise RuntimeError("disk on fire")

    registry = ToolRegistry([ToolSpec(
        name="calculator", description="d", func=boom,
        parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    )])
    assistant, llm, memory = build(
        [tool_reply("calculator", {"expression": "1+1"}), text_reply("Sorry, the calculator failed.")], registry=registry
    )
    events = []
    assert ask(assistant, "1+1?", events=events) == "Sorry, the calculator failed."
    assert events[-1].kind == "tool_error" and "disk on fire" not in events[-1].detail
    assert memory.rows[-1].content == "Sorry, the calculator failed."


def test_unknown_tool_requested_by_the_model():
    assistant, llm, _ = build([tool_reply("browser", {"url": "x"}), text_reply("I can't browse.")])
    events = []
    assert ask(assistant, "open a website", events=events) == "I can't browse."
    assert "unknown tool" in events[-1].detail


def test_invalid_arguments_from_the_model_are_rejected():
    assistant, llm, _ = build([tool_reply("calculator", {"expr": "1+1"}), text_reply("Let me retry.")])
    events = []
    ask(assistant, "1+1", events=events)
    assert "unexpected argument" in events[-1].detail or "missing required" in events[-1].detail


def test_text_written_with_a_tool_call_is_not_shown_or_saved():
    assistant, _, memory = build([
        tool_reply("calculator", {"expression": "1+1"}, content="Sure, calculating now..."),
        text_reply("2."),
    ])
    assert ask(assistant, "1+1?") == "2."
    assert memory.rows[-1].content == "2."


def test_text_form_tool_call_is_recovered_and_executed():
    assistant, llm, _ = build([
        text_reply('<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"}}</tool_call>'),
        text_reply("42."),
    ])
    assert ask(assistant, "6 times 7") == "42."
    assert tool_messages(llm.calls[1])[0].content == "6*7 = 42"


def test_reasoning_is_removed_from_agent_replies_and_not_saved():
    assistant, _, memory = build([text_reply("Okay, the user wants a greeting. Let me think.</think>\n\nHello!")])
    assert ask(assistant, "hi") == "Hello!"
    assert memory.rows[-1].content == "Hello!"


def test_too_many_tool_calls_in_one_turn_are_skipped():
    many = text_reply("")
    many.tool_calls = [
        {"name": "calculator", "args": {"expression": f"{i}+1"}, "id": f"c{i}", "type": "tool_call"} for i in range(6)
    ]
    cfg = replace(CFG, agent_max_tool_calls_per_turn=2)
    assistant, llm, _ = build([many, text_reply("done")], cfg=cfg)
    ask(assistant, "many")
    contents = [m.content for m in tool_messages(llm.calls[1])]
    assert len(contents) == 6 and contents[0].endswith("= 1") and "too many tool calls" in contents[5]


# --- iteration limit
def test_agent_iteration_limit_stops_a_tool_loop():
    cfg = replace(CFG, agent_max_iterations=3)
    assistant, llm, memory = build([tool_reply("calculator", {"expression": "1+1"})], cfg=cfg, repeat_last=True)
    events = []
    reply = ask(assistant, "loop forever", events=events)
    assert reply == LIMIT_MESSAGE.format(limit=3)
    assert len(llm.calls) == 3                                   # exactly max_iterations model calls
    assert [e.kind for e in events].count("tool_result") == 2    # no tool run after the final model call
    assert memory.rows[-1].content == reply


def test_iteration_limit_of_one_allows_a_direct_answer():
    cfg = replace(CFG, agent_max_iterations=1)
    assistant, llm, _ = build([text_reply("Direct answer.")], cfg=cfg)
    assert ask(assistant, "hello") == "Direct answer."


# --- failures and persistence
def test_model_failure_saves_nothing():
    assistant, _, memory = build([ConnectionError("connection refused")])
    try:
        ask(assistant, "hello")
    except OllamaUnavailableError:
        assert memory.rows == []
        return
    raise AssertionError("OllamaUnavailableError expected")


def test_empty_input_is_rejected():
    assistant, llm, _ = build([text_reply("x")])
    try:
        ask(assistant, "   ")
    except ValueError:
        assert llm.calls == []
        return
    raise AssertionError("ValueError expected")


def test_saved_history_never_contains_tool_traces_or_switches():
    assistant, _, memory = build([
        tool_reply("calculator", {"expression": "1+1"}),
        text_reply("Answer: 2."),
    ])
    ask(assistant, "1+1?")
    assert [(m.role, m.content) for m in memory.rows] == [("user", "1+1?"), ("assistant", "Answer: 2.")]


def test_agent_can_be_switched_off_for_plain_chat():
    llm = ScriptedToolLLM([text_reply("Plain reply.")])
    assistant = Assistant(FakeMemory(), llm=llm, cfg=replace(CFG, agent_enabled=False))
    assert ask(assistant, "hello") == "Plain reply."
    assert llm.bound == []                                   # no tools bound
    assert "calculator" not in llm.calls[0][0].content       # no tool rules in the prompt


# --- backstops around the graph itself
class _RaisingGraph:
    def __init__(self, exc):
        self._exc = exc

    def stream(self, *args, **kwargs):
        raise self._exc
        yield  # pragma: no cover


def test_langgraph_recursion_backstop_returns_the_limit_message():
    class GraphRecursionError(Exception):  # matched by class name, as LangGraph's own error is
        pass

    assistant, _, memory = build([])
    assistant._graph = _RaisingGraph(GraphRecursionError("limit"))
    reply = ask(assistant, "loop")
    assert reply == LIMIT_MESSAGE.format(limit=CFG.agent_max_iterations)
    assert memory.rows[-1].content == reply


def test_unexpected_agent_failure_is_wrapped_and_nothing_is_saved():
    from agent.agent import AgentError

    assistant, _, memory = build([])
    assistant._graph = _RaisingGraph(KeyError("bug"))
    try:
        ask(assistant, "hello")
    except AgentError:
        assert memory.rows == []
        return
    raise AssertionError("AgentError expected")
