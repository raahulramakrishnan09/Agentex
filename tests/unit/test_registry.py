from tools import default_registry
from tools.registry import ToolInputError, ToolRegistry, ToolSpec


def make_spec(func, name="demo"):
    return ToolSpec(
        name=name,
        description="demo tool",
        parameters={"type": "object", "properties": {"text": {"type": "string"}, "count": {"type": "integer"}}, "required": ["text"]},
        func=func,
    )


def test_default_registry_exposes_calculator_schema():
    schemas = default_registry().openai_schemas()
    assert [s["function"]["name"] for s in schemas] == ["calculator"]
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["parameters"]["required"] == ["expression"]


def test_execute_runs_the_calculator():
    result = default_registry().execute("calculator", {"expression": "8500 * 15%"})
    assert result.ok and result.content == "8500 * 15% = 1275"


def test_arguments_may_arrive_as_a_json_string():
    result = default_registry().execute("calculator", '{"expression": "1 + 1"}')
    assert result.ok and result.content.endswith("= 2")


def test_unknown_tool_is_reported_not_raised():
    result = default_registry().execute("browser", {"url": "x"})
    assert not result.ok and "unknown tool 'browser'" in result.content and "calculator" in result.content


def test_invalid_arguments_are_rejected():
    registry = default_registry()
    assert "missing required" in registry.execute("calculator", {}).content
    assert "unexpected argument" in registry.execute("calculator", {"expression": "1", "x": 2}).content
    assert "must be of type string" in registry.execute("calculator", {"expression": 5}).content
    assert "not valid JSON" in registry.execute("calculator", "{oops").content
    assert "must be an object" in registry.execute("calculator", [1, 2]).content


def test_invalid_expression_becomes_an_error_result():
    result = default_registry().execute("calculator", {"expression": "2 +"})
    assert not result.ok and result.content.startswith("Error:")


def test_division_by_zero_becomes_an_error_result():
    result = default_registry().execute("calculator", {"expression": "5 / 0"})
    assert not result.ok and "division by zero" in result.content


def test_unexpected_tool_exception_is_contained():
    def boom(text):
        raise RuntimeError("secret internal detail")

    result = ToolRegistry([make_spec(boom)]).execute("demo", {"text": "x"})
    assert not result.ok
    assert "failed unexpectedly" in result.content
    assert "secret internal detail" not in result.content


def test_tool_input_error_message_is_passed_through():
    def reject(text):
        raise ToolInputError("that is not allowed")

    result = ToolRegistry([make_spec(reject)]).execute("demo", {"text": "x"})
    assert result.content == "Error: that is not allowed"


def test_integer_argument_rejects_booleans_and_strings():
    registry = ToolRegistry([make_spec(lambda text, count=0: "ok")])
    assert not registry.execute("demo", {"text": "a", "count": True}).ok
    assert not registry.execute("demo", {"text": "a", "count": "3"}).ok
    assert registry.execute("demo", {"text": "a", "count": 3}).ok


def test_duplicate_registration_is_rejected():
    registry = ToolRegistry([make_spec(lambda text: "x")])
    try:
        registry.register(make_spec(lambda text: "y"))
    except ValueError:
        return
    raise AssertionError("duplicate tool name accepted")
