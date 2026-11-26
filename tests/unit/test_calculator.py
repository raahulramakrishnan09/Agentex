from tools.calculator_tool import CalculatorError, calculate


def error_of(expression):
    try:
        calculate(expression)
    except CalculatorError as exc:
        return str(exc)
    raise AssertionError(f"{expression!r} should have been rejected")


# --- basic arithmetic
def test_addition_and_subtraction():
    assert calculate("1250 + 3750") == "5000"
    assert calculate("100 - 42") == "58"


def test_multiplication_and_division():
    assert calculate("12 * 12") == "144"
    assert calculate("10 / 4") == "2.5"
    assert calculate("1 / 3") == "0.3333333333"


def test_decimals_are_exact():
    assert calculate("0.1 + 0.2") == "0.3"
    assert calculate("19.99 * 3") == "59.97"


def test_parentheses_and_precedence():
    assert calculate("2 + 3 * 4") == "14"
    assert calculate("(2 + 3) * 4") == "20"
    assert calculate("((1 + 2) * (3 + 4)) / 7") == "3"


def test_unary_signs_and_power():
    assert calculate("-5 + 3") == "-2"
    assert calculate("-2^2") == "-4"
    assert calculate("2^10") == "1024"
    assert calculate("2 ** 3 ** 2") == "512"
    assert calculate("2^-1") == "0.5"


def test_thousands_separators_and_unicode_operators():
    assert calculate("1,250 + 3,750") == "5000"
    assert calculate("6 × 7") == "42"
    assert calculate("8 ÷ 2") == "4"


# --- percentages
def test_percentage_of_a_number():
    assert calculate("8500 * 15%") == "1275"
    assert calculate("15% of 8500") == "1275"
    assert calculate("15% * 8500") == "1275"


def test_percentage_added_to_a_number():
    assert calculate("100 + 100 * 20%") == "120"
    assert calculate("2 + 15% of 8500") == "1277"


# --- multi-step
def test_multi_step_percentage_then_addition():
    assert calculate("20% of 5000 + 750") == "1750"
    assert calculate("5000 * 20% + 750") == "1750"


# --- invalid expressions
def test_empty_and_incomplete_expressions():
    assert "empty" in error_of("   ")
    error_of("2 +")
    error_of("* 3")
    error_of("()")
    error_of("2 3")


def test_unbalanced_parentheses():
    assert "parenthesis" in error_of("(2 + 3")
    error_of("2 + 3)")


def test_unsupported_characters_and_words():
    error_of("2 $ 3")
    error_of("2 + abc")
    error_of("1,2 + 3")


def test_of_requires_a_percentage():
    assert "percentage" in error_of("10 of 5")


def test_code_injection_is_rejected():
    for attempt in ("__import__('os').system('dir')", "open('x')", "1; import os", "[1,2][0]", "exec('1')"):
        error_of(attempt)


def test_oversized_inputs_are_rejected():
    assert "too long" in error_of("1+" * 150 + "1")
    assert "nested" in error_of("(" * 60 + "1" + ")" * 60)
    assert "exponent" in error_of("2^1000")
    assert "too large" in error_of("10^50")


# --- division by zero
def test_division_by_zero():
    assert "division by zero" in error_of("5 / 0")
    assert "division by zero" in error_of("0 / 0")
    assert "division by zero" in error_of("1 / (2 - 2)")
    assert "division by zero" in error_of("0 ^ -1")


def test_undefined_power():
    error_of("(-8) ^ 0.5")
