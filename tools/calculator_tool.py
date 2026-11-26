"""Safe calculator tool.

Uses a small hand-written parser with exact ``Decimal`` arithmetic. Nothing is
ever passed to ``eval``/``exec``, so model-generated text cannot run code.

Grammar (lowest to highest precedence):
    expression := term (("+" | "-") term)*
    term       := unary (("*" | "/" | "of") unary)*      # "of" only after a percentage
    unary      := ("+" | "-") unary | power
    power      := postfix ("^" | "**") unary
    postfix    := primary "%"*                            # 15% means 15/100
    primary    := number | "(" expression ")"
"""
from __future__ import annotations

import re
from decimal import Context, Decimal, InvalidOperation, localcontext

from tools.registry import ToolInputError, ToolSpec

MAX_EXPRESSION_LENGTH = 200
MAX_NESTING = 30
MAX_EXPONENT = 100
MAX_ABS_RESULT = Decimal("1e30")
RESULT_PLACES = Decimal("1e-10")


class CalculatorError(ToolInputError):
    """The expression is invalid or cannot be evaluated."""


_TOKEN_RE = re.compile(
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"|(?P<pow>\*\*|\^)"
    r"|(?P<op>[-+*/%()])"
    r"|(?P<word>[A-Za-z]+)"
)


def _tokenize(expression: str) -> list[tuple[str, str]]:
    text = expression.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            raise CalculatorError(
                f"unsupported character '{text[pos]}'. Use numbers, + - * / ^ ( ) and %."
            )
        kind, value = match.lastgroup or "", match.group()
        if kind == "word":
            if value.lower() != "of":
                raise CalculatorError(f"unsupported word '{value}'. Only numeric expressions are allowed.")
            kind = "of"
        tokens.append((kind, value))
        pos = match.end()
    return tokens


def _check(value: Decimal) -> Decimal:
    if value.is_nan() or value.is_infinite() or abs(value) > MAX_ABS_RESULT:
        raise CalculatorError("the result is too large to compute.")
    return value


def _power(base: Decimal, exponent: Decimal) -> Decimal:
    if abs(exponent) > MAX_EXPONENT:
        raise CalculatorError(f"the exponent is too large (maximum {MAX_EXPONENT}).")
    if base == 0 and exponent < 0:
        raise CalculatorError("division by zero is undefined.")
    try:
        return _check(base**exponent)
    except InvalidOperation as exc:
        raise CalculatorError("that power is not defined for these numbers.") from exc


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._depth = 0
        self._after_percent = False  # was the operand just parsed written with a "%"?

    def _peek(self) -> tuple[str, str] | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _take(self) -> tuple[str, str] | None:
        token = self._peek()
        self._pos += 1
        return token

    def _enter(self) -> None:
        self._depth += 1
        if self._depth > MAX_NESTING:
            raise CalculatorError("the expression is nested too deeply.")

    def parse(self) -> Decimal:
        if not self._tokens:
            raise CalculatorError("the expression is empty.")
        value = self._expression()
        leftover = self._peek()
        if leftover is not None:
            raise CalculatorError(f"unexpected '{leftover[1]}'.")
        return value

    def _expression(self) -> Decimal:
        value = self._term()
        while (tok := self._peek()) is not None and tok[0] == "op" and tok[1] in ("+", "-"):
            self._take()
            right = self._term()
            value = _check(value + right if tok[1] == "+" else value - right)
        return value

    def _term(self) -> Decimal:
        value = self._unary()
        left_was_percent = self._after_percent
        while (tok := self._peek()) is not None and (
            tok[0] == "of" or (tok[0] == "op" and tok[1] in ("*", "/"))
        ):
            self._take()
            if tok[0] == "of" and not left_was_percent:
                raise CalculatorError("'of' can only follow a percentage, for example '15% of 8500'.")
            right = self._unary()
            if tok[1] == "/":
                if right == 0:
                    raise CalculatorError("division by zero is undefined.")
                value = _check(value / right)
            else:
                value = _check(value * right)
            left_was_percent = False
        return value

    def _unary(self) -> Decimal:
        tok = self._peek()
        if tok is not None and tok[0] == "op" and tok[1] in ("+", "-"):
            self._take()
            self._enter()
            value = self._unary()
            self._depth -= 1
            return -value if tok[1] == "-" else value
        return self._power()

    def _power(self) -> Decimal:
        base = self._postfix()
        tok = self._peek()
        if tok is not None and tok[0] == "pow":
            self._take()
            self._enter()
            exponent = self._unary()  # right-associative, allows 2^-1
            self._depth -= 1
            self._after_percent = False
            return _power(base, exponent)
        return base

    def _postfix(self) -> Decimal:
        value = self._primary()
        self._after_percent = False
        while self._peek() == ("op", "%"):
            self._take()
            value = value / 100
            self._after_percent = True
        return value

    def _primary(self) -> Decimal:
        tok = self._take()
        if tok is None:
            raise CalculatorError("the expression ends unexpectedly.")
        kind, text = tok
        if kind == "num":
            return Decimal(text.replace(",", ""))
        if kind == "op" and text == "(":
            self._enter()
            value = self._expression()
            self._depth -= 1
            if self._take() != ("op", ")"):
                raise CalculatorError("missing closing parenthesis.")
            return value
        raise CalculatorError(f"unexpected '{text}'.")


def _format(value: Decimal) -> str:
    text = format(value.quantize(RESULT_PLACES), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result as text.

    Raises:
        CalculatorError: for invalid input, division by zero or oversized results.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(f"the expression is too long (maximum {MAX_EXPRESSION_LENGTH} characters).")
    tokens = _tokenize(expression)
    with localcontext(Context(prec=50)):
        try:
            return _format(_Parser(tokens).parse())
        except RecursionError as exc:
            raise CalculatorError("the expression is nested too deeply.") from exc
        except ArithmeticError as exc:
            raise CalculatorError("the calculation could not be completed.") from exc


def _run(expression: str) -> str:
    return f"{expression.strip()} = {calculate(expression)}"


CALCULATOR_TOOL = ToolSpec(
    name="calculator",
    description=(
        "Evaluate an arithmetic expression exactly. Supports + - * / ^ (power), parentheses, "
        "decimals and percentages (15% means 15/100). For '15% of 8500' pass '8500 * 15%'. "
        "To add a percentage of a number to it, write '100 + 100 * 20%'. Returns 'expression = result'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The arithmetic expression, for example '(1250 + 3750) / 4'.",
            }
        },
        "required": ["expression"],
    },
    func=_run,
    usage_hint=(
        "calculator: use it whenever the user asks you to calculate or evaluate arithmetic (sums, "
        "differences, products, quotients, percentages, expressions with parentheses) and for any "
        "multi-step arithmetic; never do such arithmetic yourself. Write the expression plainly, "
        "e.g. `8500 * 15%` for '15% of 8500'. Do not use it for general knowledge, explanations, "
        "dates or years, or questions that merely mention a number."
    ),
)
