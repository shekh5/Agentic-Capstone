"""Small, bounded arithmetic-expression evaluator used by agent tools."""

import ast
import math
import operator

MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 64
MAX_INTEGER_BITS = 4096
MAX_ABS_EXPONENT = 100

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_arithmetic(expression: str) -> str:
    """Evaluate basic arithmetic without executing Python code."""
    if not expression.strip():
        raise ValueError("Expression cannot be empty")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Expression exceeds {MAX_EXPRESSION_LENGTH} characters")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Could not parse expression: {exc.msg}") from exc

    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("Expression is too complex")

    try:
        result = _evaluate_node(tree.body)
    except (ArithmeticError, OverflowError) as exc:
        raise ValueError(f"Could not evaluate expression: {exc}") from exc

    _validate_result(result)
    return str(result)


def _evaluate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        _validate_result(node.value)
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        result = _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand))
        _validate_result(result)
        return result

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_ABS_EXPONENT:
            raise ValueError(f"Exponent magnitude cannot exceed {MAX_ABS_EXPONENT}")
        result = _BINARY_OPERATORS[type(node.op)](left, right)
        _validate_result(result)
        return result

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def _validate_result(value: int | float) -> None:
    if type(value) not in (int, float):
        raise ValueError("Result must be a real number")
    if isinstance(value, int) and value.bit_length() > MAX_INTEGER_BITS:
        raise ValueError("Integer result is too large")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Result must be finite")
