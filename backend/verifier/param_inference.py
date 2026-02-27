"""
Tyr — Parameter type inference from AST heuristics.
"""
from __future__ import annotations

import ast
import textwrap


def infer_param_types(
    code: str, param_names: list[str],
) -> dict[str, str]:
    """Heuristically infer whether each parameter is ``'list'`` or ``'int'``."""
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return {p: "int" for p in param_names}

    param_types: dict[str, str] = {}

    _LIST_HINTS = (
        "list", "arr", "nums", "items", "elements", "values",
        "data", "numbers", "seq", "sequence", "collection",
        "strings", "strs", "chars",
    )

    for name in param_names:
        is_list = any(h in name.lower() for h in _LIST_HINTS)

        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "len"
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == name):
                is_list = True
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == name):
                is_list = True
            if (isinstance(node, ast.For)
                    and isinstance(node.iter, ast.Name)
                    and node.iter.id == name):
                is_list = True
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == name
                    and node.attr in (
                        "append", "extend", "sort", "pop",
                        "insert", "remove", "index", "count",
                    )):
                is_list = True

        param_types[name] = "list" if is_list else "int"

    return param_types
