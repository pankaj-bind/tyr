"""
Tyr — Concrete testing fallback.

When Z3 returns UNKNOWN / times out / symbolic execution cannot handle
the code, this module runs both functions on a large grid of concrete
inputs to search for output divergence.
"""
from __future__ import annotations

import ast
import ctypes
import logging
import itertools
import random
import textwrap
import threading
from typing import Any

from config import CONCRETE_EXEC_TIMEOUT_S
from verifier.param_inference import infer_param_types

logger = logging.getLogger("tyr.verifier.concrete")

# ── Safety whitelist ──────────────────────────────────────────────────

_DANGEROUS_BUILTINS = frozenset({
    "exec", "eval", "compile", "__import__", "open",
    "exit", "quit", "breakpoint", "globals", "locals",
    "getattr", "setattr", "delattr",
    "vars", "dir", "type", "super",
    "memoryview", "bytearray", "classmethod", "staticmethod",
    "property", "input",
})

_DANGEROUS_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "ctypes", "signal", "importlib",
    "io", "pickle", "shelve", "tempfile", "multiprocessing",
    "threading", "asyncio", "webbrowser", "code", "codeop",
    "compileall", "py_compile",
})

_DANGEROUS_ATTRS = frozenset({
    "__class__", "__subclasses__", "__bases__", "__mro__",
    "__builtins__", "__globals__", "__code__", "__func__",
    "__self__", "__module__", "__dict__", "__init_subclass__",
    "__set_name__", "__reduce__", "__reduce_ex__",
    "__getattr__", "__setattr__", "__delattr__",
    "__import__",
})


def validate_code_safety(code: str) -> bool:
    """Reject code with imports, exec/eval, OS access, or dunder escapes."""
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _DANGEROUS_BUILTINS:
                return False
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in _DANGEROUS_MODULES:
                    return False
            if node.attr in _DANGEROUS_ATTRS:
                return False
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return False
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in _DANGEROUS_ATTRS:
                return False
        if isinstance(node, ast.Subscript):
            if (isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                    and node.slice.value in _DANGEROUS_ATTRS):
                return False
    return True


# ── Thread-based execution with timeout ──────────────────────────────

def _raise_in_thread(thread_id: int) -> None:
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id),
        ctypes.py_object(SystemExit),
    )


def _run_func_in_process(
    func: Any, args: tuple, timeout: int,
) -> tuple[bool, Any]:
    result_holder: list[Any] = [None, None]
    tid_holder: list[int | None] = [None]

    def _worker() -> None:
        tid_holder[0] = threading.current_thread().ident
        try:
            result_holder[0] = "OK"
            result_holder[1] = func(*args)
        except SystemExit:
            result_holder[0] = "KILLED"
        except Exception:
            result_holder[0] = "EXCEPTION"

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        tid = tid_holder[0]
        if tid is not None:
            _raise_in_thread(tid)
        t.join(timeout=1)
        return (False, None)

    if result_holder[0] == "OK":
        return (True, result_holder[1])
    return (False, None)


# ── Test-input generation ────────────────────────────────────────────

_TEST_VALUES: list[int] = [
    -100, -50, -10, -5, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    15, 20, 25, 30, 50, 100, 127, 128, 255, 256, 500, 1000,
]

_TEST_LISTS: list[list[int]] = [
    [],
    [0],
    [1],
    [-1],
    [1, 2, 3],
    [3, 2, 1],
    [1, 1, 1],
    [1, 2, 2, 3],
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [1, 1, 2, 2, 3, 3],
    [-5, -3, -1, 0, 1, 3, 5],
    [10, 20, 30, 40, 50],
    [1, 2, 3, 2, 1],
    [100, -100, 50, -50, 0],
    list(range(10)),
    list(range(10, -1, -1)),
    [1] * 10,
    list(range(20)),
    [random.Random(42).randint(-100, 100) for _ in range(15)],
    [random.Random(43).randint(-50, 50) for _ in range(25)],
    [random.Random(44).randint(0, 10) for _ in range(30)],
    [1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
    [random.Random(45).randint(-1000, 1000) for _ in range(50)],
]


def generate_test_inputs(
    param_names: list[str],
    param_types: dict[str, str],
) -> list[tuple]:
    rng = random.Random(42)
    pools: list[list] = []
    for name in param_names:
        if param_types.get(name) == "list":
            pools.append(_TEST_LISTS)
        else:
            pools.append(_TEST_VALUES)

    n = len(param_names)
    if n == 1:
        return [(v,) for v in pools[0]]
    if n == 2:
        p0 = pools[0][:20]
        p1 = pools[1][:20]
        return list(itertools.product(p0, p1))
    # 3+ params: random sample
    inputs = []
    for _ in range(500):
        inputs.append(tuple(rng.choice(pool) for pool in pools))
    return inputs


# ── Output normalisation ─────────────────────────────────────────────

def _normalize(val: Any) -> Any:
    if isinstance(val, set):
        return sorted(val, key=lambda x: (str(type(x).__name__), x))
    if isinstance(val, tuple):
        return list(val)
    return val


def _outputs_equal(a: Any, b: Any) -> bool:
    an, bn = _normalize(a), _normalize(b)
    if isinstance(a, (set, frozenset)) or isinstance(b, (set, frozenset)):
        try:
            return sorted(a, key=lambda x: (str(type(x).__name__), x)) == \
                   sorted(b, key=lambda x: (str(type(x).__name__), x))
        except TypeError:
            pass
    return an == bn


def _find_callable(ns: dict[str, Any]) -> Any:
    for v in ns.values():
        if callable(v) and not isinstance(v, type):
            return v
    return None


# ── Main fallback entry point ────────────────────────────────────────

def concrete_test_fallback(
    original_code: str,
    optimized_code: str,
    param_names: list[str],
) -> dict[str, Any]:
    logger.info("Concrete fallback: testing %d params.", len(param_names))

    for label, code_str in [("original", original_code),
                            ("optimized", optimized_code)]:
        if not validate_code_safety(code_str):
            return _error(
                f"Concrete fallback: {label} code contains disallowed "
                f"constructs. Refusing to execute."
            )

    try:
        orig_ns: dict[str, Any] = {}
        exec(compile(textwrap.dedent(original_code), "<original>", "exec"),
             orig_ns)
        opt_ns: dict[str, Any] = {}
        exec(compile(textwrap.dedent(optimized_code), "<optimized>", "exec"),
             opt_ns)
    except Exception as exc:
        return _error(f"Concrete fallback: compilation error — {exc}")

    orig_func = _find_callable(orig_ns)
    opt_func = _find_callable(opt_ns)
    if orig_func is None or opt_func is None:
        return _error(
            "Concrete fallback: could not find callable function in code."
        )

    param_types = infer_param_types(original_code, param_names)
    logger.info("Inferred param types: %s", param_types)

    test_inputs = generate_test_inputs(param_names, param_types)
    n_params = len(param_names)

    divergences = 0
    first_ce: dict[str, Any] | None = None

    for inputs in test_inputs:
        ok_o, res_o = _run_func_in_process(
            orig_func, inputs, CONCRETE_EXEC_TIMEOUT_S,
        )
        if not ok_o:
            continue
        ok_p, res_p = _run_func_in_process(
            opt_func, inputs, CONCRETE_EXEC_TIMEOUT_S,
        )
        if not ok_p:
            res_p = "__TIMEOUT__"

        if not _outputs_equal(res_o, res_p):
            divergences += 1
            if first_ce is None:
                first_ce = {
                    param_names[i]: str(inputs[i]) for i in range(n_params)
                }
                first_ce["original_output"] = str(res_o)
                first_ce["optimized_output"] = str(res_p)

    if divergences > 0:
        logger.warning(
            "Concrete fallback: SAT — %d divergences in %d tests.",
            divergences, len(test_inputs),
        )
        return {
            "status": "SAT",
            "message": (
                f"Tyr: Verification Failed (SAT via concrete testing). "
                f"{divergences} divergence(s) found across "
                f"{len(test_inputs)} test inputs."
            ),
            "counterexample": first_ce,
        }

    logger.info(
        "Concrete fallback: WARNING — no divergences in %d tests.",
        len(test_inputs),
    )
    return {
        "status": "WARNING",
        "message": (
            f"Tyr: WARNING — Z3 symbolic solving timed out. "
            f"Concrete testing across {len(test_inputs)} inputs found no "
            f"divergence, but this is empirical testing only, NOT a formal "
            f"proof of equivalence. Edge-case bugs may still exist."
        ),
        "counterexample": None,
    }


def _error(msg: str) -> dict[str, Any]:
    logger.error("Verification error: %s", msg)
    return {"status": "ERROR", "message": f"Tyr: {msg}", "counterexample": None}
