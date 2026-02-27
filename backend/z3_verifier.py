"""
Tyr Backend — z3_verifier.py
Symbolic equivalence verification of Original vs. Optimized code using Z3.

Pipeline:
  1. Parse both code strings into Python ASTs.
  2. Extract the first function definition from each AST.
  3. Create Z3 symbolic variables for every function parameter.
  4. Symbolically execute both functions via AST → Z3 translation.
  5. Assert  Original_return(x) ≠ Optimized_return(x).
  6. If Z3 returns UNSAT → proven equivalent (accept).
     If Z3 returns SAT  → counterexample found (reject).
"""

from __future__ import annotations

import ast
import ctypes
import logging
import threading
import textwrap
from typing import Any

import z3

from ast_to_z3 import (ASTToZ3Translator, SymbolicExecError, SymbolicList,
                       SymbolicDict, SymbolicSet, MAX_BMC_LENGTH,
                       MAX_LOOP_UNROLL, MAX_SYMBOLIC_RANGE)

logger = logging.getLogger("tyr.z3")

# Solver timeout (milliseconds) — kept short so we fall back to
# concrete testing quickly for complex symbolic expressions.
Z3_TIMEOUT_MS: int = 30_000

# Per-input timeout for concrete test execution (seconds).
# Prevents infinite-loop DoS from malicious or buggy user code.
CONCRETE_EXEC_TIMEOUT_S: int = 5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_equivalence(
    original_code: str,
    optimized_code: str,
) -> dict[str, Any]:
    """
    Check whether *original_code* and *optimized_code* are semantically
    equivalent by translating both into Z3 symbolic constraints and solving
    ``Original(x) != Optimized(x)``.

    Returns
    -------
    dict with keys:
        status : str
            ``"UNSAT"`` — no counterexample exists → **accepted**.
            ``"SAT"``   — counterexample found → **rejected**.
            ``"ERROR"``  — verification could not complete.
        message : str
            Human-readable summary.
        counterexample : dict | None
            If SAT, the concrete inputs that cause divergence.
    """
    logger.info(
        "verify_equivalence called  |  original=%d chars  |  optimized=%d chars",
        len(original_code),
        len(optimized_code),
    )

    # ------------------------------------------------------------------
    # 1. Parse both code strings into ASTs
    # ------------------------------------------------------------------
    try:
        original_ast = ast.parse(textwrap.dedent(original_code))
    except SyntaxError as exc:
        return _error(f"Syntax error in original code: {exc}")

    try:
        optimized_ast = ast.parse(textwrap.dedent(optimized_code))
    except SyntaxError as exc:
        return _error(f"Syntax error in optimized code: {exc}")

    # ------------------------------------------------------------------
    # 2. Extract the first function definition from each
    # ------------------------------------------------------------------
    orig_func = _extract_function(original_ast)
    opt_func = _extract_function(optimized_ast)

    if orig_func is None:
        return _error(
            "No function definition found in the original code. "
            "Tyr currently verifies function-level equivalence."
        )
    if opt_func is None:
        return _error(
            "No function definition found in the optimized code. "
            "The LLM response may not contain a valid function."
        )

    # ------------------------------------------------------------------
    # 3. Build symbolic parameters (must match between original & optimized)
    # ------------------------------------------------------------------
    orig_params = _get_param_names(orig_func)
    opt_params = _get_param_names(opt_func)

    if orig_params != opt_params:
        return _error(
            f"Parameter mismatch: original has {orig_params}, "
            f"optimized has {opt_params}. Cannot verify equivalence."
        )

    # Create Z3 symbolic variables for each parameter.
    # Use AST heuristics to detect list-typed parameters and create
    # SymbolicList (BMC) representations for them.
    param_types = _infer_param_types(original_code, orig_params)
    param_symbols: dict[str, Any] = {}
    bmc_constraints: list[Any] = []  # length bounds for SymbolicList params

    for name in orig_params:
        if param_types.get(name) == "list":
            # Create a SymbolicList with bounded length
            arr = z3.Array(f"{name}_arr", z3.IntSort(), z3.IntSort())
            length = z3.Int(f"{name}_len")
            param_symbols[name] = SymbolicList(array=arr, length=length)
            bmc_constraints.append(length >= 0)
            bmc_constraints.append(length <= MAX_BMC_LENGTH)
            logger.info("  param '%s' → SymbolicList (BMC, max len %d)", name, MAX_BMC_LENGTH)
        else:
            param_symbols[name] = z3.Int(name)  # default: integer
            logger.info("  param '%s' → z3.Int", name)

    logger.info(
        "Symbolic params: %s", {k: str(v) for k, v in param_symbols.items()}
    )

    # ------------------------------------------------------------------
    # 4. Symbolically execute both functions
    # ------------------------------------------------------------------
    translator_orig = ASTToZ3Translator()
    translator_opt = ASTToZ3Translator()

    symbolic_exec_failed = False

    try:
        env_orig = translator_orig.execute_function(orig_func, param_symbols)
    except SymbolicExecError as exc:
        logger.warning("Symbolic execution of original code failed: %s — falling back to concrete testing.", exc)
        symbolic_exec_failed = True

    if not symbolic_exec_failed:
        try:
            env_opt = translator_opt.execute_function(opt_func, param_symbols)
        except SymbolicExecError as exc:
            logger.warning("Symbolic execution of optimized code failed: %s — falling back to concrete testing.", exc)
            symbolic_exec_failed = True

    if not symbolic_exec_failed:
        ret_orig = env_orig.return_value
        ret_opt = env_opt.return_value
        if ret_orig is None or ret_opt is None:
            logger.warning("Missing return value — falling back to concrete testing.")
            symbolic_exec_failed = True

    # If symbolic execution failed for any reason, fall back to concrete testing
    if symbolic_exec_failed:
        logger.info("Using concrete testing fallback for verification.")
        return _concrete_test_fallback(original_code, optimized_code, orig_params)

    logger.info("Original  return expression: %s", ret_orig)
    logger.info("Optimized return expression: %s", ret_opt)

    # ------------------------------------------------------------------
    # 5. Assert   ret_orig != ret_opt   and check satisfiability
    # ------------------------------------------------------------------
    solver = z3.Solver()
    solver.set("timeout", Z3_TIMEOUT_MS)

    # Use appropriate solver logic depending on whether arrays are involved.
    # QF_NIA = quantifier-free nonlinear integer arithmetic (no arrays).
    # QF_ANIA = quantifier-free arrays + nonlinear integer arithmetic.
    has_arrays = any(isinstance(v, SymbolicList) for v in param_symbols.values())

    # If both return SymbolicList, compare element-wise within BMC bound
    if isinstance(ret_orig, SymbolicList) and isinstance(ret_opt, SymbolicList):
        has_arrays = True

    # If both return SymbolicSet, compare presence arrays
    if isinstance(ret_orig, SymbolicSet) and isinstance(ret_opt, SymbolicSet):
        has_arrays = True

    # If both return SymbolicDict, compare presence + values arrays
    if isinstance(ret_orig, SymbolicDict) and isinstance(ret_opt, SymbolicDict):
        has_arrays = True

    if has_arrays:
        # Let Z3 auto-detect logic (QF_ANIA is not always available)
        logger.info("BMC mode: using auto solver logic (arrays present).")
    # Let Z3 auto-select logic — QF_NIA can be slower on nested If-expressions

    # Add any path constraints accumulated during symbolic execution
    for c in env_orig.constraints:
        solver.add(c)
    for c in env_opt.constraints:
        solver.add(c)

    # Add BMC length bounds for SymbolicList parameters
    for c in bmc_constraints:
        solver.add(c)

    try:
        if isinstance(ret_orig, SymbolicList) and isinstance(ret_opt, SymbolicList):
            # Compare lengths and elements within BMC bound
            diff_clauses = [ret_orig.length != ret_opt.length]
            for k in range(MAX_BMC_LENGTH):
                k_idx = z3.IntVal(k)
                in_bounds = z3.And(k_idx < ret_orig.length, k_idx < ret_opt.length)
                diff_clauses.append(
                    z3.And(in_bounds,
                           z3.Select(ret_orig.array, k_idx) != z3.Select(ret_opt.array, k_idx))
                )
            solver.add(z3.Or(*diff_clauses))
        elif isinstance(ret_orig, SymbolicSet) and isinstance(ret_opt, SymbolicSet):
            # Sets are equal iff their presence arrays agree on every key.
            # Ask Z3 whether there exists ANY key where they differ.
            diff_key = z3.Int("__set_diff_key")
            solver.add(
                z3.Select(ret_orig.presence, diff_key)
                != z3.Select(ret_opt.presence, diff_key)
            )
        elif isinstance(ret_orig, SymbolicDict) and isinstance(ret_opt, SymbolicDict):
            # Dicts are equal iff for every key, presence and value agree.
            diff_key = z3.Int("__dict_diff_key")
            solver.add(z3.Or(
                z3.Select(ret_orig.presence, diff_key)
                != z3.Select(ret_opt.presence, diff_key),
                z3.And(
                    z3.Select(ret_orig.presence, diff_key),
                    z3.Select(ret_opt.presence, diff_key),
                    z3.Select(ret_orig.values, diff_key)
                    != z3.Select(ret_opt.values, diff_key),
                ),
            ))
        else:
            # Coerce Bool↔Int if one function returns Bool and the other Int
            ro, rp = ret_orig, ret_opt
            if isinstance(ro, z3.BoolRef) and isinstance(rp, z3.ArithRef):
                ro = z3.If(ro, z3.IntVal(1), z3.IntVal(0))
            elif isinstance(rp, z3.BoolRef) and isinstance(ro, z3.ArithRef):
                rp = z3.If(rp, z3.IntVal(1), z3.IntVal(0))
            difference = z3.simplify(ro != rp)
            solver.add(difference)
    except (z3.Z3Exception, TypeError) as exc:
        return _error(
            f"Could not construct equivalence assertion (type mismatch?): {exc}"
        )

    result = solver.check()

    # ------------------------------------------------------------------
    # 6. Interpret result
    # ------------------------------------------------------------------
    if result == z3.unsat:
        logger.info("Z3 verdict: UNSAT — codes are provably equivalent.")
        return {
            "status": "UNSAT",
            "message": (
                f"Tyr: Verification Passed (UNSAT). "
                f"The optimized code is provably equivalent to the original "
                f"within the verification bounds "
                f"(lists ≤ {MAX_BMC_LENGTH} elements, "
                f"ranges ≤ {MAX_SYMBOLIC_RANGE}, "
                f"while-loops ≤ {MAX_LOOP_UNROLL} iterations)."
            ),
            "counterexample": None,
            "verification_bounds": {
                "max_bmc_length": MAX_BMC_LENGTH,
                "max_symbolic_range": MAX_SYMBOLIC_RANGE,
                "max_loop_unroll": MAX_LOOP_UNROLL,
            },
        }

    if result == z3.sat:
        model = solver.model()
        counterexample = {
            str(d): str(model[d]) for d in model.decls()
            if not str(d).startswith("__")  # skip internal SSA vars
        }
        logger.warning("Z3 verdict: SAT — counterexample: %s", counterexample)
        return {
            "status": "SAT",
            "message": (
                "Tyr: Verification Failed (SAT). "
                "A counterexample was found — the optimized code "
                "changes behaviour for the inputs shown."
            ),
            "counterexample": counterexample,
        }

    # z3.unknown — fallback to concrete testing
    reason = solver.reason_unknown()
    logger.warning("Z3 verdict: UNKNOWN (%s) — falling back to concrete testing (result will be WARNING, not UNSAT).", reason)
    return _concrete_test_fallback(original_code, optimized_code, orig_params)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_function(tree: ast.Module) -> ast.FunctionDef | None:
    """Return the first ``FunctionDef`` node found at the top level."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _get_param_names(func: ast.FunctionDef) -> list[str]:
    """Extract parameter names (positional) from a function definition."""
    return [arg.arg for arg in func.args.args]


def _error(message: str) -> dict[str, Any]:
    logger.error("Verification error: %s", message)
    return {
        "status": "ERROR",
        "message": f"Tyr: {message}",
        "counterexample": None,
    }


# ---------------------------------------------------------------------------
# Code Safety Validation — AST whitelist before exec()
# ---------------------------------------------------------------------------

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

# Dunder attributes that enable sandbox escape via the MRO chain
_DANGEROUS_ATTRS = frozenset({
    "__class__", "__subclasses__", "__bases__", "__mro__",
    "__builtins__", "__globals__", "__code__", "__func__",
    "__self__", "__module__", "__dict__", "__init_subclass__",
    "__set_name__", "__reduce__", "__reduce_ex__",
    "__getattr__", "__setattr__", "__delattr__",
    "__import__",
})


def _validate_code_safety(code: str) -> bool:
    """
    Reject code containing imports, exec/eval, OS access, or dunder attribute
    chains that could escape the sandbox.  This is a defence-in-depth measure:
    the Z3 translation never calls ``exec()`` on user code, but the concrete
    testing fallback does, so we must whitelist aggressively.
    """
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        # Block all imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False

        # Block dangerous built-in calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _DANGEROUS_BUILTINS:
                return False

        # Block dangerous module access (e.g., os.system)
        if isinstance(node, ast.Attribute):
            # Direct module attribute: os.system, sys.exit, etc.
            if isinstance(node.value, ast.Name):
                if node.value.id in _DANGEROUS_MODULES:
                    return False
            # Dunder attribute access anywhere: obj.__class__, etc.
            if node.attr in _DANGEROUS_ATTRS:
                return False
            # Any attribute starting with __ (broad dunder catch-all)
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return False

        # Block string constants that look like dunder smuggling
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in _DANGEROUS_ATTRS:
                return False

        # Block subscript-based access to __builtins__ etc.
        # e.g., globals()["__builtins__"] or x["__class__"]
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                if node.slice.value in _DANGEROUS_ATTRS:
                    return False

    return True


# ---------------------------------------------------------------------------
# Concrete Testing Fallback
# When Z3 returns UNKNOWN (e.g., solver timeout on complex symbolic
# expressions from loop unrolling), we fall back to executing both
# functions on a large set of concrete inputs to search for divergence.
# If no divergence is found, we report "conditionally verified."
# ---------------------------------------------------------------------------

# Test values covering edge cases: negatives, zero, small, boundary
_TEST_VALUES: list[int] = [
    -100, -50, -10, -5, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    15, 20, 25, 30, 50, 100, 127, 128, 255, 256, 500, 1000,
]


def _raise_in_thread(thread_id: int) -> None:
    """Raise ``SystemExit`` asynchronously in a running CPython thread.

    Uses ``PyThreadState_SetAsyncExc`` which is CPython-specific but
    works reliably on both Windows and Linux for pure-Python code.
    The exception is raised between bytecode instructions, so it
    cleanly terminates infinite loops in user-generated algorithms
    without leaking zombie threads (unlike ``ThreadPoolExecutor``
    where the worker thread continues running after a timeout).
    """
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id),
        ctypes.py_object(SystemExit),
    )


def _run_func_in_process(
    func: Any,
    args: tuple,
    timeout: int,
) -> tuple[bool, Any]:
    """
    Run *func(*args)* in a separate daemon thread with a hard timeout.

    If the function does not complete within *timeout* seconds, the
    worker thread is **forcibly terminated** via
    ``PyThreadState_SetAsyncExc(SystemExit)`` — no leaked threads,
    no zombie CPU consumption if user code contains an infinite loop.

    Returns ``(True, result)`` on success, ``(False, None)`` on
    timeout/error.
    """
    result_holder: list[Any] = [None, None]  # [tag, value]
    thread_id_holder: list[int | None] = [None]

    def _worker() -> None:
        thread_id_holder[0] = threading.current_thread().ident
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
        # Thread still running — kill it via async exception
        tid = thread_id_holder[0]
        if tid is not None:
            _raise_in_thread(tid)
        t.join(timeout=1)  # give it a moment to react
        return (False, None)

    if result_holder[0] == "OK":
        return (True, result_holder[1])
    return (False, None)


def _concrete_test_fallback(
    original_code: str,
    optimized_code: str,
    param_names: list[str],
) -> dict[str, Any]:
    """
    Execute both code snippets on a grid of concrete inputs.
    If any input produces different outputs, return SAT with the
    counterexample.  Otherwise, return UNSAT with a caveat.
    """
    import inspect
    import itertools
    import random

    logger.info(
        "Concrete fallback: testing %d params.",
        len(param_names),
    )

    # Compile both code blocks — with safety validation
    for label, code_str in [("original", original_code), ("optimized", optimized_code)]:
        if not _validate_code_safety(code_str):
            return _error(
                f"Concrete fallback: {label} code contains disallowed "
                f"constructs (imports, exec, OS access). Refusing to execute."
            )

    try:
        orig_ns: dict[str, Any] = {}
        exec(compile(textwrap.dedent(original_code), "<original>", "exec"), orig_ns)  # noqa: S102
        opt_ns: dict[str, Any] = {}
        exec(compile(textwrap.dedent(optimized_code), "<optimized>", "exec"), opt_ns)  # noqa: S102
    except Exception as exc:
        return _error(f"Concrete fallback: compilation error — {exc}")

    # Find the function objects
    orig_func = _find_callable(orig_ns)
    opt_func = _find_callable(opt_ns)
    if orig_func is None or opt_func is None:
        return _error("Concrete fallback: could not find callable function in code.")

    # ---- Detect parameter types from function body heuristics ----
    # Inspect the original AST to guess if params are lists or ints
    param_types = _infer_param_types(original_code, param_names)
    logger.info("Inferred param types: %s", param_types)

    # ---- Generate test inputs based on inferred types ----
    test_inputs = _generate_test_inputs(param_names, param_types)
    n_params = len(param_names)

    divergences = 0
    first_counterexample: dict[str, Any] | None = None

    for inputs in test_inputs:
        ok_orig, result_orig = _run_func_in_process(orig_func, inputs, CONCRETE_EXEC_TIMEOUT_S)
        if not ok_orig:
            continue  # skip inputs that time out or crash in original
        ok_opt, result_opt = _run_func_in_process(opt_func, inputs, CONCRETE_EXEC_TIMEOUT_S)
        if not ok_opt:
            result_opt = "__TIMEOUT__"

        # Normalize outputs for comparison:
        # - Convert sets to sorted lists (order doesn't matter for equivalence)
        # - Convert tuples to lists
        if not _outputs_equal(result_orig, result_opt):
            divergences += 1
            if first_counterexample is None:
                first_counterexample = {
                    param_names[i]: str(inputs[i]) for i in range(n_params)
                }
                first_counterexample["original_output"] = str(result_orig)
                first_counterexample["optimized_output"] = str(result_opt)

    if divergences > 0:
        logger.warning(
            "Concrete fallback: SAT — %d divergences found out of %d tests.",
            divergences, len(test_inputs),
        )
        return {
            "status": "SAT",
            "message": (
                f"Tyr: Verification Failed (SAT via concrete testing). "
                f"{divergences} divergence(s) found across {len(test_inputs)} test inputs."
            ),
            "counterexample": first_counterexample,
        }

    logger.info(
        "Concrete fallback: WARNING — no divergences in %d tests (not a formal proof).", len(test_inputs),
    )
    return {
        "status": "WARNING",
        "message": (
            f"Tyr: WARNING — Z3 symbolic solving timed out. "
            f"Concrete testing across {len(test_inputs)} inputs found no divergence, "
            f"but this is empirical testing only, NOT a formal proof of equivalence. "
            f"Edge-case bugs may still exist."
        ),
        "counterexample": None,
    }


def _find_callable(namespace: dict[str, Any]) -> Any:
    """Find the first user-defined function in a namespace."""
    for v in namespace.values():
        if callable(v) and not isinstance(v, type):
            return v
    return None


def _normalize_output(val: Any) -> Any:
    """Normalize function output for equivalence comparison."""
    if isinstance(val, set):
        return sorted(val, key=lambda x: (str(type(x).__name__), x))
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, list):
        return val
    return val


def _outputs_equal(a: Any, b: Any) -> bool:
    """Compare two function outputs, normalizing sets/tuples to sorted lists."""
    a_norm = _normalize_output(a)
    b_norm = _normalize_output(b)
    # If one returned a set and the other a list, compare as sorted lists
    if isinstance(a, (set, frozenset)) or isinstance(b, (set, frozenset)):
        try:
            a_sorted = sorted(a, key=lambda x: (str(type(x).__name__), x))
            b_sorted = sorted(b, key=lambda x: (str(type(x).__name__), x))
            return a_sorted == b_sorted
        except TypeError:
            pass
    return a_norm == b_norm


# ---------------------------------------------------------------------------
# Parameter type inference from AST
# ---------------------------------------------------------------------------

def _infer_param_types(code: str, param_names: list[str]) -> dict[str, str]:
    """
    Heuristically infer whether each parameter is a 'list' or 'int'
    by examining how it's used in the code body.
    """
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return {p: "int" for p in param_names}

    source = textwrap.dedent(code)
    param_types: dict[str, str] = {}

    for name in param_names:
        # Heuristics that suggest the parameter is a list:
        # 1. len(param) is called
        # 2. param[i] is used (subscript)
        # 3. param is iterated with 'for x in param'
        # 4. param name contains 'list', 'arr', 'nums', 'items', 'elements', 'values', 'data'
        is_list = False

        list_hints = ["list", "arr", "nums", "items", "elements", "values",
                       "data", "numbers", "seq", "sequence", "collection",
                       "strings", "strs", "chars"]
        if any(hint in name.lower() for hint in list_hints):
            is_list = True

        # Check AST for usage patterns
        for node in ast.walk(tree):
            # len(param)
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "len"
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == name):
                is_list = True

            # param[i]
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == name):
                is_list = True

            # for x in param
            if (isinstance(node, ast.For)
                    and isinstance(node.iter, ast.Name)
                    and node.iter.id == name):
                is_list = True

            # param.append / param.sort / param.extend etc.
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == name
                    and node.attr in ("append", "extend", "sort", "pop",
                                       "insert", "remove", "index", "count")):
                is_list = True

        param_types[name] = "list" if is_list else "int"

    return param_types


def _generate_test_inputs(
    param_names: list[str],
    param_types: dict[str, str],
) -> list[tuple]:
    """
    Generate a comprehensive set of test inputs based on inferred types.
    Returns a list of tuples, one per test case.
    """
    import random
    rng = random.Random(42)

    # Pre-built test lists covering edge cases
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
        [i for i in range(20)],
        [rng.randint(-100, 100) for _ in range(15)],
        [rng.randint(-50, 50) for _ in range(25)],
        [rng.randint(0, 10) for _ in range(30)],
        [1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
        [rng.randint(-1000, 1000) for _ in range(50)],
    ]

    # Build per-parameter value pools
    pools: list[list] = []
    for name in param_names:
        ptype = param_types.get(name, "int")
        if ptype == "list":
            pools.append(_TEST_LISTS)
        else:
            pools.append(_TEST_VALUES)

    # Generate test inputs
    n_params = len(param_names)

    if n_params == 1:
        test_inputs = [(v,) for v in pools[0]]
    elif n_params == 2:
        import itertools
        # Limit combinatorial explosion
        p0 = pools[0][:20] if len(pools[0]) > 20 else pools[0]
        p1 = pools[1][:20] if len(pools[1]) > 20 else pools[1]
        test_inputs = list(itertools.product(p0, p1))
    else:
        # Random sampling for 3+ params
        test_inputs = []
        for _ in range(500):
            sample = tuple(rng.choice(pool) for pool in pools)
            test_inputs.append(sample)

    return test_inputs