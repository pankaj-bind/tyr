"""
Tyr — Symbolic equivalence verification pipeline.

1. Parse both code strings into ASTs.
2. Extract the first function definition from each.
3. Create Z3 symbolic variables for every parameter.
4. Symbolically execute both functions via AST → Z3 translation.
5. Assert ``Original_return(x) != Optimized_return(x)``.
6. UNSAT → proven equivalent.  SAT → counterexample found.

CRG-3 fix: after symbolic execution, if the output is a SymbolicList
whose length *could* exceed MAX_BMC_LENGTH, fall back to concrete
testing (the BMC array doesn't cover those slots).
"""
from __future__ import annotations

import ast
import logging
import textwrap
from typing import Any

import z3

from config import (
    MAX_BMC_LENGTH, MAX_SYMBOLIC_RANGE, MAX_LOOP_UNROLL,
    Z3_TIMEOUT_MS, INF_SENTINEL, STRING_ID_BASE,
)
from symbolic import (
    ASTToZ3Translator, SymbolicExecError,
    SymbolicList, SymbolicDict, SymbolicSet,
)
from verifier.param_inference import infer_param_types
from verifier.concrete_fallback import concrete_test_fallback

logger = logging.getLogger("tyr.verifier.equivalence")


# ── Public API ────────────────────────────────────────────────────────

def verify_equivalence(
    original_code: str,
    optimized_code: str,
) -> dict[str, Any]:
    logger.info(
        "verify_equivalence  |  orig=%d chars  |  opt=%d chars",
        len(original_code), len(optimized_code),
    )

    # 1. Parse
    try:
        orig_ast = ast.parse(textwrap.dedent(original_code))
    except SyntaxError as exc:
        return _error(f"Syntax error in original code: {exc}")
    try:
        opt_ast = ast.parse(textwrap.dedent(optimized_code))
    except SyntaxError as exc:
        return _error(f"Syntax error in optimized code: {exc}")

    # 2. Extract functions
    orig_func = _extract_function(orig_ast)
    opt_func = _extract_function(opt_ast)
    if orig_func is None:
        return _error("No function definition found in the original code.")
    if opt_func is None:
        return _error("No function definition found in the optimized code.")

    # 3. Build symbolic parameters
    orig_params = _get_param_names(orig_func)
    opt_params = _get_param_names(opt_func)
    if orig_params != opt_params:
        return _error(
            f"Parameter mismatch: original has {orig_params}, "
            f"optimized has {opt_params}."
        )

    param_types = infer_param_types(original_code, orig_params)
    param_symbols: dict[str, Any] = {}
    bmc_constraints: list[Any] = []

    for name in orig_params:
        if param_types.get(name) == "list":
            arr = z3.Array(f"{name}_arr", z3.IntSort(), z3.IntSort())
            length = z3.Int(f"{name}_len")
            param_symbols[name] = SymbolicList(array=arr, length=length)
            bmc_constraints.append(length >= 0)
            bmc_constraints.append(length <= MAX_BMC_LENGTH)
            # Bound elements to stay below infinity sentinels so that
            # float('inf') / float('-inf') behave as true extremes.
            for k in range(MAX_BMC_LENGTH):
                elem = z3.Select(arr, z3.IntVal(k))
                bmc_constraints.append(elem < INF_SENTINEL)
                bmc_constraints.append(elem > -INF_SENTINEL)
            logger.info("  param '%s' → SymbolicList (max len %d)",
                        name, MAX_BMC_LENGTH)
        else:
            param_symbols[name] = z3.Int(name)
            logger.info("  param '%s' → z3.Int", name)

    # 4. Symbolic execution
    smap: dict[str, int] = {}
    snext: list[int] = [STRING_ID_BASE]

    t_orig = ASTToZ3Translator(shared_string_map=smap,
                                shared_next_string_id=snext)
    t_opt = ASTToZ3Translator(shared_string_map=smap,
                               shared_next_string_id=snext)

    sym_failed = False

    try:
        env_orig = t_orig.execute_function(
            orig_func, param_symbols,
            initial_constraints=list(bmc_constraints),
        )
    except SymbolicExecError as exc:
        logger.warning("Symbolic exec of original failed: %s", exc)
        sym_failed = True

    if not sym_failed:
        try:
            env_opt = t_opt.execute_function(
                opt_func, param_symbols,
                initial_constraints=list(bmc_constraints),
            )
        except SymbolicExecError as exc:
            logger.warning("Symbolic exec of optimized failed: %s", exc)
            sym_failed = True

    if not sym_failed:
        ret_orig = env_orig.return_value
        ret_opt = env_opt.return_value
        if ret_orig is None or ret_opt is None:
            logger.warning("Missing return value — falling back.")
            sym_failed = True

    if sym_failed:
        return concrete_test_fallback(original_code, optimized_code,
                                      orig_params)

    # ── CRG-3: output-list overflow probe ────────────────────────────
    for ret in (ret_orig, ret_opt):
        if isinstance(ret, SymbolicList):
            probe = z3.Solver()
            probe.set("timeout", 1000)
            for c in bmc_constraints:
                probe.add(c)
            for c in env_orig.constraints:
                probe.add(c)
            for c in env_opt.constraints:
                probe.add(c)
            probe.add(ret.length > z3.IntVal(MAX_BMC_LENGTH))
            if probe.check() != z3.unsat:
                logger.warning(
                    "Output SymbolicList may exceed BMC bound — "
                    "falling back to concrete testing."
                )
                return concrete_test_fallback(
                    original_code, optimized_code, orig_params,
                )

    logger.info("Original  return: %s", ret_orig)
    logger.info("Optimized return: %s", ret_opt)

    # 5. Solve  ret_orig != ret_opt
    solver = z3.Solver()
    solver.set("timeout", Z3_TIMEOUT_MS)

    for c in env_orig.constraints:
        solver.add(c)
    for c in env_opt.constraints:
        solver.add(c)
    for c in bmc_constraints:
        solver.add(c)

    try:
        _add_diff_assertion(solver, ret_orig, ret_opt)
    except (z3.Z3Exception, TypeError) as exc:
        return _error(
            f"Could not construct equivalence assertion: {exc}"
        )

    result = solver.check()

    # 6. Interpret
    if result == z3.unsat:
        logger.info("Z3: UNSAT — bounded equivalence proven.")
        return {
            "status": "UNSAT",
            "message": (
                f"Tyr: Formal Proof Successful (UNSAT). "
                f"The optimized code is mathematically equivalent to the "
                f"original STRICTLY WITHIN the Bounded Model Checking "
                f"limits (lists ≤ {MAX_BMC_LENGTH} elements, "
                f"symbolic ranges ≤ {MAX_SYMBOLIC_RANGE} iterations, "
                f"while-loops ≤ {MAX_LOOP_UNROLL} iterations). "
                f"No counterexample exists in this bounded domain."
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
        ce = {
            str(d): str(model[d])
            for d in model.decls()
            if not str(d).startswith("__")
        }
        logger.warning("Z3: SAT — counterexample: %s", ce)
        return {
            "status": "SAT",
            "message": (
                "Tyr: Verification Failed (SAT). "
                "A counterexample was found — the optimized code "
                "changes behaviour for the inputs shown."
            ),
            "counterexample": ce,
        }

    # UNKNOWN → concrete fallback
    logger.warning("Z3: UNKNOWN (%s) — concrete fallback.",
                    solver.reason_unknown())
    return concrete_test_fallback(original_code, optimized_code,
                                  orig_params)


# ── Helpers ───────────────────────────────────────────────────────────

def _add_diff_assertion(
    solver: z3.Solver, ret_orig: Any, ret_opt: Any,
) -> None:
    """Add ``ret_orig != ret_opt`` in a sort-aware way."""
    if isinstance(ret_orig, SymbolicList) and isinstance(ret_opt, SymbolicList):
        clauses = [ret_orig.length != ret_opt.length]
        for k in range(MAX_BMC_LENGTH):
            ki = z3.IntVal(k)
            in_bounds = z3.And(ki < ret_orig.length, ki < ret_opt.length)
            clauses.append(z3.And(
                in_bounds,
                z3.Select(ret_orig.array, ki) != z3.Select(ret_opt.array, ki),
            ))
        solver.add(z3.Or(*clauses))
        return

    if isinstance(ret_orig, SymbolicSet) and isinstance(ret_opt, SymbolicSet):
        dk = z3.Int("__set_diff_key")
        solver.add(
            z3.Select(ret_orig.presence, dk) !=
            z3.Select(ret_opt.presence, dk)
        )
        return

    if isinstance(ret_orig, SymbolicDict) and isinstance(ret_opt, SymbolicDict):
        dk = z3.Int("__dict_diff_key")
        solver.add(z3.Or(
            z3.Select(ret_orig.presence, dk) !=
            z3.Select(ret_opt.presence, dk),
            z3.And(
                z3.Select(ret_orig.presence, dk),
                z3.Select(ret_opt.presence, dk),
                z3.Select(ret_orig.values, dk) !=
                z3.Select(ret_opt.values, dk),
            ),
        ))
        return

    # Scalar / Bool ↔ Int coercion
    ro, rp = ret_orig, ret_opt
    if isinstance(ro, z3.BoolRef) and isinstance(rp, z3.ArithRef):
        ro = z3.If(ro, z3.IntVal(1), z3.IntVal(0))
    elif isinstance(rp, z3.BoolRef) and isinstance(ro, z3.ArithRef):
        rp = z3.If(rp, z3.IntVal(1), z3.IntVal(0))

    # Real ↔ Int coercion
    if (isinstance(ro, z3.ArithRef) and isinstance(rp, z3.ArithRef)
            and ro.sort() != rp.sort()):
        if ro.sort() == z3.IntSort():
            ro = z3.ToReal(ro)
        if rp.sort() == z3.IntSort():
            rp = z3.ToReal(rp)

    solver.add(z3.simplify(ro != rp))


def _extract_function(tree: ast.Module) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _get_param_names(func: ast.FunctionDef) -> list[str]:
    return [arg.arg for arg in func.args.args]


def _error(msg: str) -> dict[str, Any]:
    logger.error("Verification error: %s", msg)
    return {"status": "ERROR", "message": f"Tyr: {msg}", "counterexample": None}
