"""
Tyr — Loop execution mixin.

All ``for`` and ``while`` loop handlers with CRG-1 (continue) and
CRG-2 (break in all loop types) fixes.
"""
from __future__ import annotations

import ast
import logging
from typing import Any

import z3

from config import (
    MAX_LOOP_UNROLL, MAX_BMC_LENGTH, MAX_SYMBOLIC_RANGE,
    MAX_PARTIAL_RETURNS,
)
from symbolic.types import (
    SymbolicEnv, SymbolicExecError,
    SymbolicList, SymbolicDict, SymbolicSet,
    EnumerateResult,
)

logger = logging.getLogger("tyr.symbolic.loops")


class LoopExecutor:
    """Mixin: every loop handler on ASTToZ3Translator."""

    # ── Dispatcher ────────────────────────────────────────────────────

    def _exec_for(self, node: ast.For, env: SymbolicEnv) -> None:
        loop_var: str | None = None
        if isinstance(node.target, ast.Name):
            loop_var = node.target.id

        # Fast path: concrete range
        iter_range = None
        if loop_var is not None:
            iter_range = self._try_extract_range(node.iter, env)
        if iter_range is not None:
            start, stop, step = iter_range
            iterations = 0
            not_yet_broken = None          # accumulates break guards
            i = start
            while (step > 0 and i < stop) or (step < 0 and i > stop):
                if iterations >= MAX_LOOP_UNROLL:
                    logger.warning("For-loop unrolling capped at %d.", MAX_LOOP_UNROLL)
                    break

                guard = z3.BoolVal(True) if not_yet_broken is None else not_yet_broken

                body_env = env.copy()
                body_env.partial_returns = []
                body_env.set(loop_var, z3.IntVal(i))
                self._exec_body(node.body, body_env)

                # ── return inside loop body ──
                if body_env.has_returned:
                    if len(env.partial_returns) < MAX_PARTIAL_RETURNS:
                        env.partial_returns.append((guard, body_env.return_value))
                    # If return is unconditional, stop the whole loop.
                    i += step
                    iterations += 1
                    continue

                # ── continue (CRG-1 fix) ──
                if body_env.has_continued:
                    body_env.has_continued = False
                    skip = {loop_var}
                    self._merge_env_guarded(env, body_env, guard, skip_vars=skip)
                    self._propagate_partial_returns(env, body_env, guard)
                    i += step
                    iterations += 1
                    continue

                # ── break (CRG-2 fix) ──
                if body_env.has_broken:
                    body_env.has_broken = False
                    # The break may be conditional.  Merge effects under
                    # the current guard and tighten not_yet_broken.
                    skip = {loop_var}
                    self._merge_env_guarded(env, body_env, guard, skip_vars=skip)
                    self._propagate_partial_returns(env, body_env, guard)
                    # Retrieve the break condition from _exec_body's guard.
                    # For unconditional break, break_cond is True.
                    break_cond = body_env._break_guard
                    if break_cond is not None:
                        # Conditional break — tighten the guard.
                        try:
                            new_guard = z3.And(guard, z3.Not(break_cond))
                            not_yet_broken = new_guard
                        except (z3.Z3Exception, TypeError):
                            break
                    else:
                        # Unconditional break — exit loop.
                        break
                    i += step
                    iterations += 1
                    continue

                # Normal iteration — merge under guard.
                skip = {loop_var}
                self._merge_env_guarded(env, body_env, guard, skip_vars=skip)
                self._propagate_partial_returns(env, body_env, guard)
                i += step
                iterations += 1
            return

        # Symbolic range
        if loop_var is not None:
            symbolic_range = self._try_extract_symbolic_range(node.iter, env)
            if symbolic_range is not None:
                self._exec_symbolic_for(
                    loop_var, *symbolic_range, node.body, env,
                )
                return

        # dict.items() / .keys() / .values()
        if (isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Attribute)
                and node.iter.func.attr in ("items", "keys", "values")):
            dict_obj = self._eval_expr(node.iter.func.value, env)
            if isinstance(dict_obj, SymbolicDict):
                self._exec_for_over_dict_items(
                    node, dict_obj, node.iter.func.attr, env,
                )
                return

        iter_val = self._eval_expr(node.iter, env)

        if isinstance(iter_val, SymbolicDict):
            self._exec_for_over_dict_items(node, iter_val, "keys", env)
            return
        if isinstance(iter_val, SymbolicList):
            if loop_var is None:
                raise SymbolicExecError(
                    "For-loop over SymbolicList requires a simple variable target."
                )
            self._exec_for_over_symbolic_list(loop_var, iter_val, node.body, env)
            return
        if isinstance(iter_val, EnumerateResult):
            self._exec_for_enumerate(node, iter_val, env)
            return
        if isinstance(iter_val, (list, tuple)):
            self._exec_for_over_python_list(node, iter_val, env)
            return

        raise SymbolicExecError(
            f"For-loop iteration must be range(), a SymbolicList, or a "
            f"tracked collection. Got: {type(iter_val).__name__}"
        )

    # ── Symbolic for (guarded) ────────────────────────────────────────

    def _exec_symbolic_for(
        self,
        loop_var: str,
        start: Any, stop: Any, step: Any,
        body: list[ast.stmt],
        env: SymbolicEnv,
    ) -> None:
        current_i = start

        for _it in range(MAX_SYMBOLIC_RANGE):
            try:
                guard = current_i < stop
            except (z3.Z3Exception, TypeError):
                break
            if isinstance(guard, bool):
                guard = z3.BoolVal(guard)

            body_env = env.copy()
            body_env.partial_returns = []
            body_env.set(loop_var, current_i)
            self._exec_body(body, body_env)

            # ── continue (CRG-1) ──
            if body_env.has_continued:
                body_env.has_continued = False
                self._merge_env_guarded(env, body_env, guard,
                                        skip_vars={loop_var})
                self._propagate_partial_returns(env, body_env, guard)
                current_i = self._advance_loop_var(current_i, step)
                continue

            # ── break (CRG-2) ──
            if body_env.has_broken:
                body_env.has_broken = False
                self._merge_env_guarded(env, body_env, guard,
                                        skip_vars={loop_var})
                self._propagate_partial_returns(env, body_env, guard)
                break

            # ── return ──
            if body_env.has_returned:
                if len(env.partial_returns) < MAX_PARTIAL_RETURNS:
                    env.partial_returns.append((guard, body_env.return_value))
                current_i = self._advance_loop_var(current_i, step)
                continue

            self._propagate_partial_returns(env, body_env, guard)
            self._merge_env_guarded(env, body_env, guard,
                                    skip_vars={loop_var})
            current_i = self._advance_loop_var(current_i, step)

        # ── Soundness guard ──────────────────────────────────────────
        try:
            residual = z3.simplify(current_i < stop)
            if z3.is_false(residual):
                pass
            else:
                probe = z3.Solver()
                probe.set("timeout", 1000)
                for c in env.constraints:
                    probe.add(c)
                probe.add(current_i < stop)
                if probe.check() != z3.unsat:
                    raise SymbolicExecError(
                        f"Symbolic for-loop may exceed {MAX_SYMBOLIC_RANGE} "
                        f"unrolled iterations — falling back to concrete testing."
                    )
        except SymbolicExecError:
            raise
        except (z3.Z3Exception, TypeError, AttributeError):
            raise SymbolicExecError(
                "Symbolic for-loop bound check inconclusive — "
                "falling back to concrete testing."
            )

    # ── BMC: for over SymbolicList ────────────────────────────────────

    def _exec_for_over_symbolic_list(
        self,
        loop_var: str,
        lst: SymbolicList,
        body: list[ast.stmt],
        env: SymbolicEnv,
    ) -> None:
        for i in range(MAX_BMC_LENGTH):
            idx = z3.IntVal(i)
            guard = idx < lst.length

            body_env = env.copy()
            body_env.partial_returns = []
            body_env.set(loop_var, z3.Select(lst.array, idx))
            self._exec_body(body, body_env)

            # ── continue (CRG-1/CRG-2 fix) ──
            if body_env.has_continued:
                body_env.has_continued = False
                self._merge_env_guarded(env, body_env, guard,
                                        skip_vars={loop_var})
                self._propagate_partial_returns(env, body_env, guard)
                continue

            # ── break (CRG-2 fix) ──
            if body_env.has_broken:
                body_env.has_broken = False
                self._merge_env_guarded(env, body_env, guard,
                                        skip_vars={loop_var})
                self._propagate_partial_returns(env, body_env, guard)
                break

            # ── return ──
            if body_env.has_returned:
                if len(env.partial_returns) < MAX_PARTIAL_RETURNS:
                    env.partial_returns.append((guard, body_env.return_value))
                continue

            self._propagate_partial_returns(env, body_env, guard)
            self._merge_env_guarded(env, body_env, guard,
                                    skip_vars={loop_var})

    # ── for over concrete Python list ─────────────────────────────────

    def _exec_for_over_python_list(
        self,
        node: ast.For,
        items: list | tuple,
        env: SymbolicEnv,
    ) -> None:
        for item in items:
            self._bind_for_target(node.target, item, env)
            self._exec_body(node.body, env)
            if env.has_returned:
                break
            if env.has_continued:              # CRG-1/CRG-2 fix
                env.has_continued = False
                continue
            if env.has_broken:                 # CRG-2 fix
                env.has_broken = False
                break

    # ── for over enumerate ────────────────────────────────────────────

    def _exec_for_enumerate(
        self,
        node: ast.For,
        enum_result: EnumerateResult,
        env: SymbolicEnv,
    ) -> None:
        for idx_z3, elem_z3 in enum_result.pairs:
            guard = idx_z3 < enum_result.source_length

            body_env = env.copy()
            body_env.partial_returns = []

            if isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2:
                i_elt, v_elt = node.target.elts
                if isinstance(i_elt, ast.Name):
                    body_env.set(i_elt.id, idx_z3)
                if isinstance(v_elt, ast.Name):
                    body_env.set(v_elt.id, elem_z3)
            elif isinstance(node.target, ast.Name):
                body_env.set(node.target.id, (idx_z3, elem_z3))
            else:
                raise SymbolicExecError(
                    "enumerate() iteration requires `for i, v in …` "
                    "or `for item in …`"
                )

            self._exec_body(node.body, body_env)

            # ── continue (CRG-1/CRG-2) ──
            if body_env.has_continued:
                body_env.has_continued = False
                skip = self._loop_skip_vars(node.target)
                self._merge_env_guarded(env, body_env, guard, skip_vars=skip)
                self._propagate_partial_returns(env, body_env, guard)
                continue

            # ── break (CRG-2) ──
            if body_env.has_broken:
                body_env.has_broken = False
                skip = self._loop_skip_vars(node.target)
                self._merge_env_guarded(env, body_env, guard, skip_vars=skip)
                self._propagate_partial_returns(env, body_env, guard)
                break

            if body_env.has_returned:
                if len(env.partial_returns) < MAX_PARTIAL_RETURNS:
                    env.partial_returns.append((guard, body_env.return_value))
                continue

            self._propagate_partial_returns(env, body_env, guard)
            skip = self._loop_skip_vars(node.target)
            self._merge_env_guarded(env, body_env, guard, skip_vars=skip)

    # ── for over SymbolicDict ─────────────────────────────────────────

    def _exec_for_over_dict_items(
        self,
        node: ast.For,
        dct: SymbolicDict,
        method: str,
        env: SymbolicEnv,
    ) -> None:
        for key in dct.tracked_keys:
            guard = z3.Select(dct.presence, key)

            body_env = env.copy()
            body_env.partial_returns = []

            # Bind iteration variable(s)
            if method == "items":
                val = z3.Select(dct.values, key)
                if isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2:
                    k_elt, v_elt = node.target.elts
                    if isinstance(k_elt, ast.Name):
                        body_env.set(k_elt.id, key)
                    if isinstance(v_elt, ast.Name):
                        body_env.set(v_elt.id, val)
                elif isinstance(node.target, ast.Name):
                    body_env.set(node.target.id, (key, val))
                else:
                    raise SymbolicExecError(
                        "dict.items() iteration requires `for k, v in …`"
                    )
            elif method == "keys":
                if isinstance(node.target, ast.Name):
                    body_env.set(node.target.id, key)
                else:
                    raise SymbolicExecError(
                        "dict.keys() iteration requires a simple variable target"
                    )
            elif method == "values":
                val = z3.Select(dct.values, key)
                if isinstance(node.target, ast.Name):
                    body_env.set(node.target.id, val)
                else:
                    raise SymbolicExecError(
                        "dict.values() iteration requires a simple variable target"
                    )

            self._exec_body(node.body, body_env)

            # ── continue (CRG-1/CRG-2) ──
            if body_env.has_continued:
                body_env.has_continued = False
                skip = self._loop_skip_vars(node.target)
                self._merge_env_guarded(env, body_env, guard, skip_vars=skip)
                self._propagate_partial_returns(env, body_env, guard)
                continue

            # ── break (CRG-2) ──
            if body_env.has_broken:
                body_env.has_broken = False
                skip = self._loop_skip_vars(node.target)
                self._merge_env_guarded(env, body_env, guard, skip_vars=skip)
                self._propagate_partial_returns(env, body_env, guard)
                break

            if body_env.has_returned:
                if len(env.partial_returns) < MAX_PARTIAL_RETURNS:
                    env.partial_returns.append((guard, body_env.return_value))
                continue

            self._propagate_partial_returns(env, body_env, guard)
            skip = self._loop_skip_vars(node.target)
            self._merge_env_guarded(env, body_env, guard, skip_vars=skip)

    # ── While ─────────────────────────────────────────────────────────

    def _exec_while(self, node: ast.While, env: SymbolicEnv) -> None:
        terminated = False

        for _ in range(MAX_LOOP_UNROLL):
            cond = self._eval_expr(node.test, env)
            concrete = self._try_concrete_bool(cond)

            if concrete is False:
                terminated = True
                break

            if concrete is True:
                self._exec_body(node.body, env)
                if env.has_returned:
                    terminated = True
                    break
                if env.has_continued:          # CRG-1
                    env.has_continued = False
                    continue
                if env.has_broken:
                    env.has_broken = False
                    terminated = True
                    break
                continue

            # Symbolic condition
            body_env = env.copy()
            body_env.partial_returns = []
            self._exec_body(node.body, body_env)

            if body_env.has_continued:
                body_env.has_continued = False
                self._merge_env_guarded(env, body_env, cond)
                self._propagate_partial_returns(env, body_env, cond)
                continue

            if body_env.has_broken:
                body_env.has_broken = False
                self._merge_env_guarded(env, body_env, cond)
                self._propagate_partial_returns(env, body_env, cond)
                terminated = True
                break

            if body_env.has_returned:
                if len(env.partial_returns) < MAX_PARTIAL_RETURNS:
                    env.partial_returns.append((cond, body_env.return_value))
            else:
                self._propagate_partial_returns(env, body_env, cond)
                self._merge_env_guarded(env, body_env, cond)

        if not terminated:
            raise SymbolicExecError(
                f"While-loop exceeded {MAX_LOOP_UNROLL} unroll iterations — "
                f"falling back to concrete testing."
            )

        if node.orelse and not env.has_returned:
            self._exec_body(node.orelse, env)

    # ── Guarded env merge ─────────────────────────────────────────────

    def _merge_env_guarded(
        self,
        env: SymbolicEnv,
        body_env: SymbolicEnv,
        guard: Any,
        skip_vars: set[str] | None = None,
    ) -> None:
        skip = skip_vars or set()
        for var_name in body_env.bindings:
            if var_name in skip:
                continue
            body_val = body_env.bindings[var_name]
            old_val = env.bindings.get(var_name)

            if old_val is None:
                env.set(var_name, body_val)
                continue

            if isinstance(body_val, SymbolicList) and isinstance(old_val, SymbolicList):
                merged_len = z3.If(guard, body_val.length, old_val.length)
                merged_arr = old_val.array
                for k in range(MAX_BMC_LENGTH):
                    ki = z3.IntVal(k)
                    merged_arr = z3.Store(
                        merged_arr, ki,
                        z3.If(guard,
                              z3.Select(body_val.array, ki),
                              z3.Select(old_val.array, ki)),
                    )
                env.set(var_name, SymbolicList(array=merged_arr, length=merged_len))
                continue

            if isinstance(body_val, SymbolicDict) and isinstance(old_val, SymbolicDict):
                merged = self._if_merge(guard, body_val, old_val)
                if merged is not None:
                    env.set(var_name, merged)
                    continue

            if isinstance(body_val, SymbolicSet) and isinstance(old_val, SymbolicSet):
                merged = self._if_merge(guard, body_val, old_val)
                if merged is not None:
                    env.set(var_name, merged)
                    continue

            try:
                merged = z3.If(guard, body_val, old_val)
                try:
                    merged = z3.simplify(merged)
                except (z3.Z3Exception, TypeError, AttributeError):
                    pass
                env.set(var_name, merged)
            except (z3.Z3Exception, TypeError):
                env.set(var_name, body_val)

    # ── Helpers ───────────────────────────────────────────────────────

    def _propagate_partial_returns(
        self, env: SymbolicEnv, body_env: SymbolicEnv, guard: Any,
    ) -> None:
        for pg, pv in body_env.partial_returns:
            if len(env.partial_returns) >= MAX_PARTIAL_RETURNS:
                break
            try:
                env.partial_returns.append((z3.And(guard, pg), pv))
            except (z3.Z3Exception, TypeError):
                env.partial_returns.append((guard, pv))

    @staticmethod
    def _advance_loop_var(current_i: Any, step: Any) -> Any:
        try:
            result = current_i + step
            try:
                result = z3.simplify(result)
            except (z3.Z3Exception, TypeError, AttributeError):
                pass
            return result
        except (z3.Z3Exception, TypeError):
            raise SymbolicExecError("Cannot advance symbolic loop variable.")

    @staticmethod
    def _loop_skip_vars(target: ast.expr) -> set[str]:
        if isinstance(target, ast.Tuple):
            return {e.id for e in target.elts if isinstance(e, ast.Name)}
        if isinstance(target, ast.Name):
            return {target.id}
        return set()

    def _bind_for_target(self, target: ast.expr, item: Any,
                         env: SymbolicEnv) -> None:
        if isinstance(target, ast.Name):
            env.set(target.id, item)
        elif isinstance(target, ast.Tuple):
            if isinstance(item, (list, tuple)):
                for j, elt in enumerate(target.elts):
                    if isinstance(elt, ast.Name):
                        env.set(elt.id, item[j] if j < len(item) else z3.IntVal(0))
            else:
                if target.elts and isinstance(target.elts[0], ast.Name):
                    env.set(target.elts[0].id, item)
        else:
            raise SymbolicExecError(
                f"Unsupported for-loop target type: {type(target).__name__}"
            )

    def _try_extract_range(
        self, node: ast.expr, env: SymbolicEnv,
    ) -> tuple[int, int, int] | None:
        if not isinstance(node, ast.Call):
            return None
        if not (isinstance(node.func, ast.Name) and node.func.id == "range"):
            return None
        args = [self._try_concrete(self._eval_expr(a, env)) for a in node.args]
        if any(a is None for a in args):
            return None
        if len(args) == 1:
            return (0, args[0], 1)
        if len(args) == 2:
            return (args[0], args[1], 1)
        if len(args) == 3:
            return (args[0], args[1], args[2])
        return None

    def _try_extract_symbolic_range(
        self, node: ast.expr, env: SymbolicEnv,
    ) -> tuple[Any, Any, Any] | None:
        if not isinstance(node, ast.Call):
            return None
        if not (isinstance(node.func, ast.Name) and node.func.id == "range"):
            return None
        args = [self._eval_expr(a, env) for a in node.args]
        if len(args) == 1:
            return (z3.IntVal(0), args[0], z3.IntVal(1))
        if len(args) == 2:
            return (args[0], args[1], z3.IntVal(1))
        if len(args) == 3:
            return (args[0], args[1], args[2])
        return None
