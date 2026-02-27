"""
Tyr — Statement execution mixin.

Handles return, assignment, augmented assignment, and if/else —
everything that is NOT a loop.
"""
from __future__ import annotations

import ast
import logging
from typing import Any

import z3

from config import (
    NONE_SENTINEL, MAX_BMC_LENGTH, MAX_PARTIAL_RETURNS,
)
from symbolic.types import (
    SymbolicEnv, SymbolicExecError,
    SymbolicList, SymbolicDict, SymbolicSet,
)

logger = logging.getLogger("tyr.symbolic.statements")


class StatementExecutor:
    """Mixin: return, assign, aug-assign, if/else."""

    # ── Return ────────────────────────────────────────────────────────

    def _exec_return(self, node: ast.Return, env: SymbolicEnv) -> None:
        val = (
            self._eval_expr(node.value, env)
            if node.value is not None
            else z3.IntVal(NONE_SENTINEL)
        )
        for guard, partial_val in reversed(env.partial_returns):
            try:
                val = z3.If(guard, partial_val, val)
            except (z3.Z3Exception, TypeError):
                pass
        env.partial_returns.clear()
        env.return_value = val
        env.has_returned = True

    # ── Assignment ────────────────────────────────────────────────────

    def _exec_assign(self, node: ast.Assign, env: SymbolicEnv) -> None:
        value = self._eval_expr(node.value, env)
        for target in node.targets:
            self._assign_target(target, value, env)

    def _assign_target(self, target: ast.expr, value: Any, env: SymbolicEnv) -> None:
        if isinstance(target, ast.Name):
            env.set(target.id, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(value, (list, tuple)):
                for i, elt in enumerate(target.elts):
                    if isinstance(elt, ast.Name):
                        env.set(elt.id, value[i] if i < len(value) else z3.IntVal(0))
            else:
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        env.set(elt.id, self.fresh_var(elt.id))
        elif isinstance(target, ast.Subscript):
            self._assign_subscript(target, value, env)
        else:
            raise SymbolicExecError(
                f"Unsupported assignment target: {type(target).__name__}"
            )

    def _assign_subscript(self, target: ast.Subscript, value: Any,
                          env: SymbolicEnv) -> None:
        arr = self._eval_expr(target.value, env)
        idx = self._eval_expr(target.slice, env)

        if isinstance(arr, SymbolicDict):
            new_p = z3.Store(arr.presence, idx, z3.BoolVal(True))
            new_v = z3.Store(arr.values, idx, value)
            tracked = list(arr.tracked_keys) + [idx]
            updated = SymbolicDict(presence=new_p, values=new_v,
                                   tracked_keys=tracked)
        elif isinstance(arr, SymbolicList):
            updated = SymbolicList(array=z3.Store(arr.array, idx, value),
                                   length=arr.length)
        elif z3.is_array(arr):
            updated = z3.Store(arr, idx, value)
        else:
            raise SymbolicExecError(
                f"Subscript assignment on unsupported type: {type(arr).__name__}"
            )
        if isinstance(target.value, ast.Name):
            env.set(target.value.id, updated)

    # ── Augmented assignment ──────────────────────────────────────────

    def _exec_augassign(self, node: ast.AugAssign, env: SymbolicEnv) -> None:
        current = self._eval_expr(node.target, env)
        rhs = self._eval_expr(node.value, env)
        result = self._apply_binop(node.op, current, rhs)

        if isinstance(node.target, ast.Name):
            env.set(node.target.id, result)
        elif isinstance(node.target, ast.Subscript):
            arr = self._eval_expr(node.target.value, env)
            idx = self._eval_expr(node.target.slice, env)
            if isinstance(arr, SymbolicDict):
                new_v = z3.Store(arr.values, idx, result)
                new_p = z3.Store(arr.presence, idx, z3.BoolVal(True))
                tracked = list(arr.tracked_keys)
                if not any(self._z3_eq(idx, tk) for tk in tracked):
                    tracked.append(idx)
                updated = SymbolicDict(presence=new_p, values=new_v,
                                       tracked_keys=tracked)
                if isinstance(node.target.value, ast.Name):
                    env.set(node.target.value.id, updated)
            elif isinstance(arr, SymbolicList):
                updated = SymbolicList(array=z3.Store(arr.array, idx, result),
                                       length=arr.length)
                if isinstance(node.target.value, ast.Name):
                    env.set(node.target.value.id, updated)
            elif z3.is_array(arr):
                if isinstance(node.target.value, ast.Name):
                    env.set(node.target.value.id, z3.Store(arr, idx, result))

    # ── If / Else ─────────────────────────────────────────────────────

    def _exec_if(self, node: ast.If, env: SymbolicEnv) -> None:
        cond = self._eval_expr(node.test, env)
        if isinstance(cond, bool):
            cond = z3.BoolVal(cond)
        elif isinstance(cond, int):
            cond = z3.BoolVal(bool(cond))

        then_env = env.copy()
        self._exec_body(node.body, then_env)

        else_env = env.copy()
        if node.orelse:
            self._exec_body(node.orelse, else_env)

        # ── Merge variable bindings ───────────────────────────────────
        all_vars = set(then_env.bindings) | set(else_env.bindings)
        for var in all_vars:
            tv = then_env.bindings.get(var)
            ev = else_env.bindings.get(var)
            if tv is not None and ev is not None:
                merged = self._if_merge(cond, tv, ev)
                env.set(var, merged if merged is not None else tv)
            elif tv is not None:
                old = env.bindings.get(var)
                if old is not None:
                    merged = self._if_merge(cond, tv, old)
                    env.set(var, merged if merged is not None else tv)
            elif ev is not None:
                old = env.bindings.get(var)
                if old is not None:
                    merged = self._if_merge(cond, old, ev)
                    if merged is not None:
                        env.set(var, merged)

        # ── Merge returns ─────────────────────────────────────────────
        if then_env.has_returned and else_env.has_returned:
            try:
                env.return_value = z3.If(cond, then_env.return_value,
                                         else_env.return_value)
            except (z3.Z3Exception, TypeError):
                env.return_value = then_env.return_value
            env.has_returned = True
        elif then_env.has_returned and not else_env.has_returned:
            env.partial_returns.append((cond, then_env.return_value))
        elif else_env.has_returned and not then_env.has_returned:
            env.partial_returns.append((z3.Not(cond), else_env.return_value))

        # Propagate sub-partials
        if len(env.partial_returns) < MAX_PARTIAL_RETURNS:
            for guard, val in then_env.partial_returns:
                if len(env.partial_returns) >= MAX_PARTIAL_RETURNS:
                    break
                try:
                    env.partial_returns.append((z3.And(cond, guard), val))
                except (z3.Z3Exception, TypeError):
                    env.partial_returns.append((cond, val))
            for guard, val in else_env.partial_returns:
                if len(env.partial_returns) >= MAX_PARTIAL_RETURNS:
                    break
                try:
                    env.partial_returns.append((z3.And(z3.Not(cond), guard), val))
                except (z3.Z3Exception, TypeError):
                    env.partial_returns.append((z3.Not(cond), val))

        # ── Merge break (CRG-1/CRG-2 fix) ────────────────────────────
        if then_env.has_broken and else_env.has_broken:
            env.has_broken = True
        elif then_env.has_broken and not else_env.has_broken:
            env._break_guard = cond
        elif else_env.has_broken and not then_env.has_broken:
            env._break_guard = z3.Not(cond)

        # ── Merge continue (CRG-1 fix) ───────────────────────────────
        if then_env.has_continued and else_env.has_continued:
            env.has_continued = True
        elif then_env.has_continued and not else_env.has_continued:
            env._continue_guard = cond
        elif else_env.has_continued and not then_env.has_continued:
            env._continue_guard = z3.Not(cond)

    # ── If-merge helper ───────────────────────────────────────────────

    def _if_merge(self, cond: Any, true_val: Any, false_val: Any) -> Any | None:
        if isinstance(true_val, SymbolicList) and isinstance(false_val, SymbolicList):
            merged_len = z3.If(cond, true_val.length, false_val.length)
            merged_arr = false_val.array
            for k in range(MAX_BMC_LENGTH):
                ki = z3.IntVal(k)
                merged_arr = z3.Store(
                    merged_arr, ki,
                    z3.If(cond,
                          z3.Select(true_val.array, ki),
                          z3.Select(false_val.array, ki)),
                )
            return SymbolicList(array=merged_arr, length=merged_len)

        if isinstance(true_val, SymbolicDict) and isinstance(false_val, SymbolicDict):
            all_keys = list(true_val.tracked_keys)
            for k in false_val.tracked_keys:
                if not any(self._z3_eq(k, tk) for tk in all_keys):
                    all_keys.append(k)
            mp = false_val.presence
            mv = false_val.values
            for key in all_keys:
                mp = z3.Store(mp, key,
                              z3.If(cond,
                                    z3.Select(true_val.presence, key),
                                    z3.Select(false_val.presence, key)))
                mv = z3.Store(mv, key,
                              z3.If(cond,
                                    z3.Select(true_val.values, key),
                                    z3.Select(false_val.values, key)))
            return SymbolicDict(presence=mp, values=mv, tracked_keys=all_keys)

        if isinstance(true_val, SymbolicSet) and isinstance(false_val, SymbolicSet):
            x = z3.Int("__set_merge_key")
            merged = z3.Lambda(
                [x],
                z3.If(cond,
                      z3.Select(true_val.presence, x),
                      z3.Select(false_val.presence, x)),
            )
            return SymbolicSet(presence=merged)

        # Sort coercion before z3.If
        true_val, false_val = self._coerce_sorts(true_val, false_val)
        try:
            return z3.If(cond, true_val, false_val)
        except (z3.Z3Exception, TypeError):
            return None
