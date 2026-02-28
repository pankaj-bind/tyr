"""
Tyr — Comprehension evaluation mixin.

Provides ``_eval_listcomp``, ``_eval_dictcomp``, ``_eval_setcomp``,
``_eval_generatorexp`` and shared helpers ``_comprehension_items``
and ``_bind_comp_target``.

Extracted from ``expressions.py`` for SRP compliance — comprehensions
are a distinct semantic unit with their own iteration/binding logic.
"""
from __future__ import annotations

import ast
import logging
from typing import Any

import z3

from config import MAX_BMC_LENGTH
from symbolic.types import (
    SymbolicEnv, SymbolicExecError,
    SymbolicList, SymbolicDict, SymbolicSet,
)

logger = logging.getLogger("tyr.symbolic.comprehensions")


class ComprehensionEvaluator:
    """Mixin: list / dict / set comprehensions and generator expressions."""

    # ── List comprehension ────────────────────────────────────────────

    def _eval_listcomp(self, node: ast.ListComp, env: SymbolicEnv) -> SymbolicList:
        if len(node.generators) != 1:
            raise SymbolicExecError(
                "Only single-generator list comprehensions are supported"
            )
        gen = node.generators[0]
        if not isinstance(gen.target, ast.Name):
            raise SymbolicExecError(
                "List comprehension target must be a simple variable"
            )
        var_name = gen.target.id
        iter_val = self._eval_expr(gen.iter, env)

        result_arr = z3.Array(self.fresh_array_name("listcomp"),
                              z3.IntSort(), z3.IntSort())
        result_len = z3.IntVal(0)
        items: list[tuple[Any, Any]] = []

        if isinstance(iter_val, SymbolicList):
            for k in range(MAX_BMC_LENGTH):
                ki = z3.IntVal(k)
                in_bounds = ki < iter_val.length
                comp_env = env.copy()
                comp_env.set(var_name, z3.Select(iter_val.array, ki))
                include: Any = in_bounds
                for if_clause in gen.ifs:
                    include = z3.And(include, self._eval_expr(if_clause, comp_env))
                items.append((include, self._eval_expr(node.elt, comp_env)))
        elif isinstance(iter_val, (list, tuple)):
            for item in iter_val:
                comp_env = env.copy()
                comp_env.set(var_name, item)
                include = z3.BoolVal(True)
                for if_clause in gen.ifs:
                    include = z3.And(include, self._eval_expr(if_clause, comp_env))
                items.append((include, self._eval_expr(node.elt, comp_env)))
        else:
            raise SymbolicExecError(
                f"List comprehension over unsupported iterable: "
                f"{type(iter_val).__name__}"
            )

        for guard, val in items:
            result_arr = z3.If(guard, z3.Store(result_arr, result_len, val), result_arr)
            result_len = z3.If(guard, result_len + 1, result_len)
        return SymbolicList(array=result_arr, length=result_len)

    # ── Dict comprehension ────────────────────────────────────────────

    def _eval_dictcomp(self, node: ast.DictComp, env: SymbolicEnv) -> SymbolicDict:
        if len(node.generators) != 1:
            raise SymbolicExecError(
                "Only single-generator dict comprehensions are supported"
            )
        gen = node.generators[0]
        iter_val = self._eval_expr(gen.iter, env)

        presence = z3.K(z3.IntSort(), z3.BoolVal(False))
        values = z3.K(z3.IntSort(), z3.IntVal(0))
        tracked: list[Any] = []

        element_list = self._comprehension_items(gen, iter_val, env)
        for item, in_bounds in element_list:
            comp_env = env.copy()
            self._bind_comp_target(gen.target, item, comp_env)
            include: Any = in_bounds
            for if_clause in gen.ifs:
                cond = self._eval_expr(if_clause, comp_env)
                include = z3.And(include, self._to_z3_bool(cond))
            key = self._eval_expr(node.key, comp_env)
            val = self._eval_expr(node.value, comp_env)
            presence = z3.If(include,
                             z3.Store(presence, key, z3.BoolVal(True)),
                             presence)
            values = z3.If(include,
                           z3.Store(values, key, val),
                           values)
            tracked.append(key)
        return SymbolicDict(presence=presence, values=values,
                            tracked_keys=tracked)

    # ── Set comprehension ─────────────────────────────────────────────

    def _eval_setcomp(self, node: ast.SetComp, env: SymbolicEnv) -> SymbolicSet:
        if len(node.generators) != 1:
            raise SymbolicExecError(
                "Only single-generator set comprehensions are supported"
            )
        gen = node.generators[0]
        iter_val = self._eval_expr(gen.iter, env)

        presence = z3.K(z3.IntSort(), z3.BoolVal(False))

        element_list = self._comprehension_items(gen, iter_val, env)
        for item, in_bounds in element_list:
            comp_env = env.copy()
            self._bind_comp_target(gen.target, item, comp_env)
            include: Any = in_bounds
            for if_clause in gen.ifs:
                cond = self._eval_expr(if_clause, comp_env)
                include = z3.And(include, self._to_z3_bool(cond))
            elt = self._eval_expr(node.elt, comp_env)
            presence = z3.If(include,
                             z3.Store(presence, elt, z3.BoolVal(True)),
                             presence)
        return SymbolicSet(presence=presence)

    # ── Generator expression (materialised as list) ───────────────────

    def _eval_generatorexp(self, node: ast.GeneratorExp,
                           env: SymbolicEnv) -> SymbolicList:
        """Materialise a generator expression into a SymbolicList.

        This is exact for the bounded domain and allows ``sum(x for ...)``,
        ``any(x for ...)``, etc. to work seamlessly.
        """
        if len(node.generators) != 1:
            raise SymbolicExecError(
                "Only single-generator generator expressions are supported"
            )
        gen = node.generators[0]
        iter_val = self._eval_expr(gen.iter, env)

        result_arr = z3.Array(self.fresh_array_name("genexp"),
                              z3.IntSort(), z3.IntSort())
        result_len = z3.IntVal(0)
        items: list[tuple[Any, Any]] = []

        element_list = self._comprehension_items(gen, iter_val, env)
        for item, in_bounds in element_list:
            comp_env = env.copy()
            self._bind_comp_target(gen.target, item, comp_env)
            include: Any = in_bounds
            for if_clause in gen.ifs:
                cond = self._eval_expr(if_clause, comp_env)
                include = z3.And(include, self._to_z3_bool(cond))
            items.append((include, self._eval_expr(node.elt, comp_env)))

        for guard, val in items:
            result_arr = z3.If(guard, z3.Store(result_arr, result_len, val),
                               result_arr)
            result_len = z3.If(guard, result_len + 1, result_len)
        return SymbolicList(array=result_arr, length=result_len)

    # ── Shared comprehension helpers ──────────────────────────────────

    def _comprehension_items(
        self,
        gen: ast.comprehension,
        iter_val: Any,
        env: SymbolicEnv,
    ) -> list[tuple[Any, Any]]:
        """Return ``(element, in_bounds_guard)`` pairs for a comprehension."""
        if isinstance(iter_val, SymbolicList):
            result = []
            for k in range(MAX_BMC_LENGTH):
                ki = z3.IntVal(k)
                result.append((z3.Select(iter_val.array, ki),
                               ki < iter_val.length))
            return result
        if isinstance(iter_val, (list, tuple)):
            return [(item, z3.BoolVal(True)) for item in iter_val]
        raise SymbolicExecError(
            f"Comprehension over unsupported iterable: "
            f"{type(iter_val).__name__}"
        )

    def _bind_comp_target(self, target: ast.expr, item: Any,
                          env: SymbolicEnv) -> None:
        """Bind a comprehension iteration variable (supports tuples)."""
        if isinstance(target, ast.Name):
            env.set(target.id, item)
        elif isinstance(target, ast.Tuple):
            if isinstance(item, (list, tuple)):
                for j, elt in enumerate(target.elts):
                    if isinstance(elt, ast.Name):
                        env.set(elt.id,
                                item[j] if j < len(item) else z3.IntVal(0))
        else:
            raise SymbolicExecError(
                f"Unsupported comprehension target: {type(target).__name__}"
            )
