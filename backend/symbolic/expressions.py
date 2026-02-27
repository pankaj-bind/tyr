"""
Tyr — Expression evaluation mixin.

Provides ``_eval_expr`` and all sub-dispatchers for constants,
binary/unary/boolean ops, comparisons, subscripts, literals,
list comprehensions, and lambdas.
"""
from __future__ import annotations

import ast
import logging
from typing import Any

import z3

from config import (
    NONE_SENTINEL, INF_SENTINEL, NEG_INF_SENTINEL,
    MAX_BMC_LENGTH, MAX_PARTIAL_RETURNS,
)
from symbolic.types import (
    SymbolicEnv, SymbolicExecError,
    SymbolicList, SymbolicDict, SymbolicSet,
    EnumerateResult, LambdaClosure,
)

logger = logging.getLogger("tyr.symbolic.expressions")


class ExpressionEvaluator:
    """Mixin: every ``_eval_*`` method that lives on ASTToZ3Translator."""

    # ── Main dispatch ─────────────────────────────────────────────────

    def _eval_expr(self, node: ast.expr, env: SymbolicEnv) -> Any:
        if isinstance(node, ast.Constant):
            return self._eval_constant(node)
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.BinOp):
            return self._eval_binop(node, env)
        if isinstance(node, ast.UnaryOp):
            return self._eval_unaryop(node, env)
        if isinstance(node, ast.BoolOp):
            return self._eval_boolop(node, env)
        if isinstance(node, ast.Compare):
            return self._eval_compare(node, env)
        if isinstance(node, ast.IfExp):
            return self._eval_ifexp(node, env)
        if isinstance(node, ast.Call):
            return self._eval_call(node, env)
        if isinstance(node, ast.Subscript):
            return self._eval_subscript(node, env)
        if isinstance(node, (ast.List, ast.Tuple)):
            return self._eval_list_literal(node, env)
        if isinstance(node, ast.Dict):
            return self._eval_dict_literal(node, env)
        if isinstance(node, ast.Set):
            return self._eval_set_literal(node, env)
        if isinstance(node, ast.ListComp):
            return self._eval_listcomp(node, env)
        if isinstance(node, ast.Lambda):
            return LambdaClosure(node=node, env=env, translator=self)
        if isinstance(node, ast.Attribute):
            raise SymbolicExecError(
                f"Attribute access not fully supported: {ast.dump(node)}"
            )
        raise SymbolicExecError(
            f"Unsupported expression type: {type(node).__name__} "
            f"(line {getattr(node, 'lineno', '?')})"
        )

    # ── Constants ─────────────────────────────────────────────────────

    def _eval_constant(self, node: ast.Constant) -> Any:
        v = node.value
        if isinstance(v, bool):
            return z3.BoolVal(v)
        if isinstance(v, int):
            return z3.IntVal(v)
        if isinstance(v, float):
            if v == float("inf"):
                return z3.IntVal(INF_SENTINEL)
            if v == float("-inf"):
                return z3.IntVal(NEG_INF_SENTINEL)
            return z3.RealVal(v)
        if v is None:
            return z3.IntVal(NONE_SENTINEL)
        if isinstance(v, str):
            if v not in self._string_map:
                self._string_map[v] = self._next_string_id_ref[0]
                self._next_string_id_ref[0] += 1
            return z3.IntVal(self._string_map[v])
        raise SymbolicExecError(f"Unsupported constant type: {type(v).__name__}")

    # ── Binary operations ─────────────────────────────────────────────

    def _eval_binop(self, node: ast.BinOp, env: SymbolicEnv) -> Any:
        left = self._eval_expr(node.left, env)
        right = self._eval_expr(node.right, env)
        return self._apply_binop(node.op, left, right)

    @staticmethod
    def _apply_binop(op: ast.operator, left: Any, right: Any) -> Any:
        # Bool → Int coercion
        if isinstance(left, z3.BoolRef):
            left = z3.If(left, z3.IntVal(1), z3.IntVal(0))
        if isinstance(right, z3.BoolRef):
            right = z3.If(right, z3.IntVal(1), z3.IntVal(0))
        # Real↔Int coercion (CRG-7 fix)
        left, right = ExpressionEvaluator._coerce_sorts(left, right)

        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.FloorDiv):
            trunc_q = left / right
            concrete_right = None
            if isinstance(right, int) and right > 0:
                concrete_right = right
            elif z3.is_int_value(right) and right.as_long() > 0:
                concrete_right = right.as_long()
            if concrete_right is not None:
                remainder = left - trunc_q * z3.IntVal(concrete_right)
                needs_adjust = z3.And(left < 0, remainder != 0)
                return z3.simplify(z3.If(needs_adjust, trunc_q - 1, trunc_q))
            remainder = left - trunc_q * right
            signs_differ = z3.Or(
                z3.And(left < 0, right > 0),
                z3.And(left > 0, right < 0),
            )
            needs_adjust = z3.And(signs_differ, remainder != 0)
            return z3.simplify(z3.If(needs_adjust, trunc_q - 1, trunc_q))
        if isinstance(op, ast.Mod):
            trunc_mod = left % right
            signs_differ = z3.Or(
                z3.And(left < 0, right > 0),
                z3.And(left > 0, right < 0),
            )
            needs_adjust = z3.And(signs_differ, trunc_mod != 0)
            return z3.simplify(z3.If(needs_adjust, trunc_mod + right, trunc_mod))
        if isinstance(op, ast.Pow):
            concrete_r = None
            if isinstance(right, int):
                concrete_r = right
            elif z3.is_int_value(right):
                concrete_r = right.as_long()
            if concrete_r is not None and 0 <= concrete_r <= 10:
                result = z3.IntVal(1)
                for _ in range(concrete_r):
                    result = result * left
                return result
            raise SymbolicExecError(
                "Symbolic exponentiation not supported for non-concrete exponents."
            )
        if isinstance(op, ast.BitAnd):
            return left & right
        if isinstance(op, ast.BitOr):
            return left | right
        if isinstance(op, ast.BitXor):
            return left ^ right
        if isinstance(op, ast.LShift):
            return left << right
        if isinstance(op, ast.RShift):
            return left >> right
        if isinstance(op, ast.Div):
            return z3.ToReal(left) / z3.ToReal(right)
        raise SymbolicExecError(f"Unsupported binary op: {type(op).__name__}")

    # ── Sort coercion ─────────────────────────────────────────────────

    @staticmethod
    def _coerce_sorts(left: Any, right: Any) -> tuple[Any, Any]:
        """Promote both operands to the same Z3 sort (Int vs Real)."""
        try:
            if isinstance(left, z3.ArithRef) and isinstance(right, z3.ArithRef):
                if left.sort() != right.sort():
                    if left.sort() == z3.IntSort():
                        left = z3.ToReal(left)
                    if right.sort() == z3.IntSort():
                        right = z3.ToReal(right)
        except (z3.Z3Exception, TypeError, AttributeError):
            pass
        return left, right

    # ── Unary operations ──────────────────────────────────────────────

    def _eval_unaryop(self, node: ast.UnaryOp, env: SymbolicEnv) -> Any:
        operand = self._eval_expr(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.Not):
            if isinstance(operand, z3.ArithRef):
                return operand == z3.IntVal(0)
            if isinstance(operand, SymbolicList):
                return operand.length == z3.IntVal(0)
            if isinstance(operand, SymbolicDict):
                if not operand.tracked_keys:
                    return z3.BoolVal(True)
                any_present = z3.Or(*[
                    z3.Select(operand.presence, k)
                    for k in operand.tracked_keys
                ])
                return z3.Not(any_present)
            if isinstance(operand, SymbolicSet):
                return self.fresh_bool("set_empty")
            return z3.Not(operand)
        if isinstance(node.op, ast.Invert):
            return ~operand
        raise SymbolicExecError(f"Unsupported unary op: {type(node.op).__name__}")

    # ── Boolean operations ────────────────────────────────────────────

    def _eval_boolop(self, node: ast.BoolOp, env: SymbolicEnv) -> Any:
        if isinstance(node.op, ast.And):
            result = self._eval_expr(node.values[0], env)
            for val_node in node.values[1:]:
                try:
                    nv = self._eval_expr(val_node, env)
                    result = z3.And(result, nv)
                except SymbolicExecError:
                    break
            return result
        if isinstance(node.op, ast.Or):
            result = self._eval_expr(node.values[0], env)
            for val_node in node.values[1:]:
                try:
                    nv = self._eval_expr(val_node, env)
                    result = z3.Or(result, nv)
                except SymbolicExecError:
                    break
            return result
        raise SymbolicExecError(f"Unsupported bool op: {type(node.op).__name__}")

    # ── Comparisons ───────────────────────────────────────────────────

    def _eval_compare(self, node: ast.Compare, env: SymbolicEnv) -> Any:
        left = self._eval_expr(node.left, env)
        parts: list[Any] = []
        for op, comp in zip(node.ops, node.comparators):
            right = self._eval_expr(comp, env)
            parts.append(self._apply_cmpop(op, left, right))
            left = right
        return parts[0] if len(parts) == 1 else z3.And(*parts)

    def _apply_cmpop(self, op: ast.cmpop, left: Any, right: Any) -> Any:
        if isinstance(op, ast.In):
            return self._symbolic_in(left, right)
        if isinstance(op, ast.NotIn):
            return z3.Not(self._symbolic_in(left, right))
        left, right = self._coerce_pair(left, right)
        left, right = self._coerce_sorts(left, right)
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        raise SymbolicExecError(f"Unsupported comparison: {type(op).__name__}")

    def _symbolic_in(self, needle: Any, haystack: Any) -> Any:
        if isinstance(haystack, SymbolicDict):
            return z3.Select(haystack.presence, needle)
        if isinstance(haystack, SymbolicSet):
            return z3.Select(haystack.presence, needle)
        if isinstance(haystack, SymbolicList):
            clauses = []
            for k in range(MAX_BMC_LENGTH):
                ki = z3.IntVal(k)
                clauses.append(z3.And(ki < haystack.length,
                                      z3.Select(haystack.array, ki) == needle))
            return z3.Or(*clauses) if clauses else z3.BoolVal(False)
        if isinstance(haystack, list):
            if not haystack:
                return z3.BoolVal(False)
            return z3.Or(*[needle == e for e in haystack])
        raise SymbolicExecError(
            f"'in' operator not supported for type: {type(haystack).__name__}"
        )

    @staticmethod
    def _coerce_to_int(val: Any) -> Any:
        if isinstance(val, z3.BoolRef):
            return z3.If(val, z3.IntVal(1), z3.IntVal(0))
        return val

    @staticmethod
    def _coerce_pair(left: Any, right: Any) -> tuple[Any, Any]:
        l_bool = isinstance(left, z3.BoolRef)
        r_bool = isinstance(right, z3.BoolRef)
        l_int = isinstance(left, (z3.ArithRef, int))
        r_int = isinstance(right, (z3.ArithRef, int))
        if l_bool and r_int:
            return z3.If(left, z3.IntVal(1), z3.IntVal(0)), right
        if r_bool and l_int:
            return left, z3.If(right, z3.IntVal(1), z3.IntVal(0))
        return left, right

    # ── Ternary / If-expression ───────────────────────────────────────

    def _eval_ifexp(self, node: ast.IfExp, env: SymbolicEnv) -> Any:
        cond = self._eval_expr(node.test, env)
        tv = self._eval_expr(node.body, env)
        fv = self._eval_expr(node.orelse, env)
        tv, fv = self._coerce_sorts(tv, fv)
        return z3.If(cond, tv, fv)

    # ── Subscript ─────────────────────────────────────────────────────

    def _eval_subscript(self, node: ast.Subscript, env: SymbolicEnv) -> Any:
        obj = self._eval_expr(node.value, env)
        idx = self._eval_expr(node.slice, env)

        if isinstance(obj, SymbolicDict):
            present = z3.Select(obj.presence, idx)
            value = z3.Select(obj.values, idx)
            fresh = self.fresh_var("keyerror")
            return z3.If(present, value, fresh)

        if isinstance(obj, SymbolicList):
            try:
                adjusted = z3.If(idx < 0, idx + obj.length, idx)
            except (z3.Z3Exception, TypeError):
                adjusted = idx
            return z3.Select(obj.array, adjusted)

        if isinstance(obj, list):
            concrete_idx = self._try_concrete(idx)
            if concrete_idx is not None:
                if concrete_idx < 0:
                    concrete_idx += len(obj)
                if 0 <= concrete_idx < len(obj):
                    return obj[concrete_idx]
            if obj:
                list_len = z3.IntVal(len(obj))
                try:
                    adjusted = z3.If(idx < 0, idx + list_len, idx)
                except (z3.Z3Exception, TypeError):
                    adjusted = idx
                result = obj[-1]
                for i in range(len(obj) - 1):
                    result = z3.If(adjusted == z3.IntVal(i), obj[i], result)
                return result
            raise SymbolicExecError("Index into empty list")

        if z3.is_array(obj):
            return z3.Select(obj, idx)
        raise SymbolicExecError(
            f"Subscript on unsupported type: {type(obj).__name__}"
        )

    # ── Literal constructors ──────────────────────────────────────────

    def _eval_list_literal(self, node: ast.List | ast.Tuple, env: SymbolicEnv) -> SymbolicList:
        elements = [self._eval_expr(elt, env) for elt in node.elts]
        n = len(elements)
        if n > MAX_BMC_LENGTH:
            raise SymbolicExecError(
                f"List literal has {n} elements, exceeds MAX_BMC_LENGTH={MAX_BMC_LENGTH}"
            )
        arr = z3.Array(self.fresh_array_name("litarr"), z3.IntSort(), z3.IntSort())
        for i, elem in enumerate(elements):
            arr = z3.Store(arr, z3.IntVal(i), elem)
        return SymbolicList(array=arr, length=z3.IntVal(n))

    def _eval_dict_literal(self, node: ast.Dict, env: SymbolicEnv) -> SymbolicDict:
        presence = z3.K(z3.IntSort(), z3.BoolVal(False))
        values = z3.K(z3.IntSort(), z3.IntVal(0))
        tracked: list[Any] = []
        for key_node, val_node in zip(node.keys, node.values):
            if key_node is None:
                raise SymbolicExecError("Dict unpacking (**) not supported")
            key = self._eval_expr(key_node, env)
            val = self._eval_expr(val_node, env)
            presence = z3.Store(presence, key, z3.BoolVal(True))
            values = z3.Store(values, key, val)
            tracked.append(key)
        return SymbolicDict(presence=presence, values=values, tracked_keys=tracked)

    def _eval_set_literal(self, node: ast.Set, env: SymbolicEnv) -> SymbolicSet:
        presence = z3.K(z3.IntSort(), z3.BoolVal(False))
        for elt_node in node.elts:
            elt = self._eval_expr(elt_node, env)
            presence = z3.Store(presence, elt, z3.BoolVal(True))
        return SymbolicSet(presence=presence)

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

    # ── Static helpers ────────────────────────────────────────────────

    @staticmethod
    def _try_concrete(val: Any) -> int | None:
        if isinstance(val, int):
            return val
        if isinstance(val, z3.IntNumRef):
            return val.as_long()
        if z3.is_int_value(val):
            return val.as_long()
        try:
            s = z3.simplify(val)
            if z3.is_int_value(s):
                return s.as_long()
        except (z3.Z3Exception, TypeError, AttributeError):
            pass
        return None

    @staticmethod
    def _try_concrete_bool(val: Any) -> bool | None:
        if isinstance(val, bool):
            return val
        if isinstance(val, int):
            return bool(val)
        if z3.is_true(val):
            return True
        if z3.is_false(val):
            return False
        try:
            s = z3.simplify(val)
            if z3.is_true(s):
                return True
            if z3.is_false(s):
                return False
        except (z3.Z3Exception, TypeError, AttributeError):
            pass
        return None

    @staticmethod
    def _z3_eq(a: Any, b: Any) -> bool:
        try:
            if a is b:
                return True
            if z3.is_expr(a) and z3.is_expr(b):
                return z3.eq(a, b)
            return str(a) == str(b)
        except Exception:
            return False
