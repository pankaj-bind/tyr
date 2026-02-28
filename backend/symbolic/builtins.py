"""
Tyr — Built-in function dispatch mixin.

Handles ``_eval_call`` for top-level function calls like
``abs``, ``min``, ``max``, ``len``, ``sum``, ``sorted``, ``enumerate``,
``range``, ``map``, ``filter``, ``all``, ``any``, etc.

CRG-4 fix: ``min`` / ``max`` on a SymbolicList no longer inject
``length > 0`` into ``env.constraints``.  Instead, the result is
wrapped in ``If(length > 0, computed, fresh_var)`` so the solver
domain is never artificially shrunk.
"""
from __future__ import annotations

import ast
import logging
from typing import Any

import z3

from config import MAX_BMC_LENGTH, INF_SENTINEL, NEG_INF_SENTINEL
from symbolic.types import (
    SymbolicEnv, SymbolicExecError,
    SymbolicList, SymbolicDict, SymbolicSet,
    EnumerateResult, LambdaClosure,
)

logger = logging.getLogger("tyr.symbolic.builtins")


class BuiltinDispatcher:
    """Mixin: ``_eval_call`` and all built-in function handlers."""

    def _eval_call(self, node: ast.Call, env: SymbolicEnv) -> Any:
        # Method call on an object → delegate
        if isinstance(node.func, ast.Attribute):
            return self._eval_method_call(node, env)

        if not isinstance(node.func, ast.Name):
            raise SymbolicExecError(
                f"Unsupported callable: {ast.dump(node.func)}"
            )

        fname = node.func.id
        args = [self._eval_expr(a, env) for a in node.args]

        # ── Dispatch table ────────────────────────────────────────────
        handler = _BUILTIN_TABLE.get(fname)
        if handler is not None:
            return handler(self, fname, args, node, env)

        raise SymbolicExecError(f"Unsupported function call: {fname}()")


# ── Individual handlers ──────────────────────────────────────────────

def _builtin_abs(self, _fn, args, _node, _env):
    if len(args) != 1:
        raise SymbolicExecError("abs() requires exactly 1 argument")
    x = args[0]
    return z3.If(x >= 0, x, -x)


def _builtin_min(self, _fn, args, node, env):
    if len(args) == 1 and isinstance(args[0], (list, SymbolicList)):
        result = self._symbolic_min_of(args[0])
        # CRG-4 fix: do NOT inject length > 0 into env.constraints.
        # Wrap result with empty-list guard so the solver stays free.
        if isinstance(args[0], SymbolicList):
            fresh = self.fresh_var("min_empty")
            result = z3.If(args[0].length > 0, result, fresh)
        return result
    if len(args) >= 2:
        return self._symbolic_min(args)
    raise SymbolicExecError("min() requires at least 2 arguments or a list")


def _builtin_max(self, _fn, args, node, env):
    if len(args) == 1 and isinstance(args[0], (list, SymbolicList)):
        result = self._symbolic_max_of(args[0])
        # CRG-4 fix: same guard approach as min.
        if isinstance(args[0], SymbolicList):
            fresh = self.fresh_var("max_empty")
            result = z3.If(args[0].length > 0, result, fresh)
        return result
    if len(args) >= 2:
        return self._symbolic_max(args)
    raise SymbolicExecError("max() requires at least 2 arguments or a list")


def _builtin_len(self, _fn, args, _node, _env):
    if len(args) != 1:
        raise SymbolicExecError("len() requires exactly 1 argument")
    a = args[0]
    if isinstance(a, SymbolicList):
        return a.length
    if isinstance(a, list):
        return z3.IntVal(len(a))
    if isinstance(a, SymbolicDict):
        # Count tracked keys whose presence flag is True
        result = z3.IntVal(0)
        for k in a.tracked_keys:
            result = result + z3.If(
                z3.Select(a.presence, k), z3.IntVal(1), z3.IntVal(0),
            )
        return result
    if isinstance(a, SymbolicSet):
        # SymbolicSet has no tracked key list; return a fresh symbolic
        # variable (over-approximation).  This is sound for equivalence
        # checking because both sides see the same SymbolicSet.
        return self.fresh_var("set_len")
    raise SymbolicExecError(f"len() on unsupported type: {type(a).__name__}")


def _builtin_sum(self, _fn, args, _node, _env):
    if len(args) != 1:
        raise SymbolicExecError("sum() requires exactly 1 argument")
    a = args[0]
    if isinstance(a, SymbolicList):
        result = z3.IntVal(0)
        for k in range(MAX_BMC_LENGTH):
            elem = z3.Select(a.array, z3.IntVal(k))
            in_bounds = z3.IntVal(k) < a.length
            result = z3.If(in_bounds, result + elem, result)
        return result
    if isinstance(a, list):
        if not a:
            return z3.IntVal(0)
        result = a[0]
        for v in a[1:]:
            result = result + v
        return result
    raise SymbolicExecError("sum() on non-list not supported")


def _builtin_enumerate(self, _fn, args, _node, _env):
    if len(args) != 1:
        raise SymbolicExecError("enumerate() requires exactly 1 argument")
    a = args[0]
    if isinstance(a, SymbolicList):
        pairs = []
        for k in range(MAX_BMC_LENGTH):
            ki = z3.IntVal(k)
            pairs.append((ki, z3.Select(a.array, ki)))
        return EnumerateResult(pairs=pairs, source_length=a.length)
    raise SymbolicExecError("enumerate() only supported for SymbolicList")


def _builtin_range(self, _fn, args, _node, _env):
    concrete = [self._try_concrete(a) for a in args]
    if any(a is None for a in concrete):
        raise SymbolicExecError(
            "range() with symbolic args not supported in expression context"
        )
    if len(concrete) == 1:
        return [z3.IntVal(i) for i in range(concrete[0])]
    if len(concrete) == 2:
        return [z3.IntVal(i) for i in range(concrete[0], concrete[1])]
    if len(concrete) == 3:
        return [z3.IntVal(i) for i in range(*concrete)]
    raise SymbolicExecError("range() requires 1-3 arguments")


def _builtin_sorted(self, _fn, args, node, _env):
    if len(args) < 1 or not isinstance(args[0], SymbolicList):
        raise SymbolicExecError("sorted() only supported for SymbolicList")
    reverse = False
    for kw in node.keywords:
        if kw.arg == "reverse":
            kw_val = self._eval_expr(kw.value, _env)
            concrete_r = self._try_concrete_bool(kw_val)
            if concrete_r is True:
                reverse = True
        elif kw.arg == "key":
            raise SymbolicExecError(
                "sorted() with key= not supported"
            )
    arr = args[0].array
    n = args[0].length
    for _ in range(MAX_BMC_LENGTH):
        for j in range(MAX_BMC_LENGTH - 1):
            ji = z3.IntVal(j)
            ni = z3.IntVal(j + 1)
            valid = z3.And(ji < n, ni < n)
            v1 = z3.Select(arr, ji)
            v2 = z3.Select(arr, ni)
            swap = z3.And(valid, v1 < v2) if reverse else z3.And(valid, v1 > v2)
            arr = z3.If(
                swap,
                z3.Store(z3.Store(arr, ji, v2), ni, v1),
                arr,
            )
    return SymbolicList(array=arr, length=n)


def _builtin_int(self, _fn, args, _node, _env):
    if len(args) == 1:
        return args[0]
    raise SymbolicExecError("int() requires exactly 1 argument")


def _builtin_bool(self, _fn, args, _node, _env):
    if len(args) == 1:
        x = args[0]
        return z3.If(x != z3.IntVal(0), z3.BoolVal(True), z3.BoolVal(False))
    raise SymbolicExecError("bool() requires exactly 1 argument")


def _builtin_set(self, _fn, args, _node, _env):
    if len(args) == 0:
        return SymbolicSet(presence=z3.K(z3.IntSort(), z3.BoolVal(False)))
    if len(args) == 1:
        if isinstance(args[0], SymbolicList):
            presence = z3.K(z3.IntSort(), z3.BoolVal(False))
            for k in range(MAX_BMC_LENGTH):
                ki = z3.IntVal(k)
                in_bounds = ki < args[0].length
                elem = z3.Select(args[0].array, ki)
                presence = z3.If(
                    in_bounds,
                    z3.Store(presence, elem, z3.BoolVal(True)),
                    presence,
                )
            return SymbolicSet(presence=presence)
        if isinstance(args[0], SymbolicSet):
            return args[0].copy()
    raise SymbolicExecError("set() requires 0 or 1 arguments")


def _builtin_dict(self, _fn, args, _node, _env):
    if len(args) == 0:
        return SymbolicDict(
            presence=z3.K(z3.IntSort(), z3.BoolVal(False)),
            values=z3.K(z3.IntSort(), z3.IntVal(0)),
            tracked_keys=[],
        )
    raise SymbolicExecError("dict() with arguments not yet supported")


def _builtin_list(self, _fn, args, _node, _env):
    if len(args) == 0:
        arr = z3.Array(
            self.fresh_array_name("emptylist"), z3.IntSort(), z3.IntSort(),
        )
        return SymbolicList(array=arr, length=z3.IntVal(0))
    if len(args) == 1 and isinstance(args[0], SymbolicList):
        return args[0].copy()
    raise SymbolicExecError("list() requires 0 args or a SymbolicList")


def _builtin_map(self, _fn, args, _node, _env):
    if len(args) != 2:
        raise SymbolicExecError("map() requires exactly 2 arguments")
    fn, iterable = args
    if not isinstance(fn, LambdaClosure):
        raise SymbolicExecError("map() only supported with lambda functions")
    if isinstance(iterable, SymbolicList):
        new_arr = z3.Array(
            self.fresh_array_name("map"), z3.IntSort(), z3.IntSort(),
        )
        for k in range(MAX_BMC_LENGTH):
            ki = z3.IntVal(k)
            elem = z3.Select(iterable.array, ki)
            mapped = fn.apply(elem)
            in_bounds = ki < iterable.length
            new_arr = z3.If(
                in_bounds, z3.Store(new_arr, ki, mapped), new_arr,
            )
        return SymbolicList(array=new_arr, length=iterable.length)
    if isinstance(iterable, list):
        return [fn.apply(e) for e in iterable]
    raise SymbolicExecError("map() only supported for lists")


def _builtin_filter(self, _fn, args, _node, _env):
    if len(args) != 2:
        raise SymbolicExecError("filter() requires exactly 2 arguments")
    fn, iterable = args
    if not isinstance(fn, LambdaClosure):
        raise SymbolicExecError("filter() only supported with lambda functions")
    if isinstance(iterable, SymbolicList):
        new_arr = z3.Array(
            self.fresh_array_name("filter"), z3.IntSort(), z3.IntSort(),
        )
        new_len = z3.IntVal(0)
        for k in range(MAX_BMC_LENGTH):
            ki = z3.IntVal(k)
            elem = z3.Select(iterable.array, ki)
            pred = fn.apply(elem)
            in_bounds = ki < iterable.length
            keep = (
                z3.And(in_bounds, pred)
                if isinstance(pred, z3.BoolRef)
                else z3.And(in_bounds, pred != z3.IntVal(0))
            )
            new_arr = z3.If(keep, z3.Store(new_arr, new_len, elem), new_arr)
            new_len = z3.If(keep, new_len + 1, new_len)
        return SymbolicList(array=new_arr, length=new_len)
    raise SymbolicExecError("filter() only supported for lists")


# ── NEW: all() and any() — CRG-5 fix ─────────────────────────────

def _builtin_all(self, _fn, args, _node, _env):
    """``all(lst)`` — True iff every in-bounds element is truthy."""
    if len(args) != 1:
        raise SymbolicExecError("all() requires exactly 1 argument")
    a = args[0]
    if isinstance(a, SymbolicList):
        result = z3.BoolVal(True)
        for k in range(MAX_BMC_LENGTH):
            ki = z3.IntVal(k)
            in_bounds = ki < a.length
            elem = z3.Select(a.array, ki)
            result = z3.And(result, z3.Implies(in_bounds, elem != z3.IntVal(0)))
        return result
    if isinstance(a, list):
        if not a:
            return z3.BoolVal(True)
        parts = [e != z3.IntVal(0) if isinstance(e, z3.ArithRef) else e for e in a]
        return z3.And(*parts)
    raise SymbolicExecError("all() only supported for lists")


def _builtin_any(self, _fn, args, _node, _env):
    """``any(lst)`` — True iff at least one in-bounds element is truthy."""
    if len(args) != 1:
        raise SymbolicExecError("any() requires exactly 1 argument")
    a = args[0]
    if isinstance(a, SymbolicList):
        clauses = []
        for k in range(MAX_BMC_LENGTH):
            ki = z3.IntVal(k)
            in_bounds = ki < a.length
            elem = z3.Select(a.array, ki)
            clauses.append(z3.And(in_bounds, elem != z3.IntVal(0)))
        return z3.Or(*clauses) if clauses else z3.BoolVal(False)
    if isinstance(a, list):
        if not a:
            return z3.BoolVal(False)
        parts = [e != z3.IntVal(0) if isinstance(e, z3.ArithRef) else e for e in a]
        return z3.Or(*parts)
    raise SymbolicExecError("any() only supported for lists")


# ── float() handler ───────────────────────────────────────────────

def _builtin_float(self, _fn, args, _node, _env):
    """Handle float('inf'), float('-inf'), and plain float(x)."""
    if len(args) != 1:
        raise SymbolicExecError("float() requires exactly 1 argument")
    a = args[0]
    if isinstance(a, str):
        if a in ("inf", "+inf"):
            return z3.IntVal(INF_SENTINEL)
        if a == "-inf":
            return z3.IntVal(NEG_INF_SENTINEL)
        raise SymbolicExecError(f"float('{a}') not supported symbolically")
    # numeric arg → treat as identity (integer approx)
    return a


# ── min / max helpers ─────────────────────────────────────────────

def _symbolic_min(self, values):
    result = values[0]
    for v in values[1:]:
        result = z3.If(v < result, v, result)
    return result

BuiltinDispatcher._symbolic_min = _symbolic_min


def _symbolic_max(self, values):
    result = values[0]
    for v in values[1:]:
        result = z3.If(v > result, v, result)
    return result

BuiltinDispatcher._symbolic_max = _symbolic_max


def _symbolic_min_of(self, arg):
    if isinstance(arg, list):
        return self._symbolic_min(arg)
    if isinstance(arg, SymbolicList):
        result = z3.Select(arg.array, z3.IntVal(0))
        for k in range(1, MAX_BMC_LENGTH):
            ki = z3.IntVal(k)
            in_bounds = ki < arg.length
            elem = z3.Select(arg.array, ki)
            result = z3.If(z3.And(in_bounds, elem < result), elem, result)
        return result
    raise SymbolicExecError(f"min() on unsupported type: {type(arg).__name__}")

BuiltinDispatcher._symbolic_min_of = _symbolic_min_of


def _symbolic_max_of(self, arg):
    if isinstance(arg, list):
        return self._symbolic_max(arg)
    if isinstance(arg, SymbolicList):
        result = z3.Select(arg.array, z3.IntVal(0))
        for k in range(1, MAX_BMC_LENGTH):
            ki = z3.IntVal(k)
            in_bounds = ki < arg.length
            elem = z3.Select(arg.array, ki)
            result = z3.If(z3.And(in_bounds, elem > result), elem, result)
        return result
    raise SymbolicExecError(f"max() on unsupported type: {type(arg).__name__}")

BuiltinDispatcher._symbolic_max_of = _symbolic_max_of


# ── reversed() handler ───────────────────────────────────────────────

def _builtin_reversed(self, _fn, args, _node, _env):
    if len(args) != 1:
        raise SymbolicExecError("reversed() requires exactly 1 argument")
    a = args[0]
    if isinstance(a, SymbolicList):
        new_arr = z3.Array(
            self.fresh_array_name("reversed"), z3.IntSort(), z3.IntSort(),
        )
        for k in range(MAX_BMC_LENGTH):
            ki = z3.IntVal(k)
            in_bounds = ki < a.length
            src_idx = a.length - z3.IntVal(1) - ki
            new_arr = z3.If(
                in_bounds,
                z3.Store(new_arr, ki, z3.Select(a.array, src_idx)),
                new_arr,
            )
        return SymbolicList(array=new_arr, length=a.length)
    if isinstance(a, (list, tuple)):
        return list(reversed(a))
    raise SymbolicExecError(f"reversed() on unsupported type: {type(a).__name__}")


# ── zip() handler ───────────────────────────────────────────────────

def _builtin_zip(self, _fn, args, _node, _env):
    """zip(list_a, list_b) → list of (a_i, b_i) tuples."""
    if len(args) != 2:
        raise SymbolicExecError("zip() only supported with exactly 2 arguments")
    a, b = args
    if isinstance(a, SymbolicList) and isinstance(b, SymbolicList):
        # Minimum length
        min_len = z3.If(a.length < b.length, a.length, b.length)
        pairs = []
        for k in range(MAX_BMC_LENGTH):
            ki = z3.IntVal(k)
            pairs.append((z3.Select(a.array, ki), z3.Select(b.array, ki)))
        return EnumerateResult(pairs=pairs, source_length=min_len)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return [list(p) for p in zip(a, b)]
    raise SymbolicExecError("zip() only supported for two lists")


# ── tuple() handler ─────────────────────────────────────────────────

def _builtin_tuple(self, _fn, args, _node, _env):
    if len(args) == 0:
        arr = z3.Array(
            self.fresh_array_name("emptytuple"), z3.IntSort(), z3.IntSort(),
        )
        return SymbolicList(array=arr, length=z3.IntVal(0))
    if len(args) == 1 and isinstance(args[0], SymbolicList):
        return args[0].copy()
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return list(args[0])
    raise SymbolicExecError("tuple() requires 0 args or a list/SymbolicList")


# ── print() no-op ──────────────────────────────────────────────────

def _builtin_print(self, _fn, args, _node, _env):
    """print() is a side-effect — ignore for equivalence checking."""
    return z3.IntVal(0)


# ── Dispatch table ────────────────────────────────────────────────

_BUILTIN_TABLE: dict[str, Any] = {
    "abs": _builtin_abs,
    "min": _builtin_min,
    "max": _builtin_max,
    "len": _builtin_len,
    "sum": _builtin_sum,
    "enumerate": _builtin_enumerate,
    "range": _builtin_range,
    "sorted": _builtin_sorted,
    "int": _builtin_int,
    "bool": _builtin_bool,
    "set": _builtin_set,
    "dict": _builtin_dict,
    "list": _builtin_list,
    "map": _builtin_map,
    "filter": _builtin_filter,
    "all": _builtin_all,
    "any": _builtin_any,
    "float": _builtin_float,
    "reversed": _builtin_reversed,
    "zip": _builtin_zip,
    "tuple": _builtin_tuple,
    "print": _builtin_print,
}
