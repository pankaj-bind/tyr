"""
Tyr — Method-call dispatch mixin.

Handles ``_eval_method_call`` for attribute-style calls:
``obj.append(…)``, ``obj.get(…)``, ``obj.add(…)``, etc.
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

logger = logging.getLogger("tyr.symbolic.methods")


class MethodDispatcher:
    """Mixin: all method-call handlers on ASTToZ3Translator."""

    def _eval_method_call(self, node: ast.Call, env: SymbolicEnv) -> Any:
        attr = node.func
        assert isinstance(attr, ast.Attribute)
        method_name = attr.attr
        obj = self._eval_expr(attr.value, env)
        args = [self._eval_expr(a, env) for a in node.args]

        if isinstance(obj, SymbolicDict):
            return self._dict_method(attr, method_name, obj, args, env)
        if isinstance(obj, SymbolicSet):
            return self._set_method(attr, method_name, obj, args, env)
        if isinstance(obj, SymbolicList):
            return self._list_method(attr, method_name, obj, args, env)

        raise SymbolicExecError(
            f"Method call .{method_name}() on unsupported type: "
            f"{type(obj).__name__}"
        )

    # ── SymbolicDict ──────────────────────────────────────────────────

    def _dict_method(
        self, attr: ast.Attribute, name: str,
        obj: SymbolicDict, args: list[Any], env: SymbolicEnv,
    ) -> Any:
        if name == "get":
            if len(args) < 1:
                raise SymbolicExecError(".get() requires at least 1 argument")
            key = args[0]
            default = args[1] if len(args) >= 2 else z3.IntVal(0)
            present = z3.Select(obj.presence, key)
            return z3.If(present, z3.Select(obj.values, key), default)

        if name == "keys":
            return obj.tracked_keys

        if name == "values":
            return [z3.Select(obj.values, k) for k in obj.tracked_keys]

        if name == "items":
            return [(k, z3.Select(obj.values, k)) for k in obj.tracked_keys]

        if name == "pop":
            if len(args) < 1:
                raise SymbolicExecError(".pop() requires at least 1 argument")
            key = args[0]
            default = args[1] if len(args) >= 2 else z3.IntVal(0)
            present = z3.Select(obj.presence, key)
            result = z3.If(present, z3.Select(obj.values, key), default)
            new_p = z3.Store(obj.presence, key, z3.BoolVal(False))
            new_tk = [k for k in obj.tracked_keys if not self._z3_eq(k, key)]
            updated = SymbolicDict(presence=new_p, values=obj.values,
                                   tracked_keys=new_tk)
            if isinstance(attr.value, ast.Name):
                env.set(attr.value.id, updated)
            return result

        if name == "update":
            if len(args) != 1 or not isinstance(args[0], SymbolicDict):
                raise SymbolicExecError(
                    ".update() requires a SymbolicDict argument"
                )
            other = args[0]
            new_p, new_v = obj.presence, obj.values
            tracked = list(obj.tracked_keys)
            for key in other.tracked_keys:
                p = z3.Select(other.presence, key)
                new_p = z3.If(
                    p, z3.Store(new_p, key, z3.BoolVal(True)), new_p,
                )
                new_v = z3.If(
                    p,
                    z3.Store(new_v, key, z3.Select(other.values, key)),
                    new_v,
                )
                if not any(self._z3_eq(key, tk) for tk in tracked):
                    tracked.append(key)
            updated = SymbolicDict(presence=new_p, values=new_v,
                                   tracked_keys=tracked)
            if isinstance(attr.value, ast.Name):
                env.set(attr.value.id, updated)
            return z3.IntVal(0)

        if name == "setdefault":
            if len(args) < 1:
                raise SymbolicExecError(
                    ".setdefault() requires at least 1 argument"
                )
            key = args[0]
            default = args[1] if len(args) >= 2 else z3.IntVal(0)
            present = z3.Select(obj.presence, key)
            result = z3.If(present, z3.Select(obj.values, key), default)
            new_p = z3.If(
                present, obj.presence,
                z3.Store(obj.presence, key, z3.BoolVal(True)),
            )
            new_v = z3.If(
                present, obj.values,
                z3.Store(obj.values, key, default),
            )
            tracked = list(obj.tracked_keys)
            if not any(self._z3_eq(key, tk) for tk in tracked):
                tracked.append(key)
            updated = SymbolicDict(presence=new_p, values=new_v,
                                   tracked_keys=tracked)
            if isinstance(attr.value, ast.Name):
                env.set(attr.value.id, updated)
            return result

        raise SymbolicExecError(
            f"Unsupported method on SymbolicDict: .{name}()"
        )

    # ── SymbolicSet ───────────────────────────────────────────────────

    def _set_method(
        self, attr: ast.Attribute, name: str,
        obj: SymbolicSet, args: list[Any], env: SymbolicEnv,
    ) -> Any:
        if name == "add":
            if len(args) != 1:
                raise SymbolicExecError(".add() requires exactly 1 argument")
            new_p = z3.Store(obj.presence, args[0], z3.BoolVal(True))
            updated = SymbolicSet(presence=new_p)
            if isinstance(attr.value, ast.Name):
                env.set(attr.value.id, updated)
            return z3.IntVal(0)

        if name in ("remove", "discard"):
            if len(args) != 1:
                raise SymbolicExecError(
                    f".{name}() requires exactly 1 argument"
                )
            new_p = z3.Store(obj.presence, args[0], z3.BoolVal(False))
            updated = SymbolicSet(presence=new_p)
            if isinstance(attr.value, ast.Name):
                env.set(attr.value.id, updated)
            return z3.IntVal(0)

        raise SymbolicExecError(
            f"Unsupported method on SymbolicSet: .{name}()"
        )

    # ── SymbolicList ──────────────────────────────────────────────────

    def _list_method(
        self, attr: ast.Attribute, name: str,
        obj: SymbolicList, args: list[Any], env: SymbolicEnv,
    ) -> Any:
        if name == "append":
            if len(args) != 1:
                raise SymbolicExecError(
                    ".append() requires exactly 1 argument"
                )
            val = args[0]
            new_arr = z3.Store(obj.array, obj.length, val)
            can_grow = obj.length < z3.IntVal(MAX_BMC_LENGTH)
            new_len = z3.If(can_grow, obj.length + 1, obj.length)
            new_arr_g = z3.If(can_grow, new_arr, obj.array)
            updated = SymbolicList(array=new_arr_g, length=new_len)
            if isinstance(attr.value, ast.Name):
                env.set(attr.value.id, updated)
            return z3.IntVal(0)

        if name == "count":
            if len(args) != 1:
                raise SymbolicExecError(
                    ".count() requires exactly 1 argument"
                )
            target = args[0]
            result = z3.IntVal(0)
            for k in range(MAX_BMC_LENGTH):
                ki = z3.IntVal(k)
                in_bounds = ki < obj.length
                matches = z3.Select(obj.array, ki) == target
                result = result + z3.If(
                    z3.And(in_bounds, matches), z3.IntVal(1), z3.IntVal(0),
                )
            return result

        if name == "pop":
            if len(args) == 0:
                new_len = z3.If(obj.length > 0, obj.length - 1, z3.IntVal(0))
                popped = z3.Select(obj.array, obj.length - 1)
                updated = SymbolicList(array=obj.array, length=new_len)
                if isinstance(attr.value, ast.Name):
                    env.set(attr.value.id, updated)
                return popped
            raise SymbolicExecError(".pop(index) not supported, only .pop()")

        if name == "extend":
            if len(args) != 1 or not isinstance(args[0], SymbolicList):
                raise SymbolicExecError(
                    ".extend() requires a SymbolicList argument"
                )
            other = args[0]
            new_arr, new_len = obj.array, obj.length
            for k in range(MAX_BMC_LENGTH):
                si = z3.IntVal(k)
                in_bounds = si < other.length
                can_grow = new_len < z3.IntVal(MAX_BMC_LENGTH)
                should_copy = z3.And(in_bounds, can_grow)
                val = z3.Select(other.array, si)
                new_arr = z3.If(
                    should_copy, z3.Store(new_arr, new_len, val), new_arr,
                )
                new_len = z3.If(should_copy, new_len + 1, new_len)
            updated = SymbolicList(array=new_arr, length=new_len)
            if isinstance(attr.value, ast.Name):
                env.set(attr.value.id, updated)
            return z3.IntVal(0)

        raise SymbolicExecError(
            f"Unsupported method on SymbolicList: .{name}()"
        )
