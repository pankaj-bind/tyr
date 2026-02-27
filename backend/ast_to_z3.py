"""
Tyr Backend — ast_to_z3.py
Translates Python AST nodes into Z3 symbolic expressions.

Supports:
  - Integer arithmetic (+, -, *, //, %, **)
  - Boolean logic (and, or, not)
  - Comparisons (<, <=, >, >=, ==, !=)
  - If/else expressions and statements → Z3 If()
  - For-loops over range() → bounded unrolling
  - For-loops over SymbolicList → bounded BMC unrolling
  - While-loops → bounded unrolling
  - Variable assignments (SSA-style environment tracking)
  - Return statements
  - Function calls: len(), sum(), min(), max(), abs(), sorted()
  - SymbolicList: len(), subscript read/write, append(), iteration, .count()
  - List literals and subscript access → Z3 Arrays
  - Augmented assignments (+=, -=, *=, etc.)
  - Tuple unpacking (limited)
  - Bounded Model Checking (BMC) for dynamic lists
  - SymbolicDict: subscript read/write, key in dict, .get(), .items(), .keys(), .values()
  - SymbolicSet: add(), remove(), element in set
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any

import z3

logger = logging.getLogger("tyr.z3.ast")

# Maximum iterations for loop unrolling to keep solving tractable
MAX_LOOP_UNROLL: int = 30

# ---------------------------------------------------------------------------
# Bounded Model Checking — maximum list length
# ---------------------------------------------------------------------------
MAX_BMC_LENGTH: int = 5


# ---------------------------------------------------------------------------
# SymbolicList — bounded representation of a Python list in Z3
# ---------------------------------------------------------------------------
@dataclass
class SymbolicList:
    """
    Represents a Python list as a Z3 Array (values) + Z3 Int (length).

    - array: Z3 Array(IntSort → IntSort), maps index → element value
    - length: Z3 ArithRef, tracks the current dynamic length

    Bounded by MAX_BMC_LENGTH: length is always in [0, MAX_BMC_LENGTH].
    """
    array: z3.ArrayRef
    length: z3.ArithRef

    def copy(self) -> SymbolicList:
        """Shallow copy — Z3 objects are immutable, so this is safe."""
        return SymbolicList(array=self.array, length=self.length)


# ---------------------------------------------------------------------------
# SymbolicSet — bounded representation of a Python set in Z3
# ---------------------------------------------------------------------------
@dataclass
class SymbolicSet:
    """
    Represents a Python set as a Z3 Array mapping element (Int) → presence (Bool).

    Uses Z3 Array(IntSort → BoolSort).  `presence[x] == True` means x is in
    the set.  This is an *exact* model for integer sets — no BMC length bound
    needed because membership is checked via Select, not iteration.
    """
    presence: z3.ArrayRef

    def copy(self) -> SymbolicSet:
        return SymbolicSet(presence=self.presence)


# ---------------------------------------------------------------------------
# SymbolicDict — bounded representation of a Python dict in Z3
# ---------------------------------------------------------------------------
@dataclass
class SymbolicDict:
    """
    Represents a Python dict as two parallel Z3 Arrays:

    - presence: Array(IntSort → BoolSort) — tracks which keys exist
    - values:   Array(IntSort → IntSort)  — maps key → stored value

    `presence[k] == True` means key k is in the dict, and `values[k]`
    gives the associated value.

    Additionally, `tracked_keys` records which concrete/symbolic key
    expressions have ever been stored so that bounded iteration over
    `.items()` / `.keys()` / `.values()` is possible.
    """
    presence: z3.ArrayRef
    values: z3.ArrayRef
    tracked_keys: list[Any] = field(default_factory=list)

    def copy(self) -> SymbolicDict:
        return SymbolicDict(
            presence=self.presence,
            values=self.values,
            tracked_keys=list(self.tracked_keys),
        )


# ---------------------------------------------------------------------------
# Symbolic Environment — tracks variable bindings during execution
# ---------------------------------------------------------------------------

@dataclass
class SymbolicEnv:
    """Tracks symbolic variable bindings during AST interpretation."""
    bindings: dict[str, Any] = field(default_factory=dict)
    constraints: list[Any] = field(default_factory=list)
    return_value: Any = None
    has_returned: bool = False
    # Partial returns: list of (guard_condition, return_value) from branches
    # where only one side of an if-statement returned.
    partial_returns: list[tuple[Any, Any]] = field(default_factory=list)

    def copy(self) -> SymbolicEnv:
        return SymbolicEnv(
            bindings=dict(self.bindings),
            constraints=list(self.constraints),
            return_value=self.return_value,
            has_returned=self.has_returned,
            partial_returns=list(self.partial_returns),
        )

    def set(self, name: str, value: Any) -> None:
        self.bindings[name] = value

    def get(self, name: str) -> Any:
        if name in self.bindings:
            return self.bindings[name]
        raise SymbolicExecError(f"Undefined variable: '{name}'")


class SymbolicExecError(Exception):
    """Raised when AST→Z3 translation encounters unsupported constructs."""
    pass


# ---------------------------------------------------------------------------
# AST → Z3 Translator
# ---------------------------------------------------------------------------

class ASTToZ3Translator:
    """
    Walks a Python AST and produces Z3 symbolic expressions.

    Usage:
        translator = ASTToZ3Translator()
        env = translator.execute_function(func_ast, symbolic_params)
        # env.return_value is the Z3 expression for the function's output
    """

    def __init__(self) -> None:
        self._fresh_counter: int = 0

    def fresh_var(self, prefix: str = "tmp") -> z3.ArithRef:
        """Create a fresh Z3 integer variable for SSA-style assignments."""
        self._fresh_counter += 1
        return z3.Int(f"__{prefix}_{self._fresh_counter}")

    def fresh_bool(self, prefix: str = "btmp") -> z3.BoolRef:
        self._fresh_counter += 1
        return z3.Bool(f"__{prefix}_{self._fresh_counter}")

    # ------------------------------------------------------------------
    # Top-level: execute a function body with symbolic parameters
    # ------------------------------------------------------------------

    def execute_function(
        self,
        func_node: ast.FunctionDef,
        param_symbols: dict[str, Any],
    ) -> SymbolicEnv:
        """
        Symbolically execute *func_node* with *param_symbols* as inputs.
        Returns the final SymbolicEnv (including return_value).
        """
        env = SymbolicEnv(bindings=dict(param_symbols))
        self._exec_body(func_node.body, env)
        return env

    # ------------------------------------------------------------------
    # Statement dispatch
    # ------------------------------------------------------------------

    def _exec_body(self, stmts: list[ast.stmt], env: SymbolicEnv) -> None:
        for stmt in stmts:
            if env.has_returned:
                break
            self._exec_stmt(stmt, env)

    def _exec_stmt(self, node: ast.stmt, env: SymbolicEnv) -> None:
        if isinstance(node, ast.Return):
            self._exec_return(node, env)
        elif isinstance(node, ast.Assign):
            self._exec_assign(node, env)
        elif isinstance(node, ast.AugAssign):
            self._exec_augassign(node, env)
        elif isinstance(node, ast.If):
            self._exec_if(node, env)
        elif isinstance(node, ast.For):
            self._exec_for(node, env)
        elif isinstance(node, ast.While):
            self._exec_while(node, env)
        elif isinstance(node, ast.Expr):
            # standalone expression (e.g., function call with side effect)
            self._eval_expr(node.value, env)
        elif isinstance(node, ast.Pass):
            pass
        elif isinstance(node, ast.AnnAssign):
            # Annotated assignment: x: int = 5
            if node.value is not None and node.target is not None:
                val = self._eval_expr(node.value, env)
                if isinstance(node.target, ast.Name):
                    env.set(node.target.id, val)
        else:
            raise SymbolicExecError(
                f"Unsupported statement type: {type(node).__name__} "
                f"(line {getattr(node, 'lineno', '?')})"
            )

    # ------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------

    def _exec_return(self, node: ast.Return, env: SymbolicEnv) -> None:
        if node.value is not None:
            val = self._eval_expr(node.value, env)
        else:
            val = z3.IntVal(0)  # None → 0 for Z3

        # If there are pending partial returns from earlier if-branches,
        # merge them:  final = If(guard1, partial1, If(guard2, partial2, val))
        for guard, partial_val in reversed(env.partial_returns):
            try:
                val = z3.If(guard, partial_val, val)
            except (z3.Z3Exception, TypeError):
                pass  # type mismatch — skip
        env.partial_returns.clear()

        env.return_value = val
        env.has_returned = True

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def _exec_assign(self, node: ast.Assign, env: SymbolicEnv) -> None:
        value = self._eval_expr(node.value, env)

        for target in node.targets:
            if isinstance(target, ast.Name):
                env.set(target.id, value)
            elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                # Tuple unpacking: a, b = expr
                # Best effort — if value is a Python tuple/list of Z3 exprs
                if isinstance(value, (list, tuple)):
                    for i, elt in enumerate(target.elts):
                        if isinstance(elt, ast.Name):
                            if i < len(value):
                                env.set(elt.id, value[i])
                            else:
                                env.set(elt.id, z3.IntVal(0))
                else:
                    # Cannot unpack Z3 expression
                    for i, elt in enumerate(target.elts):
                        if isinstance(elt, ast.Name):
                            env.set(elt.id, self.fresh_var(elt.id))
            elif isinstance(target, ast.Subscript):
                # arr[i] = val → Z3 Store (supports SymbolicList, SymbolicDict, raw arrays)
                arr = self._eval_expr(target.value, env)
                idx = self._eval_expr(target.slice, env)
                if isinstance(arr, SymbolicDict):
                    # dict[key] = val → update presence + values
                    new_presence = z3.Store(arr.presence, idx, z3.BoolVal(True))
                    new_values = z3.Store(arr.values, idx, value)
                    tracked = list(arr.tracked_keys)
                    tracked.append(idx)
                    updated = SymbolicDict(presence=new_presence, values=new_values,
                                           tracked_keys=tracked)
                    if isinstance(target.value, ast.Name):
                        env.set(target.value.id, updated)
                elif isinstance(arr, SymbolicList):
                    new_arr = z3.Store(arr.array, idx, value)
                    updated = SymbolicList(array=new_arr, length=arr.length)
                    if isinstance(target.value, ast.Name):
                        env.set(target.value.id, updated)
                elif z3.is_array(arr):
                    new_arr = z3.Store(arr, idx, value)
                    if isinstance(target.value, ast.Name):
                        env.set(target.value.id, new_arr)
                else:
                    raise SymbolicExecError(
                        f"Subscript assignment on unsupported type: {type(arr).__name__}"
                    )
            else:
                raise SymbolicExecError(
                    f"Unsupported assignment target: {type(target).__name__}"
                )

    def _exec_augassign(self, node: ast.AugAssign, env: SymbolicEnv) -> None:
        """Handle +=, -=, *=, //=, %="""
        current = self._eval_expr(node.target, env)
        rhs = self._eval_expr(node.value, env)
        result = self._apply_binop(node.op, current, rhs)

        if isinstance(node.target, ast.Name):
            env.set(node.target.id, result)
        elif isinstance(node.target, ast.Subscript):
            arr = self._eval_expr(node.target.value, env)
            idx = self._eval_expr(node.target.slice, env)
            if isinstance(arr, SymbolicDict):
                # dict[key] += val → read, add, write back
                new_values = z3.Store(arr.values, idx, result)
                new_presence = z3.Store(arr.presence, idx, z3.BoolVal(True))
                tracked = list(arr.tracked_keys)
                if not any(self._z3_eq(idx, tk) for tk in tracked):
                    tracked.append(idx)
                updated = SymbolicDict(presence=new_presence, values=new_values,
                                       tracked_keys=tracked)
                if isinstance(node.target.value, ast.Name):
                    env.set(node.target.value.id, updated)
            elif isinstance(arr, SymbolicList):
                new_arr = z3.Store(arr.array, idx, result)
                updated = SymbolicList(array=new_arr, length=arr.length)
                if isinstance(node.target.value, ast.Name):
                    env.set(node.target.value.id, updated)
            elif z3.is_array(arr):
                new_arr = z3.Store(arr, idx, result)
                if isinstance(node.target.value, ast.Name):
                    env.set(node.target.value.id, new_arr)

    # ------------------------------------------------------------------
    # If / Else
    # ------------------------------------------------------------------

    def _exec_if(self, node: ast.If, env: SymbolicEnv) -> None:
        cond = self._eval_expr(node.test, env)

        # Ensure it's a Z3 boolean
        if isinstance(cond, bool):
            cond = z3.BoolVal(cond)
        elif isinstance(cond, int):
            cond = z3.BoolVal(bool(cond))

        # Execute both branches symbolically
        then_env = env.copy()
        self._exec_body(node.body, then_env)

        else_env = env.copy()
        if node.orelse:
            self._exec_body(node.orelse, else_env)

        # Merge variable bindings using Z3 If (with SymbolicList support)
        all_vars = set(then_env.bindings.keys()) | set(else_env.bindings.keys())
        for var in all_vars:
            then_val = then_env.bindings.get(var)
            else_val = else_env.bindings.get(var)
            if then_val is not None and else_val is not None:
                merged = self._if_merge(cond, then_val, else_val)
                if merged is not None:
                    env.set(var, merged)
                else:
                    env.set(var, then_val)
            elif then_val is not None:
                if var in env.bindings:
                    merged = self._if_merge(cond, then_val, env.bindings[var])
                    if merged is not None:
                        env.set(var, merged)
                    else:
                        env.set(var, then_val)
            elif else_val is not None:
                if var in env.bindings:
                    merged = self._if_merge(cond, env.bindings[var], else_val)
                    if merged is not None:
                        env.set(var, merged)

        # Merge return values
        if then_env.has_returned and else_env.has_returned:
            # Both branches return → function definitely returns here
            try:
                env.return_value = z3.If(cond, then_env.return_value, else_env.return_value)
            except (z3.Z3Exception, TypeError):
                env.return_value = then_env.return_value
            env.has_returned = True
        elif then_env.has_returned and not else_env.has_returned:
            # Only the if-branch returned (e.g., early return / guard).
            # Code after the if-block runs when cond is False.
            # Record this as a partial (conditional) return.
            env.partial_returns.append((cond, then_env.return_value))
            # Do NOT set has_returned — execution continues.
        elif else_env.has_returned and not then_env.has_returned:
            # Only the else-branch returned.
            # Code after the if-block runs when cond is True.
            env.partial_returns.append((z3.Not(cond), else_env.return_value))
            # Do NOT set has_returned.

    # ------------------------------------------------------------------
    # If-merge helper: merges two values under a Z3 condition.
    # Handles SymbolicList element-wise merging.
    # ------------------------------------------------------------------

    def _if_merge(self, cond: Any, true_val: Any, false_val: Any) -> Any | None:
        """
        Return z3.If(cond, true_val, false_val), handling SymbolicList,
        SymbolicDict, and SymbolicSet by merging their internal arrays.
        Returns None on failure.
        """
        if isinstance(true_val, SymbolicList) and isinstance(false_val, SymbolicList):
            merged_len = z3.If(cond, true_val.length, false_val.length)
            merged_arr = false_val.array
            for k in range(MAX_BMC_LENGTH):
                k_idx = z3.IntVal(k)
                merged_arr = z3.Store(
                    merged_arr, k_idx,
                    z3.If(cond,
                          z3.Select(true_val.array, k_idx),
                          z3.Select(false_val.array, k_idx))
                )
            return SymbolicList(array=merged_arr, length=merged_len)

        if isinstance(true_val, SymbolicDict) and isinstance(false_val, SymbolicDict):
            # Merge presence and values arrays using z3.If on each tracked key
            all_keys = list(set(id(k) for k in true_val.tracked_keys) and
                           true_val.tracked_keys)
            for k in false_val.tracked_keys:
                if not any(self._z3_eq(k, tk) for tk in all_keys):
                    all_keys.append(k)
            merged_presence = false_val.presence
            merged_values = false_val.values
            for key in all_keys:
                merged_presence = z3.Store(
                    merged_presence, key,
                    z3.If(cond,
                          z3.Select(true_val.presence, key),
                          z3.Select(false_val.presence, key))
                )
                merged_values = z3.Store(
                    merged_values, key,
                    z3.If(cond,
                          z3.Select(true_val.values, key),
                          z3.Select(false_val.values, key))
                )
            return SymbolicDict(presence=merged_presence, values=merged_values,
                                tracked_keys=all_keys)

        if isinstance(true_val, SymbolicSet) and isinstance(false_val, SymbolicSet):
            # SymbolicSet uses a presence array (Int → Bool).
            # For a point-free merge without tracked keys, use z3.Lambda
            # or accept a best-effort: the merged array selects per-key.
            # Since we don't iterate sets, a functional merge suffices:
            merged = z3.If(cond, true_val.presence, false_val.presence)
            return SymbolicSet(presence=merged)

        try:
            return z3.If(cond, true_val, false_val)
        except (z3.Z3Exception, TypeError):
            return None

    @staticmethod
    def _z3_eq(a: Any, b: Any) -> bool:
        """Best-effort structural equality check for Z3 expressions."""
        try:
            return a is b or str(a) == str(b)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # For loop — bounded unrolling over range()
    # Supports both concrete and symbolic bounds.
    # ------------------------------------------------------------------

    def _exec_for(self, node: ast.For, env: SymbolicEnv) -> None:
        # For range() and SymbolicList, target must be a simple Name.
        # For Python-list iteration (dict.keys/items/values), tuple targets
        # are allowed and handled in _exec_for_over_python_list.
        loop_var: str | None = None
        if isinstance(node.target, ast.Name):
            loop_var = node.target.id

        # Try concrete range first (fast path) — requires simple Name target
        iter_range = None
        if loop_var is not None:
            iter_range = self._try_extract_range(node.iter, env)

        if iter_range is not None:
            start, stop, step = iter_range
            iterations = 0
            i = start
            while (step > 0 and i < stop) or (step < 0 and i > stop):
                if iterations >= MAX_LOOP_UNROLL:
                    logger.warning("For-loop unrolling capped at %d iterations.", MAX_LOOP_UNROLL)
                    break
                env.set(loop_var, z3.IntVal(i))
                self._exec_body(node.body, env)
                if env.has_returned:
                    break
                i += step
                iterations += 1
            return

        # Symbolic range — unroll with Z3 If-guards
        if loop_var is not None:
            symbolic_range = self._try_extract_symbolic_range(node.iter, env)
            if symbolic_range is not None:
                sym_start, sym_stop, sym_step = symbolic_range
                self._exec_symbolic_for(loop_var, sym_start, sym_stop, sym_step, node.body, env)
                return

        # ── Detect dict.items() / dict.keys() / dict.values() on SymbolicDict ──
        if (isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Attribute)
                and node.iter.func.attr in ("items", "keys", "values")):
            dict_obj = self._eval_expr(node.iter.func.value, env)
            if isinstance(dict_obj, SymbolicDict):
                self._exec_for_over_dict_items(
                    node, dict_obj, node.iter.func.attr, env,
                )
                return

        # ── BMC: For-loop over a SymbolicList ──
        iter_val = self._eval_expr(node.iter, env)
        if isinstance(iter_val, SymbolicList):
            if loop_var is None:
                raise SymbolicExecError("For-loop over SymbolicList requires a simple variable target.")
            self._exec_for_over_symbolic_list(loop_var, iter_val, node.body, env)
            return

        # ── For-loop over tracked dict keys/values/items (Python list) ──
        if isinstance(iter_val, (list, tuple)):
            self._exec_for_over_python_list(node, iter_val, env)
            return

        raise SymbolicExecError(
            f"For-loop iteration must be range(), a SymbolicList, or a tracked collection. "
            f"Got: {type(iter_val).__name__}"
        )

    def _exec_symbolic_for(
        self,
        loop_var: str,
        start: Any,
        stop: Any,
        step: Any,
        body: list[ast.stmt],
        env: SymbolicEnv,
    ) -> None:
        """
        Unroll a for-loop with symbolic bounds up to MAX_LOOP_UNROLL iterations.
        Each iteration is guarded by Z3 If: the body effects only apply
        when the loop variable is still within range.
        """
        current_i = start

        for iteration in range(MAX_LOOP_UNROLL):
            # Guard: is current_i < stop (for step > 0)?
            # We assume step = 1 for most symbolic ranges
            try:
                guard = current_i < stop
            except (z3.Z3Exception, TypeError):
                break

            if isinstance(guard, bool):
                guard = z3.BoolVal(guard)

            # Execute body in a copy
            body_env = env.copy()
            body_env.set(loop_var, current_i)
            self._exec_body(body, body_env)

            if body_env.has_returned:
                # Conditional return: only if guard is true
                if env.return_value is not None:
                    try:
                        env.return_value = z3.If(guard, body_env.return_value, env.return_value)
                    except (z3.Z3Exception, TypeError):
                        env.return_value = body_env.return_value
                else:
                    env.return_value = body_env.return_value
                # Don't set has_returned — later iterations might also matter
                break

            # Merge modified variables: var = If(guard, body_val, old_val)
            self._merge_env_guarded(env, body_env, guard, skip_vars={loop_var})

            # Advance loop variable
            try:
                current_i = current_i + step
            except (z3.Z3Exception, TypeError):
                break

    # ------------------------------------------------------------------
    # BMC: For-loop over a SymbolicList
    # ------------------------------------------------------------------

    def _exec_for_over_symbolic_list(
        self,
        loop_var: str,
        lst: SymbolicList,
        body: list[ast.stmt],
        env: SymbolicEnv,
    ) -> None:
        """
        Bounded unrolling of `for <loop_var> in <SymbolicList>`.

        Unrolls exactly MAX_BMC_LENGTH iterations. Iteration i executes
        only when i < lst.length (guarded by Z3 If on every variable merge).

        The loop variable is set to Select(lst.array, i) on each iteration.
        """
        for i in range(MAX_BMC_LENGTH):
            idx = z3.IntVal(i)
            guard = idx < lst.length  # True iff this iteration is live

            body_env = env.copy()
            body_env.set(loop_var, z3.Select(lst.array, idx))
            self._exec_body(body, body_env)

            if body_env.has_returned:
                if env.return_value is not None:
                    try:
                        env.return_value = z3.If(guard, body_env.return_value, env.return_value)
                    except (z3.Z3Exception, TypeError):
                        env.return_value = body_env.return_value
                else:
                    env.return_value = body_env.return_value
                break

            self._merge_env_guarded(env, body_env, guard, skip_vars={loop_var})

    # ------------------------------------------------------------------
    # For-loop over a concrete Python list (e.g., dict.keys() result)
    # ------------------------------------------------------------------

    def _exec_for_over_python_list(
        self,
        node: ast.For,
        items: list | tuple,
        env: SymbolicEnv,
    ) -> None:
        """
        Execute `for target in items` where items is a concrete Python
        list of Z3 expressions (e.g., from dict.keys(), dict.items()).

        Handles tuple unpacking for `for k, v in dict.items()`.
        """
        for item in items:
            if isinstance(node.target, ast.Name):
                env.set(node.target.id, item)
            elif isinstance(node.target, ast.Tuple):
                # Tuple unpacking: for k, v in dict.items()
                if isinstance(item, (list, tuple)):
                    for j, elt in enumerate(node.target.elts):
                        if isinstance(elt, ast.Name):
                            if j < len(item):
                                env.set(elt.id, item[j])
                            else:
                                env.set(elt.id, z3.IntVal(0))
                else:
                    # Single value — assign to first target
                    if node.target.elts and isinstance(node.target.elts[0], ast.Name):
                        env.set(node.target.elts[0].id, item)
            else:
                raise SymbolicExecError(
                    f"Unsupported for-loop target type: {type(node.target).__name__}"
                )

            self._exec_body(node.body, env)
            if env.has_returned:
                break

    # ------------------------------------------------------------------
    # BMC: For-loop over a SymbolicDict (.items / .keys / .values)
    # ------------------------------------------------------------------

    def _exec_for_over_dict_items(
        self,
        node: ast.For,
        dct: SymbolicDict,
        method: str,
        env: SymbolicEnv,
    ) -> None:
        """
        Bounded, presence-guarded iteration over a SymbolicDict.

        Unlike `_exec_for_over_python_list`, every iteration is wrapped
        in a Z3 guard derived from `Select(dct.presence, key)` so that
        body effects are only applied when the key actually exists.
        This is critical for BMC correctness: tracked_keys may include
        keys that are only *conditionally* present.

        Parameters
        ----------
        node   : The `ast.For` node (used for target unpacking).
        dct    : The SymbolicDict being iterated.
        method : One of "items", "keys", "values".
        env    : The live symbolic environment.
        """
        for key in dct.tracked_keys:
            guard = z3.Select(dct.presence, key)

            body_env = env.copy()

            # ── Bind loop variable(s) ──
            if method == "items":
                val = z3.Select(dct.values, key)
                if isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2:
                    k_elt, v_elt = node.target.elts
                    if isinstance(k_elt, ast.Name):
                        body_env.set(k_elt.id, key)
                    if isinstance(v_elt, ast.Name):
                        body_env.set(v_elt.id, val)
                elif isinstance(node.target, ast.Name):
                    # `for item in d.items()` — bind as tuple
                    body_env.set(node.target.id, (key, val))
                else:
                    raise SymbolicExecError(
                        "dict.items() iteration requires `for k, v in …` "
                        "or `for item in …`"
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

            # ── Execute body ──
            self._exec_body(node.body, body_env)

            if body_env.has_returned:
                if env.return_value is not None:
                    try:
                        env.return_value = z3.If(
                            guard, body_env.return_value, env.return_value,
                        )
                    except (z3.Z3Exception, TypeError):
                        env.return_value = body_env.return_value
                else:
                    env.return_value = body_env.return_value
                break

            # ── Guarded merge — effects only apply when key is present ──
            skip: set[str] = set()
            if isinstance(node.target, ast.Tuple):
                skip = {
                    e.id for e in node.target.elts if isinstance(e, ast.Name)
                }
            elif isinstance(node.target, ast.Name):
                skip = {node.target.id}
            self._merge_env_guarded(env, body_env, guard, skip_vars=skip)

    # ------------------------------------------------------------------
    # Env merging helper (shared by symbolic for & BMC list for)
    # ------------------------------------------------------------------

    def _merge_env_guarded(
        self,
        env: SymbolicEnv,
        body_env: SymbolicEnv,
        guard: Any,
        skip_vars: set[str] | None = None,
    ) -> None:
        """
        Merge body_env back into env for all modified bindings,
        guarded by `guard`: var = If(guard, body_val, old_val).

        Handles SymbolicList merging via element-wise If on array and length.
        """
        skip = skip_vars or set()
        for var_name in body_env.bindings:
            if var_name in skip:
                continue
            body_val = body_env.bindings[var_name]
            old_val = env.bindings.get(var_name)

            if old_val is None:
                env.set(var_name, body_val)
                continue

            # Both are SymbolicList → merge array and length separately
            if isinstance(body_val, SymbolicList) and isinstance(old_val, SymbolicList):
                merged_len = z3.If(guard, body_val.length, old_val.length)
                # Merge arrays element-wise for indices [0, MAX_BMC_LENGTH)
                merged_arr = old_val.array
                for k in range(MAX_BMC_LENGTH):
                    k_idx = z3.IntVal(k)
                    merged_arr = z3.Store(
                        merged_arr, k_idx,
                        z3.If(guard,
                              z3.Select(body_val.array, k_idx),
                              z3.Select(old_val.array, k_idx))
                    )
                env.set(var_name, SymbolicList(array=merged_arr, length=merged_len))
                continue

            # Both are SymbolicDict → merge presence + values arrays
            if isinstance(body_val, SymbolicDict) and isinstance(old_val, SymbolicDict):
                merged = self._if_merge(guard, body_val, old_val)
                if merged is not None:
                    env.set(var_name, merged)
                    continue

            # Both are SymbolicSet → merge presence arrays
            if isinstance(body_val, SymbolicSet) and isinstance(old_val, SymbolicSet):
                merged = self._if_merge(guard, body_val, old_val)
                if merged is not None:
                    env.set(var_name, merged)
                    continue

            try:
                env.set(var_name, z3.If(guard, body_val, old_val))
            except (z3.Z3Exception, TypeError):
                env.set(var_name, body_val)

    def _try_extract_range(
        self, node: ast.expr, env: SymbolicEnv
    ) -> tuple[int, int, int] | None:
        """Extract (start, stop, step) from a range() call with concrete args."""
        if not isinstance(node, ast.Call):
            return None
        if not (isinstance(node.func, ast.Name) and node.func.id == "range"):
            return None

        args = [self._try_concrete(self._eval_expr(a, env)) for a in node.args]

        if any(a is None for a in args):
            return None

        if len(args) == 1:
            return (0, args[0], 1)
        elif len(args) == 2:
            return (args[0], args[1], 1)
        elif len(args) == 3:
            return (args[0], args[1], args[2])
        return None

    def _try_extract_symbolic_range(
        self, node: ast.expr, env: SymbolicEnv
    ) -> tuple[Any, Any, Any] | None:
        """Extract (start, stop, step) from range() allowing symbolic args."""
        if not isinstance(node, ast.Call):
            return None
        if not (isinstance(node.func, ast.Name) and node.func.id == "range"):
            return None

        args = [self._eval_expr(a, env) for a in node.args]

        if len(args) == 1:
            return (z3.IntVal(0), args[0], z3.IntVal(1))
        elif len(args) == 2:
            return (args[0], args[1], z3.IntVal(1))
        elif len(args) == 3:
            return (args[0], args[1], args[2])
        return None

    @staticmethod
    def _try_concrete(val: Any) -> int | None:
        """Try to extract a concrete Python int from a Z3 expression or int."""
        if isinstance(val, int):
            return val
        if isinstance(val, z3.IntNumRef):
            return val.as_long()
        if z3.is_int_value(val):
            return val.as_long()
        return None

    # ------------------------------------------------------------------
    # While loop — bounded unrolling
    # ------------------------------------------------------------------

    def _exec_while(self, node: ast.While, env: SymbolicEnv) -> None:
        for _ in range(MAX_LOOP_UNROLL):
            cond = self._eval_expr(node.test, env)

            # If we can concretely evaluate the condition, use it
            concrete = self._try_concrete_bool(cond)
            if concrete is False:
                break

            if concrete is True:
                self._exec_body(node.body, env)
                if env.has_returned:
                    break
                continue

            # Symbolic condition — unroll one more iteration with guard
            body_env = env.copy()
            self._exec_body(node.body, body_env)

            # Merge
            for var in body_env.bindings:
                if var in env.bindings:
                    try:
                        env.set(var, z3.If(cond, body_env.bindings[var], env.bindings[var]))
                    except (z3.Z3Exception, TypeError):
                        env.set(var, body_env.bindings[var])
                else:
                    env.set(var, body_env.bindings[var])

            if body_env.has_returned:
                env.return_value = body_env.return_value
                env.has_returned = True
                break

        if node.orelse and not env.has_returned:
            self._exec_body(node.orelse, env)

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
        return None

    # ------------------------------------------------------------------
    # Expression dispatch
    # ------------------------------------------------------------------

    def _eval_expr(self, node: ast.expr, env: SymbolicEnv) -> Any:
        if isinstance(node, ast.Constant):
            return self._eval_constant(node)
        elif isinstance(node, ast.Name):
            return env.get(node.id)
        elif isinstance(node, ast.BinOp):
            return self._eval_binop(node, env)
        elif isinstance(node, ast.UnaryOp):
            return self._eval_unaryop(node, env)
        elif isinstance(node, ast.BoolOp):
            return self._eval_boolop(node, env)
        elif isinstance(node, ast.Compare):
            return self._eval_compare(node, env)
        elif isinstance(node, ast.IfExp):
            return self._eval_ifexp(node, env)
        elif isinstance(node, ast.Call):
            return self._eval_call(node, env)
        elif isinstance(node, ast.Subscript):
            return self._eval_subscript(node, env)
        elif isinstance(node, ast.List) or isinstance(node, ast.Tuple):
            return self._eval_list_literal(node, env)
        elif isinstance(node, ast.Dict):
            return self._eval_dict_literal(node, env)
        elif isinstance(node, ast.Set):
            return self._eval_set_literal(node, env)
        elif isinstance(node, ast.Attribute):
            # Attribute access on SymbolicList — return a marker for method dispatch
            # (actual dispatch happens in _eval_call for method calls)
            raise SymbolicExecError(
                f"Attribute access not fully supported: "
                f"{ast.dump(node)}"
            )
        else:
            raise SymbolicExecError(
                f"Unsupported expression type: {type(node).__name__} "
                f"(line {getattr(node, 'lineno', '?')})"
            )

    # ------------------------------------------------------------------
    # List literal → SymbolicList
    # ------------------------------------------------------------------

    def _eval_list_literal(self, node: ast.List | ast.Tuple, env: SymbolicEnv) -> SymbolicList:
        """
        Convert a list literal like `[]`, `[1, 2, 3]` into a SymbolicList.

        This bridges the gap between Python list literals and our BMC
        representation. All subsequent operations (len, subscript, append,
        iteration) work uniformly on SymbolicList.
        """
        elements = [self._eval_expr(elt, env) for elt in node.elts]
        n = len(elements)

        if n > MAX_BMC_LENGTH:
            raise SymbolicExecError(
                f"List literal has {n} elements, exceeds MAX_BMC_LENGTH={MAX_BMC_LENGTH}"
            )

        # Build the array with concrete stores
        arr_name = f"__litarr_{self._fresh_counter}"
        self._fresh_counter += 1
        arr = z3.Array(arr_name, z3.IntSort(), z3.IntSort())
        for i, elem in enumerate(elements):
            arr = z3.Store(arr, z3.IntVal(i), elem)

        length = z3.IntVal(n)
        return SymbolicList(array=arr, length=length)

    # ------------------------------------------------------------------
    # Dict literal → SymbolicDict
    # ------------------------------------------------------------------

    def _eval_dict_literal(self, node: ast.Dict, env: SymbolicEnv) -> SymbolicDict:
        """
        Convert a dict literal like `{}`, `{1: 2, 3: 4}` into a SymbolicDict.

        Empty `{}` → presence all-False, values all-zero.
        Non-empty dict → Store each key-value pair.
        """
        tag = f"__dictlit_{self._fresh_counter}"
        self._fresh_counter += 1

        presence = z3.K(z3.IntSort(), z3.BoolVal(False))
        values = z3.K(z3.IntSort(), z3.IntVal(0))
        tracked: list[Any] = []

        for key_node, val_node in zip(node.keys, node.values):
            if key_node is None:
                # ** unpacking — not supported
                raise SymbolicExecError("Dict unpacking (**) not supported")
            key = self._eval_expr(key_node, env)
            val = self._eval_expr(val_node, env)
            presence = z3.Store(presence, key, z3.BoolVal(True))
            values = z3.Store(values, key, val)
            tracked.append(key)

        return SymbolicDict(presence=presence, values=values, tracked_keys=tracked)

    # ------------------------------------------------------------------
    # Set literal → SymbolicSet
    # ------------------------------------------------------------------

    def _eval_set_literal(self, node: ast.Set, env: SymbolicEnv) -> SymbolicSet:
        """
        Convert a set literal like `{1, 2, 3}` into a SymbolicSet.
        """
        presence = z3.K(z3.IntSort(), z3.BoolVal(False))
        for elt_node in node.elts:
            elt = self._eval_expr(elt_node, env)
            presence = z3.Store(presence, elt, z3.BoolVal(True))
        return SymbolicSet(presence=presence)

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    @staticmethod
    def _eval_constant(node: ast.Constant) -> Any:
        v = node.value
        if isinstance(v, bool):
            return z3.BoolVal(v)
        if isinstance(v, int):
            return z3.IntVal(v)
        if isinstance(v, float):
            # Approximate floats as Z3 reals
            return z3.RealVal(v)
        if v is None:
            return z3.IntVal(0)
        # Strings and other types — not easily representable
        raise SymbolicExecError(f"Unsupported constant type: {type(v).__name__}")

    # ------------------------------------------------------------------
    # Binary operations
    # ------------------------------------------------------------------

    def _eval_binop(self, node: ast.BinOp, env: SymbolicEnv) -> Any:
        left = self._eval_expr(node.left, env)
        right = self._eval_expr(node.right, env)
        return self._apply_binop(node.op, left, right)

    @staticmethod
    def _apply_binop(op: ast.operator, left: Any, right: Any) -> Any:
        if isinstance(op, ast.Add):
            return left + right
        elif isinstance(op, ast.Sub):
            return left - right
        elif isinstance(op, ast.Mult):
            return left * right
        elif isinstance(op, ast.FloorDiv):
            return left / right  # Z3 integer division
        elif isinstance(op, ast.Mod):
            return left % right
        elif isinstance(op, ast.Pow):
            # Z3 doesn't natively support symbolic exponentiation well;
            # for concrete exponents, unroll
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
            raise SymbolicExecError("Symbolic exponentiation not supported for non-concrete exponents.")
        elif isinstance(op, ast.BitAnd):
            return left & right
        elif isinstance(op, ast.BitOr):
            return left | right
        elif isinstance(op, ast.BitXor):
            return left ^ right
        elif isinstance(op, ast.LShift):
            return left << right
        elif isinstance(op, ast.RShift):
            return left >> right
        elif isinstance(op, ast.Div):
            # True division → use Z3 Real division
            return z3.ToReal(left) / z3.ToReal(right)
        else:
            raise SymbolicExecError(f"Unsupported binary op: {type(op).__name__}")

    # ------------------------------------------------------------------
    # Unary operations
    # ------------------------------------------------------------------

    def _eval_unaryop(self, node: ast.UnaryOp, env: SymbolicEnv) -> Any:
        operand = self._eval_expr(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return operand
        elif isinstance(node.op, ast.Not):
            return z3.Not(operand)
        elif isinstance(node.op, ast.Invert):
            return ~operand
        raise SymbolicExecError(f"Unsupported unary op: {type(node.op).__name__}")

    # ------------------------------------------------------------------
    # Boolean operations (and, or)
    # ------------------------------------------------------------------

    def _eval_boolop(self, node: ast.BoolOp, env: SymbolicEnv) -> Any:
        values = [self._eval_expr(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            return z3.And(*values)
        elif isinstance(node.op, ast.Or):
            return z3.Or(*values)
        raise SymbolicExecError(f"Unsupported bool op: {type(node.op).__name__}")

    # ------------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------------

    def _eval_compare(self, node: ast.Compare, env: SymbolicEnv) -> Any:
        left = self._eval_expr(node.left, env)
        parts = []
        for op, comparator in zip(node.ops, node.comparators):
            right = self._eval_expr(comparator, env)
            parts.append(self._apply_cmpop(op, left, right))
            left = right
        if len(parts) == 1:
            return parts[0]
        return z3.And(*parts)

    def _apply_cmpop(self, op: ast.cmpop, left: Any, right: Any) -> Any:
        # ── Handle `x in list` and `x not in list` ──
        if isinstance(op, ast.In):
            return self._symbolic_in(left, right)
        if isinstance(op, ast.NotIn):
            return z3.Not(self._symbolic_in(left, right))

        if isinstance(op, ast.Lt):
            return left < right
        elif isinstance(op, ast.LtE):
            return left <= right
        elif isinstance(op, ast.Gt):
            return left > right
        elif isinstance(op, ast.GtE):
            return left >= right
        elif isinstance(op, ast.Eq):
            return left == right
        elif isinstance(op, ast.NotEq):
            return left != right
        raise SymbolicExecError(f"Unsupported comparison: {type(op).__name__}")

    def _symbolic_in(self, needle: Any, haystack: Any) -> Any:
        """
        Symbolically evaluate `needle in haystack`.
        Supports SymbolicList, Python list of Z3 values, and raw Z3 arrays.
        """
        if isinstance(haystack, SymbolicDict):
            # key in dict → Select(presence, key)
            return z3.Select(haystack.presence, needle)

        if isinstance(haystack, SymbolicSet):
            # element in set → Select(presence, element)
            return z3.Select(haystack.presence, needle)

        if isinstance(haystack, SymbolicList):
            # OR of (needle == arr[i]) for i in [0, MAX_BMC_LENGTH) guarded by i < length
            clauses = []
            for k in range(MAX_BMC_LENGTH):
                k_idx = z3.IntVal(k)
                in_bounds = k_idx < haystack.length
                matches = z3.Select(haystack.array, k_idx) == needle
                clauses.append(z3.And(in_bounds, matches))
            return z3.Or(*clauses) if clauses else z3.BoolVal(False)

        if isinstance(haystack, list):
            if len(haystack) == 0:
                return z3.BoolVal(False)
            clauses = [needle == elem for elem in haystack]
            return z3.Or(*clauses)

        raise SymbolicExecError(
            f"'in' operator not supported for type: {type(haystack).__name__}"
        )

    # ------------------------------------------------------------------
    # Ternary / If-expression
    # ------------------------------------------------------------------

    def _eval_ifexp(self, node: ast.IfExp, env: SymbolicEnv) -> Any:
        cond = self._eval_expr(node.test, env)
        then_val = self._eval_expr(node.body, env)
        else_val = self._eval_expr(node.orelse, env)
        return z3.If(cond, then_val, else_val)

    # ------------------------------------------------------------------
    # Function calls — built-in support
    # ------------------------------------------------------------------

    def _eval_call(self, node: ast.Call, env: SymbolicEnv) -> Any:
        # ── Method calls on objects (e.g., list.append, list.count) ──
        if isinstance(node.func, ast.Attribute):
            return self._eval_method_call(node, env)

        if isinstance(node.func, ast.Name):
            fname = node.func.id
            args = [self._eval_expr(a, env) for a in node.args]

            if fname == "abs":
                if len(args) != 1:
                    raise SymbolicExecError("abs() requires exactly 1 argument")
                x = args[0]
                return z3.If(x >= 0, x, -x)

            if fname == "min":
                if len(args) == 1 and isinstance(args[0], (list, SymbolicList)):
                    return self._symbolic_min_of(args[0])
                if len(args) >= 2:
                    return self._symbolic_min(args)
                raise SymbolicExecError("min() requires at least 2 arguments or a list")

            if fname == "max":
                if len(args) == 1 and isinstance(args[0], (list, SymbolicList)):
                    return self._symbolic_max_of(args[0])
                if len(args) >= 2:
                    return self._symbolic_max(args)
                raise SymbolicExecError("max() requires at least 2 arguments or a list")

            if fname == "len":
                if len(args) != 1:
                    raise SymbolicExecError("len() requires exactly 1 argument")
                arg = args[0]
                if isinstance(arg, SymbolicList):
                    return arg.length
                if isinstance(arg, list):
                    return z3.IntVal(len(arg))
                raise SymbolicExecError(f"len() on unsupported type: {type(arg).__name__}")

            if fname == "sum":
                if len(args) != 1:
                    raise SymbolicExecError("sum() requires exactly 1 argument")
                arg = args[0]
                if isinstance(arg, SymbolicList):
                    # Bounded sum over SymbolicList
                    result = z3.IntVal(0)
                    for k in range(MAX_BMC_LENGTH):
                        elem = z3.Select(arg.array, z3.IntVal(k))
                        in_bounds = z3.IntVal(k) < arg.length
                        result = z3.If(in_bounds, result + elem, result)
                    return result
                if isinstance(arg, list):
                    if len(arg) == 0:
                        return z3.IntVal(0)
                    result = arg[0]
                    for v in arg[1:]:
                        result = result + v
                    return result
                raise SymbolicExecError("sum() on non-list not supported")

            if fname == "range":
                # range() used in expression context (not for-loop)
                concrete_args = [self._try_concrete(a) for a in args]
                if any(a is None for a in concrete_args):
                    raise SymbolicExecError("range() with symbolic args not supported in expression context")
                if len(concrete_args) == 1:
                    return [z3.IntVal(i) for i in range(concrete_args[0])]
                elif len(concrete_args) == 2:
                    return [z3.IntVal(i) for i in range(concrete_args[0], concrete_args[1])]
                elif len(concrete_args) == 3:
                    return [z3.IntVal(i) for i in range(concrete_args[0], concrete_args[1], concrete_args[2])]

            if fname == "sorted":
                # Cannot truly sort symbolically — return input unchanged
                if len(args) == 1:
                    return args[0]

            if fname == "int":
                if len(args) == 1:
                    return args[0]  # identity for Z3 ints

            if fname == "bool":
                if len(args) == 1:
                    x = args[0]
                    return z3.If(x != z3.IntVal(0), z3.BoolVal(True), z3.BoolVal(False))

            if fname == "set":
                if len(args) == 0:
                    # Empty set → SymbolicSet with all-False presence
                    return SymbolicSet(presence=z3.K(z3.IntSort(), z3.BoolVal(False)))
                if len(args) == 1 and isinstance(args[0], SymbolicList):
                    # set(list) → populate presence from list elements
                    presence = z3.K(z3.IntSort(), z3.BoolVal(False))
                    for k in range(MAX_BMC_LENGTH):
                        k_idx = z3.IntVal(k)
                        in_bounds = k_idx < args[0].length
                        elem = z3.Select(args[0].array, k_idx)
                        presence = z3.If(in_bounds,
                                         z3.Store(presence, elem, z3.BoolVal(True)),
                                         presence)
                    return SymbolicSet(presence=presence)
                if len(args) == 1 and isinstance(args[0], SymbolicSet):
                    return args[0].copy()

            if fname == "dict":
                if len(args) == 0:
                    # Empty dict → SymbolicDict
                    return SymbolicDict(
                        presence=z3.K(z3.IntSort(), z3.BoolVal(False)),
                        values=z3.K(z3.IntSort(), z3.IntVal(0)),
                        tracked_keys=[],
                    )

            if fname == "list":
                if len(args) == 0:
                    arr_name = f"__emptylist_{self._fresh_counter}"
                    self._fresh_counter += 1
                    arr = z3.Array(arr_name, z3.IntSort(), z3.IntSort())
                    return SymbolicList(array=arr, length=z3.IntVal(0))
                if len(args) == 1 and isinstance(args[0], SymbolicList):
                    return args[0].copy()

            raise SymbolicExecError(f"Unsupported function call: {fname}()")

        raise SymbolicExecError(f"Unsupported callable: {ast.dump(node.func)}")

    # ------------------------------------------------------------------
    # Method calls — .append(), .count(), etc.
    # ------------------------------------------------------------------

    def _eval_method_call(self, node: ast.Call, env: SymbolicEnv) -> Any:
        """
        Handle method calls like obj.append(val), obj.count(val),
        nums.count(x), etc.
        """
        attr = node.func  # ast.Attribute
        assert isinstance(attr, ast.Attribute)
        method_name = attr.attr
        obj = self._eval_expr(attr.value, env)
        args = [self._eval_expr(a, env) for a in node.args]

        # ── SymbolicDict methods ──
        if isinstance(obj, SymbolicDict):
            if method_name == "get":
                # dict.get(key) or dict.get(key, default)
                if len(args) < 1:
                    raise SymbolicExecError(".get() requires at least 1 argument")
                key = args[0]
                default = args[1] if len(args) >= 2 else z3.IntVal(0)
                present = z3.Select(obj.presence, key)
                return z3.If(present, z3.Select(obj.values, key), default)

            if method_name == "keys":
                # Return tracked_keys as a Python list (for iteration)
                return obj.tracked_keys

            if method_name == "values":
                # Return tracked values as expressions guarded by presence
                return [z3.Select(obj.values, k) for k in obj.tracked_keys]

            if method_name == "items":
                # Return list of (key, value) tuples for tracked keys
                return [(k, z3.Select(obj.values, k)) for k in obj.tracked_keys]

            if method_name == "pop":
                # dict.pop(key) or dict.pop(key, default)
                if len(args) < 1:
                    raise SymbolicExecError(".pop() requires at least 1 argument")
                key = args[0]
                default = args[1] if len(args) >= 2 else z3.IntVal(0)
                present = z3.Select(obj.presence, key)
                result = z3.If(present, z3.Select(obj.values, key), default)
                # Remove the key
                new_presence = z3.Store(obj.presence, key, z3.BoolVal(False))
                updated = SymbolicDict(presence=new_presence, values=obj.values,
                                       tracked_keys=obj.tracked_keys)
                if isinstance(attr.value, ast.Name):
                    env.set(attr.value.id, updated)
                return result

            if method_name == "update":
                # dict.update(other_dict)
                if len(args) != 1 or not isinstance(args[0], SymbolicDict):
                    raise SymbolicExecError(".update() requires a SymbolicDict argument")
                other = args[0]
                new_presence = obj.presence
                new_values = obj.values
                tracked = list(obj.tracked_keys)
                for key in other.tracked_keys:
                    p = z3.Select(other.presence, key)
                    new_presence = z3.If(p,
                                         z3.Store(new_presence, key, z3.BoolVal(True)),
                                         new_presence)
                    new_values = z3.If(p,
                                       z3.Store(new_values, key, z3.Select(other.values, key)),
                                       new_values)
                    if not any(self._z3_eq(key, tk) for tk in tracked):
                        tracked.append(key)
                updated = SymbolicDict(presence=new_presence, values=new_values,
                                       tracked_keys=tracked)
                if isinstance(attr.value, ast.Name):
                    env.set(attr.value.id, updated)
                return z3.IntVal(0)

            if method_name == "setdefault":
                # dict.setdefault(key, default)
                if len(args) < 1:
                    raise SymbolicExecError(".setdefault() requires at least 1 argument")
                key = args[0]
                default = args[1] if len(args) >= 2 else z3.IntVal(0)
                present = z3.Select(obj.presence, key)
                result = z3.If(present, z3.Select(obj.values, key), default)
                # Only store if not present
                new_presence = z3.If(present, obj.presence,
                                     z3.Store(obj.presence, key, z3.BoolVal(True)))
                new_values = z3.If(present, obj.values,
                                   z3.Store(obj.values, key, default))
                tracked = list(obj.tracked_keys)
                if not any(self._z3_eq(key, tk) for tk in tracked):
                    tracked.append(key)
                updated = SymbolicDict(presence=new_presence, values=new_values,
                                       tracked_keys=tracked)
                if isinstance(attr.value, ast.Name):
                    env.set(attr.value.id, updated)
                return result

            raise SymbolicExecError(
                f"Unsupported method on SymbolicDict: .{method_name}()"
            )

        # ── SymbolicSet methods ──
        if isinstance(obj, SymbolicSet):
            if method_name == "add":
                if len(args) != 1:
                    raise SymbolicExecError(".add() requires exactly 1 argument")
                val = args[0]
                new_presence = z3.Store(obj.presence, val, z3.BoolVal(True))
                updated = SymbolicSet(presence=new_presence)
                if isinstance(attr.value, ast.Name):
                    env.set(attr.value.id, updated)
                return z3.IntVal(0)

            if method_name == "remove" or method_name == "discard":
                if len(args) != 1:
                    raise SymbolicExecError(f".{method_name}() requires exactly 1 argument")
                val = args[0]
                new_presence = z3.Store(obj.presence, val, z3.BoolVal(False))
                updated = SymbolicSet(presence=new_presence)
                if isinstance(attr.value, ast.Name):
                    env.set(attr.value.id, updated)
                return z3.IntVal(0)

            raise SymbolicExecError(
                f"Unsupported method on SymbolicSet: .{method_name}()"
            )

        # ── SymbolicList methods ──
        if isinstance(obj, SymbolicList):
            if method_name == "append":
                if len(args) != 1:
                    raise SymbolicExecError(".append() requires exactly 1 argument")
                val = args[0]
                # Guard: only append if length < MAX_BMC_LENGTH
                new_arr = z3.Store(obj.array, obj.length, val)
                can_grow = obj.length < z3.IntVal(MAX_BMC_LENGTH)
                new_len = z3.If(can_grow, obj.length + 1, obj.length)
                new_arr_guarded = z3.If(can_grow, new_arr, obj.array)
                updated = SymbolicList(array=new_arr_guarded, length=new_len)
                # Update the binding in env
                if isinstance(attr.value, ast.Name):
                    env.set(attr.value.id, updated)
                return z3.IntVal(0)  # append returns None; use 0 placeholder

            if method_name == "count":
                if len(args) != 1:
                    raise SymbolicExecError(".count() requires exactly 1 argument")
                target_val = args[0]
                # Bounded count: sum of (1 if arr[i]==val else 0) for i in [0,length)
                count_expr = z3.IntVal(0)
                for k in range(MAX_BMC_LENGTH):
                    k_idx = z3.IntVal(k)
                    in_bounds = k_idx < obj.length
                    matches = z3.Select(obj.array, k_idx) == target_val
                    count_expr = count_expr + z3.If(
                        z3.And(in_bounds, matches), z3.IntVal(1), z3.IntVal(0)
                    )
                return count_expr

            if method_name == "pop":
                # pop() removes and returns last element
                if len(args) == 0:
                    # Pop from end
                    new_len = z3.If(obj.length > 0, obj.length - 1, z3.IntVal(0))
                    popped_val = z3.Select(obj.array, obj.length - 1)
                    updated = SymbolicList(array=obj.array, length=new_len)
                    if isinstance(attr.value, ast.Name):
                        env.set(attr.value.id, updated)
                    return popped_val
                raise SymbolicExecError(".pop(index) not supported, only .pop()")

            if method_name == "extend":
                if len(args) != 1 or not isinstance(args[0], SymbolicList):
                    raise SymbolicExecError(".extend() requires a SymbolicList argument")
                other = args[0]
                # Copy elements from other into obj
                new_arr = obj.array
                new_len = obj.length
                for k in range(MAX_BMC_LENGTH):
                    src_idx = z3.IntVal(k)
                    in_bounds = src_idx < other.length
                    can_grow = new_len < z3.IntVal(MAX_BMC_LENGTH)
                    should_copy = z3.And(in_bounds, can_grow)
                    val = z3.Select(other.array, src_idx)
                    new_arr = z3.If(should_copy, z3.Store(new_arr, new_len, val), new_arr)
                    new_len = z3.If(should_copy, new_len + 1, new_len)
                updated = SymbolicList(array=new_arr, length=new_len)
                if isinstance(attr.value, ast.Name):
                    env.set(attr.value.id, updated)
                return z3.IntVal(0)

            raise SymbolicExecError(
                f"Unsupported method on SymbolicList: .{method_name}()"
            )

        raise SymbolicExecError(
            f"Method call .{method_name}() on unsupported type: {type(obj).__name__}"
        )

    @staticmethod
    def _symbolic_min(values: list) -> Any:
        result = values[0]
        for v in values[1:]:
            result = z3.If(v < result, v, result)
        return result

    def _symbolic_min_of(self, arg: Any) -> Any:
        """min() of a list or SymbolicList."""
        if isinstance(arg, list):
            return self._symbolic_min(arg)
        if isinstance(arg, SymbolicList):
            # Bounded min: reduce over [0, MAX_BMC_LENGTH)
            result = z3.Select(arg.array, z3.IntVal(0))
            for k in range(1, MAX_BMC_LENGTH):
                k_idx = z3.IntVal(k)
                in_bounds = k_idx < arg.length
                elem = z3.Select(arg.array, k_idx)
                result = z3.If(z3.And(in_bounds, elem < result), elem, result)
            return result
        raise SymbolicExecError(f"min() on unsupported type: {type(arg).__name__}")

    @staticmethod
    def _symbolic_max(values: list) -> Any:
        result = values[0]
        for v in values[1:]:
            result = z3.If(v > result, v, result)
        return result

    def _symbolic_max_of(self, arg: Any) -> Any:
        """max() of a list or SymbolicList."""
        if isinstance(arg, list):
            return self._symbolic_max(arg)
        if isinstance(arg, SymbolicList):
            result = z3.Select(arg.array, z3.IntVal(0))
            for k in range(1, MAX_BMC_LENGTH):
                k_idx = z3.IntVal(k)
                in_bounds = k_idx < arg.length
                elem = z3.Select(arg.array, k_idx)
                result = z3.If(z3.And(in_bounds, elem > result), elem, result)
            return result
        raise SymbolicExecError(f"max() on unsupported type: {type(arg).__name__}")

    # ------------------------------------------------------------------
    # Subscript (indexing) — supports SymbolicList, raw lists, raw arrays
    # ------------------------------------------------------------------

    def _eval_subscript(self, node: ast.Subscript, env: SymbolicEnv) -> Any:
        obj = self._eval_expr(node.value, env)
        idx = self._eval_expr(node.slice, env)

        if isinstance(obj, SymbolicDict):
            return z3.Select(obj.values, idx)

        if isinstance(obj, SymbolicList):
            return z3.Select(obj.array, idx)

        if isinstance(obj, list):
            # Concrete index into Python list of Z3 values
            concrete_idx = self._try_concrete(idx)
            if concrete_idx is not None and 0 <= concrete_idx < len(obj):
                return obj[concrete_idx]
            # Symbolic index — build If-chain
            if len(obj) > 0:
                result = obj[-1]
                for i in range(len(obj) - 1):
                    result = z3.If(idx == z3.IntVal(i), obj[i], result)
                return result
            raise SymbolicExecError("Index into empty list")

        if z3.is_array(obj):
            return z3.Select(obj, idx)

        raise SymbolicExecError(f"Subscript on unsupported type: {type(obj).__name__}")
