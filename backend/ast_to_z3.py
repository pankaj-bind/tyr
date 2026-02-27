"""
Tyr Backend — ast_to_z3.py
Translates Python AST nodes into Z3 symbolic expressions.

Supports:
  - Integer arithmetic (+, -, *, //, %, **)
  - Boolean logic (and, or, not)
  - Comparisons (<, <=, >, >=, ==, !=)
  - If/else expressions and statements → Z3 If()
  - For-loops over range() → bounded unrolling
  - While-loops → bounded unrolling
  - Variable assignments (SSA-style environment tracking)
  - Return statements
  - Function calls: len(), sum(), min(), max(), abs(), sorted()
  - List literals and subscript access → Z3 Arrays
  - Augmented assignments (+=, -=, *=, etc.)
  - Tuple unpacking (limited)
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
                # arr[i] = val → Z3 Store
                arr = self._eval_expr(target.value, env)
                idx = self._eval_expr(target.slice, env)
                if z3.is_array(arr):
                    new_arr = z3.Store(arr, idx, value)
                    if isinstance(target.value, ast.Name):
                        env.set(target.value.id, new_arr)
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
            if z3.is_array(arr):
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

        # Merge variable bindings using Z3 If
        all_vars = set(then_env.bindings.keys()) | set(else_env.bindings.keys())
        for var in all_vars:
            then_val = then_env.bindings.get(var)
            else_val = else_env.bindings.get(var)
            if then_val is not None and else_val is not None:
                try:
                    merged = z3.If(cond, then_val, else_val)
                    env.set(var, merged)
                except (z3.Z3Exception, TypeError):
                    # Different types; take the then-branch value
                    env.set(var, then_val)
            elif then_val is not None:
                if var in env.bindings:
                    try:
                        env.set(var, z3.If(cond, then_val, env.bindings[var]))
                    except (z3.Z3Exception, TypeError):
                        env.set(var, then_val)
            elif else_val is not None:
                if var in env.bindings:
                    try:
                        env.set(var, z3.If(cond, env.bindings[var], else_val))
                    except (z3.Z3Exception, TypeError):
                        pass

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
    # For loop — bounded unrolling over range()
    # Supports both concrete and symbolic bounds.
    # ------------------------------------------------------------------

    def _exec_for(self, node: ast.For, env: SymbolicEnv) -> None:
        if not isinstance(node.target, ast.Name):
            raise SymbolicExecError("For-loop target must be a simple variable.")

        loop_var = node.target.id

        # Try concrete range first (fast path)
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
        symbolic_range = self._try_extract_symbolic_range(node.iter, env)
        if symbolic_range is not None:
            sym_start, sym_stop, sym_step = symbolic_range
            self._exec_symbolic_for(loop_var, sym_start, sym_stop, sym_step, node.body, env)
            return

        raise SymbolicExecError(
            "For-loop iteration must be a range() call for bounded unrolling."
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
            for var_name in body_env.bindings:
                if var_name == loop_var:
                    continue
                body_val = body_env.bindings[var_name]
                old_val = env.bindings.get(var_name)
                if old_val is not None:
                    try:
                        env.set(var_name, z3.If(guard, body_val, old_val))
                    except (z3.Z3Exception, TypeError):
                        env.set(var_name, body_val)
                else:
                    env.set(var_name, body_val)

            # Advance loop variable
            try:
                current_i = current_i + step
            except (z3.Z3Exception, TypeError):
                break

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
            return [self._eval_expr(elt, env) for elt in node.elts]
        elif isinstance(node, ast.Attribute):
            # Limited support: e.g., x.append — mostly skip
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

    @staticmethod
    def _apply_cmpop(op: ast.cmpop, left: Any, right: Any) -> Any:
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
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            args = [self._eval_expr(a, env) for a in node.args]

            if fname == "abs":
                if len(args) != 1:
                    raise SymbolicExecError("abs() requires exactly 1 argument")
                x = args[0]
                return z3.If(x >= 0, x, -x)

            if fname == "min":
                if len(args) == 1 and isinstance(args[0], list):
                    return self._symbolic_min(args[0])
                if len(args) >= 2:
                    return self._symbolic_min(args)
                raise SymbolicExecError("min() requires at least 2 arguments or a list")

            if fname == "max":
                if len(args) == 1 and isinstance(args[0], list):
                    return self._symbolic_max(args[0])
                if len(args) >= 2:
                    return self._symbolic_max(args)
                raise SymbolicExecError("max() requires at least 2 arguments or a list")

            if fname == "len":
                if len(args) != 1:
                    raise SymbolicExecError("len() requires exactly 1 argument")
                arg = args[0]
                if isinstance(arg, list):
                    return z3.IntVal(len(arg))
                # If it's a Z3 array, we may have stored length
                raise SymbolicExecError("len() on non-list/symbolic arrays not supported")

            if fname == "sum":
                if len(args) != 1:
                    raise SymbolicExecError("sum() requires exactly 1 argument")
                arg = args[0]
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
                # (sorted is semantics-altering only by ordering; if both
                #  functions call sorted on the same input, equivalence holds)
                if len(args) == 1:
                    return args[0]

            if fname == "int":
                if len(args) == 1:
                    return args[0]  # identity for Z3 ints

            if fname == "bool":
                if len(args) == 1:
                    x = args[0]
                    return z3.If(x != z3.IntVal(0), z3.BoolVal(True), z3.BoolVal(False))

            raise SymbolicExecError(f"Unsupported function call: {fname}()")

        elif isinstance(node.func, ast.Attribute):
            # Method calls like list.append, dict.get, etc.
            raise SymbolicExecError(
                f"Method calls not fully supported: {ast.dump(node.func)}"
            )

        raise SymbolicExecError(f"Unsupported callable: {ast.dump(node.func)}")

    @staticmethod
    def _symbolic_min(values: list) -> Any:
        result = values[0]
        for v in values[1:]:
            result = z3.If(v < result, v, result)
        return result

    @staticmethod
    def _symbolic_max(values: list) -> Any:
        result = values[0]
        for v in values[1:]:
            result = z3.If(v > result, v, result)
        return result

    # ------------------------------------------------------------------
    # Subscript (indexing)
    # ------------------------------------------------------------------

    def _eval_subscript(self, node: ast.Subscript, env: SymbolicEnv) -> Any:
        obj = self._eval_expr(node.value, env)
        idx = self._eval_expr(node.slice, env)

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

        raise SymbolicExecError(f"Subscript on unsupported type: {type(obj)}")
