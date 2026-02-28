"""
Tyr — AST → Z3 Translator (coordinator).

This is the thin top-level class that owns shared state
(fresh-var counter, string interning map) and delegates to
mixin classes in sibling modules.
"""
from __future__ import annotations

import ast
import logging
from typing import Any

import z3

from config import NONE_SENTINEL, MAX_PARTIAL_RETURNS, STRING_ID_BASE
from symbolic.types import (
    SymbolicEnv, SymbolicExecError,
)
# Mixin imports – each provides a block of _exec_* / _eval_* methods.
from symbolic.expressions import ExpressionEvaluator
from symbolic.statements import StatementExecutor
from symbolic.loops import LoopExecutor
from symbolic.builtins import BuiltinDispatcher
from symbolic.methods import MethodDispatcher

logger = logging.getLogger("tyr.symbolic.translator")


class ASTToZ3Translator(
    ExpressionEvaluator,
    StatementExecutor,
    LoopExecutor,
    BuiltinDispatcher,
    MethodDispatcher,
):
    """Walks a Python AST and produces Z3 symbolic expressions.

    Usage::

        translator = ASTToZ3Translator()
        env = translator.execute_function(func_ast, symbolic_params)
        # env.return_value is the Z3 expression for the function output
    """

    def __init__(
        self,
        shared_string_map: dict[str, int] | None = None,
        shared_next_string_id: list[int] | None = None,
    ) -> None:
        self._fresh_counter: int = 0
        self._string_map: dict[str, int] = (
            shared_string_map if shared_string_map is not None else {}
        )
        self._next_string_id_ref: list[int] = (
            shared_next_string_id if shared_next_string_id is not None
            else [STRING_ID_BASE]
        )

    # ── Fresh-variable factories ──────────────────────────────────────

    def fresh_var(self, prefix: str = "tmp") -> z3.ArithRef:
        self._fresh_counter += 1
        return z3.Int(f"__{prefix}_{self._fresh_counter}")

    def fresh_bool(self, prefix: str = "btmp") -> z3.BoolRef:
        self._fresh_counter += 1
        return z3.Bool(f"__{prefix}_{self._fresh_counter}")

    def fresh_array_name(self, prefix: str = "arr") -> str:
        self._fresh_counter += 1
        return f"__{prefix}_{self._fresh_counter}"

    # ── Top-level entry point ─────────────────────────────────────────

    def execute_function(
        self,
        func_node: ast.FunctionDef,
        param_symbols: dict[str, Any],
        initial_constraints: list[Any] | None = None,
    ) -> SymbolicEnv:
        """Symbolically execute *func_node* with *param_symbols* as inputs."""
        env = SymbolicEnv(bindings=dict(param_symbols))
        if initial_constraints:
            env.constraints.extend(initial_constraints)
        self._exec_body(func_node.body, env)

        # Merge pending partial returns with an implicit ``None`` return.
        if not env.has_returned and env.partial_returns:
            val = z3.IntVal(NONE_SENTINEL)
            for guard, partial_val in reversed(env.partial_returns):
                try:
                    val = z3.If(guard, partial_val, val)
                except (z3.Z3Exception, TypeError):
                    pass
            env.partial_returns.clear()
            env.return_value = val
            env.has_returned = True

        return env

    # ── Body / statement dispatch ─────────────────────────────────────

    def _exec_body(self, stmts: list[ast.stmt], env: SymbolicEnv) -> None:
        """Execute a list of statements, stopping on return/break/continue.

        When an ``if``-statement produces a *conditional* break or
        continue (only one branch fires), subsequent statements are
        executed inside a guarded sub-environment so their effects only
        apply on the non-break/non-continue path.  This is how we get
        correct semantics for ``if cond: continue`` inside a loop body.
        """
        for idx, stmt in enumerate(stmts):
            if env.has_returned or env.has_broken or env.has_continued:
                break
            self._exec_stmt(stmt, env)

            # ── Conditional break / continue guard ────────────────────
            guard = (env._continue_guard if env._continue_guard is not None
                     else env._break_guard)
            if guard is None:
                continue  # no guard — move to next statement

            is_break = env._break_guard is not None
            env._continue_guard = None
            env._break_guard = None

            remaining = stmts[idx + 1:]
            if remaining:
                sub = env.copy()
                sub._continue_guard = None
                sub._break_guard = None
                sub.has_broken = False
                sub.has_continued = False
                self._exec_body(remaining, sub)

                try:
                    neg = z3.Not(guard)
                except (z3.Z3Exception, TypeError):
                    neg = z3.BoolVal(True)
                self._merge_env_guarded(env, sub, neg)

                # Propagate control-flow flags from the sub-body.
                if sub.has_returned:
                    if len(env.partial_returns) < MAX_PARTIAL_RETURNS:
                        env.partial_returns.append((neg, sub.return_value))
                for pg, pv in sub.partial_returns:
                    if len(env.partial_returns) >= MAX_PARTIAL_RETURNS:
                        break
                    try:
                        env.partial_returns.append((z3.And(neg, pg), pv))
                    except (z3.Z3Exception, TypeError):
                        pass
                if sub.has_continued:
                    env.has_continued = True
                if sub.has_broken:
                    env.has_broken = True

            # Set the final flag for the enclosing loop.
            if is_break:
                env.has_broken = True
                env._break_guard = guard   # preserve for loop handler
            else:
                env.has_continued = True
                env._continue_guard = guard
            break  # remaining stmts handled recursively

    def _exec_stmt(self, node: ast.stmt, env: SymbolicEnv) -> None:
        """Dispatch a single AST statement."""
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
            self._eval_expr(node.value, env)
        elif isinstance(node, ast.Pass):
            pass
        elif isinstance(node, ast.Break):
            env.has_broken = True
        elif isinstance(node, ast.Continue):
            env.has_continued = True
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None and node.target is not None:
                val = self._eval_expr(node.value, env)
                if isinstance(node.target, ast.Name):
                    env.set(node.target.id, val)
        else:
            raise SymbolicExecError(
                f"Unsupported statement type: {type(node).__name__} "
                f"(line {getattr(node, 'lineno', '?')})"
            )
