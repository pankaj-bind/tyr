"""
Tyr — Symbolic data types and environment.

All data-classes that flow through the symbolic execution engine are
defined here, ensuring zero circular imports.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import z3

if TYPE_CHECKING:
    pass  # forward refs resolved by `from __future__ import annotations`

logger = logging.getLogger("tyr.symbolic.types")


# ── Errors ────────────────────────────────────────────────────────────

class SymbolicExecError(Exception):
    """Raised when AST→Z3 translation hits an unsupported construct."""


# ── SymbolicList ──────────────────────────────────────────────────────

@dataclass
class SymbolicList:
    """Python list modelled as Z3 ``Array(Int→Int)`` + Z3 ``Int`` length."""
    array: z3.ArrayRef
    length: z3.ArithRef

    def copy(self) -> SymbolicList:
        return SymbolicList(array=self.array, length=self.length)


# ── SymbolicSet ───────────────────────────────────────────────────────

@dataclass
class SymbolicSet:
    """Python set modelled as Z3 ``Array(Int→Bool)``."""
    presence: z3.ArrayRef

    def copy(self) -> SymbolicSet:
        return SymbolicSet(presence=self.presence)


# ── SymbolicDict ──────────────────────────────────────────────────────

@dataclass
class SymbolicDict:
    """Python dict modelled as two Z3 arrays + tracked key list."""
    presence: z3.ArrayRef
    values: z3.ArrayRef
    tracked_keys: list[Any] = field(default_factory=list)

    def copy(self) -> SymbolicDict:
        return SymbolicDict(
            presence=self.presence,
            values=self.values,
            tracked_keys=list(self.tracked_keys),
        )


# ── _EnumerateResult ─────────────────────────────────────────────────

@dataclass
class EnumerateResult:
    """Intermediate output of ``enumerate(SymbolicList)``."""
    pairs: list[tuple[Any, Any]]
    source_length: Any


# ── LambdaClosure ────────────────────────────────────────────────────

class LambdaClosure:
    """Callable closure wrapping an ``ast.Lambda`` node."""

    def __init__(self, node: ast.Lambda, env: SymbolicEnv,
                 translator: Any) -> None:
        self._node = node
        self._env = env
        self._translator = translator

    def apply(self, *args: Any) -> Any:
        params = [a.arg for a in self._node.args.args]
        if len(args) != len(params):
            raise SymbolicExecError(
                f"Lambda expects {len(params)} args, got {len(args)}"
            )
        child = self._env.copy()
        for name, val in zip(params, args):
            child.set(name, val)
        return self._translator._eval_expr(self._node.body, child)


# ── SymbolicEnv ───────────────────────────────────────────────────────

@dataclass
class SymbolicEnv:
    """Tracks symbolic variable bindings during AST interpretation.

    Flags
    -----
    has_returned   – function has returned (unconditionally)
    has_broken     – a ``break`` fired (unconditionally or after guard merge)
    has_continued  – a ``continue`` fired (unconditionally or after guard merge)
    _break_guard   – condition under which a *conditional* break fired
    _continue_guard – condition under which a *conditional* continue fired
    """
    bindings: dict[str, Any] = field(default_factory=dict)
    constraints: list[Any] = field(default_factory=list)
    return_value: Any = None
    has_returned: bool = False
    has_broken: bool = False
    has_continued: bool = False
    _break_guard: Any = None
    _continue_guard: Any = None
    partial_returns: list[tuple[Any, Any]] = field(default_factory=list)

    def copy(self) -> SymbolicEnv:
        return SymbolicEnv(
            bindings=dict(self.bindings),
            constraints=list(self.constraints),
            return_value=self.return_value,
            has_returned=self.has_returned,
            has_broken=self.has_broken,
            has_continued=self.has_continued,
            _break_guard=self._break_guard,
            _continue_guard=self._continue_guard,
            partial_returns=list(self.partial_returns),
        )

    def set(self, name: str, value: Any) -> None:
        self.bindings[name] = value

    def get(self, name: str) -> Any:
        if name in self.bindings:
            return self.bindings[name]
        raise SymbolicExecError(f"Undefined variable: '{name}'")
