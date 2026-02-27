"""
Tyr — Shared configuration constants.

All tunable parameters live here so that every module imports from
one canonical source.  Environment variables override the defaults.
"""
from __future__ import annotations

import os

# ── BMC / Loop bounds ──────────────────────────────────────────────────
MAX_LOOP_UNROLL: int = int(os.getenv("TYR_MAX_LOOP_UNROLL", "30"))
MAX_BMC_LENGTH: int = int(os.getenv("TYR_MAX_BMC_LENGTH", "5"))
MAX_SYMBOLIC_RANGE: int = int(os.getenv("TYR_MAX_SYMBOLIC_RANGE", "10"))

# ── Expression-tree caps ──────────────────────────────────────────────
MAX_PARTIAL_RETURNS: int = 32

# ── Sentinels ─────────────────────────────────────────────────────────
NONE_SENTINEL: int = -(2**62)        # Python None → unique integer
INF_SENTINEL: int = 2**60            # float('inf')  → large integer
NEG_INF_SENTINEL: int = -(2**60)     # float('-inf') → large negative

# ── Solver ────────────────────────────────────────────────────────────
Z3_TIMEOUT_MS: int = 30_000
CONCRETE_EXEC_TIMEOUT_S: int = 5

# ── CGSC ──────────────────────────────────────────────────────────────
MAX_CORRECTION_ROUNDS: int = 3
ENGINE_VERSION: str = "tyr-0.4.0"
