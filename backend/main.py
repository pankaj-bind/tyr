"""
Tyr Backend — main.py
FastAPI server with Counterexample-Guided Self-Correction (CGSC) loop.

Architecture:
  1. LLM generates optimized code
  2. Z3/concrete verifier checks semantic equivalence
  3. If SAT (bug found), counterexample is fed back to LLM for correction
  4. Steps 2-3 repeat up to MAX_CORRECTION_ROUNDS
  5. Returns final verdict with full audit trail
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_service import optimize_code, optimize_with_correction, analyze_complexity
from verifier import verify_equivalence
from config import (
    MAX_BMC_LENGTH, MAX_LOOP_UNROLL, MAX_SYMBOLIC_RANGE,
    MAX_CORRECTION_ROUNDS, ENGINE_VERSION,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("tyr")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Tyr — LLM Hallucination Bounding Engine",
    version=ENGINE_VERSION,
    description=(
        "Counterexample-Guided Self-Correction for LLM code optimization. "
        "Uses Z3 SMT solving + concrete testing to bound LLM hallucinations."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CorrectionRound(BaseModel):
    """One round in the CGSC audit trail."""
    round: int
    optimized_code: str
    status: str
    message: str
    counterexample: dict[str, Any] | None = None


class ComplexityInfo(BaseModel):
    """Big-O complexity for a code snippet."""
    time: str = "N/A"
    space: str = "N/A"
    explanation: str = ""


class VerifyRequest(BaseModel):
    """Payload from the VS Code extension."""
    code: str = Field(..., min_length=1)
    language: str = Field(default="python")


class VerifyResponse(BaseModel):
    """Full response including CGSC audit trail."""
    original_code: str
    optimized_code: str
    status: str = Field(
        ...,
        description=(
            "'UNSAT' — formally proven equivalent via Z3 (bounded). "
            "'SAT' — counterexample found, semantics differ. "
            "'WARNING' — Z3 timed out; empirically tested only (not a proof). "
            "'ERROR' — verification could not complete."
        ),
    )
    message: str
    counterexample: dict[str, Any] | None = None

    # --- Advanced fields ---
    correction_rounds: list[CorrectionRound] = Field(default_factory=list)
    total_rounds: int = 0
    original_complexity: ComplexityInfo | None = None
    optimized_complexity: ComplexityInfo | None = None
    complexity_improved: bool | None = None
    elapsed_ms: int = 0
    verification_bounds: dict[str, int] | None = None


# ---------------------------------------------------------------------------
# Core endpoint — Counterexample-Guided Self-Correction Loop
# ---------------------------------------------------------------------------

@app.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest) -> VerifyResponse:
    """
    CGSC Loop:
      1. LLM optimizes code
      2. Z3/concrete verifier checks equivalence
      3. If SAT → feed counterexample back to LLM for correction
      4. Repeat up to MAX_CORRECTION_ROUNDS
      5. Return final verdict + full audit trail
    """
    t0 = time.perf_counter()
    logger.info("━━━ /verify request (%d chars, lang=%s) ━━━", len(req.code), req.language)

    audit: list[CorrectionRound] = []

    # --- Round 0: Initial LLM optimization --------------------------------
    try:
        optimized_code = optimize_code(req.code, language=req.language)
    except Exception as exc:
        logger.exception("LLM optimization failed")
        raise HTTPException(status_code=502, detail=f"LLM service error: {exc}") from exc

    if not optimized_code or not optimized_code.strip():
        raise HTTPException(status_code=502, detail="LLM returned empty optimized code.")

    logger.info("Round 0: LLM returned %d chars", len(optimized_code))

    # --- Verification + Correction rounds ---------------------------------
    current_optimized = optimized_code
    final_verification: dict[str, Any] = {}

    for round_num in range(MAX_CORRECTION_ROUNDS + 1):
        logger.info("── Verification round %d ──", round_num)

        try:
            verification = verify_equivalence(req.code, current_optimized)
        except Exception as exc:
            logger.exception("Verification engine error in round %d", round_num)
            verification = {
                "status": "ERROR",
                "message": f"Verification engine error: {exc}",
                "counterexample": None,
            }

        status = verification["status"]
        logger.info("Round %d verdict: %s", round_num, status)

        audit.append(CorrectionRound(
            round=round_num,
            optimized_code=current_optimized,
            status=status,
            message=verification["message"],
            counterexample=verification.get("counterexample"),
        ))

        final_verification = verification

        # UNSAT or ERROR → stop (no correction possible/needed)
        if status != "SAT":
            break

        # SAT on last round → we've exhausted correction attempts
        if round_num >= MAX_CORRECTION_ROUNDS:
            logger.warning(
                "Exhausted %d correction rounds. Final status: SAT",
                MAX_CORRECTION_ROUNDS,
            )
            break

        # --- CGSC: Feed counterexample back to LLM -----------------------
        counterexample = verification.get("counterexample", {})
        logger.info(
            "Round %d: SAT — attempting correction with counterexample: %s",
            round_num, counterexample,
        )

        try:
            corrected = optimize_with_correction(
                original_code=req.code,
                failed_optimized_code=current_optimized,
                counterexample=counterexample or {},
                language=req.language,
            )
        except Exception as exc:
            logger.exception("LLM correction failed in round %d", round_num)
            break

        if not corrected or not corrected.strip():
            logger.warning("LLM returned empty correction in round %d", round_num)
            break

        # If the LLM returns the same code, no point retrying
        if corrected.strip() == current_optimized.strip():
            logger.warning("LLM returned identical code in round %d — stopping", round_num)
            break

        current_optimized = corrected
        logger.info("Round %d: Got corrected code (%d chars)", round_num, len(corrected))

    # --- Complexity analysis (parallel for original & optimized) -----------
    original_cx = ComplexityInfo()
    optimized_cx = ComplexityInfo()
    try:
        cx_orig = analyze_complexity(req.code, language=req.language)
        original_cx = ComplexityInfo(**cx_orig)
    except Exception:
        logger.warning("Original complexity analysis failed", exc_info=True)
    try:
        cx_opt = analyze_complexity(current_optimized, language=req.language)
        optimized_cx = ComplexityInfo(**cx_opt)
    except Exception:
        logger.warning("Optimized complexity analysis failed", exc_info=True)

    # --- Complexity gate: detect no-improvement ---
    complexity_improved = _check_complexity_improved(
        original_cx.time, optimized_cx.time
    )
    if not complexity_improved and final_verification.get("status") == "UNSAT":
        logger.warning(
            "Optimization is equivalent but complexity did NOT improve: %s → %s",
            original_cx.time, optimized_cx.time,
        )

    elapsed = int((time.perf_counter() - t0) * 1000)
    logger.info("━━━ Done in %dms — Final: %s (%d rounds) ━━━",
                elapsed, final_verification.get("status", "?"), len(audit))

    return VerifyResponse(
        original_code=req.code,
        optimized_code=current_optimized,
        status=final_verification.get("status", "ERROR"),
        message=final_verification.get("message", "Unknown error"),
        counterexample=final_verification.get("counterexample"),
        correction_rounds=audit,
        total_rounds=len(audit),
        original_complexity=original_cx,
        optimized_complexity=optimized_cx,
        complexity_improved=complexity_improved,
        elapsed_ms=elapsed,
        verification_bounds={
            "max_list_length": MAX_BMC_LENGTH,
            "max_symbolic_range": MAX_SYMBOLIC_RANGE,
            "max_loop_unroll": MAX_LOOP_UNROLL,
        },
    )


# ---------------------------------------------------------------------------
# Health-check
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Complexity comparison helper
# ---------------------------------------------------------------------------

def _normalize_complexity(s: str) -> str | None:
    """
    Normalize a Big-O string to a canonical form for comparison.
    E.g. 'O(N^2)', 'O(n ** 2)', 'O(n²)' all → 'n^2'
    Returns the canonical inner expression, or None if unparseable.
    """
    import re as _re
    s = s.strip().lower()
    # Extract inner part: O(...) → ...
    m = _re.match(r"o\((.+)\)", s)
    if not m:
        return None
    inner = m.group(1).strip()
    # Normalize whitespace
    inner = _re.sub(r"\s+", "", inner)
    # n² → n^2, n³ → n^3
    inner = inner.replace("²", "^2").replace("³", "^3")
    # n**2 → n^2
    inner = inner.replace("**", "^")
    # nlogn, n*log(n), n*logn → nlogn
    inner = inner.replace("*", "")
    inner = _re.sub(r"log\(n\)", "logn", inner)
    inner = _re.sub(r"log\(n\)", "logn", inner)
    return inner


_CANONICAL_RANK: dict[str, int] = {
    "1": 0,
    "logn": 1,
    "sqrt(n)": 2, "sqrtn": 2,
    "n": 3,
    "nlogn": 4,
    "n^2": 5,
    "n^3": 6,
    "2^n": 7,
    "n!": 8,
}


def _check_complexity_improved(orig_time: str, opt_time: str) -> bool | None:
    """Return True if optimized complexity is strictly better, False if same/worse, None if unknown."""
    orig_norm = _normalize_complexity(orig_time)
    opt_norm = _normalize_complexity(opt_time)
    if orig_norm is None or opt_norm is None:
        return None
    orig_rank = _CANONICAL_RANK.get(orig_norm)
    opt_rank = _CANONICAL_RANK.get(opt_norm)
    if orig_rank is None or opt_rank is None:
        # Last resort: string comparison
        if orig_norm == opt_norm:
            return False
        return None
    return opt_rank < orig_rank


class VerifyPairRequest(BaseModel):
    """Payload for direct pair verification — skips LLM optimization."""
    original_code: str = Field(..., min_length=1)
    optimized_code: str = Field(..., min_length=1)
    language: str = Field(default="python")


@app.post("/verify-pair", response_model=VerifyResponse)
async def verify_pair(req: VerifyPairRequest) -> VerifyResponse:
    """
    Verify a pre-existing (original, optimized) code pair.
    Runs ONLY equivalence verification + complexity analysis —
    does NOT call the LLM to generate optimized code.
    Used by Stage 2 benchmarking to verify LLM-generated code.
    """
    t0 = time.perf_counter()
    logger.info(
        "━━━ /verify-pair request  |  orig=%d chars  |  opt=%d chars ━━━",
        len(req.original_code), len(req.optimized_code),
    )

    # ── Equivalence verification ──────────────────────────────────────
    try:
        verification = verify_equivalence(req.original_code, req.optimized_code)
    except Exception as exc:
        logger.exception("Verification engine error")
        verification = {
            "status": "ERROR",
            "message": f"Verification engine error: {exc}",
            "counterexample": None,
        }

    status = verification.get("status", "ERROR")
    logger.info("verify-pair verdict: %s", status)

    # ── Complexity analysis ───────────────────────────────────────────
    original_cx = ComplexityInfo()
    optimized_cx = ComplexityInfo()
    try:
        cx_orig = analyze_complexity(req.original_code, language=req.language)
        original_cx = ComplexityInfo(**cx_orig)
    except Exception:
        logger.warning("Original complexity analysis failed", exc_info=True)
    try:
        cx_opt = analyze_complexity(req.optimized_code, language=req.language)
        optimized_cx = ComplexityInfo(**cx_opt)
    except Exception:
        logger.warning("Optimized complexity analysis failed", exc_info=True)

    complexity_improved = _check_complexity_improved(
        original_cx.time, optimized_cx.time
    )

    elapsed = int((time.perf_counter() - t0) * 1000)
    logger.info("━━━ verify-pair done in %dms — %s ━━━", elapsed, status)

    return VerifyResponse(
        original_code=req.original_code,
        optimized_code=req.optimized_code,
        status=status,
        message=verification.get("message", "Unknown error"),
        counterexample=verification.get("counterexample"),
        correction_rounds=[],
        total_rounds=0,
        original_complexity=original_cx,
        optimized_complexity=optimized_cx,
        complexity_improved=complexity_improved,
        elapsed_ms=elapsed,
        verification_bounds={
            "max_list_length": MAX_BMC_LENGTH,
            "max_symbolic_range": MAX_SYMBOLIC_RANGE,
            "max_loop_unroll": MAX_LOOP_UNROLL,
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "engine": ENGINE_VERSION}
