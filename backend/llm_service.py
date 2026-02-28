"""
Tyr Backend — llm_service.py
Handles communication with the Groq API (LLM) for code optimisation.

Key features:
  - Initial code optimization with strict raw-code-only prompt
  - Counterexample-Guided Self-Correction (CGSC): when Z3 finds SAT,
    the counterexample is fed back to the LLM for a corrected attempt
  - Big-O complexity estimation via LLM analysis
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL: str = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT: int = 120  # seconds
MAX_RETRIES: int = 5  # aggressive retry budget for Groq free-tier rate limits

logger = logging.getLogger("tyr.llm")

if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY is not set. LLM calls will fail. "
        "Add it to backend/.env as GROQ_API_KEY=<your-key>."
    )

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = (
    "You are an expert algorithm engineer. "
    "The user will provide a code snippet. Your job is to refactor it to "
    "GENUINELY improve its Big-O time complexity (e.g. O(N^2) → O(N log N) or O(N)) "
    "while preserving EXACTLY the same input/output behaviour.\n\n"
    "OPTIMIZATION STRATEGIES you MUST consider:\n"
    "- Replace nested loops with hash-map / dict / set lookups (O(N^2) → O(N))\n"
    "- Use sorting + two-pointer or binary search (O(N^2) → O(N log N))\n"
    "- Use prefix sums, sliding window, or monotonic stack\n"
    "- Use Counter / defaultdict to avoid repeated .count() calls\n"
    "- Pre-compute lookup tables instead of redundant recomputation\n\n"
    "ANTI-PATTERNS — these are NOT real optimizations, NEVER do these:\n"
    "- Do NOT replace a nested loop with list.count() — .count() is O(N), "
    "so calling it in a loop is STILL O(N^2).\n"
    "- Do NOT create variables you never use (dead code).\n"
    "- Do NOT use set() if the original preserves insertion order in lists.\n"
    "- Do NOT use `in list` checks — use `in set` or `in dict` instead for O(1) lookup.\n\n"
    "RULES — follow them strictly:\n"
    "1. Return ONLY the optimized source code. No markdown fences, no "
    "explanation, no comments about what you changed.\n"
    "2. Do NOT wrap the code in ```python``` or any other code-block markers.\n"
    "3. Keep the same function signature(s) and return type(s).\n"
    "4. Do NOT add, remove, or rename any parameters.\n"
    "5. Do NOT import modules that were not already imported in the original.\n"
    "6. If the code is already optimal, return it unchanged.\n"
    "7. The return value must be EXACTLY the same type and content for ALL inputs.\n"
    "8. If the original returns a list, the optimized must return a list with "
    "the SAME order. Do NOT change list to set.\n"
    "9. Output raw source code and absolutely nothing else."
)

CORRECTION_PROMPT: str = (
    "You are an expert algorithm engineer fixing a previous optimization attempt.\n\n"
    "Your PREVIOUS optimization was REJECTED because a formal verifier found a "
    "counterexample where the original and optimized code produce different results.\n\n"
    "RULES — follow them strictly:\n"
    "1. Return ONLY the corrected optimized source code.\n"
    "2. No markdown fences, no explanation, no comments.\n"
    "3. Keep the same function signature(s) and return type(s).\n"
    "4. Fix the specific bug described in the counterexample.\n"
    "5. The return value must be EXACTLY the same type, order, and content.\n"
    "6. Output raw source code and absolutely nothing else."
)

COMPLEXITY_PROMPT: str = (
    "You are a complexity analysis expert. Analyze the Big-O time and space "
    "complexity of the given code. Respond in EXACTLY this JSON format, "
    "nothing else:\n"
    '{"time": "O(...)", "space": "O(...)", "explanation": "one sentence"}\n'
    "Do NOT wrap in markdown. Return ONLY the JSON object."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_code(source_code: str, *, language: str = "python") -> str:
    """
    Send *source_code* to the Groq LLM and return the optimized version.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured. Add it to backend/.env.")

    user_prompt = (
        f"Language: {language}\n\n"
        f"Optimize the following code for better Big-O complexity. "
        f"Return ONLY the raw optimized code, nothing else.\n\n"
        f"{source_code}"
    )

    raw = _call_groq(SYSTEM_PROMPT, user_prompt)
    return _strip_markdown_fences(raw).strip()


def optimize_with_correction(
    original_code: str,
    failed_optimized_code: str,
    counterexample: dict[str, Any],
    *,
    language: str = "python",
) -> str:
    """
    Counterexample-Guided Self-Correction (CGSC).

    Sends the original code, the failed optimization, and the concrete
    counterexample back to the LLM, asking it to produce a corrected version.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    ce_str = "\n".join(f"  {k} = {v}" for k, v in counterexample.items())

    user_prompt = (
        f"Language: {language}\n\n"
        f"ORIGINAL CODE (correct behaviour):\n{original_code}\n\n"
        f"YOUR PREVIOUS OPTIMIZATION (REJECTED — produces wrong output):\n"
        f"{failed_optimized_code}\n\n"
        f"COUNTEREXAMPLE (inputs where outputs differ):\n{ce_str}\n\n"
        f"Fix the optimization so it produces EXACTLY the same output as the "
        f"original for ALL inputs, including the counterexample above. "
        f"Return ONLY the corrected optimized code."
    )

    raw = _call_groq(CORRECTION_PROMPT, user_prompt)
    return _strip_markdown_fences(raw).strip()


def analyze_complexity(code: str, *, language: str = "python") -> dict[str, str]:
    """
    Ask the LLM to estimate the Big-O time and space complexity of *code*.
    Returns {"time": "O(...)", "space": "O(...)", "explanation": "..."}.
    """
    if not GROQ_API_KEY:
        return {"time": "N/A", "space": "N/A", "explanation": "API key not configured"}

    user_prompt = f"Language: {language}\n\nAnalyze this code:\n{code}"

    try:
        raw = _call_groq(COMPLEXITY_PROMPT, user_prompt, temperature=0.0)
        cleaned = _strip_markdown_fences(raw).strip()
        # Try to parse JSON
        import json
        return json.loads(cleaned)
    except Exception as exc:
        logger.warning("Complexity analysis failed: %s", exc)
        return {"time": "N/A", "space": "N/A", "explanation": str(exc)}


# ---------------------------------------------------------------------------
# Internal: Groq API caller
# ---------------------------------------------------------------------------

def _call_groq(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """
    Low-level Groq API call with **hardcore** retry logic.

    Rate-limit strategy (Groq free tier):
      - HTTP 429  → sleep  20 × attempt  (up to 100 s on attempt 5)
      - HTTP 5xx  → sleep   5 × attempt
      - Timeout / connection error → sleep 5 × attempt
    All transient failures are retried up to MAX_RETRIES times before raising.
    """
    import time as _time

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    total_attempts = MAX_RETRIES + 1          # e.g. 6 total with MAX_RETRIES=5
    last_exc: Exception | None = None

    for attempt in range(1, total_attempts + 1):   # 1-indexed
        logger.info(
            "Groq API call  model=%s  attempt %d/%d",
            GROQ_MODEL, attempt, total_attempts,
        )
        try:
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            # ── Success ───────────────────────────────────────────
            if response.status_code == 200:
                data = response.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as exc:
                    raise RuntimeError(
                        f"Unexpected Groq API response structure: {data}"
                    ) from exc

            # ── 429 Rate Limit — long cooldown ────────────────────
            if response.status_code == 429:
                wait = 20 * attempt
                logger.warning(
                    "RATE LIMITED (429) on attempt %d/%d — "
                    "sleeping %d s before retry …",
                    attempt, total_attempts, wait,
                )
                last_exc = RuntimeError(
                    f"HTTP 429 Rate Limited: {response.text[:300]}"
                )
                if attempt < total_attempts:
                    _time.sleep(wait)
                    continue
                break                           # exhausted

            # ── 5xx Server Error — moderate cooldown ──────────────
            if response.status_code in (500, 502, 503, 504):
                wait = 5 * attempt
                logger.warning(
                    "Server error (%d) on attempt %d/%d — "
                    "sleeping %d s before retry …",
                    response.status_code, attempt, total_attempts, wait,
                )
                last_exc = RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )
                if attempt < total_attempts:
                    _time.sleep(wait)
                    continue
                break

            # ── Non-retryable HTTP error (4xx except 429) ─────────
            raise RuntimeError(
                f"Groq API returned HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

        except requests.exceptions.Timeout:
            wait = 5 * attempt
            logger.warning(
                "Groq API timed out on attempt %d/%d — "
                "sleeping %d s …",
                attempt, total_attempts, wait,
            )
            last_exc = RuntimeError(
                f"Groq API timed out after {REQUEST_TIMEOUT}s"
            )
            if attempt < total_attempts:
                _time.sleep(wait)
                continue

        except requests.exceptions.ConnectionError as exc:
            wait = 5 * attempt
            logger.warning(
                "Connection error on attempt %d/%d: %s — "
                "sleeping %d s …",
                attempt, total_attempts, exc, wait,
            )
            last_exc = RuntimeError(f"Connection error: {exc}")
            if attempt < total_attempts:
                _time.sleep(wait)
                continue

    raise last_exc or RuntimeError(
        "Groq API call failed after all retries"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove accidental markdown code fences."""
    pattern = r"^```(?:\w+)?\s*\n(.*?)```\s*$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return text
