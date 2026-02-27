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
MAX_RETRIES: int = 2  # retry on transient failures

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
    """Low-level Groq API call with retry logic for transient failures."""
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

    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 2):  # 1-indexed, up to MAX_RETRIES+1
        logger.info("Calling Groq API (model=%s, attempt %d) …", GROQ_MODEL, attempt)
        try:
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as exc:
                    raise RuntimeError(f"Unexpected Groq API response: {data}") from exc

            # Retry on 5xx / 429 (rate limit)
            if response.status_code in (429, 500, 502, 503, 504):
                logger.warning(
                    "Groq API returned %d on attempt %d — retrying…",
                    response.status_code, attempt,
                )
                last_exc = RuntimeError(f"HTTP {response.status_code}: {response.text}")
                _time.sleep(2 * attempt)  # exponential-ish backoff
                continue

            # Non-retryable HTTP error
            raise RuntimeError(f"Groq API returned HTTP {response.status_code}: {response.text}")

        except requests.exceptions.Timeout:
            logger.warning("Groq API timed out on attempt %d/%d", attempt, MAX_RETRIES + 1)
            last_exc = RuntimeError(f"Groq API timed out after {REQUEST_TIMEOUT}s")
            if attempt <= MAX_RETRIES:
                _time.sleep(2 * attempt)
                continue

        except requests.exceptions.ConnectionError as exc:
            logger.warning("Connection error on attempt %d: %s", attempt, exc)
            last_exc = RuntimeError(f"Connection error: {exc}")
            if attempt <= MAX_RETRIES:
                _time.sleep(2 * attempt)
                continue

    raise last_exc or RuntimeError("Groq API call failed after all retries")


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
