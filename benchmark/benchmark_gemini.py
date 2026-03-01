#!/usr/bin/env python3
"""
Tyr — Standalone Gemini Benchmark Script (Google Generative AI SDK)
====================================================================
Generates optimized code via Google Gemini models, then verifies each
output through the local Tyr BMC verifier.

Produces: benchmark/data/gemini_results.csv (17 columns, incremental).

Usage
~~~~~
    python benchmark/benchmark_gemini.py \
        --model gemini-2.5-flash \
        --api-key AIzaSy... \
        --delay 4.0

    # With thinking-budget control (Gemini 2.5 Flash)
    python benchmark/benchmark_gemini.py \
        --model gemini-2.5-flash \
        --api-key AIzaSy... \
        --thinking-budget 8192

Requirements
~~~~~~~~~~~~
    pip install google-genai tqdm
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

# ─────────────────────────── PATH SETUP ──────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_BACKEND_DIR  = _PROJECT_ROOT / "backend"

for _p in (_PROJECT_ROOT, _BACKEND_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ─────────────────────────── THIRD-PARTY ─────────────────────────────
try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit(
        "ERROR: `google-genai` SDK is required.\n"
        "  pip install google-genai"
    )

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: `tqdm` is required.  pip install tqdm")

# ─────────────────────────── LOCAL VERIFIER ──────────────────────────
try:
    from backend.verifier.equivalence import verify_equivalence
except ImportError:
    from verifier.equivalence import verify_equivalence  # type: ignore[no-redef]


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_DATASET = _SCRIPT_DIR / "tyr_benchmark_150.json"
DEFAULT_CSV_OUT = _SCRIPT_DIR / "data" / "gemini_results.csv"

CSV_COLUMNS = [
    "id", "name", "category", "difficulty",
    "original_complexity", "target_complexity",
    "verdict", "original_code", "generated_code",
    "optimized_complexity_time", "complexity_improved",
    "latency_ms", "prompt_tokens", "reasoning_tokens",
    "total_tokens", "api_status", "error_detail",
]

PROMPT_TEMPLATE = (
    "You are an expert algorithm developer. Below is a naive Python "
    "implementation. Refactor this code to achieve a target time complexity "
    "of {target_complexity}. Provide ONLY the optimized Python code. "
    "Do not provide explanations.\n\nNaive Code:\n{original_code}"
)

MAX_RETRIES  = 3
BACKOFF_BASE = 2


# ═══════════════════════════════════════════════════════════════════════
# Output cleaning
# ═══════════════════════════════════════════════════════════════════════

def clean_llm_output(raw: str) -> str:
    """Strip markdown fences, XML tags, and trailing prose."""
    text = raw

    text = re.sub(r"</?[a-zA-Z][a-zA-Z0-9_-]*(?:\s[^>]*)?>", "", text)

    fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not cleaned and not stripped:
            continue
        if cleaned and stripped and not stripped.startswith("#"):
            if re.match(
                r"^(This|Note|The above|Here |I |Let me|Explanation|"
                r"Output|In this|Time complexity|Space complexity|"
                r"Example|Alternative|We can|The key|The function|"
                r"The algorithm|Complexity|##|###|\*\*)",
                stripped,
            ):
                break
        cleaned.append(line)

    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    return "\n".join(cleaned)


def _syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ═══════════════════════════════════════════════════════════════════════
# Token extraction
# ═══════════════════════════════════════════════════════════════════════

def _extract_tokens(response) -> dict[str, int]:
    """Extract token counts from Gemini response.usage_metadata."""
    out = {"prompt_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    meta = getattr(response, "usage_metadata", None)
    if not meta:
        return out

    pt = getattr(meta, "prompt_token_count", None)
    ct = getattr(meta, "candidates_token_count", None)
    tt = getattr(meta, "total_token_count", None)

    # Gemini 2.5 Flash exposes thinking_token_count
    thinking = getattr(meta, "thinking_token_count", None)

    if pt is not None:
        out["prompt_tokens"] = int(pt)
    if tt is not None:
        out["total_tokens"] = int(tt)
    elif pt is not None and ct is not None:
        out["total_tokens"] = int(pt) + int(ct)
    if thinking is not None:
        out["reasoning_tokens"] = int(thinking)

    return out


# ═══════════════════════════════════════════════════════════════════════
# Error classification
# ═══════════════════════════════════════════════════════════════════════

def _is_retriable(exc: Exception) -> bool:
    """Return True for 429 / 503 / ResourceExhausted errors."""
    msg = str(exc).lower()
    return any(tok in msg for tok in (
        "429", "503", "rate", "quota", "resourceexhausted",
        "service unavailable", "internal server error",
    ))


# ═══════════════════════════════════════════════════════════════════════
# API caller with exponential backoff
# ═══════════════════════════════════════════════════════════════════════

def call_gemini(
    client: genai.Client,
    model: str,
    prompt: str,
    thinking_budget: int | None = None,
) -> dict:
    """
    Call the Gemini API.  Returns dict with:
        text, latency_ms, prompt_tokens, reasoning_tokens, total_tokens
    """
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            config_kwargs: dict = {"temperature": 0.0}
            if thinking_budget is not None:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                )

            config = types.GenerateContentConfig(**config_kwargs)

            # ── STRICT latency isolation: timer wraps ONLY the API call ──
            t0 = time.perf_counter()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            text   = response.text or ""
            tokens = _extract_tokens(response)

            return {"text": text, "latency_ms": latency_ms, **tokens}

        except Exception as exc:
            last_exc = exc
            if _is_retriable(exc) and attempt < MAX_RETRIES:
                wait = BACKOFF_BASE ** attempt
                tqdm.write(
                    f"  ⚠  Transient error (attempt {attempt}/{MAX_RETRIES}). "
                    f"Retrying in {wait}s …  [{type(exc).__name__}: {exc}]"
                )
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"API error after {MAX_RETRIES} attempts: {last_exc}")


# ═══════════════════════════════════════════════════════════════════════
# CSV helpers
# ═══════════════════════════════════════════════════════════════════════

def _load_processed_ids(csv_path: Path) -> set[str]:
    done: set[str] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pid = row.get("id", "").strip()
            if pid:
                done.add(pid)
    return done


def _ensure_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()


def _append_row(csv_path: Path, row: dict) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════
# Local verification wrapper
# ═══════════════════════════════════════════════════════════════════════

def _run_verification(
    original_code: str,
    generated_code: str,
) -> dict:
    """Run Tyr BMC verifier locally."""
    try:
        result = verify_equivalence(original_code, generated_code)
        verdict = result.get("status", "ERROR")
        if verdict not in ("UNSAT", "SAT", "WARNING", "ERROR"):
            verdict = "WARNING"

        return {
            "verdict": verdict,
            "optimized_complexity_time": result.get("optimized_complexity_time", ""),
            "complexity_improved": result.get("complexity_improved", ""),
        }

    except Exception:
        return {
            "verdict": "ERROR",
            "optimized_complexity_time": "",
            "complexity_improved": "",
            "solver_error": traceback.format_exc(),
        }


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Tyr — Gemini Benchmark (Google Generative AI SDK). "
            "Generates + verifies optimized code for tyr_benchmark_150."
        ),
    )
    ap.add_argument(
        "--model", default="gemini-2.5-flash",
        help="Gemini model identifier (default: gemini-2.5-flash).",
    )
    ap.add_argument(
        "--api-key", required=True,
        help="Google AI API key.",
    )
    ap.add_argument(
        "--delay", type=float, default=4.5,
        help="Seconds to sleep between API calls (default: 4.5).",
    )
    ap.add_argument(
        "--thinking-budget", type=int, default=None,
        help="Thinking token budget for Gemini 2.5 models (optional).",
    )
    ap.add_argument(
        "--dataset", default=str(DEFAULT_DATASET),
        help="Path to benchmark JSON (default: tyr_benchmark_150.json).",
    )
    ap.add_argument(
        "--output", default=str(DEFAULT_CSV_OUT),
        help="Path to output CSV (default: benchmark/data/gemini_results.csv).",
    )
    return ap.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    args    = parse_args()
    model   = args.model
    csv_out = Path(args.output)

    # ── Load dataset ────────────────────────────────────────────────
    with open(args.dataset, "r", encoding="utf-8") as fh:
        dataset: list[dict] = json.load(fh)

    # ── Gemini client ──────────────────────────────────────────────
    client = genai.Client(api_key=args.api_key)

    # ── CSV setup ───────────────────────────────────────────────────
    _ensure_csv_header(csv_out)
    processed = _load_processed_ids(csv_out)

    # ── Banner ──────────────────────────────────────────────────────
    print(
        f"\n{'═' * 72}\n"
        f"  Tyr — Gemini Benchmark  |  Google Generative AI SDK\n"
        f"  Model     : {model}\n"
        f"  Thinking  : {args.thinking_budget or 'default'}\n"
        f"  Problems  : {len(dataset)}  ({len(processed)} already done)\n"
        f"  Delay     : {args.delay}s\n"
        f"  Output    : {csv_out}\n"
        f"{'═' * 72}\n"
    )

    # ── Counters ────────────────────────────────────────────────────
    ok = err = skipped = 0
    latencies: list[float] = []
    t_start = time.time()

    bar = tqdm(dataset, unit="prob", ncols=110)

    for problem in bar:
        pid  = problem["id"]
        name = problem["name"]
        bar.set_postfix_str(f"{pid} {name}")

        if pid in processed:
            skipped += 1
            continue

        prompt = PROMPT_TEMPLATE.format(
            target_complexity=problem["target_complexity"],
            original_code=problem["original_code"],
        )

        generated_code = ""
        api_status     = "OK"
        error_detail   = ""
        latency_ms     = 0.0
        prompt_tokens  = 0
        reasoning_tokens = 0
        total_tokens   = 0
        verdict        = ""
        opt_complexity  = ""
        complexity_improved = ""

        # ── Step 1: Call the API ───────────────────────────────────
        try:
            result = call_gemini(
                client, model, prompt,
                thinking_budget=args.thinking_budget,
            )

            generated_code   = clean_llm_output(result["text"])
            latency_ms       = result["latency_ms"]
            prompt_tokens    = result["prompt_tokens"]
            reasoning_tokens = result["reasoning_tokens"]
            total_tokens     = result["total_tokens"]

            if not generated_code or not _syntax_ok(generated_code):
                api_status   = "SYNTAX_ERROR"
                error_detail = "Generated code failed ast.parse()"
                verdict      = "ERROR"
            else:
                api_status = "OK"

        except Exception as exc:
            api_status   = "ERROR"
            error_detail = traceback.format_exc()[:1000]
            verdict      = "ERROR"
            tqdm.write(f"  ✖  {pid} ({name}): {exc}")
            err += 1

        # ── Step 2: Verify (only if API succeeded with valid code) ─
        if api_status == "OK" and generated_code:
            try:
                vr = _run_verification(problem["original_code"], generated_code)
                verdict             = vr["verdict"]
                opt_complexity      = vr.get("optimized_complexity_time", "")
                complexity_improved = vr.get("complexity_improved", "")

                if "solver_error" in vr:
                    error_detail = vr["solver_error"][:1000]

            except Exception:
                verdict      = "ERROR"
                error_detail = traceback.format_exc()[:1000]

            latencies.append(latency_ms)
            ok += 1

        # ── Step 3: Write CSV row immediately ─────────────────────
        row = {
            "id":                        pid,
            "name":                      name,
            "category":                  problem.get("category", ""),
            "difficulty":                problem.get("difficulty", ""),
            "original_complexity":       problem.get("original_complexity", ""),
            "target_complexity":         problem.get("target_complexity", ""),
            "verdict":                   verdict,
            "original_code":             problem.get("original_code", ""),
            "generated_code":            generated_code,
            "optimized_complexity_time": opt_complexity,
            "complexity_improved":       complexity_improved,
            "latency_ms":                f"{latency_ms:.1f}",
            "prompt_tokens":             prompt_tokens,
            "reasoning_tokens":          reasoning_tokens,
            "total_tokens":              total_tokens,
            "api_status":                api_status,
            "error_detail":              error_detail,
        }
        _append_row(csv_out, row)

        # ── Throttle ──────────────────────────────────────────────
        time.sleep(args.delay)

    bar.close()

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)

    avg_lat = f"{sum(latencies)/len(latencies):.1f}" if latencies else "N/A"

    print(
        f"\n{'═' * 72}\n"
        f"  BENCHMARK COMPLETE — Gemini ({model})\n"
        f"  Processed : {ok}  |  Errors : {err}  |  Skipped : {skipped}\n"
        f"  Avg Latency (API only) : {avg_lat} ms\n"
        f"  Wall Clock : {h:02d}h {m:02d}m {s:02d}s\n"
        f"  Output     : {csv_out}\n"
        f"{'═' * 72}\n"
    )


if __name__ == "__main__":
    main()
