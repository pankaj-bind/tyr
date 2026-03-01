#!/usr/bin/env python3
"""
Tyr — Stage 2: BMC Verification of LLM-Generated Code
========================================================
Reads llm_results.csv (Stage 1 output), sends each row's
(original_code, generated_code) pair through the Tyr BMC verifier
(/verify-pair endpoint), and fills in:

  • verdict            — UNSAT / SAT / WARNING / ERROR
  • optimized_complexity_time — Big-O of the generated code
  • complexity_improved       — True / False / None

Usage
~~~~~
    # 1. Start the Tyr backend
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

    # 2. Run Stage 2 verification
    python benchmark/verify_llm_results.py
    python benchmark/verify_llm_results.py --csv path/to/results.csv
    python benchmark/verify_llm_results.py --ids TYR-001,TYR-005
    python benchmark/verify_llm_results.py --batch 20
    python benchmark/verify_llm_results.py --force     # re-verify all rows

Features
~~~~~~~~
  • Incremental: skips rows already verified (unless --force)
  • Resume-safe: writes results one row at a time to a temp file, swaps on
    completion — no data loss on Ctrl-C
  • Retry logic with back-off for transient errors / rate limits
  • tqdm progress bar with live verdict stats
  • Full summary report at the end

Requires
~~~~~~~~
    pip install requests tqdm
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("ERROR: `requests` is required.  pip install requests")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: `tqdm` is required.  pip install tqdm")


# ═══════════════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════════════

_PROJECT_ROOT     = Path(__file__).resolve().parent.parent
DEFAULT_CSV       = _PROJECT_ROOT / "Research_Paper" / "data" / "llm_results.csv"
DEFAULT_API_URL   = "http://localhost:8000/verify-pair"
DEFAULT_TIMEOUT   = 300          # seconds per problem (Z3 can be slow)
MAX_RETRIES       = 3
RETRY_BACKOFF     = 5            # base seconds
INTER_REQUEST_DELAY = 2          # seconds between requests

# CSV columns in the unified schema
CSV_FIELDS = [
    "id", "name", "category", "difficulty",
    "original_complexity", "target_complexity",
    "verdict",
    "original_code", "generated_code",
    "optimized_complexity_time", "complexity_improved",
    "latency_ms",
    "prompt_tokens", "reasoning_tokens", "total_tokens",
    "api_status", "error_detail",
]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _call_verify_pair(
    api_url: str,
    original_code: str,
    optimized_code: str,
    timeout: int,
) -> dict[str, Any]:
    """POST to /verify-pair with retry logic."""
    payload = {
        "original_code": original_code,
        "optimized_code": optimized_code,
        "language": "python",
    }
    last_err: Exception | None = None

    for attempt in range(1 + MAX_RETRIES):
        try:
            resp = requests.post(api_url, json=payload, timeout=timeout)

            if resp.status_code == 429:
                wait = 20 * (attempt + 1)
                print(f"\n  ⚠ 429 Rate Limited — sleeping {wait}s "
                      f"(attempt {attempt + 1}/{1 + MAX_RETRIES})",
                      file=sys.stderr)
                last_err = Exception("HTTP 429 rate limited")
                time.sleep(wait)
                continue

            if resp.status_code in (500, 502, 503, 504):
                wait = RETRY_BACKOFF * (attempt + 1)
                print(f"\n  ⚠ HTTP {resp.status_code} — sleeping {wait}s "
                      f"(attempt {attempt + 1}/{1 + MAX_RETRIES})",
                      file=sys.stderr)
                last_err = Exception(
                    f"HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
                    continue
                return {
                    "status": "ERROR",
                    "message": f"HTTP {resp.status_code} after "
                               f"{MAX_RETRIES} retries",
                }

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            return {"status": "ERROR",
                    "message": f"Request timed out ({timeout}s)"}
        except requests.exceptions.ConnectionError as exc:
            last_err = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
        except requests.exceptions.HTTPError as exc:
            return {
                "status": "ERROR",
                "message": f"HTTP {getattr(exc.response, 'status_code', '?')}: "
                           f"{str(exc)[:200]}",
            }
        except (json.JSONDecodeError, ValueError) as exc:
            return {
                "status": "ERROR",
                "message": f"Invalid JSON in API response: {str(exc)[:200]}",
            }
        except Exception as exc:
            return {"status": "ERROR", "message": str(exc)[:300]}

    return {"status": "ERROR", "message": f"Exhausted retries: {last_err}"}


def _safe_get(d: dict, *keys: str, default: Any = "") -> Any:
    """Safely traverse nested dict."""
    cur: Any = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k, default)
        else:
            return default
    return cur


# ═══════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════

def run(
    api_url: str,
    csv_path: Path,
    timeout: int,
    batch: int = 0,
    ids: list[str] | None = None,
    force: bool = False,
) -> None:
    # ── Load CSV ──────────────────────────────────────────────────────
    if not csv_path.exists():
        sys.exit(f"ERROR: CSV not found: {csv_path}")

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows: list[dict[str, str]] = list(reader)

    n = len(all_rows)
    if n == 0:
        sys.exit("ERROR: CSV is empty — run Stage 1 first.")

    print(f"\n{'═' * 64}")
    print(f"  Tyr Stage 2 — BMC Verification")
    print(f"  CSV    : {csv_path.absolute()}")
    print(f"  Rows   : {n}")
    print(f"  API    : {api_url}")
    print(f"  Timeout: {timeout}s per problem")
    print(f"  Force  : {force}")
    print(f"{'═' * 64}\n")

    # ── Identify rows needing verification ────────────────────────────
    to_verify: list[int] = []  # indices into all_rows
    for i, row in enumerate(all_rows):
        # Skip rows with SYNTAX_ERROR or empty generated code
        api_status = row.get("api_status", "").strip()
        gen_code = row.get("generated_code", "").strip()
        if api_status == "SYNTAX_ERROR" or not gen_code:
            continue
        # Skip already-verified unless --force
        verdict = row.get("verdict", "").strip()
        if verdict and not force:
            continue
        to_verify.append(i)

    # ── Filter by --ids ───────────────────────────────────────────────
    if ids:
        id_set = set(ids)
        to_verify = [i for i in to_verify
                     if all_rows[i].get("id", "") in id_set]
        print(f"  🎯 Precision Mode: {len(to_verify)} matching IDs")

    # ── Batch slicing ─────────────────────────────────────────────────
    if batch > 0 and len(to_verify) > batch:
        to_verify = to_verify[:batch]
        print(f"  ⏱ Batch Mode: Verifying {batch} problems this run")

    skipped = n - len(to_verify)
    print(f"  → {len(to_verify)} problems to verify "
          f"({skipped} skipped — already verified or invalid)\n")

    if not to_verify:
        print("  Nothing to do. Use --force to re-verify all rows.")
        return

    # ── Counters ──────────────────────────────────────────────────────
    verdicts: dict[str, int] = {}
    category_pass: dict[str, list[int]] = {}
    latencies: list[float] = []
    improved_count = 0
    total_assessed = 0

    # ── Progress bar ──────────────────────────────────────────────────
    bar = tqdm(to_verify, desc="Verifying", unit="prob",
               bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]"))

    for idx in bar:
        row = all_rows[idx]
        pid  = row.get("id", "TYR-???")
        name = row.get("name", "unknown")
        cat  = row.get("category", "unknown")
        original_code = row.get("original_code", "")
        generated_code = row.get("generated_code", "")

        bar.set_postfix_str(f"{pid} {name[:22]}")

        try:
            if not original_code.strip() or not generated_code.strip():
                raise ValueError("Empty original_code or generated_code")

            # ── Call /verify-pair ─────────────────────────────────────
            t0 = time.perf_counter()
            data = _call_verify_pair(api_url, original_code,
                                     generated_code, timeout)
            verify_latency_ms = (time.perf_counter() - t0) * 1000.0

            verdict = data.get("status", "ERROR")
            opt_time = _safe_get(data, "optimized_complexity",
                                 "time", default="N/A")
            comp_imp = data.get("complexity_improved", None)

            if comp_imp is True:
                improved_count += 1

        except Exception as exc:
            verify_latency_ms = 0.0
            verdict = "ERROR"
            opt_time = "N/A"
            comp_imp = None
            print(f"\n  ⚠ {pid}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

        total_assessed += 1

        # ── Update the row in-memory ─────────────────────────────────
        all_rows[idx]["verdict"] = verdict
        all_rows[idx]["optimized_complexity_time"] = opt_time
        if comp_imp is not None:
            all_rows[idx]["complexity_improved"] = str(comp_imp)
        else:
            all_rows[idx]["complexity_improved"] = ""

        # ── Bookkeeping ──────────────────────────────────────────────
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        latencies.append(verify_latency_ms)
        is_pass = 1 if verdict == "UNSAT" else 0
        category_pass.setdefault(cat, []).append(is_pass)

        # ── Incremental save (write ENTIRE CSV after each row) ────────
        _write_csv(csv_path, all_rows)

        # ── Throttle ─────────────────────────────────────────────────
        time.sleep(INTER_REQUEST_DELAY)

    bar.close()

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    evaluated = len(to_verify)

    print(f"\n{'═' * 64}")
    print("  STAGE 2 RESULTS — VERDICT DISTRIBUTION")
    print(f"{'═' * 64}")
    for v in ["UNSAT", "SAT", "WARNING", "ERROR"]:
        cnt = verdicts.get(v, 0)
        pct = cnt / evaluated * 100 if evaluated else 0
        bar_str = "█" * int(pct / 2)
        print(f"  {v:8s}  {cnt:4d} / {evaluated}  "
              f"({pct:5.1f}%)  {bar_str}")
    total_pass = verdicts.get("UNSAT", 0)
    if evaluated:
        print(f"\n  PASS RATE: {total_pass}/{evaluated}"
              f" = {total_pass / evaluated * 100:.1f}%")

    # Category breakdown
    print(f"\n{'─' * 64}")
    print("  CATEGORY BREAKDOWN")
    print(f"{'─' * 64}")
    print(f"  {'Category':<25s}  {'Pass':>4s} / {'Tot':>3s}  {'Rate':>6s}")
    for cat in sorted(category_pass):
        passes = sum(category_pass[cat])
        total  = len(category_pass[cat])
        rate   = passes / total * 100 if total else 0
        print(f"  {cat:<25s}  {passes:4d} / {total:3d}  {rate:5.1f}%")

    # Latency stats
    if latencies:
        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p95_idx = int(len(sorted_lat) * 0.95)
        p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]
        print(f"\n{'─' * 64}")
        print("  VERIFICATION LATENCY")
        print(f"{'─' * 64}")
        print(f"  Mean : {statistics.mean(latencies):>10.1f} ms")
        print(f"  p50  : {p50:>10.1f} ms")
        print(f"  p95  : {p95:>10.1f} ms")
        print(f"  Min  : {min(latencies):>10.1f} ms")
        print(f"  Max  : {max(latencies):>10.1f} ms")

    # Complexity improvement
    if total_assessed:
        print(f"\n{'─' * 64}")
        print("  COMPLEXITY IMPROVEMENT")
        print(f"{'─' * 64}")
        print(f"  Improved: {improved_count}/{total_assessed}"
              f" ({improved_count / total_assessed * 100:.1f}%)")

    print(f"\n{'═' * 64}")
    print(f"  CSV updated → {csv_path}")
    print(f"{'═' * 64}\n")


def _write_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    """Atomically write all rows to CSV (write tmp → rename)."""
    tmp = csv_path.with_suffix(".csv.tmp")
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        # Atomic replace (Windows: need to remove first)
        if csv_path.exists():
            csv_path.unlink()
        shutil.move(str(tmp), str(csv_path))
    except Exception as exc:
        print(f"\n  ⚠ CSV write failed: {exc}", file=sys.stderr)
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tyr Stage 2 — BMC Verification of LLM Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", default=str(DEFAULT_CSV),
                    help=f"Path to llm_results.csv (default: {DEFAULT_CSV})")
    ap.add_argument("--url", default=DEFAULT_API_URL,
                    help=f"Tyr /verify-pair endpoint (default: {DEFAULT_API_URL})")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help=f"Per-problem timeout in seconds (default: {DEFAULT_TIMEOUT})")
    ap.add_argument("--batch", type=int, default=0,
                    help="Max problems to verify this run (0 = all)")
    ap.add_argument("--ids", type=str, default=None,
                    help="Comma-separated problem IDs (e.g. TYR-001,TYR-005)")
    ap.add_argument("--force", action="store_true",
                    help="Re-verify rows that already have a verdict")

    args = ap.parse_args()

    parsed_ids: list[str] | None = None
    if args.ids:
        parsed_ids = [x.strip() for x in args.ids.split(",") if x.strip()]

    run(
        api_url=args.url,
        csv_path=Path(args.csv),
        timeout=args.timeout,
        batch=args.batch,
        ids=parsed_ids,
        force=args.force,
    )


if __name__ == "__main__":
    main()
