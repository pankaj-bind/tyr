"""
Tyr — Enterprise-Grade Benchmark Evaluator
=============================================
Drives ``tyr_benchmark_150.json`` through the CGSC pipeline (``POST /verify``)
and emits a detailed CSV + CLI summary suitable for ICSE / PLDI paper tables.

Key Features
~~~~~~~~~~~~
* **Incremental CSV** — each result row is flushed immediately; safe to Ctrl-C.
* **``tqdm`` progress bar** — real-time ETA, latency, pass-rate.
* **Robust error handling** — one bad problem never crashes the run.
* **Retry with back-off** — transient network / 503 failures are retried twice.
* **Summary statistics** — category-level breakdown, verdict distribution,
  mean / p50 / p95 latency, complexity-improvement rate.

Usage
~~~~~
    # 1. Start the Tyr backend
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

    # 2. Run the evaluator
    python evaluate_dataset.py                         # defaults
    python evaluate_dataset.py --url http://X:8000     # custom host
    python evaluate_dataset.py --timeout 180           # per-problem timeout
    python evaluate_dataset.py --out my_results.csv    # custom output path

Requires
~~~~~~~~
    pip install requests tqdm
"""
from __future__ import annotations

import argparse
import csv
import json
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

DEFAULT_BENCHMARK = Path(__file__).resolve().parent / "tyr_benchmark_150.json"
_PROJECT_ROOT     = Path(__file__).resolve().parent.parent
DEFAULT_CSV_OUT   = _PROJECT_ROOT / "Research_Paper" / "data" / "paper_results_150.csv"
DEFAULT_API_URL   = "http://localhost:8000/verify"
DEFAULT_TIMEOUT   = 600            # seconds per problem (server may sleep 100s+ on 429)
MAX_RETRIES       = 4              # retries on transient / rate-limit failures
RETRY_BACKOFF     = 5              # base seconds between retries
INTER_REQUEST_DELAY = 12           # seconds between problems (~5 req/min free-tier cap)

CSV_FIELDS = [
    "id",
    "name",
    "category",
    "difficulty",
    "original_complexity",
    "target_complexity",
    "verdict",
    "optimized_complexity_time",
    "complexity_improved",
    "total_rounds",
    "message",
    "has_counterexample",
    "latency_ms",
]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _call_verify(api_url: str, code: str, timeout: int) -> dict[str, Any]:
    """POST to /verify with retry logic.  Returns parsed JSON or error dict.

    Handles 429 (rate limit) with aggressive back-off so the benchmark
    never crashes due to Groq free-tier throttling.
    """
    payload = {"code": code, "language": "python"}
    last_err: Exception | None = None

    for attempt in range(1 + MAX_RETRIES):
        try:
            resp = requests.post(api_url, json=payload, timeout=timeout)

            # ── Rate-limited by backend / Groq ───────────────────
            if resp.status_code == 429:
                wait = 20 * (attempt + 1)
                print(f"\n  ⚠ 429 Rate Limited — sleeping {wait}s "
                      f"(attempt {attempt + 1}/{1 + MAX_RETRIES})",
                      file=sys.stderr)
                last_err = Exception(f"HTTP 429 rate limited")
                time.sleep(wait)
                continue

            # ── Server errors (often Groq 429 surfacing as 502) ────
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
            return {"status": "TIMEOUT",
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
# Main evaluator
# ═══════════════════════════════════════════════════════════════════════

def run(api_url: str, benchmark_path: Path, csv_out: Path, timeout: int,
       batch: int = 0, ids: list[str] | None = None,
       index_range: tuple[int, int] | None = None) -> None:
    # ── Load dataset ──────────────────────────────────────────────────
    if not benchmark_path.exists():
        sys.exit(f"ERROR: Benchmark file not found: {benchmark_path}\n"
                 f"       Run  build_dataset.py  first.")

    with open(benchmark_path, encoding="utf-8") as f:
        problems: list[dict] = json.load(f)

    n = len(problems)
    print(f"\n{'═' * 64}")
    print(f"  Tyr Benchmark Evaluator  —  {n} problems")
    print(f"  API   : {api_url}")
    print(f"  Output: {csv_out.absolute()}")
    print(f"  Timeout: {timeout}s per problem")
    print(f"{'═' * 64}\n")

    # ── Resume logic: detect already-evaluated problems ───────────────
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    resuming = False

    if csv_out.exists() and csv_out.stat().st_size > 0:
        try:
            with open(csv_out, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = row.get("id", "")
                    if rid:
                        done_ids.add(rid)
            resuming = bool(done_ids)
        except Exception as exc:
            print(f"  ⚠ Could not parse existing CSV ({exc}) — starting fresh",
                  file=sys.stderr)
            done_ids.clear()

    if resuming:
        remaining = [p for p in problems if p.get("id") not in done_ids]
        print(f"  ↻ Resuming benchmark: Skipping {len(done_ids)} already "
              f"evaluated problems. {len(remaining)} remaining.")
    else:
        remaining = problems
        # Fresh start — write header
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
        print(f"  ✓ Fresh run — CSV initialised with header")

    # ── Precision filter: --ids or --range ────────────────────────────
    if ids:
        id_set = set(ids)
        remaining = [p for p in remaining if p.get("id") in id_set]
        matched = [p.get("id") for p in remaining]
        missing = id_set - set(matched)
        print(f"  🎯 Precision Mode: Running specific IDs "
              f"{matched}")
        if missing:
            print(f"     ⚠ IDs not found or already done: "
                  f"{sorted(missing)}")
    elif index_range:
        start, end = index_range
        # Apply range to the FULL problem list (1-based), then intersect
        # with remaining (resume-filtered) to skip already-done entries.
        ranged_ids = {
            p.get("id") for p in problems[start - 1 : end]
        }
        remaining = [p for p in remaining if p.get("id") in ranged_ids]
        print(f"  📍 Range Mode: Running problems {start} to {end} "
              f"({len(remaining)} after resume filter).")

    # ── Batch slicing ─────────────────────────────────────────────────
    if batch > 0 and len(remaining) > batch:
        deferred = len(remaining) - batch
        remaining = remaining[:batch]
        print(f"  ⏱ Batch Mode: Executing the next {batch} problems. "
              f"{deferred} will be left for later.")
    elif batch > 0:
        print(f"  ⏱ Batch Mode: {len(remaining)} problems remaining "
              f"(≤ batch size {batch}). This is the final batch.")

    # ── Counters ──────────────────────────────────────────────────────
    verdicts: dict[str, int] = {}
    category_pass: dict[str, list[int]] = {}   # cat -> [1/0 per problem]
    latencies: list[float] = []
    improved_count = 0
    total_assessed = 0

    # ── Progress bar ──────────────────────────────────────────────────
    bar = tqdm(remaining, desc="Evaluating", unit="prob",
               bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]"))

    for problem in bar:
        # ── Safe field extraction (tolerates malformed entries) ────────
        pid  = problem.get("id", f"TYR-???")
        name = problem.get("name", "unknown")
        cat  = problem.get("category", "unknown")
        diff = problem.get("difficulty", "unknown")
        oc   = problem.get("original_complexity", "N/A")
        tc   = problem.get("target_complexity", "N/A")
        code = problem.get("original_code", "")

        bar.set_postfix_str(f"{pid} {name[:22]}")

        try:
            if not code.strip():
                raise ValueError("Empty or missing original_code")

            # ── Call API ──────────────────────────────────────────────
            t0 = time.perf_counter()
            data = _call_verify(api_url, code, timeout)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            verdict = data.get("status", "ERROR")
            msg     = str(data.get("message", ""))[:250].replace("\n", " ")
            has_cx  = bool(data.get("counterexample"))
            rounds  = data.get("total_rounds", 0)

            # Complexity
            opt_time = _safe_get(data, "optimized_complexity", "time",
                                 default="N/A")
            comp_imp = data.get("complexity_improved", None)
            if comp_imp is True:
                improved_count += 1

        except Exception as exc:
            # ── Impenetrable guard: log ERROR, never crash ────────────
            latency_ms = 0.0
            verdict    = "ERROR"
            msg        = (f"Script exception: {type(exc).__name__}: "
                          f"{str(exc)[:200]}")
            has_cx     = False
            rounds     = 0
            opt_time   = "N/A"
            comp_imp   = None

        total_assessed += 1

        # ── Build row ────────────────────────────────────────────────
        row = {
            "id":                      pid,
            "name":                    name,
            "category":                cat,
            "difficulty":              diff,
            "original_complexity":     oc,
            "target_complexity":       tc,
            "verdict":                 verdict,
            "optimized_complexity_time": opt_time,
            "complexity_improved":     comp_imp,
            "total_rounds":            rounds,
            "message":                 msg,
            "has_counterexample":      has_cx,
            "latency_ms":             f"{latency_ms:.1f}",
        }

        # ── Write row (append-mode, one-at-a-time) ───────────────────
        try:
            with open(csv_out, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS,
                                        extrasaction="ignore")
                writer.writerow(row)
        except Exception as write_exc:
            print(f"\n  WARNING: CSV write failed for {pid}: {write_exc}",
                  file=sys.stderr)

        # ── Bookkeeping ──────────────────────────────────────────────
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        latencies.append(latency_ms)

        is_pass = 1 if verdict == "UNSAT" else 0
        category_pass.setdefault(cat, []).append(is_pass)

        # ── Client-side throttle (~15 req/min for free tier) ─────────
        time.sleep(INTER_REQUEST_DELAY)

    bar.close()

    # ══════════════════════════════════════════════════════════════════
    # Summary (covers THIS run only; full results are in the CSV)
    # ══════════════════════════════════════════════════════════════════
    evaluated_this_run = len(remaining)
    total_in_csv = len(done_ids) + evaluated_this_run

    print(f"\n{'═' * 64}")
    print("  BENCHMARK RESULTS — VERDICT DISTRIBUTION (this run)")
    print(f"{'═' * 64}")
    for v in ["UNSAT", "SAT", "WARNING", "TIMEOUT", "ERROR"]:
        cnt = verdicts.get(v, 0)
        pct = cnt / evaluated_this_run * 100 if evaluated_this_run else 0
        bar_str = "█" * int(pct / 2)
        print(f"  {v:8s}  {cnt:4d} / {evaluated_this_run}  ({pct:5.1f}%)  {bar_str}")
    total_pass = verdicts.get("UNSAT", 0)
    print(f"\n  PASS RATE (this run): {total_pass}/{evaluated_this_run}"
          f" = {total_pass / evaluated_this_run * 100:.1f}%"
          if evaluated_this_run else "\n  No problems evaluated this run.")
    if resuming:
        print(f"  TOTAL IN CSV: {total_in_csv}/{n} problems evaluated")

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
        print("  LATENCY")
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
    print(f"  CSV saved → {csv_out}")
    print(f"{'═' * 64}\n")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tyr — Enterprise Benchmark Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--url",     default=DEFAULT_API_URL,
                    help=f"Tyr /verify endpoint (default: {DEFAULT_API_URL})")
    ap.add_argument("--bench",   default=str(DEFAULT_BENCHMARK),
                    help="Path to benchmark JSON")
    ap.add_argument("--out",     default=str(DEFAULT_CSV_OUT),
                    help="Output CSV path")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help=f"Per-problem timeout in seconds (default: {DEFAULT_TIMEOUT})")
    ap.add_argument("--batch",   type=int, default=0,
                    help="Max problems to evaluate this run (0 = all remaining)")

    precision = ap.add_mutually_exclusive_group()
    precision.add_argument(
        "--ids", type=str, default=None,
        help="Comma-separated problem IDs (e.g. TYR-001,TYR-005,TYR-010)")
    precision.add_argument(
        "--range", type=str, default=None, dest="idx_range",
        help="1-based start,end index range (e.g. 1,10)")

    args = ap.parse_args()

    # Parse --ids / --range into typed values
    parsed_ids: list[str] | None = None
    parsed_range: tuple[int, int] | None = None

    if args.ids:
        parsed_ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    if args.idx_range:
        parts = args.idx_range.split(",")
        if len(parts) != 2:
            ap.error("--range requires exactly two values: start,end (e.g. 1,10)")
        try:
            parsed_range = (int(parts[0]), int(parts[1]))
        except ValueError:
            ap.error("--range values must be integers")
        if parsed_range[0] < 1 or parsed_range[1] < parsed_range[0]:
            ap.error("--range: start must be >= 1 and end must be >= start")

    run(
        api_url=args.url,
        benchmark_path=Path(args.bench),
        csv_out=Path(args.out),
        timeout=args.timeout,
        batch=args.batch,
        ids=parsed_ids,
        index_range=parsed_range,
    )


if __name__ == "__main__":
    main()
