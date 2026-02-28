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
DEFAULT_CSV_OUT   = _PROJECT_ROOT / "Research Paper" / "data" / "paper_results_150.csv"
DEFAULT_API_URL   = "http://localhost:8000/verify"
DEFAULT_TIMEOUT   = 120            # seconds per problem
MAX_RETRIES       = 2              # retries on transient failure
RETRY_BACKOFF     = 3              # seconds between retries

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
    """POST to /verify with retry logic.  Returns parsed JSON or error dict."""
    payload = {"code": code, "language": "python"}
    last_err: Exception | None = None

    for attempt in range(1 + MAX_RETRIES):
        try:
            resp = requests.post(api_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            return {"status": "TIMEOUT", "message": f"Request timed out ({timeout}s)"}
        except requests.exceptions.ConnectionError as exc:
            last_err = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
        except requests.exceptions.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", 0)
            if status_code in (502, 503, 504) and attempt < MAX_RETRIES:
                last_err = exc
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            return {
                "status": "ERROR",
                "message": f"HTTP {status_code}: {str(exc)[:200]}",
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

def run(api_url: str, benchmark_path: Path, csv_out: Path, timeout: int) -> None:
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
    print(f"  Output: {csv_out}")
    print(f"  Timeout: {timeout}s per problem")
    print(f"{'═' * 64}\n")

    # ── Prepare CSV (truncate + write header) ─────────────────────────
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()

    # ── Counters ──────────────────────────────────────────────────────
    verdicts: dict[str, int] = {}
    category_pass: dict[str, list[int]] = {}   # cat -> [1/0 per problem]
    latencies: list[float] = []
    improved_count = 0
    total_assessed = 0

    # ── Progress bar ──────────────────────────────────────────────────
    bar = tqdm(problems, desc="Evaluating", unit="prob",
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

    bar.close()

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 64}")
    print("  BENCHMARK RESULTS — VERDICT DISTRIBUTION")
    print(f"{'═' * 64}")
    for v in ["UNSAT", "SAT", "WARNING", "TIMEOUT", "ERROR"]:
        cnt = verdicts.get(v, 0)
        pct = cnt / n * 100 if n else 0
        bar_str = "█" * int(pct / 2)
        print(f"  {v:8s}  {cnt:4d} / {n}  ({pct:5.1f}%)  {bar_str}")
    total_pass = verdicts.get("UNSAT", 0)
    print(f"\n  PASS RATE: {total_pass}/{n} = {total_pass / n * 100:.1f}%")

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
    args = ap.parse_args()

    run(
        api_url=args.url,
        benchmark_path=Path(args.bench),
        csv_out=Path(args.out),
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
