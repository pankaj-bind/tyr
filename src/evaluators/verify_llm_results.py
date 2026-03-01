#!/usr/bin/env python3
"""
Tyr — Unified Stage 2: Formal Verification of LLM-Generated Code
=================================================================
Reads Stage 1 CSV(s) from ``data/raw/``, runs the Tyr Bounded Model
Checking verifier on each (original_code, generated_code) pair, and
writes the fully annotated results to ``data/verified/``.

Verification can be called two ways (auto-detected):
  1. Direct import — no server needed (default when backend/ is importable)
  2. HTTP  — POST to /verify-pair (when --url is passed or import fails)

Stage 2 writes the IDENTICAL Stage 1 columns PLUS:
    verdict, optimized_complexity_time, complexity_improved, verify_error

verdict values: UNSAT | SAT | WARNING | ERROR

Usage
~~~~~
    # Verify a single raw CSV
    python src/evaluators/verify_llm_results.py \\
        --input data/raw/github_gpt_4o.csv

    # Verify all CSVs in data/raw/ (batch mode)
    python src/evaluators/verify_llm_results.py --all

    # Use HTTP backend instead of direct import
    python src/evaluators/verify_llm_results.py --all \\
        --url http://localhost:8000/verify-pair

    # Re-verify rows already verified
    python src/evaluators/verify_llm_results.py --input <csv> --force
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ─────────────────────────── PATH SETUP ──────────────────────────────
_SRC_DIR      = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent

for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: `tqdm` is required.  pip install tqdm")

# ─────────────────────────── VERIFIER IMPORT ─────────────────────────
# Try direct import first (zero-overhead, no server required).
# Falls back to HTTP if the backend isn't importable (e.g., z3 missing).
_DIRECT_IMPORT_OK = False
try:
    from backend.verifier.equivalence import verify_equivalence as _verify_fn
    _DIRECT_IMPORT_OK = True
except ImportError:
    try:
        from verifier.equivalence import verify_equivalence as _verify_fn  # type: ignore
        _DIRECT_IMPORT_OK = True
    except ImportError:
        pass  # will use HTTP


# ═══════════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════════

# Stage 1 columns (must match generate_llm_benchmark.py exactly)
_STAGE1_COLS: list[str] = [
    "id", "name", "model_name", "category", "difficulty",
    "original_complexity", "target_complexity",
    "original_code", "generated_code",
    "latency_ms", "prompt_tokens", "reasoning_tokens", "total_tokens",
    "api_status", "error_detail",
]

# Stage 2 adds these columns
_STAGE2_EXTRA: list[str] = [
    "verdict",
    "optimized_complexity_time",
    "complexity_improved",
]

CSV_COLUMNS: list[str] = _STAGE1_COLS + _STAGE2_EXTRA


# ═══════════════════════════════════════════════════════════════════════
# Complexity estimator (static AST-based — no LLM call required)
# ═══════════════════════════════════════════════════════════════════════
_COMPLEXITY_RANK: dict[str, int] = {
    "1": 0,
    "logn": 1,
    "sqrtn": 2,
    "n": 3,
    "nlogn": 4,
    "n^2": 5,
    "n^3": 6,
    "2^n": 7,
    "n!": 8,
}


def _normalize_complexity(s: str) -> str | None:
    """Normalize a Big-O string to a canonical form for ranking."""
    s = s.strip().lower()
    m = re.match(r"o\((.+)\)", s)
    if not m:
        return None
    inner = m.group(1).strip()
    inner = re.sub(r"\s+", "", inner)
    inner = inner.replace("\u00b2", "^2").replace("\u00b3", "^3")
    inner = inner.replace("**", "^")
    inner = inner.replace("*", "")
    inner = re.sub(r"log\(n\)", "logn", inner)
    inner = re.sub(r"sqrt\(n\)", "sqrtn", inner)
    return inner


def _compare_complexity(original: str, generated: str) -> str:
    """Compare two Big-O strings.

    Returns:
        "True"  — generated is strictly better (lower rank)
        "False" — generated is strictly worse  (higher rank)
        "Same"  — identical complexity
    """
    orig_n = _normalize_complexity(original)
    gen_n  = _normalize_complexity(generated)
    if orig_n is None or gen_n is None:
        return "Same"
    orig_r = _COMPLEXITY_RANK.get(orig_n)
    gen_r  = _COMPLEXITY_RANK.get(gen_n)
    if orig_r is None or gen_r is None:
        return "Same" if orig_n == gen_n else "Same"
    if gen_r < orig_r:
        return "True"
    if gen_r > orig_r:
        return "False"
    return "Same"


def estimate_complexity(code: str) -> str:
    """Heuristic AST-based time-complexity estimation.

    Analyses loop nesting depth, recursive calls, and built-in
    calls like sorted()/sort() to produce a Big-O estimate.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "N/A"

    func_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_names.add(node.name)

    max_depth = 0
    has_sort = False
    has_recursion = False
    has_log_pattern = False

    class _LoopVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.depth = 0

        def _enter_loop(self, node: ast.AST) -> None:
            nonlocal max_depth, has_log_pattern
            self.depth += 1
            if self.depth > max_depth:
                max_depth = self.depth
            if isinstance(node, ast.While):
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign):
                        if isinstance(child.op, (ast.FloorDiv, ast.RShift)):
                            has_log_pattern = True
            self.generic_visit(node)
            self.depth -= 1

        def visit_For(self, node: ast.For) -> None:
            self._enter_loop(node)

        def visit_While(self, node: ast.While) -> None:
            self._enter_loop(node)

        def visit_ListComp(self, node: ast.ListComp) -> None:
            nonlocal max_depth
            self.depth += len(node.generators)
            if self.depth > max_depth:
                max_depth = self.depth
            self.generic_visit(node)
            self.depth -= len(node.generators)

        def visit_SetComp(self, node: ast.SetComp) -> None:
            self.visit_ListComp(node)  # type: ignore[arg-type]

        def visit_DictComp(self, node: ast.DictComp) -> None:
            self.visit_ListComp(node)  # type: ignore[arg-type]

        def visit_Call(self, node: ast.Call) -> None:
            nonlocal has_sort, has_recursion
            fname = ""
            if isinstance(node.func, ast.Name):
                fname = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fname = node.func.attr
            if fname in ("sorted", "sort"):
                has_sort = True
            if fname in func_names:
                has_recursion = True
            self.generic_visit(node)

    _LoopVisitor().visit(tree)

    if has_recursion:
        if max_depth >= 1:
            return "O(2^n)"
        return "O(2^n)"

    if max_depth == 0 and not has_sort:
        if has_log_pattern:
            return "O(log n)"
        return "O(1)"

    if has_sort:
        if max_depth >= 1:
            return "O(n^2)"
        return "O(n log n)"

    if has_log_pattern:
        return "O(n log n)"

    depth_map = {1: "O(n)", 2: "O(n^2)", 3: "O(n^3)"}
    return depth_map.get(max_depth, f"O(n^{max_depth})")

DEFAULT_RAW_DIR      = _PROJECT_ROOT / "data" / "raw"
DEFAULT_VERIFIED_DIR = _PROJECT_ROOT / "data" / "verified"
DEFAULT_API_URL      = "http://localhost:8000/verify-pair"
DEFAULT_TIMEOUT      = 300
INTER_CALL_DELAY     = 0.5   # seconds between BMC calls (avoid Z3 starvation)
MAX_HTTP_RETRIES     = 3


# ═══════════════════════════════════════════════════════════════════════
# Verification backend
# ═══════════════════════════════════════════════════════════════════════

def _verify_direct(original_code: str, generated_code: str) -> dict[str, Any]:
    """
    Call verify_equivalence() in-process.
    latency_ms is NOT measured here — that's Stage 1's job.
    """
    try:
        result = _verify_fn(original_code, generated_code)
        verdict = result.get("status", "ERROR")
        if verdict not in ("UNSAT", "SAT", "WARNING", "ERROR"):
            verdict = "WARNING"
        return {
            "verdict": verdict,
        }
    except Exception:
        return {
            "verdict": "ERROR",
        }


def _verify_http(
    api_url: str,
    original_code: str,
    generated_code: str,
    timeout: int,
) -> dict[str, Any]:
    """POST to /verify-pair with retry logic."""
    try:
        import requests as _req
    except ImportError:
        sys.exit("ERROR: pip install requests")

    payload = {
        "original_code": original_code,
        "optimized_code": generated_code,
        "language": "python",
    }

    for attempt in range(1 + MAX_HTTP_RETRIES):
        try:
            resp = _req.post(api_url, json=payload, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(20 * (attempt + 1))
                continue
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            verdict = data.get("status", "ERROR")
            if verdict not in ("UNSAT", "SAT", "WARNING", "ERROR"):
                verdict = "WARNING"
            return {"verdict": verdict}
        except Exception as exc:
            if attempt < MAX_HTTP_RETRIES:
                time.sleep(5 * (attempt + 1))
                continue
            return {"verdict": "ERROR"}

    return {"verdict": "ERROR"}


def verify_pair(
    original_code: str,
    generated_code: str,
    api_url: str | None,
    timeout: int,
) -> dict[str, Any]:
    """Route to direct or HTTP backend."""
    if _DIRECT_IMPORT_OK and not api_url:
        return _verify_direct(original_code, generated_code)
    url = api_url or DEFAULT_API_URL
    return _verify_http(url, original_code, generated_code, timeout)


# ═══════════════════════════════════════════════════════════════════════
# CSV helpers
# ═══════════════════════════════════════════════════════════════════════

def _read_raw_csv(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _verified_path(raw_path: Path, verified_dir: Path) -> Path:
    verified_dir.mkdir(parents=True, exist_ok=True)
    return verified_dir / raw_path.name


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    """Write CSV atomically via tmp file → rename (crash-safe)."""
    tmp = path.with_suffix(".csv.tmp")
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=CSV_COLUMNS, extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        if path.exists():
            path.unlink()
        shutil.move(str(tmp), str(path))
    except Exception as exc:
        tqdm.write(f"\n  ⚠ CSV write failed: {exc}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _load_verified_dict(path: Path) -> dict[str, dict]:
    """Return {id → row} for rows already in the verified CSV."""
    if not path.exists():
        return {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return {r["id"]: r for r in csv.DictReader(fh) if r.get("id")}


# ═══════════════════════════════════════════════════════════════════════
# Per-file processor
# ═══════════════════════════════════════════════════════════════════════

def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    return s[f] + (k - f) * (s[min(f + 1, len(s) - 1)] - s[f])


def process_file(
    raw_path: Path,
    verified_dir: Path,
    api_url: str | None,
    timeout: int,
    force: bool,
) -> dict[str, int]:
    """
    Verify all rows in ``raw_path`` and write to ``verified_dir``.
    Returns counters dict.
    """
    print(f"\n{'═' * 70}")
    print(f"  Input   : {raw_path}")

    raw_rows = _read_raw_csv(raw_path)
    if not raw_rows:
        print("  SKIP: empty CSV.")
        return {}

    model_name = raw_rows[0].get("model_name", raw_path.stem)
    out_path   = _verified_path(raw_path, verified_dir)

    existing = _load_verified_dict(out_path) if not force else {}

    print(f"  Model   : {model_name}")
    print(f"  Rows    : {len(raw_rows)}")
    print(f"  Output  : {out_path}")
    print(f"  Mode    : {'DIRECT' if (_DIRECT_IMPORT_OK and not api_url) else f'HTTP ({api_url or DEFAULT_API_URL})'}")
    print(f"  Force   : {force}")
    print(f"{'─' * 70}\n")

    # Merge existing verified rows into all_rows
    all_rows: list[dict] = []
    for r in raw_rows:
        pid = r.get("id", "")
        if pid in existing:
            all_rows.append(existing[pid])
        else:
            merged = {col: r.get(col, "") for col in _STAGE1_COLS}
            for col in _STAGE2_EXTRA:
                merged[col] = ""
            all_rows.append(merged)

    # Which rows need verification?
    to_verify = [
        i for i, r in enumerate(all_rows)
        if (
            r.get("api_status", "") not in ("SYNTAX_ERROR",)
            and r.get("generated_code", "").strip()
            and (force or not r.get("verdict", "").strip())
        )
    ]

    print(f"  → {len(to_verify)} rows to verify "
          f"({len(all_rows) - len(to_verify)} already done or invalid)\n")

    if not to_verify:
        _write_csv_atomic(out_path, all_rows)
        return {"ok": 0, "err": 0, "skipped": len(all_rows)}

    # ── Progress bar — same format as Stage 1 ─────────────────────
    bar = tqdm(
        to_verify,
        unit="prob",
        ncols=100,
        bar_format=(
            f"  [{model_name}] {{l_bar}}{{bar}}| {{n_fmt}}/{{total_fmt}} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )

    verdicts: dict[str, int] = {}
    latencies: list[float] = []
    ok = err_count = 0

    for idx in bar:
        row  = all_rows[idx]
        pid  = row.get("id", "?")
        name = row.get("name", "?")

        bar.set_postfix(
            id=pid,
            verdict=row.get("verdict", "—") or "—",
            unsat=verdicts.get("UNSAT", 0),
            err=verdicts.get("ERROR", 0),
        )

        orig_code = row.get("original_code", "")
        gen_code  = row.get("generated_code", "")

        if not orig_code.strip() or not gen_code.strip():
            row["verdict"]                  = "ERROR"
            row["optimized_complexity_time"] = "N/A"
            row["complexity_improved"]       = ""
            if not row.get("error_detail", "").strip():
                row["error_detail"] = "None"
            verdicts["ERROR"] = verdicts.get("ERROR", 0) + 1
            err_count += 1
        else:
            t0 = time.perf_counter()
            vr = verify_pair(orig_code, gen_code, api_url, timeout)
            verify_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(verify_ms)

            row["verdict"] = vr["verdict"]

            # Compute complexity of generated code via AST analysis
            gen_cx = estimate_complexity(gen_code)
            row["optimized_complexity_time"] = gen_cx

            # Compare original vs generated complexity
            orig_cx = row.get("original_complexity", "")
            if gen_cx != "N/A" and orig_cx:
                row["complexity_improved"] = _compare_complexity(orig_cx, gen_cx)
            else:
                row["complexity_improved"] = "Same"

            # Normalize error_detail: empty → "None"
            if not row.get("error_detail", "").strip():
                row["error_detail"] = "None"

            verdicts[vr["verdict"]] = verdicts.get(vr["verdict"], 0) + 1
            if vr["verdict"] == "ERROR":
                err_count += 1
                tqdm.write(f"\n    ✖  {pid} ({name}): verification error")
            else:
                ok += 1

        # Atomic CSV write after every row — crash-safe
        _write_csv_atomic(out_path, all_rows)

        time.sleep(INTER_CALL_DELAY)

    bar.close()

    # ── Per-file summary ─────────────────────────────────────────
    total   = len(to_verify)
    skipped = len(all_rows) - total
    unsat   = verdicts.get("UNSAT", 0)
    rate    = unsat / total * 100 if total else 0.0

    avg = f"{sum(latencies)/len(latencies):.0f}" if latencies else "N/A"
    p50 = f"{_percentile(latencies, 50):.0f}" if latencies else "N/A"
    p95 = f"{_percentile(latencies, 95):.0f}" if latencies else "N/A"

    print(
        f"\n  UNSAT   : {verdicts.get('UNSAT', 0)}/{total}  ({rate:.1f}% pass)\n"
        f"  SAT     : {verdicts.get('SAT', 0)}\n"
        f"  WARNING : {verdicts.get('WARNING', 0)}\n"
        f"  ERROR   : {verdicts.get('ERROR', 0)}\n"
        f"  Skipped : {skipped}\n"
        f"  Verify latency  avg={avg}ms  p50={p50}ms  p95={p95}ms\n"
        f"  Output  → {out_path}\n"
        f"{'═' * 70}"
    )

    return {"ok": ok, "err": err_count, "skipped": skipped,
            "unsat": unsat, "total": total}


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Tyr Stage 2 — Formal Verification of LLM Results",
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--input",
        help="Path to a single Stage 1 CSV in data/raw/.",
    )
    grp.add_argument(
        "--all",
        action="store_true",
        help="Verify ALL *.csv files in data/raw/.",
    )
    ap.add_argument(
        "--raw-dir",
        default=str(DEFAULT_RAW_DIR),
        dest="raw_dir",
        help=f"Directory of raw Stage 1 CSVs (default: {DEFAULT_RAW_DIR}).",
    )
    ap.add_argument(
        "--verified-dir",
        default=str(DEFAULT_VERIFIED_DIR),
        dest="verified_dir",
        help=f"Output directory (default: {DEFAULT_VERIFIED_DIR}).",
    )
    ap.add_argument(
        "--url",
        default=None,
        help=(
            "Force HTTP mode: POST to /verify-pair at this URL. "
            "If omitted, direct import is used when available."
        ),
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-problem timeout for HTTP mode, seconds (default: {DEFAULT_TIMEOUT}).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-verify rows that already have a verdict.",
    )
    return ap.parse_args()


def main() -> None:
    args         = parse_args()
    verified_dir = Path(args.verified_dir)
    raw_dir      = Path(args.raw_dir)

    if args.input:
        targets = [Path(args.input)]
    else:
        targets = sorted(raw_dir.glob("*.csv"))
        if not targets:
            sys.exit(f"ERROR: No CSV files found in {raw_dir}")

    if not _DIRECT_IMPORT_OK and not args.url:
        print(
            "  ℹ  Direct import unavailable (z3 not installed?). "
            f"Falling back to HTTP → {DEFAULT_API_URL}.\n"
            "     Start the Tyr backend first, or pass --url.\n"
        )

    print(
        f"\n{'═' * 70}\n"
        f"  Tyr Stage 2 — Formal Verification\n"
        f"  Backend : {'DIRECT' if (_DIRECT_IMPORT_OK and not args.url) else f'HTTP ({args.url or DEFAULT_API_URL})'}\n"
        f"  Files   : {len(targets)}\n"
        f"  Force   : {args.force}\n"
        f"{'═' * 70}"
    )

    grand: dict[str, int] = {
        "ok": 0, "err": 0, "skipped": 0, "unsat": 0, "total": 0,
    }
    for raw_path in targets:
        counts = process_file(
            raw_path=raw_path,
            verified_dir=verified_dir,
            api_url=args.url,
            timeout=args.timeout,
            force=args.force,
        )
        for k, v in counts.items():
            grand[k] = grand.get(k, 0) + v

    if len(targets) > 1:
        total = grand.get("total", 0)
        unsat = grand.get("unsat", 0)
        print(
            f"\n{'═' * 70}\n"
            f"  GRAND TOTAL — {len(targets)} files\n"
            f"  UNSAT   : {unsat}/{total} ({unsat/total*100:.1f}%)\n"
            f"  Errors  : {grand.get('err', 0)}\n"
            f"  Skipped : {grand.get('skipped', 0)}\n"
            f"{'═' * 70}\n"
        )


if __name__ == "__main__":
    main()
