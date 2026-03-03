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
    verdict, optimized_complexity_time, complexity_improved, verify_latency_ms

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
import os
import re
import sys
import time
import threading
import traceback
from pathlib import Path
from typing import Any
import multiprocessing

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
    "latency_ms", "prompt_tokens", "reasoning_tokens",
    "completion_tokens", "total_tokens",
    "api_status", "error_detail",
]

# Stage 2 adds these columns
_STAGE2_EXTRA: list[str] = [
    "verdict",
    "optimized_complexity_time",
    "complexity_improved",
    "verify_latency_ms",
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
        # Recursion + loop nesting → exponential (e.g., naive fib with loop)
        # Recursion alone → conservatively O(2^n); callers should note this
        # heuristic cannot detect memoization or tail-call patterns.
        if max_depth >= 2:
            return "O(2^n)"
        if max_depth == 1:
            return "O(n^2)"   # recursion inside single loop
        return "O(2^n)"      # bare recursion (worst-case assumption)

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
# ASCII Art Banner
# ═══════════════════════════════════════════════════════════════════════

_BANNER_ART: tuple[str, ...] = (
    "████████╗██╗   ██╗██████╗     ██╗   ██╗███████╗██████╗ ██╗███████╗██╗ ██████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗",
    "╚══██╔══╝╚██╗ ██╔╝██╔══██╗    ██║   ██║██╔════╝██╔══██╗██║██╔════╝██║██╔════╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║",
    "   ██║    ╚████╔╝ ██████╔╝    ██║   ██║█████╗  ██████╔╝██║█████╗  ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║",
    "   ██║     ╚██╔╝  ██╔══██╗    ╚██╗ ██╔╝██╔══╝  ██╔══██╗██║██╔══╝  ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║",
    "   ██║      ██║   ██║  ██║     ╚████╔╝ ███████╗██║  ██║██║██║     ██║╚██████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║",
    "   ╚═╝      ╚═╝   ╚═╝  ╚═╝      ╚═══╝  ╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝",
)


def _print_banner(
    *,
    model: str,
    n_rows: int,
    n_done: int,
    backend: str,
    force: bool,
    csv_path: Path | str,
    out_path: Path | str,
) -> None:
    """Print the TYR VERIFICATION ASCII art banner with run details."""
    art_widths  = [len("   " + a) for a in _BANNER_ART]
    info_widths = [
        len(f"   FORMAL VERIFICATION ENGINE  │  Stage 2  │  {n_rows} problems"),
        len(f"   Output  : {out_path}"),
    ]
    inner = max(*art_widths, *info_widths) + 3
    inner = max(inner, 78)

    def row(text: str = "") -> str:
        return "║" + (f"   {text}" if text else "").ljust(inner) + "║"

    top = "╔" + "═" * inner + "╗"
    bot = "╚" + "═" * inner + "╝"
    sep = "║" + "─" * inner + "║"

    lines = [
        "",
        top,
        row(),
        *(row(a) for a in _BANNER_ART),
        row(),
        sep,
        row(f"FORMAL VERIFICATION ENGINE  │  Stage 2  │  {n_rows} problems"),
        sep,
        row(f"Model   : {model}"),
        row(f"Backend : {backend:<20s}  Force : {force}"),
        row(f"Progress: {n_done}/{n_rows} done"),
        row(f"Input   : {csv_path}"),
        row(f"Output  : {out_path}"),
        row(),
        bot,
        "",
    ]
    print("\n".join(lines))


def _print_art_header(title: str) -> None:
    """Print a lighter banner with just the ASCII art and a title line."""
    art_widths = [len("   " + a) for a in _BANNER_ART]
    inner = max(*art_widths, len(f"   {title}")) + 3
    inner = max(inner, 78)

    def row(text: str = "") -> str:
        return "║" + (f"   {text}" if text else "").ljust(inner) + "║"

    top = "╔" + "═" * inner + "╗"
    bot = "╚" + "═" * inner + "╝"
    sep = "║" + "─" * inner + "║"

    lines = [
        "",
        top,
        row(),
        *(row(a) for a in _BANNER_ART),
        row(),
        sep,
        row(title),
        bot,
    ]
    print("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# Keyboard / Terminal helpers  (shared with interactive picker)
# ═══════════════════════════════════════════════════════════════════════

def _read_key() -> str:
    """Read a single keypress. Returns 'UP', 'DOWN', 'ENTER', 'Q', etc."""
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            return {"H": "UP", "P": "DOWN"}.get(ch2, "")
        if ch == "\r":
            return "ENTER"
        if ch == "\x1b":
            return "Q"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch.upper()
    else:
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                return {"[A": "UP", "[B": "DOWN"}.get(seq, "")
            if ch in ("\r", "\n"):
                return "ENTER"
            if ch == "\x03":
                raise KeyboardInterrupt
            return ch.upper()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _get_terminal_width() -> int:
    """Get the current terminal width, default 120."""
    try:
        return os.get_terminal_size().columns
    except (OSError, ValueError):
        return 120


def _clear_screen() -> None:
    """Clear the entire terminal screen reliably across platforms."""
    if sys.platform == "win32":
        os.system("cls")
    else:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════
# Interactive file picker  (shown when run with no arguments)
# ═══════════════════════════════════════════════════════════════════════

def _interactive_file_picker(raw_dir: Path) -> str | None:
    """Arrow-key driven interactive menu to pick a CSV file to verify."""

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        print(f"\n  ⚠  No CSV files found in {raw_dir}")
        return None

    # ── Build menu items ───────────────────────────────────────────
    items: list[dict] = []

    # Header
    items.append({"label": "RAW STAGE 1 CSVs ─ Select a File to Verify",
                  "value": None, "is_header": True, "disabled": True})

    for csv_path in csv_files:
        # Extract model name from the CSV if possible
        model = csv_path.stem  # e.g. "github_gpt_4_1"
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                first = next(reader, None)
                if first and first.get("model_name"):
                    model = first["model_name"]
                rows = sum(1 for _ in fh) + (1 if first else 0)
        except Exception:
            rows = 0

        label = f"{csv_path.name:<45s}  Model: {model:<30s}  Rows: {rows}"
        items.append({"label": label, "value": str(csv_path),
                      "is_header": False, "disabled": False})

    # Batch options
    items.append({"label": "BATCH OPTIONS", "value": None,
                  "is_header": True, "disabled": True})
    items.append({"label": f"Verify ALL {len(csv_files)} files in {raw_dir.name}/",
                  "value": "__ALL__", "is_header": False, "disabled": False})
    items.append({"label": "Quit",
                  "value": "__QUIT__", "is_header": False, "disabled": False})

    # Selectable indices
    selectable = [i for i, it in enumerate(items) if not it["disabled"]]
    cursor_idx = 0

    def _draw() -> None:
        _clear_screen()
        _print_art_header("Select a File to Verify")

        tw = _get_terminal_width()
        cur = selectable[cursor_idx]

        for i, it in enumerate(items):
            if it["is_header"]:
                print()
                print(f"  \033[1;36m{it['label']}\033[0m")
                print(f"  {'─' * min(74, tw - 4)}")
                continue

            is_cur = (i == cur)
            marker = " ▸ " if is_cur else "   "

            raw_line = f"  {marker}{it['label']}"
            if len(raw_line) > tw - 1:
                raw_line = raw_line[: tw - 4] + "..."

            if it["disabled"]:
                print(f"\033[90m{raw_line}\033[0m")
            elif is_cur:
                print(f"\033[1;33m{raw_line}\033[0m")
            else:
                print(raw_line)

        print()
        print(f"  \033[2m↑/↓ Navigate  •  Enter Select  •  Q/Esc Quit\033[0m")

    # ── Interaction loop ───────────────────────────────────────────
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    try:
        _draw()

        while True:
            key = _read_key()
            if key == "UP":
                cursor_idx = (cursor_idx - 1) % len(selectable)
                _draw()
            elif key == "DOWN":
                cursor_idx = (cursor_idx + 1) % len(selectable)
                _draw()
            elif key == "ENTER":
                break
            elif key == "Q":
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()
                print("\n  Bye!")
                return None

    except (KeyboardInterrupt, EOFError):
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        print("\n  Aborted.")
        return None

    sys.stdout.write("\033[?25h")
    sys.stdout.flush()

    selected = items[selectable[cursor_idx]]
    value = selected["value"]

    if value == "__QUIT__":
        print("\n  Bye!")
        return None

    if value == "__ALL__":
        print(f"\n  ✔ Selected: Verify all {len(csv_files)} files\n")
        return "__ALL__"

    print(f"\n  ✔ Selected: {Path(value).name}\n")
    return value


# ═══════════════════════════════════════════════════════════════════════
# Verification backend
# ═══════════════════════════════════════════════════════════════════════

# Per-problem timeout for the in-process Z3 solver (seconds).
# Z3's internal timeout is best-effort; this thread-level kill switch
# guarantees the pipeline never hangs on a single problem.
_DIRECT_VERIFY_TIMEOUT_S: int = 120


def _verify_worker(original_code: str, generated_code: str, result_queue):
    """Worker function that runs in a completely separate OS process.

    Uses a multiprocessing.Queue instead of Manager.dict() — avoids
    spawning a background Manager server process for every single
    verification call (was: 3000+ leaked processes in a full run).
    """
    try:
        res = _verify_fn(original_code, generated_code)
        result_queue.put({"ok": True, "data": res})
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})


def _verify_direct(original_code: str, generated_code: str, timeout: int) -> dict[str, Any]:
    """
    Call verify_equivalence() using multiprocessing + Queue.
    Guarantees that hanging Z3 SMT solvers are forcefully killed.

    Uses a lightweight multiprocessing.Queue (pipe-based) instead of
    Manager.dict() — eliminates the per-call Manager server process
    that previously leaked ~3000 child processes across a full run.
    """
    result_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)

    p = multiprocessing.Process(
        target=_verify_worker,
        args=(original_code, generated_code, result_queue),
    )
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join(timeout=10)  # bounded wait after kill — never hang
        if p.is_alive():
            p.kill()       # SIGKILL / TerminateProcess hard kill
            p.join(timeout=5)
        tqdm.write(
            f"    ⏱  verify_equivalence exceeded {timeout}s "
            f"hard timeout — Process Terminated (TIMEOUT)"
        )
        # Drain queue to prevent broken-pipe warnings
        try:
            result_queue.get_nowait()
        except Exception:
            pass
        result_queue.close()
        result_queue.join_thread()
        return {"verdict": "TIMEOUT", "error_detail": f"Z3 SMT Solver Hung > {timeout}s"}

    # Process finished — retrieve result from queue
    try:
        msg = result_queue.get_nowait()
    except Exception:
        result_queue.close()
        result_queue.join_thread()
        return {"verdict": "ERROR", "error_detail": "Process died without returning data"}

    result_queue.close()
    result_queue.join_thread()

    if not msg.get("ok"):
        return {"verdict": "ERROR", "error_detail": msg.get("error", "Unknown worker error")}

    data = msg["data"]
    verdict = data.get("status", "ERROR")
    if verdict not in ("UNSAT", "SAT", "WARNING", "ERROR", "TIMEOUT"):
        verdict = "WARNING"

    return {
        "verdict": verdict,
        "error_detail": data.get("message", ""),
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
            if verdict not in ("UNSAT", "SAT", "WARNING", "ERROR", "TIMEOUT"):
                verdict = "WARNING"
            result: dict[str, Any] = {"verdict": verdict}
            if data.get("message"):
                result["error_detail"] = data["message"]
            return result
        except Exception as exc:
            if attempt < MAX_HTTP_RETRIES:
                time.sleep(5 * (attempt + 1))
                continue
            return {"verdict": "ERROR", "error_detail": f"HTTP error: {exc}"}

    return {"verdict": "ERROR", "error_detail": "Max HTTP retries exhausted"}


def verify_pair(
    original_code: str,
    generated_code: str,
    api_url: str | None,
    timeout: int,
) -> dict[str, Any]:
    """Route to direct or HTTP backend."""
    if _DIRECT_IMPORT_OK and not api_url:
        return _verify_direct(original_code, generated_code, timeout)
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
    """Write CSV atomically via tmp file → os.replace (crash-safe).

    Uses ``os.replace()`` which is atomic on BOTH POSIX and Windows
    (MoveFileExW + MOVEFILE_REPLACE_EXISTING).  The previous
    ``unlink()`` + ``shutil.move()`` pattern had a TOCTOU race: if the
    process was killed between the two calls, the verified CSV was
    deleted with no replacement — permanent data loss.
    """
    tmp = path.with_suffix(".csv.tmp")
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=CSV_COLUMNS, extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())  # guarantee data is on disk before rename
        # os.replace() is atomic on all platforms — no TOCTOU gap.
        os.replace(str(tmp), str(path))
        # Fsync the directory on POSIX to persist the rename metadata.
        # Without this, ext4/XFS may lose the directory entry on crash.
        # On Windows (NTFS) this is a no-op as metadata is journalled.
        if sys.platform != "win32":
            try:
                dirfd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(dirfd)
                finally:
                    os.close(dirfd)
            except OSError:
                pass  # best-effort; Windows may not support O_RDONLY on dirs
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
    raw_rows = _read_raw_csv(raw_path)
    if not raw_rows:
        print(f"\n  ⚠  SKIP: {raw_path.name} is empty.")
        return {}

    model_name = raw_rows[0].get("model_name", raw_path.stem)
    out_path   = _verified_path(raw_path, verified_dir)

    existing = _load_verified_dict(out_path) if not force else {}

    backend_str = 'DIRECT' if (_DIRECT_IMPORT_OK and not api_url) else f'HTTP ({api_url or DEFAULT_API_URL})'
    _print_banner(
        model=model_name,
        n_rows=len(raw_rows),
        n_done=len(existing),
        backend=backend_str,
        force=force,
        csv_path=raw_path,
        out_path=out_path,
    )

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
            row["verify_latency_ms"]         = "0.00"
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
            row["verify_latency_ms"] = f"{verify_ms:.2f}"

            # Capture verification error detail (TIMEOUT / ERROR reason)
            if vr.get("error_detail"):
                row["error_detail"] = vr["error_detail"]

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

    timeout_n = verdicts.get('TIMEOUT', 0)
    print(
        f"\n  UNSAT   : {verdicts.get('UNSAT', 0)}/{total}  ({rate:.1f}% pass)\n"
        f"  SAT     : {verdicts.get('SAT', 0)}\n"
        f"  WARNING : {verdicts.get('WARNING', 0)}\n"
        f"  TIMEOUT : {timeout_n}\n"
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
    grp = ap.add_mutually_exclusive_group(required=False)
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

    # ── Interactive picker when no --input / --all is provided ─────
    if not args.input and not args.all:
        pick = _interactive_file_picker(raw_dir)
        if not pick:
            return
        if pick == "__ALL__":
            args.all = True
        else:
            args.input = pick

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

    backend_str = 'DIRECT' if (_DIRECT_IMPORT_OK and not args.url) else f'HTTP ({args.url or DEFAULT_API_URL})'

    _print_art_header(
        f"Tyr Stage 2 — Formal Verification  │  "
        f"{len(targets)} file{'s' if len(targets) > 1 else ''}  │  "
        f"Backend: {backend_str}"
    )

    print(
        f"\n  Force   : {args.force}\n"
        f"  Timeout : {args.timeout}s\n"
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
