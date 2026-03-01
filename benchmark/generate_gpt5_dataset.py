#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║         GPT-5 Reasoning Pipeline  —  Tyr Stage 1 Generator            ║
║         GitHub Models Endpoint  |  Tier-1 ICSE/ASE Research           ║
╚══════════════════════════════════════════════════════════════════════════╝
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
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# ─────────────────────────── ENV ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / "backend" / ".env")

# ─────────────────────────── CONSTANTS ──────────────────────────────
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"

CSV_COLUMNS = [
    "id",
    "name",
    "category",
    "difficulty",
    "original_complexity",
    "target_complexity",
    "verdict",
    "original_code",
    "generated_code",
    "optimized_complexity_time",
    "complexity_improved",
    "latency_ms",
    "prompt_tokens",
    "reasoning_tokens",
    "total_tokens",
    "api_status",
    "error_detail",
]

PROMPT_TEMPLATE = (
    "You are an expert algorithm developer. Below is a naive Python "
    "implementation. Refactor this code to achieve a target time complexity "
    "of {target_complexity}. Provide ONLY the optimized Python code. "
    "Do not provide explanations. \n\nNaive Code:\n{original_code}"
)

MAX_RETRIES = 3
BACKOFF_BASE = 2

BANNER = r"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║      ██████╗ ██████╗ ████████╗   ███████╗    ██████╗ ██████╗           ║
║     ██╔════╝ ██╔══██╗╚══██╔══╝   ██╔════╝    ██╔══██╗██╔══██╗         ║
║     ██║  ███╗██████╔╝   ██║█████╗███████╗    ██████╔╝██████╔╝         ║
║     ██║   ██║██╔═══╝    ██║╚════╝╚════██║    ██╔═══╝ ██╔══██╗         ║
║     ╚██████╔╝██║        ██║      ███████║    ██║     ██║  ██║         ║
║      ╚═════╝ ╚═╝        ╚═╝      ╚══════╝    ╚═╝     ╚═╝  ╚═╝       ║
║                                                                        ║
║     GPT-5 REASONING PIPELINE  |  TIER-1 RESEARCH  |  {n_problems} PROBLEMS   ║
║     Model  : {model:<20s}  Keys : {n_keys} loaded                     ║
║     Delay  : {delay:.1f}s                                                    ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

SUMMARY_TPL = """
╔══════════════════════════════════════════════════════════════════════════╗
║                   GPT-5 REASONING — BENCHMARK SUMMARY                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Model               : {model:<48s} ║
║  Keys Used           : {n_keys:<48d}║
╠══════════════════════════════════════════════════════════════════════════╣
║  Total Problems      : {total:<48d}║
║  Processed (OK)      : {ok:<48d}║
║  Syntax Errors       : {syntax_errs:<48d}║
║  API Errors          : {errors:<48d}║
║  Skipped (resumed)   : {skipped:<48d}║
╠══════════════════════════════════════════════════════════════════════════╣
║  Wall-Clock Time     : {elapsed:<48s}║
║  Avg Latency (ms)    : {avg_latency:<48s}║
║  p50 Latency (ms)    : {p50_latency:<48s}║
║  p95 Latency (ms)    : {p95_latency:<48s}║
║  Avg Reasoning Tok   : {avg_reasoning:<48s}║
║  Output CSV          : {csv_out:<48s} ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


# ══════════════════════ KEY ROTATION ═══════════════════════════════
class KeyPool:
    """Round-robin API key pool with automatic rotation on 429."""

    def __init__(self, raw: str) -> None:
        keys: list[str] = []
        # Check if it's a file path
        candidate = Path(raw)
        if candidate.is_file():
            keys = [
                line.strip()
                for line in candidate.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        else:
            # Comma-separated tokens
            keys = [k.strip() for k in raw.split(",") if k.strip()]

        if not keys:
            print("✖  No valid API keys found.")
            sys.exit(1)

        self._keys = keys
        self._idx = 0

    @property
    def current(self) -> str:
        return self._keys[self._idx]

    @property
    def index(self) -> int:
        return self._idx

    @property
    def count(self) -> int:
        return len(self._keys)

    def rotate(self) -> bool:
        """Advance to next key. Returns False if all keys exhausted (full cycle)."""
        next_idx = (self._idx + 1) % len(self._keys)
        if next_idx == 0 and self._idx != 0:
            return False  # wrapped around — all keys tried
        self._idx = next_idx
        return True


# ══════════════════════ OUTPUT CLEANING ════════════════════════════
def clean_llm_output(raw: str) -> str:
    """Aggressively strip markdown fences, XML tags, <think> blocks,
    and trailing conversational text. Returns pure Python code."""
    text = raw

    # 1. Remove <think>...</think> reasoning blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. Remove any XML-style tags (<response>, <code>, <answer>, etc.)
    text = re.sub(r"</?[a-zA-Z][a-zA-Z0-9_-]*(?:\s[^>]*)?>" , "", text)

    # 3. Extract from code fence if present
    fence_match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

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


def _syntax_check(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ══════════════════════ GPT-5 API CALLER ══════════════════════════
def call_gpt5(
    model: str,
    key_pool: KeyPool,
    prompt: str,
) -> dict:
    """
    Call GPT-5 via GitHub Models with reasoning_effort="high".
    No temperature/top_p/n — unsupported in reasoning mode.

    Returns dict with: text, latency_ms, prompt_tokens, reasoning_tokens, total_tokens.
    """
    import openai

    last_exc: Exception | None = None
    keys_tried = 0

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = openai.OpenAI(
                api_key=key_pool.current,
                base_url=GITHUB_MODELS_BASE_URL,
            )
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="high",
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            text = response.choices[0].message.content or ""
            prompt_tokens = -1
            reasoning_tokens = -1
            total_tokens = -1
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens or -1
                total_tokens = response.usage.total_tokens or -1
                # GPT-5 reasoning_tokens in completion_tokens_details
                details = getattr(response.usage, "completion_tokens_details", None)
                if details:
                    rt = getattr(details, "reasoning_tokens", None)
                    if rt is not None:
                        reasoning_tokens = int(rt)

            return {
                "text": text,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
            }

        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()

            is_rate_limit = False
            try:
                import openai as _oai
                if isinstance(exc, _oai.RateLimitError):
                    is_rate_limit = True
            except ImportError:
                pass
            if "429" in exc_str or ("rate" in exc_str and "limit" in exc_str):
                is_rate_limit = True
            if "quota" in exc_str:
                is_rate_limit = True

            # On rate limit → try rotating key first
            if is_rate_limit:
                if key_pool.count > 1 and keys_tried < key_pool.count:
                    rotated = key_pool.rotate()
                    keys_tried += 1
                    tqdm.write(
                        f"  ⚠  429 on key #{key_pool.index}. "
                        f"Rotating to key #{key_pool.index} …"
                    )
                    if rotated:
                        continue

            # Retriable errors (503, transient)
            retriable = is_rate_limit
            if "503" in exc_str or "service unavailable" in exc_str:
                retriable = True
            if "500" in exc_str or "internal server error" in exc_str:
                retriable = True

            if retriable and attempt < MAX_RETRIES:
                wait = BACKOFF_BASE ** attempt
                tqdm.write(
                    f"  ⚠  Transient error (attempt {attempt}/{MAX_RETRIES}). "
                    f"Retrying in {wait}s …  [{type(exc).__name__}]"
                )
                time.sleep(wait)
                continue

            raise RuntimeError(
                f"API Error after {attempt} attempt(s): {exc}"
            ) from exc

    raise RuntimeError(f"API Error after {MAX_RETRIES} attempts: {last_exc}")


# ══════════════════════ CSV HELPERS ════════════════════════════════
def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "-", name)


def load_processed_ids(csv_path: str) -> set[str]:
    processed: set[str] = set()
    if not os.path.exists(csv_path):
        return processed
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("id", "")
            if pid and row.get("generated_code", "").strip():
                processed.add(pid)
    return processed


def ensure_csv_header(csv_path: str) -> None:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_row(csv_path: str, row: dict) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


# ══════════════════════ PERCENTILE ════════════════════════════════
def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


# ══════════════════════ CLI ═══════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT-5 Reasoning Pipeline — Tyr Stage 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help=(
            "GitHub PAT token(s). Accepts: "
            "a single token, comma-separated tokens, "
            "or a path to a keys.txt file (one per line)."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Model identifier on GitHub Models (default: gpt-5).",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "tyr_benchmark_150.json"),
        help="Path to benchmark JSON dataset.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=6.0,
        help="Seconds between API calls (default: 6.0 for GitHub Student RPM).",
    )
    return parser.parse_args()


# ══════════════════════ MAIN ══════════════════════════════════════
def main() -> None:
    args = parse_args()
    key_pool = KeyPool(args.api_key)

    # ── Load dataset ────────────────────────────────────────────────
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)

    # ── Banner ──────────────────────────────────────────────────────
    print(BANNER.format(
        n_problems=len(dataset),
        model=args.model,
        n_keys=key_pool.count,
        delay=args.delay,
    ))

    # ── CSV path ────────────────────────────────────────────────────
    data_dir = PROJECT_ROOT / "Research_Paper" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_out = str(data_dir / "llm_results.csv")

    ensure_csv_header(csv_out)
    processed_ids = load_processed_ids(csv_out)
    if processed_ids:
        print(f"  ℹ  Resuming — {len(processed_ids)} problems already in CSV.\n")

    # ── Counters ────────────────────────────────────────────────────
    ok_count = 0
    error_count = 0
    syntax_err_count = 0
    skipped_count = len(processed_ids)
    latencies: list[float] = []
    reasoning_counts: list[int] = []
    t_start = time.time()

    # ── Progress bar ────────────────────────────────────────────────
    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    pbar = tqdm(dataset, bar_format=bar_fmt, unit="prob", ncols=110)

    for problem in pbar:
        pid: str = problem["id"]
        name: str = problem["name"]
        pbar.set_postfix_str(f"{pid} — {name} [key#{key_pool.index}]")

        if pid in processed_ids:
            continue

        prompt = PROMPT_TEMPLATE.format(
            target_complexity=problem["target_complexity"],
            original_code=problem["original_code"],
        )

        generated_code = ""
        api_status = "OK"
        error_detail = ""
        latency_ms = 0.0
        prompt_tokens = -1
        reasoning_tokens = -1
        total_tokens = -1

        try:
            result = call_gpt5(
                model=args.model,
                key_pool=key_pool,
                prompt=prompt,
            )

            raw_text = result["text"]
            latency_ms = result["latency_ms"]
            prompt_tokens = result.get("prompt_tokens", -1)
            reasoning_tokens = result.get("reasoning_tokens", -1)
            total_tokens = result.get("total_tokens", -1)

            generated_code = clean_llm_output(raw_text)

            if reasoning_tokens > 0:
                reasoning_counts.append(reasoning_tokens)

            if generated_code and not _syntax_check(generated_code):
                api_status = "SYNTAX_ERROR"
                error_detail = "ast.parse() failed on generated code"
                syntax_err_count += 1
            else:
                latencies.append(latency_ms)
                ok_count += 1

        except Exception as exc:
            api_status = "ERROR"
            error_detail = str(exc)[:500]
            tqdm.write(f"  ✖  {pid} ({name}): {exc}")
            error_count += 1

        row = {
            "id":                       pid,
            "name":                     name,
            "category":                 problem.get("category", ""),
            "difficulty":               problem.get("difficulty", ""),
            "original_complexity":      problem.get("original_complexity", ""),
            "target_complexity":        problem.get("target_complexity", ""),
            "verdict":                  "",   # Stage 2 fills this
            "original_code":            problem.get("original_code", ""),
            "generated_code":           generated_code,
            "optimized_complexity_time": "",  # Stage 2 fills this
            "complexity_improved":      "",   # Stage 2 fills this
            "latency_ms":               f"{latency_ms:.1f}",
            "prompt_tokens":            prompt_tokens,
            "reasoning_tokens":         reasoning_tokens,
            "total_tokens":             total_tokens,
            "api_status":               api_status,
            "error_detail":             error_detail,
        }
        append_row(csv_out, row)

        time.sleep(args.delay)

    pbar.close()

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)
    elapsed_str = f"{hrs:02d}h {mins:02d}m {secs:02d}s"

    avg_lat = f"{sum(latencies) / len(latencies):.1f}" if latencies else "N/A"
    p50_lat = f"{percentile(latencies, 50):.1f}" if latencies else "N/A"
    p95_lat = f"{percentile(latencies, 95):.1f}" if latencies else "N/A"
    avg_rsn = (
        f"{sum(reasoning_counts) / len(reasoning_counts):.0f}"
        if reasoning_counts else "N/A"
    )

    csv_display = csv_out
    try:
        csv_display = str(Path(csv_out).relative_to(PROJECT_ROOT))
    except ValueError:
        pass

    print(SUMMARY_TPL.format(
        model=args.model,
        n_keys=key_pool.count,
        total=len(dataset),
        ok=ok_count,
        syntax_errs=syntax_err_count,
        errors=error_count,
        skipped=skipped_count,
        elapsed=elapsed_str,
        avg_latency=avg_lat,
        p50_latency=p50_lat,
        p95_latency=p95_lat,
        avg_reasoning=avg_rsn,
        csv_out=csv_display,
    ))


if __name__ == "__main__":
    main()
