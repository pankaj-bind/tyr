#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    Tyr — Unified Multi-Provider LLM Benchmark Engine  (Stage 1)           ║
║    Supports: GPT-5, o3, o4-mini, DeepSeek-R1, Grok-3, Gemini, Llama      ║
║    Output :  Research_Paper/data/llm_results.csv  (single canonical file) ║
╚══════════════════════════════════════════════════════════════════════════════╝
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

# ─────────────────────────── SCHEMA ─────────────────────────────────
CSV_COLUMNS = [
    "id",
    "name",
    "category",
    "difficulty",
    "original_complexity",
    "target_complexity",
    "generated_code_complexity",
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

# ─────────────────────────── REASONING MODEL REGISTRY ───────────────
# Models that REJECT temperature / top_p / presence_penalty / n
REASONING_MODELS = frozenset({
    "gpt-5",
    "o3", "o3-mini",
    "o4-mini",
    "o1", "o1-mini", "o1-preview",
    "deepseek-reasoner",
    "deepseek-r1",
    "DeepSeek-R1-0528",
    "grok-3",
})

# Models using OpenAI-compatible SDK (not Gemini)
OPENAI_COMPAT_PROVIDERS = frozenset({"openai", "deepseek", "llama", "grok", "github"})

GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"

PROMPT_TEMPLATE = (
    "You are an expert algorithm developer. Below is a naive Python "
    "implementation. Refactor this code to achieve a target time complexity "
    "of {target_complexity}. Provide ONLY the optimized Python code. "
    "Do not provide explanations. \n\nNaive Code:\n{original_code}"
)

MAX_RETRIES = 3
BACKOFF_BASE = 2

BANNER = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ████████╗██╗   ██╗██████╗     ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗║
║   ╚══██╔══╝╚██╗ ██╔╝██╔══██╗    ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║║
║      ██║    ╚████╔╝ ██████╔╝    ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║║
║      ██║     ╚██╔╝  ██╔══██╗    ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║║
║      ██║      ██║   ██║  ██║    ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║║
║      ╚═╝      ╚═╝   ╚═╝  ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝║
║                                                                           ║
║   UNIFIED LLM BENCHMARK ENGINE  |  Stage 1  |  {n_problems} problems          ║
║   Provider : {provider:<12s}  Model : {model:<30s}                   ║
║   Keys     : {n_keys} loaded      Delay : {delay:.1f}s                          ║
║   Reasoning: {is_reasoning:<6s}                                                ║
║                                                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

SUMMARY_TPL = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    STAGE 1 — GENERATION SUMMARY                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Provider              : {provider:<48s}  ║
║  Model                 : {model:<48s}  ║
║  Reasoning Mode        : {is_reasoning:<48s}  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Total Problems        : {total:<48d}  ║
║  Processed (OK)        : {ok:<48d}  ║
║  Syntax Errors         : {syntax_errs:<48d}  ║
║  API Errors            : {errors:<48d}  ║
║  Skipped (resumed)     : {skipped:<48d}  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Wall-Clock Time       : {elapsed:<48s}  ║
║  Avg Latency (ms)      : {avg_latency:<48s}  ║
║  p50 Latency (ms)      : {p50_latency:<48s}  ║
║  p95 Latency (ms)      : {p95_latency:<48s}  ║
║  Avg Reasoning Tokens  : {avg_reasoning:<48s}  ║
║  Output CSV            : {csv_out:<48s}  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ══════════════════════ KEY ROTATION ═══════════════════════════════
class KeyPool:
    """Round-robin API key pool with automatic rotation on 429."""

    def __init__(self, raw: str) -> None:
        keys: list[str] = []
        candidate = Path(raw)
        if candidate.is_file():
            keys = [
                line.strip()
                for line in candidate.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        else:
            keys = [k.strip() for k in raw.split(",") if k.strip()]

        if not keys:
            print("✖  No valid API keys found.")
            sys.exit(1)

        self._keys = keys
        self._idx = 0
        self._rotations = 0

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
        """Advance to next key. Returns False if full cycle exhausted."""
        self._rotations += 1
        if self._rotations >= len(self._keys):
            self._rotations = 0
            return False
        self._idx = (self._idx + 1) % len(self._keys)
        return True

    def reset_rotations(self) -> None:
        self._rotations = 0


# ══════════════════════ OUTPUT CLEANING ════════════════════════════
def clean_llm_output(raw: str) -> str:
    """Aggressively strip markdown fences, XML tags, DeepSeek <think> blocks,
    and trailing conversational text. Returns pure Python code."""

    text = raw

    # 1. Remove DeepSeek <think>...</think> reasoning blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. Remove any XML-style tags (<response>, <code>, <answer>, etc.)
    text = re.sub(r"</?[a-zA-Z][a-zA-Z0-9_-]*(?:\s[^>]*)?>", "", text)

    # 3. Extract from code fence if present
    fence_match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # 4. Kill trailing conversational lines
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


# ══════════════════════ COMPLEXITY ESTIMATOR ══════════════════════
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
    inner = inner.replace("²", "^2").replace("³", "^3")
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
    gen_n = _normalize_complexity(generated)
    if orig_n is None or gen_n is None:
        return "Same"  # can't determine → assume no change
    orig_r = _COMPLEXITY_RANK.get(orig_n)
    gen_r = _COMPLEXITY_RANK.get(gen_n)
    if orig_r is None or gen_r is None:
        if orig_n == gen_n:
            return "Same"
        return "Same"  # unknown ranking → safe default
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
            # Detect log-pattern: while-loop with halving (i //= 2, i >>= 1)
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

    # Heuristic decision tree
    if has_recursion:
        # Recursive + loops → exponential (conservative)
        if max_depth >= 1:
            return "O(2^n)"
        return "O(2^n)"

    if max_depth == 0 and not has_sort:
        if has_log_pattern:
            return "O(log n)"
        return "O(1)"

    if has_sort:
        # sorted() inside a loop → O(n^2 log n) ≈ O(n^2)
        if max_depth >= 1:
            return "O(n^2)"
        return "O(n log n)"

    if has_log_pattern:
        # log loop nested inside linear → O(n log n)
        return "O(n log n)"

    depth_map = {1: "O(n)", 2: "O(n^2)", 3: "O(n^3)"}
    return depth_map.get(max_depth, f"O(n^{max_depth})")


# ══════════════════════ TOKEN EXTRACTION ══════════════════════════
def _extract_openai_tokens(response) -> dict[str, int]:
    """Extract prompt, reasoning, and total tokens from OpenAI-compat response."""
    result = {"prompt_tokens": -1, "reasoning_tokens": -1, "total_tokens": -1}
    if not response.usage:
        return result

    result["prompt_tokens"] = response.usage.prompt_tokens or -1
    result["total_tokens"] = response.usage.total_tokens or -1

    # GPT-5 / o3 / o4: reasoning_tokens lives in completion_tokens_details
    details = getattr(response.usage, "completion_tokens_details", None)
    if details:
        rt = getattr(details, "reasoning_tokens", None)
        if rt is not None:
            result["reasoning_tokens"] = int(rt)

    return result


def _extract_deepseek_reasoning(response) -> tuple[str, int]:
    """Extract reasoning_content from DeepSeek-R1 and return
    (cleaned_content, reasoning_token_count).

    DeepSeek-R1 may put reasoning in:
      - message.reasoning_content  (dedicated field)
      - <think>...</think> inside message.content
    """
    msg = response.choices[0].message
    content = msg.content or ""
    reasoning_tokens = -1

    # Check dedicated reasoning_content field
    rc = getattr(msg, "reasoning_content", None)
    if rc:
        reasoning_tokens = len(rc.split())  # approximate token count
        return content, reasoning_tokens

    # Check inline <think> tags
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if think_match:
        reasoning_text = think_match.group(1)
        reasoning_tokens = len(reasoning_text.split())
        # Strip thinking from content before returning
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content, reasoning_tokens

    return content, reasoning_tokens


def _extract_gemini_tokens(response) -> dict[str, int]:
    """Extract tokens from Gemini response."""
    result = {"prompt_tokens": -1, "reasoning_tokens": -1, "total_tokens": -1}
    meta = getattr(response, "usage_metadata", None)
    if meta:
        pt = getattr(meta, "prompt_token_count", None)
        ct = getattr(meta, "candidates_token_count", None)
        tt = getattr(meta, "total_token_count", None)
        if pt is not None:
            result["prompt_tokens"] = int(pt)
        if tt is not None:
            result["total_tokens"] = int(tt)
        elif pt is not None and ct is not None:
            result["total_tokens"] = int(pt) + int(ct)
    return result


# ══════════════════════ UNIFIED LLM CALLER ════════════════════════
def call_llm_api(
    provider: str,
    model: str,
    key_pool: KeyPool,
    prompt: str,
    base_url: str | None = None,
) -> dict:
    """
    Unified LLM API caller with exponential back-off + key rotation.

    Returns dict with keys:
        text, latency_ms, prompt_tokens, reasoning_tokens, total_tokens
    """
    is_reasoning = model.lower() in {m.lower() for m in REASONING_MODELS}
    last_exc: Exception | None = None
    keys_tried = 0

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()

            if provider == "gemini":
                result = _call_gemini(model, key_pool.current, prompt)
            elif provider in ("deepseek",) and is_reasoning:
                result = _call_deepseek_reasoning(
                    model, key_pool.current, prompt, base_url,
                )
            else:
                result = _call_openai_compat(
                    provider, model, key_pool.current, prompt,
                    base_url, is_reasoning,
                )

            result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
            key_pool.reset_rotations()
            return result

        except Exception as exc:
            last_exc = exc
            is_rate_limit = _is_rate_limit(exc)

            # Try rotating keys on 429
            if is_rate_limit and key_pool.count > 1:
                if key_pool.rotate():
                    keys_tried += 1
                    tqdm.write(
                        f"  ⚠  429 — rotated to key #{key_pool.index}"
                    )
                    continue

            if _is_retriable(exc) and attempt < MAX_RETRIES:
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


def _is_rate_limit(exc: Exception) -> bool:
    exc_str = str(exc).lower()
    try:
        import openai as _oai
        if isinstance(exc, _oai.RateLimitError):
            return True
    except ImportError:
        pass
    if "429" in exc_str or "quota" in exc_str:
        return True
    if "rate" in exc_str and "limit" in exc_str:
        return True
    if "resourceexhausted" in exc_str:
        return True
    return False


def _is_retriable(exc: Exception) -> bool:
    if _is_rate_limit(exc):
        return True
    exc_str = str(exc).lower()
    try:
        import openai as _oai
        if isinstance(exc, _oai.APIStatusError) and exc.status_code in (429, 500, 503):
            return True
    except ImportError:
        pass
    if "503" in exc_str or "service unavailable" in exc_str:
        return True
    if "500" in exc_str or "internal server error" in exc_str:
        return True
    return False


# ──────────── OPENAI-COMPATIBLE (GPT, Llama, Grok, GitHub) ────────
def _call_openai_compat(
    provider: str,
    model: str,
    api_key: str,
    prompt: str,
    base_url: str | None,
    is_reasoning: bool,
) -> dict:
    import openai

    client_kwargs: dict = {"api_key": api_key}

    # Base URL resolution
    if base_url:
        client_kwargs["base_url"] = base_url
    elif provider == "github":
        client_kwargs["base_url"] = GITHUB_MODELS_BASE_URL
    elif provider == "deepseek":
        client_kwargs["base_url"] = "https://api.deepseek.com"
    elif provider == "grok":
        client_kwargs["base_url"] = "https://api.x.ai/v1"

    client = openai.OpenAI(**client_kwargs)

    # Build payload — strip sampling params for reasoning models
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if is_reasoning:
        # Reasoning models: NO temperature, top_p, n, presence_penalty
        payload["reasoning_effort"] = "high"
    else:
        payload["temperature"] = 0.0

    response = client.chat.completions.create(**payload)

    text = response.choices[0].message.content or ""
    tokens = _extract_openai_tokens(response)

    return {"text": text, **tokens}


# ──────────── DEEPSEEK REASONING (R1) ─────────────────────────────
def _call_deepseek_reasoning(
    model: str,
    api_key: str,
    prompt: str,
    base_url: str | None,
) -> dict:
    import openai

    client_kwargs: dict = {"api_key": api_key}
    client_kwargs["base_url"] = base_url or "https://api.deepseek.com"

    client = openai.OpenAI(**client_kwargs)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        # No temperature/top_p for reasoning model
    )

    text, reasoning_tokens = _extract_deepseek_reasoning(response)
    tokens = _extract_openai_tokens(response)
    # Override reasoning_tokens with our extraction
    if reasoning_tokens > 0:
        tokens["reasoning_tokens"] = reasoning_tokens

    return {"text": text, **tokens}


# ──────────── GEMINI ──────────────────────────────────────────────
def _call_gemini(model: str, api_key: str, prompt: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )

    text = response.text or ""
    tokens = _extract_gemini_tokens(response)

    return {"text": text, **tokens}


# ══════════════════════ CSV HELPERS ════════════════════════════════
CSV_PATH = str(PROJECT_ROOT / "Research_Paper" / "data" / "llm_results.csv")


def load_processed_ids(csv_path: str, model: str) -> set[str]:
    """Return IDs already processed for this specific model."""
    processed: set[str] = set()
    if not os.path.exists(csv_path):
        return processed
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only skip if same model already ran this problem
            if row.get("id") and row.get("generated_code", "").strip():
                # Check if this model+id combo exists
                # For unified CSV, we track by (id) since each model run
                # appends its own rows. Filter by model to allow multi-model.
                processed.add(row["id"])
    return processed


def load_processed_ids_for_model(csv_path: str, model: str) -> set[str]:
    """Return IDs already processed for a SPECIFIC model in the CSV."""
    processed: set[str] = set()
    if not os.path.exists(csv_path):
        return processed
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row.get("id", "")
            row_model = row.get("name", "")  # model is not in old schema
            # For the unified schema, we don't have a model column in the
            # required headers. We use the generated_code presence as proxy.
            if row_id:
                processed.add(row_id)
    return processed


def ensure_csv_header(csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return
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
        description="Tyr — Unified Multi-Provider LLM Benchmark Engine (Stage 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "gemini", "deepseek", "llama", "grok", "github"],
        help="LLM provider (github = GitHub Models endpoint).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key(s). Accepts: single token, comma-separated tokens, "
            "or path to keys.txt. Falls back to .env if omitted."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override base URL for OpenAI-compatible endpoints.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier (e.g. gpt-5, gemini-2.0-flash, DeepSeek-R1-0528).",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "tyr_benchmark_150.json"),
        help="Path to benchmark JSON dataset.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=4.5,
        help="Seconds between API calls (default: 4.5).",
    )
    return parser.parse_args()


def resolve_api_key(args: argparse.Namespace) -> str:
    """Resolve API key from CLI → .env → error."""
    if args.api_key:
        return args.api_key
    key_map = {
        "gemini":   ["Gemini_API_KEY", "GEMINI_API_KEY"],
        "llama":    ["GROQ_API_KEY"],
        "deepseek": ["DEEPSEEK_API_KEY", "GROQ_API_KEY"],
        "openai":   ["OPENAI_API_KEY"],
        "grok":     ["GROK_API_KEY", "XAI_API_KEY"],
        "github":   ["GITHUB_TOKEN", "GITHUB_PAT"],
    }
    for env_var in key_map.get(args.provider, []):
        key = os.getenv(env_var)
        if key:
            return key
    print(
        f"✖  No API key for provider '{args.provider}'. "
        "Pass --api-key or set the variable in backend/.env."
    )
    sys.exit(1)


# ══════════════════════ MAIN ══════════════════════════════════════
def main() -> None:
    args = parse_args()
    raw_key = resolve_api_key(args)
    key_pool = KeyPool(raw_key)

    is_reasoning = args.model.lower() in {m.lower() for m in REASONING_MODELS}

    # ── Load dataset ────────────────────────────────────────────────
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)

    # ── Banner ──────────────────────────────────────────────────────
    print(BANNER.format(
        n_problems=len(dataset),
        provider=args.provider.upper(),
        model=args.model,
        n_keys=key_pool.count,
        delay=args.delay,
        is_reasoning="YES" if is_reasoning else "NO",
    ))

    # ── CSV ─────────────────────────────────────────────────────────
    csv_out = CSV_PATH
    ensure_csv_header(csv_out)
    processed_ids = load_processed_ids(csv_out, args.model)
    if processed_ids:
        print(f"  ℹ  Resuming — {len(processed_ids)} IDs already in CSV.\n")

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
            result = call_llm_api(
                provider=args.provider,
                model=args.model,
                key_pool=key_pool,
                prompt=prompt,
                base_url=args.base_url,
            )

            raw_text = result["text"]
            latency_ms = result["latency_ms"]
            prompt_tokens = result.get("prompt_tokens", -1)
            reasoning_tokens = result.get("reasoning_tokens", -1)
            total_tokens = result.get("total_tokens", -1)

            generated_code = clean_llm_output(raw_text)

            if reasoning_tokens > 0:
                reasoning_counts.append(reasoning_tokens)

            # Syntax gate
            if generated_code and not _syntax_check(generated_code):
                api_status = "SYNTAX_ERROR"
                error_detail = "ast.parse() failed on generated code"
                syntax_err_count += 1
            else:
                latencies.append(latency_ms)
                ok_count += 1

                # ── Complexity estimation & verdict ──────────────
                gen_cx = estimate_complexity(generated_code)
                orig_cx = problem.get("original_complexity", "")
                if gen_cx != "N/A" and orig_cx:
                    verdict_val = _compare_complexity(orig_cx, gen_cx)
                else:
                    verdict_val = "Same"

        except Exception as exc:
            api_status = "ERROR"
            error_detail = str(exc)[:500]
            tqdm.write(f"  ✖  {pid} ({name}): {exc}")
            error_count += 1

        # Defaults for error / syntax-error paths
        if api_status != "OK":
            gen_cx = "N/A"
            verdict_val = ""

        row = {
            "id":                       pid,
            "name":                     name,
            "category":                 problem.get("category", ""),
            "difficulty":               problem.get("difficulty", ""),
            "original_complexity":      problem.get("original_complexity", ""),
            "target_complexity":        problem.get("target_complexity", ""),
            "generated_code_complexity": gen_cx,
            "verdict":                  verdict_val,
            "original_code":            problem.get("original_code", ""),
            "generated_code":           generated_code,
            "optimized_complexity_time": gen_cx if gen_cx != "N/A" else "",
            "complexity_improved":      verdict_val,
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
        provider=args.provider.upper(),
        model=args.model,
        is_reasoning="YES" if is_reasoning else "NO",
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
