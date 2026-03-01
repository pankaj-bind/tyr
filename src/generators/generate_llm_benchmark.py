#!/usr/bin/env python3
"""
Tyr — Unified Stage 1: LLM Code Generation
============================================
Single entry-point for ALL providers. Routes to the correct SDK via
``--provider`` and enforces zero-discrimination between models at every
layer: prompt, temperature policy, latency measurement, cleaning, and
CSV schema.

Supported providers
~~~~~~~~~~~~~~~~~~~
    github    OpenAI SDK → https://models.inference.ai.azure.com (GitHub PAT)
    openai    OpenAI SDK → https://api.openai.com/v1
    deepseek  OpenAI SDK → https://api.deepseek.com
    grok      OpenAI SDK → https://api.x.ai/v1
    gemini    google-genai SDK → Google AI

Stage 1 output schema  (data/raw/<provider>_<model>.csv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    id, name, model_name, category, difficulty,
    original_complexity, target_complexity,
    original_code, generated_code,
    latency_ms, prompt_tokens, reasoning_tokens, total_tokens,
    api_status, error_detail

Usage
~~~~~
    python src/generators/generate_llm_benchmark.py \\
        --provider github --model gpt-4o --api-key ghp_XXX

    python src/generators/generate_llm_benchmark.py \\
        --provider gemini --model gemini-2.5-flash --api-key AIzaSy...

    python src/generators/generate_llm_benchmark.py \\
        --provider deepseek --model deepseek-reasoner --api-key sk-...
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
_SRC_DIR      = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent

for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / "backend" / ".env")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: `tqdm` is required.  pip install tqdm")


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

_PROVIDER_BASES: dict[str, str] = {
    "github":   "https://models.inference.ai.azure.com",
    "openai":   "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "grok":     "https://api.x.ai/v1",
}

_ENV_KEY_MAP: dict[str, list[str]] = {
    "github":   ["GITHUB_TOKEN", "GITHUB_PAT"],
    "openai":   ["OPENAI_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "grok":     ["GROK_API_KEY", "XAI_API_KEY"],
    "gemini":   ["GEMINI_API_KEY", "Gemini_API_KEY"],
}

# Models that reject temperature / top_p / presence_penalty / n
# These use reasoning_effort instead.
_REASONING_MODELS: frozenset[str] = frozenset({
    "o1", "o1-mini", "o1-preview",
    "o3", "o3-mini",
    "o4-mini",
    "gpt-5",
    "deepseek-reasoner",
    "deepseek-r1",
    "DeepSeek-R1-0528",
})

DEFAULT_DATASET = _PROJECT_ROOT / "dataset" / "tyr_benchmark_150.json"
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "raw"

# Stage 1 CSV schema — deliberately excludes verdict/optimized_complexity
# (those are owned by Stage 2).
CSV_COLUMNS: list[str] = [
    "id",
    "name",
    "model_name",
    "category",
    "difficulty",
    "original_complexity",
    "target_complexity",
    "original_code",
    "generated_code",
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
    "Do not provide explanations.\n\nNaive Code:\n{original_code}"
)

# Exponential: 2^1=2s, 2^2=4s, 2^3=8s
MAX_RETRIES  = 3
BACKOFF_BASE = 2

# Unified tqdm bar format — IDENTICAL for every provider/model.
_BAR_FMT = (
    "  [{model}] {l_bar}{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}]"
)


# ═══════════════════════════════════════════════════════════════════════
# Output cleaning  (identical for every provider — zero discrimination)
# ═══════════════════════════════════════════════════════════════════════

def clean_llm_output(raw: str) -> str:
    """Strip markdown fences, XML tags, <think> blocks, trailing prose."""
    text = raw

    # DeepSeek-R1 / Grok reasoning block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Generic XML tags (<response>, <code>, etc.)
    text = re.sub(r"</?[a-zA-Z][a-zA-Z0-9_-]*(?:\s[^>]*)?>", "", text)

    # ```python ... ``` or ```py ... ```
    fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        s = line.strip()
        if not cleaned and not s:
            continue
        if cleaned and s and not s.startswith("#"):
            if re.match(
                r"^(This|Note|The above|Here |I |Let me|Explanation|"
                r"Output|In this|Time complexity|Space complexity|"
                r"Example|Alternative|We can|The key|The function|"
                r"The algorithm|Complexity|##|###|\*\*)",
                s,
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
# Retriability classifier  (shared across all providers)
# ═══════════════════════════════════════════════════════════════════════

def _is_retriable(exc: Exception) -> bool:
    msg = str(exc).lower()
    try:
        import openai as _oa
        if isinstance(exc, _oa.RateLimitError):
            return True
        if isinstance(exc, _oa.APIStatusError) and exc.status_code in (429, 500, 503):
            return True
    except ImportError:
        pass
    return any(t in msg for t in (
        "429", "503", "500",
        "rate", "quota", "resourceexhausted",
        "service unavailable", "internal server error",
    ))


# ═══════════════════════════════════════════════════════════════════════
# Provider-specific API callers
#
# CONTRACT (enforced for every provider):
#   - The client object is passed in, NOT created here.
#   - `t0 = time.perf_counter()` is set IMMEDIATELY before the SDK call.
#   - `latency_ms` is IMMEDIATELY captured after the SDK call returns.
#   - No regex, no disk I/O, no output parsing occurs inside the timed block.
#   - Returns dict(text, prompt_tokens, reasoning_tokens, total_tokens,
#                  latency_ms)
# ═══════════════════════════════════════════════════════════════════════

def _call_openai_compat(
    client,          # openai.OpenAI — pre-built, passed in
    model: str,
    prompt: str,
    is_reasoning: bool,
) -> dict:
    """OpenAI / GitHub / DeepSeek / Grok call via openai SDK."""
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if is_reasoning:
        payload["reasoning_effort"] = "high"
    else:
        payload["temperature"] = 0.0   # enforced for ALL non-reasoning models

    # ── LATENCY ISOLATION: timer wraps ONLY the network/inference call ──
    t0 = time.perf_counter()
    response = client.chat.completions.create(**payload)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    # ── END TIMED SECTION ────────────────────────────────────────────

    text = response.choices[0].message.content or ""

    # Handle DeepSeek-R1 reasoning_content field
    rc = getattr(response.choices[0].message, "reasoning_content", None)
    if rc and not text:
        text = rc  # fallback if content is empty but reasoning_content has code

    pt = rt = tt = 0
    if response.usage:
        pt = response.usage.prompt_tokens or 0
        tt = response.usage.total_tokens or 0
        details = getattr(response.usage, "completion_tokens_details", None)
        if details:
            rt = getattr(details, "reasoning_tokens", None) or 0

    # DeepSeek-R1: extract reasoning token count from inline <think> blocks
    # when the dedicated field isn't populated
    if rt == 0:
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            rt = len(think_match.group(1).split())  # approximate

    return {
        "text": text,
        "latency_ms": latency_ms,
        "prompt_tokens": pt,
        "reasoning_tokens": rt,
        "total_tokens": tt,
    }


def _call_gemini(
    client,          # genai.Client — pre-built, passed in
    model: str,
    prompt: str,
    thinking_budget: int | None,
) -> dict:
    """Google Gemini call via google-genai SDK."""
    from google.genai import types

    config_kwargs: dict = {"temperature": 0.0}
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget,
        )
    config = types.GenerateContentConfig(**config_kwargs)

    # ── LATENCY ISOLATION ──────────────────────────────────────────
    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    # ── END TIMED SECTION ─────────────────────────────────────────

    text = response.text or ""

    pt = rt = tt = 0
    meta = getattr(response, "usage_metadata", None)
    if meta:
        pt = int(getattr(meta, "prompt_token_count", 0) or 0)
        ct = int(getattr(meta, "candidates_token_count", 0) or 0)
        tt_raw = getattr(meta, "total_token_count", None)
        tt = int(tt_raw) if tt_raw is not None else pt + ct
        rt = int(getattr(meta, "thinking_token_count", 0) or 0)

    return {
        "text": text,
        "latency_ms": latency_ms,
        "prompt_tokens": pt,
        "reasoning_tokens": rt,
        "total_tokens": tt,
    }


# ═══════════════════════════════════════════════════════════════════════
# Retry wrapper  (provider-agnostic)
# ═══════════════════════════════════════════════════════════════════════

def _call_with_backoff(
    fn,        # zero-arg callable that returns the result dict
    label: str,
) -> dict:
    """
    Execute ``fn()`` with exponential backoff on transient failures.
    ``fn`` must be a closure that captures the client and prompt.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if _is_retriable(exc) and attempt < MAX_RETRIES:
                wait = BACKOFF_BASE ** attempt
                tqdm.write(
                    f"    ⚠  {label} transient error "
                    f"(attempt {attempt}/{MAX_RETRIES}), "
                    f"retrying in {wait}s … [{type(exc).__name__}]"
                )
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"API failed after {MAX_RETRIES} attempts: {last_exc}")


# ═══════════════════════════════════════════════════════════════════════
# Client factory  (built ONCE, before the benchmark loop)
# ═══════════════════════════════════════════════════════════════════════

def build_client(provider: str, api_key: str):
    """
    Instantiate the SDK client exactly once before the benchmark loop.
    Cost: ~2-5ms per init. Must NOT occur inside the latency-timed block.
    """
    if provider == "gemini":
        try:
            from google import genai
        except ImportError:
            sys.exit("ERROR: pip install google-genai")
        return genai.Client(api_key=api_key)
    else:
        try:
            import openai as oa
        except ImportError:
            sys.exit("ERROR: pip install openai")
        base = _PROVIDER_BASES.get(provider)
        kwargs: dict = {"api_key": api_key}
        if base:
            kwargs["base_url"] = base
        return oa.OpenAI(**kwargs)


# ═══════════════════════════════════════════════════════════════════════
# CSV helpers
# ═══════════════════════════════════════════════════════════════════════

def _model_slug(model: str) -> str:
    """Convert 'gemini-2.5-flash' → 'gemini_2_5_flash'."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()


def _output_csv_path(provider: str, model: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{provider}_{_model_slug(model)}.csv"


def _load_processed_ids(csv_path: Path) -> set[str]:
    done: set[str] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = row.get("id", "").strip()
            if pid:
                done.add(pid)
    return done


def _ensure_header(csv_path: Path) -> None:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()


def _append_row(csv_path: Path, row: dict) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writerow(row)


# ═══════════════════════════════════════════════════════════════════════
# Percentile
# ═══════════════════════════════════════════════════════════════════════

def _pct(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    return s[f] + (k - f) * (s[min(f + 1, len(s) - 1)] - s[f])


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Tyr Stage 1 — Unified LLM Code Generation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  --provider github   --model gpt-4o          --api-key ghp_XXX\n"
            "  --provider github   --model o3-mini         --api-key ghp_XXX\n"
            "  --provider gemini   --model gemini-2.5-flash --api-key AIzaSy...\n"
            "  --provider deepseek --model deepseek-reasoner --api-key sk-...\n"
            "  --provider grok     --model grok-3          --api-key xai-...\n"
        ),
    )
    ap.add_argument(
        "--provider",
        required=True,
        choices=["github", "openai", "gemini", "deepseek", "grok"],
        help="LLM provider.",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="Model identifier (must match the provider's API).",
    )
    ap.add_argument(
        "--api-key",
        default=None,
        help="API key / token. Falls back to .env if omitted.",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=4.5,
        help="Seconds between API calls (default: 4.5).",
    )
    ap.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        dest="thinking_budget",
        help="Thinking token budget for Gemini 2.5 models (optional).",
    )
    ap.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to benchmark JSON (default: dataset/tyr_benchmark_150.json).",
    )
    ap.add_argument(
        "--output-dir",
        default=str(DEFAULT_DATA_DIR),
        dest="output_dir",
        help="Directory for raw Stage 1 CSVs (default: data/raw/).",
    )
    return ap.parse_args()


def _resolve_api_key(provider: str, cli_key: str | None) -> str:
    if cli_key:
        return cli_key
    for env in _ENV_KEY_MAP.get(provider, []):
        val = os.getenv(env)
        if val:
            return val
    sys.exit(
        f"ERROR: No API key for provider '{provider}'. "
        "Pass --api-key or set a key in backend/.env."
    )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    args     = parse_args()
    provider = args.provider
    model    = args.model
    api_key  = _resolve_api_key(provider, args.api_key)
    is_reasoning = model in _REASONING_MODELS or model.lower() in {
        m.lower() for m in _REASONING_MODELS
    }

    # ── Load dataset ────────────────────────────────────────────────
    with open(args.dataset, "r", encoding="utf-8") as fh:
        dataset: list[dict] = json.load(fh)

    # ── Build client ONCE before any timing begins ─────────────────
    client = build_client(provider, api_key)

    # ── CSV
    csv_path = _output_csv_path(provider, model, Path(args.output_dir))
    _ensure_header(csv_path)
    processed = _load_processed_ids(csv_path)

    # ── Banner ──────────────────────────────────────────────────────
    print(
        f"\n{'═' * 72}\n"
        f"  Tyr Stage 1 — Unified LLM Benchmark\n"
        f"  Provider  : {provider.upper()}\n"
        f"  Model     : {model}\n"
        f"  Reasoning : {'YES (temperature stripped)' if is_reasoning else 'NO (temperature=0.0)'}\n"
        f"  Problems  : {len(dataset)}  ({len(processed)} already done)\n"
        f"  Delay     : {args.delay}s\n"
        f"  Output    : {csv_path}\n"
        f"{'═' * 72}\n"
    )

    # ── Counters ────────────────────────────────────────────────────
    ok = err = skipped = syntax_errs = 0
    latencies: list[float] = []
    t_start = time.time()

    # ── Unified tqdm — identical format for every provider/model ───
    bar = tqdm(
        dataset,
        unit="prob",
        ncols=100,
        bar_format=(
            "  [{model}] {{l_bar}}{{bar}}| {{n_fmt}}/{{total_fmt}} "
            "[{{elapsed}}<{{remaining}}, {{rate_fmt}}]"
        ).format(model=model),
    )

    for problem in bar:
        pid  = problem["id"]
        name = problem["name"]

        # Uniform postfix: ID | latency | status
        # (updated after each problem so every model shows identical fields)
        bar.set_postfix(
            id=pid,
            lat=f"{latencies[-1]:.0f}ms" if latencies else "—",
            ok=ok,
            err=err,
        )

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

        # ── Step 1: API call (with backoff) ────────────────────────
        try:
            if provider == "gemini":
                fn = lambda: _call_gemini(         # noqa: E731
                    client, model, prompt, args.thinking_budget,
                )
            else:
                fn = lambda: _call_openai_compat(  # noqa: E731
                    client, model, prompt, is_reasoning,
                )

            result = _call_with_backoff(fn, label=f"{pid}")

            latency_ms       = result["latency_ms"]
            prompt_tokens    = result["prompt_tokens"]
            reasoning_tokens = result["reasoning_tokens"]
            total_tokens     = result["total_tokens"]
            generated_code   = clean_llm_output(result["text"])

            if not generated_code or not _syntax_ok(generated_code):
                api_status   = "SYNTAX_ERROR"
                error_detail = "Generated code failed ast.parse()"
                syntax_errs += 1
            else:
                latencies.append(latency_ms)
                ok += 1

        except Exception as exc:
            api_status   = "ERROR"
            error_detail = traceback.format_exc()[:1000]
            tqdm.write(
                f"\n    ✖  {pid} ({name}) [{provider}/{model}]: "
                f"{type(exc).__name__}: {exc}"
            )
            err += 1

        # ── Step 2: Write row immediately — crash-safe ─────────────
        _append_row(csv_path, {
            "id":                   pid,
            "name":                 name,
            "model_name":           model,
            "category":             problem.get("category", ""),
            "difficulty":           problem.get("difficulty", ""),
            "original_complexity":  problem.get("original_complexity", ""),
            "target_complexity":    problem.get("target_complexity", ""),
            "original_code":        problem.get("original_code", ""),
            "generated_code":       generated_code,
            "latency_ms":           f"{latency_ms:.2f}",
            "prompt_tokens":        prompt_tokens,
            "reasoning_tokens":     reasoning_tokens,
            "total_tokens":         total_tokens,
            "api_status":           api_status,
            "error_detail":         error_detail,
        })

        time.sleep(args.delay)

    bar.close()

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    m_e, s_e = divmod(int(elapsed), 60)
    h_e, m_e = divmod(m_e, 60)

    avg = f"{sum(latencies)/len(latencies):.1f}" if latencies else "N/A"
    p50 = f"{_pct(latencies, 50):.1f}" if latencies else "N/A"
    p95 = f"{_pct(latencies, 95):.1f}" if latencies else "N/A"

    print(
        f"\n{'═' * 72}\n"
        f"  STAGE 1 COMPLETE — {provider.upper()} / {model}\n"
        f"{'─' * 72}\n"
        f"  OK            : {ok}\n"
        f"  Syntax errors : {syntax_errs}\n"
        f"  API errors    : {err}\n"
        f"  Skipped       : {skipped}\n"
        f"{'─' * 72}\n"
        f"  Latency (API-only)  avg={avg}ms  p50={p50}ms  p95={p95}ms\n"
        f"  Wall clock          {h_e:02d}h {m_e:02d}m {s_e:02d}s\n"
        f"{'─' * 72}\n"
        f"  Output  →  {csv_path}\n"
        f"{'═' * 72}\n"
    )


if __name__ == "__main__":
    main()
