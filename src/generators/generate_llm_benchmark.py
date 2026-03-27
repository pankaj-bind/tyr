#!/usr/bin/env python3
"""
Tyr — Unified Stage 1: LLM Code Generation
============================================
Single entry-point for ALL providers.  Supports two benchmark suites:

PART 1 — System 1 (Standard) Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    gpt-4.1                        OpenAI Prior Gen
    DeepSeek-V3-0324               Open-Weight Standard SOTA
    Meta-Llama-3.1-405B-Instruct   Heavy Compute Open-Source
    Codestral-2501                 Coding-Specific Baseline
    grok-3                         xAI Standard Heavyweight
    gemini-2.5-pro                 Google Standard

PART 2 — System 2 (Reasoning) Titans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    gpt-5                          OpenAI Reasoning Baseline
    o3                             OpenAI SOTA Reasoning
    o4-mini                        OpenAI Lightweight Reasoning
    DeepSeek-R1-0528               Open-Weight Reasoning SOTA
    gemini-2.5-flash               Google Fast Reasoning

Stage 1 output schema  (data/raw/<provider>_<model>.csv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    id, name, model_name, category, difficulty,
    original_complexity, target_complexity,
    original_code, generated_code,
    latency_ms, prompt_tokens, reasoning_tokens,
    completion_tokens, total_tokens,
    api_status, error_detail

Usage
~~~~~
    # Single model
    python src/generators/generate_llm_benchmark.py \\
        --provider github --model gpt-4.1 --api-key ghp_XXX

    # Run entire System 1 suite (keys from .env)
    python src/generators/generate_llm_benchmark.py --suite system1

    # Run entire System 2 suite
    python src/generators/generate_llm_benchmark.py --suite system2

    # Run ALL 11 models
    python src/generators/generate_llm_benchmark.py --suite all

    # List registered models + key availability
    python src/generators/generate_llm_benchmark.py --list-models

Authentication
~~~~~~~~~~~~~~
    Pass your API key via --api-key or set the corresponding env var
    in .env (e.g. GITHUB_TOKEN, OPENAI_API_KEY, GEMINI_API_KEY).
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
# Root .env takes priority; backend/.env is the fallback.
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT / "backend" / ".env")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: `tqdm` is required.  pip install tqdm")


# ═══════════════════════════════════════════════════════════════════════
# Provider Configuration
# ═══════════════════════════════════════════════════════════════════════

_PROVIDER_BASES: dict[str, str] = {
    "github":   "https://models.inference.ai.azure.com",
    "openai":   "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
}

_ENV_KEY_MAP: dict[str, list[str]] = {
    "github":   ["GITHUB_TOKEN", "GITHUB_PAT"],
    "openai":   ["OPENAI_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "gemini":   ["GEMINI_API_KEY", "Gemini_API_KEY"],
}



# ═══════════════════════════════════════════════════════════════════════
# Model Registry — all benchmark models with provider & suite routing
# ═══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, dict] = {
    # ── PART 1: System 1 (Standard) Benchmarks ────────────────────
    # Pure autoregressive / instruction-tuned — no chain-of-thought
    # reasoning. Use temperature=0.0.
    "gpt-4.1": {
        "provider": "github",
        "suite":    "system1",
        "label":    "GPT-4.1 (OpenAI Prior Gen)",
    },
    "DeepSeek-V3-0324": {
        "provider": "github",
        "suite":    "system1",
        "label":    "DeepSeek-V3-0324 (Open-Weight Standard SOTA)",
    },
    "Meta-Llama-3.1-405B-Instruct": {
        "provider": "github",
        "suite":    "system1",
        "label":    "Meta-Llama-3.1-405B (Heavy Compute Open-Source)",
    },
    "Codestral-2501": {
        "provider": "github",
        "suite":    "system1",
        "label":    "Codestral 25.01 (Coding-Specific Baseline)",
    },
    "grok-3": {
        "provider": "github",
        "suite":    "system1",
        "label":    "Grok 3 (xAI Standard Heavyweight)",
    },
    "gemini-2.5-pro": {
        "provider":        "gemini",
        "suite":           "system1",
        "label":           "Gemini 2.5 Pro (Google Standard)",
        "thinking_budget": None,
    },
    "gemini-3.1-pro-preview": {
        "provider":        "gemini",
        "suite":           "system1",
        "label":           "Gemini 3.1 Pro Preview (Google Next-Gen Standard)",
        "thinking_budget": None,
    },

    # ── PART 2: System 2 (Reasoning) Titans ───────────────────────
    # Chain-of-thought / reasoning-effort models. Reject temperature;
    # use reasoning_effort="high" or thinking_budget.
    "gpt-5": {
        "provider": "github",
        "suite":    "system2",
        "label":    "GPT-5 (OpenAI Reasoning Baseline)",
    },
    "o3": {
        "provider": "github",
        "suite":    "system2",
        "label":    "o3 (OpenAI SOTA Reasoning)",
    },
    "o4-mini": {
        "provider": "github",
        "suite":    "system2",
        "label":    "o4-mini (OpenAI Lightweight Reasoning)",
    },
    "DeepSeek-R1-0528": {
        "provider": "github",
        "suite":    "system2",
        "label":    "DeepSeek-R1-0528 (Open-Weight Reasoning SOTA)",
    },
    "gemini-2.5-flash": {
        "provider":        "gemini",
        "suite":           "system2",
        "label":           "Gemini 2.5 Flash (Google Fast Reasoning)",
        "thinking_budget": 8192,
    },
    "gemini-3-flash-preview": {
        "provider":        "gemini",
        "suite":           "system2",
        "label":           "Gemini 3.0 Flash Preview (Google Next-Gen Reasoning)",
        "thinking_budget": 8192,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Reasoning-model set  (API-level: rejects temperature, uses
# reasoning_effort).  Independent of System 1 / System 2 categorization.
# ═══════════════════════════════════════════════════════════════════════

_REASONING_MODELS: frozenset[str] = frozenset({
    # OpenAI o-series
    "o1", "o1-mini", "o1-preview",
    "o3", "o3-mini",
    "o4-mini",
    # GPT-5 unified — rejects temperature via OpenAI API
    "gpt-5",
    # DeepSeek reasoning family
    "deepseek-reasoner", "deepseek-r1", "DeepSeek-R1-0528",
    # NOTE: grok-3 is standard (System 1) — accepts temperature normally.
    # NOTE: Gemini thinking (2.5-flash, 3-flash-preview) is controlled
    #       via thinking_budget in MODEL_REGISTRY, not this set.
})


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_DATASET  = _PROJECT_ROOT / "data" / "benchmarks" / "tyr_benchmark_250.json"
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "raw"

# Stage 1 CSV schema — deliberately excludes verdict / optimized_complexity
# (those are owned exclusively by Stage 2).
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
    "completion_tokens",
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

# Exponential backoff: 2^1=2s, 2^2=4s, 2^3=8s
MAX_RETRIES  = 3
BACKOFF_BASE = 2



# ═══════════════════════════════════════════════════════════════════════
# ASCII Art Banner
# ═══════════════════════════════════════════════════════════════════════

_BANNER_ART: tuple[str, ...] = (
    "████████╗██╗   ██╗██████╗     ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗███╗   ███╗ █████╗ ██████╗ ██╗  ██╗",
    "╚══██╔══╝╚██╗ ██╔╝██╔══██╗    ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝",
    "   ██║    ╚████╔╝ ██████╔╝    ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║██╔████╔██║███████║██████╔╝█████╔╝ ",
    "   ██║     ╚██╔╝  ██╔══██╗    ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗ ",
    "   ██║      ██║   ██║  ██║    ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗",
    "   ╚═╝      ╚═╝   ╚═╝  ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝",
)


def _print_banner(
    *,
    provider: str,
    model: str,
    suite_tag: str,
    n_problems: int,
    n_done: int,
    delay: float,
    is_reasoning: bool,
    thinking_budget: int | None,
    csv_path: Path | str,
) -> None:
    """Print the TYR BENCHMARK ASCII art banner with run details."""
    # Compute inner width to fit all content
    art_widths  = [len("   " + a) for a in _BANNER_ART]
    info_widths = [
        len(f"   UNIFIED LLM BENCHMARK ENGINE  │  Stage 1  │  {n_problems} problems"),
        len(f"   Output   : {csv_path}"),
    ]
    inner = max(*art_widths, *info_widths) + 3
    inner = max(inner, 78)  # minimum 78 chars

    def row(text: str = "") -> str:
        return "║" + (f"   {text}" if text else "").ljust(inner) + "║"

    top = "╔" + "═" * inner + "╗"
    bot = "╚" + "═" * inner + "╝"
    sep = "║" + "─" * inner + "║"

    reasoning_str = "YES (no temperature)" if is_reasoning else "NO (temp=0.0)"
    thinking_str  = f"{thinking_budget} tokens" if thinking_budget else "OFF"

    lines = [
        "",
        top,
        row(),
        *(row(a) for a in _BANNER_ART),
        row(),
        sep,
        row(f"UNIFIED LLM BENCHMARK ENGINE  │  Stage 1  │  {n_problems} problems"),
        sep,
        row(f"Provider : {provider.upper():<12s}  Model    : {model}"),
        row(f"Suite    : {suite_tag.upper():<12s}  Delay    : {delay:.1f}s"),
        row(f"Reasoning: {reasoning_str:<20s}  Thinking : {thinking_str}"),
        row(f"Progress : {n_done}/{n_problems} done"),
        row(f"Output   : {csv_path}"),
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
# Output cleaning  (identical for every provider — zero discrimination)
# ═══════════════════════════════════════════════════════════════════════

def clean_llm_output(raw: str) -> str:
    """Strip markdown fences, XML tags, <think> blocks, trailing prose."""
    text = raw

    # Strip null bytes — LLMs occasionally emit binary garbage that
    # would corrupt CSV output.
    text = text.replace("\x00", "")

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
        payload["max_tokens"] = 4096

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

    pt = rt = ct = tt = 0
    if response.usage:
        pt = response.usage.prompt_tokens or 0
        ct = response.usage.completion_tokens or 0
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
        "completion_tokens": ct,
        "total_tokens": tt,
    }


def _call_gemini(
    client,          # genai.Client — pre-built, passed in
    model: str,
    prompt: str,
    thinking_budget: int | None,
) -> dict:
    """Google Gemini call via google-genai SDK (Vertex AI endpoint).

    For standard (System 1) models: ``temperature=0.0``, no thinking.
    For reasoning (System 2) models: ``ThinkingConfig(thinking_budget=N)``.
    """
    from google.genai import types

    config_kwargs: dict = {}
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget,
        )
    else:
        config_kwargs["temperature"] = 0.0

    config = types.GenerateContentConfig(**config_kwargs)

    # ── LATENCY ISOLATION: timer wraps ONLY the network/inference call ──
    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    # ── END TIMED SECTION ───────────────────────────────────────────────

    text = response.text or ""

    pt = ct = rt = tt = 0
    meta = getattr(response, "usage_metadata", None)
    if meta:
        pt = int(getattr(meta, "prompt_token_count", 0) or 0)
        ct = int(getattr(meta, "candidates_token_count", 0) or 0)
        tt_raw = getattr(meta, "total_token_count", None)
        tt = int(tt_raw) if tt_raw is not None else pt + ct

        rt = int(getattr(meta, "thinking_token_count", 0) or 0)

        # MLOps FIX: Deduce invisible thinking tokens via algebra
        hidden_reasoning = tt - (pt + ct)
        if hidden_reasoning > 0 and rt == 0:
            rt = hidden_reasoning

    return {
        "text": text,
        "latency_ms": latency_ms,
        "prompt_tokens": pt,
        "reasoning_tokens": rt,
        "completion_tokens": ct,
        "total_tokens": tt,
    }


# ═══════════════════════════════════════════════════════════════════════
# Retry wrapper  (provider-agnostic)
# ═══════════════════════════════════════════════════════════════════════

def _call_with_backoff(
    fn_factory,   # callable(client) -> result dict
    client,
    label: str,
) -> dict:
    """
    Execute ``fn_factory(client)`` with exponential backoff.

    Retries up to MAX_RETRIES on transient / rate-limit errors.
    """
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn_factory(client)
        except Exception as exc:
            last_exc = exc

            if _is_retriable(exc) and attempt < MAX_RETRIES:
                wait = BACKOFF_BASE ** min(attempt, 4)  # cap at 16s
                tqdm.write(
                    f"    ⚠  {label} transient error "
                    f"(attempt {attempt}/{MAX_RETRIES}), "
                    f"retrying in {wait:.1f}s … "
                    f"[{type(exc).__name__}]"
                )
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(
        f"API failed after {MAX_RETRIES} attempts: {last_exc}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Client factory  (built ONCE, before the benchmark loop)
# ═══════════════════════════════════════════════════════════════════════

def build_client(provider: str, api_key: str):
    """
    Instantiate the SDK client exactly once before the benchmark loop.
    Cost: ~2-5ms per init. Must NOT occur inside the latency-timed block.

    For the ``gemini`` provider the builder auto-discovers a GCP Service
    Account JSON key inside ``src/generators/`` and initialises the
    google-genai client in **Vertex AI** mode (enterprise endpoint).
    """
    if provider == "gemini":
        try:
            from google import genai
        except ImportError:
            sys.exit("ERROR: pip install google-genai")

        # ── Auto-discover GCP Service Account JSON key ─────────────
        generators_dir = Path(__file__).resolve().parent
        json_files = sorted(generators_dir.glob("*.json"))

        if json_files:
            if len(json_files) > 1:
                tqdm.write(
                    f"    ⚠  Multiple JSON files found in {generators_dir}; "
                    f"using first: {json_files[0].name}"
                )
            sa_path = json_files[0]
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(sa_path)

            with open(sa_path, "r", encoding="utf-8") as fh:
                sa_data = json.load(fh)
            project_id = sa_data["project_id"]

            tqdm.write(
                f"    🔐  Vertex AI auth: project={project_id}, "
                f"key={sa_path.name}"
            )
            return genai.Client(
                vertexai=True,
                project=project_id,
                location="us-central1",
            )

        # Fallback: consumer API key if no JSON file is present
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
# Suite helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_suite_models(suite: str) -> dict[str, dict]:
    """Return models matching the given suite filter."""
    if suite == "all":
        return dict(MODEL_REGISTRY)
    return {k: v for k, v in MODEL_REGISTRY.items() if v["suite"] == suite}


def _resolve_api_key(provider: str, cli_key: str | None = None) -> str | None:
    """Resolve a single API key from CLI → .env. Returns None if not found."""
    if cli_key:
        return cli_key.strip()
    for env in _ENV_KEY_MAP.get(provider, []):
        val = os.getenv(env, "").strip()
        if val:
            return val
    return None


def list_models() -> None:
    """Print all registered models with key availability."""
    _print_art_header("Tyr — Registered Benchmark Models")

    for suite_name, suite_label in [
        ("system1", "PART 1: System 1 (Standard) Benchmarks"),
        ("system2", "PART 2: System 2 (Reasoning) Titans"),
    ]:
        models = _get_suite_models(suite_name)
        print(f"\n  {suite_label}")
        print(f"  {'─' * 74}")
        for i, (model_id, info) in enumerate(models.items(), 1):
            has_key = _resolve_api_key(info["provider"]) is not None
            key_icon = "✔" if has_key else "✖"
            reasoning = model_id in _REASONING_MODELS
            thinking  = info.get("thinking_budget") is not None
            mode = "THINK" if thinking else ("REASON" if reasoning else "STD")
            print(
                f"    {i}. {model_id:<35s}  "
                f"[{info['provider']:<8s}]  "
                f"Key: {key_icon}  "
                f"Mode: {mode:<6s}  "
                f"{info['label']}"
            )

    print(f"\n{'═' * 80}\n")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Tyr Stage 1 — Unified LLM Code Generation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Single model:\n"
            "  --provider github --model gpt-4.1 --api-key ghp_XXX\n\n"
            "Suite mode (runs all models in a suite automatically):\n"
            "  --suite system1          (6 standard models)\n"
            "  --suite system2          (5 reasoning models)\n"
            "  --suite all              (all 11 models)\n\n"
            "List registered models:\n"
            "  --list-models\n"
        ),
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--suite",
        choices=["system1", "system2", "all"],
        help="Run all models in a benchmark suite.",
    )
    mode.add_argument(
        "--list-models",
        action="store_true",
        help="List all registered benchmark models and exit.",
    )
    ap.add_argument(
        "--provider",
        choices=["github", "openai", "gemini", "deepseek"],
        help="LLM provider (required for single-model mode).",
    )
    ap.add_argument(
        "--model",
        help="Model identifier (required for single-model mode).",
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
        help="Thinking token budget for Gemini models (optional).",
    )
    ap.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to benchmark JSON (default: data/benchmarks/tyr_benchmark_250.json).",
    )
    ap.add_argument(
        "--output-dir",
        default=str(DEFAULT_DATA_DIR),
        dest="output_dir",
        help="Directory for raw Stage 1 CSVs (default: data/raw/).",
    )
    return ap.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Single-model benchmark runner
# ═══════════════════════════════════════════════════════════════════════

def run_single_benchmark(
    provider: str,
    model: str,
    api_key: str,
    dataset: list[dict],
    data_dir: Path,
    delay: float,
    thinking_budget: int | None = None,
) -> dict:
    """Run the full benchmark for one model.

    Returns a summary dict with ok/error/syntax counts and output path.
    """
    is_reasoning = model in _REASONING_MODELS or model.lower() in {
        m.lower() for m in _REASONING_MODELS
    }

    # ── Build SDK client (once, before the benchmark loop) ─────────
    client = build_client(provider, api_key)

    # ── CSV setup ──────────────────────────────────────────────────
    csv_path = _output_csv_path(provider, model, data_dir)
    _ensure_header(csv_path)
    processed = _load_processed_ids(csv_path)

    # ── Resolve suite label from registry ──────────────────────────
    reg = MODEL_REGISTRY.get(model, {})
    suite_tag = reg.get("suite", "custom").upper()

    # ── Banner ─────────────────────────────────────────────────────
    _print_banner(
        provider=provider,
        model=model,
        suite_tag=suite_tag,
        n_problems=len(dataset),
        n_done=len(processed),
        delay=delay,
        is_reasoning=is_reasoning,
        thinking_budget=thinking_budget,
        csv_path=csv_path,
    )

    # ── Counters ───────────────────────────────────────────────────
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

        generated_code   = ""
        api_status       = "OK"
        error_detail     = ""
        latency_ms         = 0.0
        prompt_tokens      = 0
        reasoning_tokens   = 0
        completion_tokens  = 0
        total_tokens       = 0

        # ── API call (with backoff) ────────────────────────────────
        try:
            if provider == "gemini":
                fn_factory = lambda c: _call_gemini(         # noqa: E731
                    c, model, prompt, thinking_budget,
                )
            else:
                fn_factory = lambda c: _call_openai_compat(  # noqa: E731
                    c, model, prompt, is_reasoning,
                )

            result = _call_with_backoff(fn_factory, client, label=pid)

            latency_ms         = result["latency_ms"]
            prompt_tokens      = result["prompt_tokens"]
            reasoning_tokens   = result["reasoning_tokens"]
            completion_tokens  = result["completion_tokens"]
            total_tokens       = result["total_tokens"]
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

        # ── Write row immediately — crash-safe ─────────────────────
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
            "completion_tokens":    completion_tokens,
            "total_tokens":         total_tokens,
            "api_status":           api_status,
            "error_detail":         error_detail,
        })

        time.sleep(delay)

    bar.close()

    # ── Summary ────────────────────────────────────────────────────
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

    return {
        "model": model,
        "provider": provider,
        "suite": reg.get("suite", "custom"),
        "ok": ok,
        "errors": err,
        "syntax_errs": syntax_errs,
        "skipped": skipped,
        "elapsed_s": elapsed,
        "csv_path": str(csv_path),
    }


# ═══════════════════════════════════════════════════════════════════════
# Interactive model picker  (shown when run with no arguments)
# ═══════════════════════════════════════════════════════════════════════

def _read_key() -> str:
    """Read a single keypress. Returns 'UP', 'DOWN', 'ENTER', 'Q', etc."""
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):          # special key prefix on Windows
            ch2 = msvcrt.getwch()
            return {"H": "UP", "P": "DOWN"}.get(ch2, "")
        if ch == "\r":
            return "ENTER"
        if ch == "\x1b":                    # Escape
            return "Q"
        if ch == "\x03":                    # Ctrl-C
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


def _interactive_model_picker() -> str | None:
    """Arrow-key driven interactive menu to pick a model or suite."""

    # ── Build menu items ───────────────────────────────────────────
    # Each item: (label_plain, value, is_header, disabled)
    items: list[dict] = []
    all_models = list(MODEL_REGISTRY.items())
    s1 = [(m, i) for m, i in all_models if i["suite"] == "system1"]
    s2 = [(m, i) for m, i in all_models if i["suite"] == "system2"]

    def _model_line(mid: str, info: dict) -> tuple[str, bool]:
        has_key = _resolve_api_key(info["provider"]) is not None
        icon = "\u2714" if has_key else "\u2716"
        thinking = info.get("thinking_budget") is not None
        reasoning = mid in _REASONING_MODELS
        mode = "THINK" if thinking else ("REASON" if reasoning else "STD")
        label = (
            f"{mid:<35s}  [{info['provider']:<8s}]  "
            f"Key: {icon}  Mode: {mode:<6s}  {info['label']}"
        )
        return label, has_key

    # System 1 header + models
    items.append({"label": "SYSTEM 1 \u2500 Standard Benchmarks", "value": None,
                  "is_header": True, "disabled": True})
    for mid, info in s1:
        label, has_key = _model_line(mid, info)
        items.append({"label": label, "value": mid,
                      "is_header": False, "disabled": not has_key})

    # System 2 header + models
    items.append({"label": "SYSTEM 2 \u2500 Reasoning Titans", "value": None,
                  "is_header": True, "disabled": True})
    for mid, info in s2:
        label, has_key = _model_line(mid, info)
        items.append({"label": label, "value": mid,
                      "is_header": False, "disabled": not has_key})

    # Suite shortcuts header + options
    items.append({"label": "SUITE SHORTCUTS", "value": None,
                  "is_header": True, "disabled": True})
    items.append({"label": "Run ALL System 1 models (standard)",
                  "value": "__SUITE__S1", "is_header": False, "disabled": False})
    items.append({"label": "Run ALL System 2 models (reasoning)",
                  "value": "__SUITE__S2", "is_header": False, "disabled": False})
    items.append({"label": "Run ALL 11 models",
                  "value": "__SUITE__ALL", "is_header": False, "disabled": False})
    items.append({"label": "Quit",
                  "value": "__QUIT__", "is_header": False, "disabled": False})

    # Selectable indices (skip headers and disabled items)
    selectable = [i for i, it in enumerate(items) if not it["disabled"]]
    cursor_idx = 0  # index into `selectable`

    def _draw() -> None:
        """Full-screen redraw: clear screen, print banner + menu."""
        _clear_screen()

        # Print compact banner
        _print_art_header("Select a Model to Benchmark")

        tw = _get_terminal_width()
        cur = selectable[cursor_idx]

        for i, it in enumerate(items):
            if it["is_header"]:
                print()
                print(f"  \033[1;36m{it['label']}\033[0m")
                print(f"  {'─' * min(74, tw - 4)}")
                continue

            is_cur = (i == cur)
            marker = " \u25b8 " if is_cur else "   "

            # Truncate line to terminal width to prevent wrapping
            raw_line = f"  {marker}{it['label']}"
            if len(raw_line) > tw - 1:
                raw_line = raw_line[: tw - 4] + "..."

            if it["disabled"]:
                print(f"\033[90m{raw_line}  (no key)\033[0m")
            elif is_cur:
                print(f"\033[1;33m{raw_line}\033[0m")
            else:
                print(raw_line)

        print()
        print(f"  \033[2m\u2191/\u2193 Navigate  \u2022  Enter Select  \u2022  Q/Esc Quit\033[0m")

    # ── Initial draw ───────────────────────────────────────────────
    # Hide cursor during interaction
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
                # Show cursor, clear, print quit
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()
                print("\n  Bye!")
                return None

    except (KeyboardInterrupt, EOFError):
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        print("\n  Aborted.")
        return None

    # ── Show cursor again ──────────────────────────────────────────
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()

    selected_item = items[selectable[cursor_idx]]
    value = selected_item["value"]

    if value == "__QUIT__":
        print("\n  Bye!")
        return None

    if value and value.startswith("__SUITE__"):
        print(f"\n  \u2714 Selected: {selected_item['label']}\n")
        return value

    # Single model
    info = MODEL_REGISTRY[value]
    print(f"\n  \u2714 Selected: {value}  ({info['label']})\n")
    return value


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── List models and exit ───────────────────────────────────────
    if args.list_models:
        list_models()
        return

    # ── Interactive picker when no flags provided ──────────────────
    if not args.model and not args.suite:
        pick = _interactive_model_picker()
        if not pick:
            return
        # Handle suite shortcuts from interactive menu
        if pick.startswith("__SUITE__"):
            suite_map = {"S1": "system1", "S2": "system2", "ALL": "all"}
            args.suite = suite_map[pick.split("__")[-1]]
        else:
            args.model = pick

    # ── Load dataset (once) ────────────────────────────────────────
    with open(args.dataset, "r", encoding="utf-8") as fh:
        dataset: list[dict] = json.load(fh)

    data_dir = Path(args.output_dir)

    # ── Suite mode: batch-run all models in the suite ──────────────
    if args.suite:
        models = _get_suite_models(args.suite)
        suite_label = {
            "system1": "System 1 (Standard)",
            "system2": "System 2 (Reasoning)",
            "all":     "All Models (System 1 + System 2)",
        }[args.suite]

        _print_art_header(
            f"Suite Benchmark: {suite_label}  │  "
            f"{len(models)} models  │  {len(dataset)} problems"
        )

        results: list[dict] = []
        skipped_models: list[str] = []

        for i, (model_id, info) in enumerate(models.items(), 1):
            provider = info["provider"]
            key = _resolve_api_key(provider, args.api_key)

            if not key:
                print(
                    f"\n  ⚠  [{i}/{len(models)}] SKIP {model_id} — "
                    f"no API key for provider '{provider}'"
                )
                skipped_models.append(model_id)
                continue

            print(
                f"\n  ▶  [{i}/{len(models)}] Starting {model_id} "
                f"({info['label']})"
            )

            thinking = info.get("thinking_budget") or args.thinking_budget
            summary = run_single_benchmark(
                provider=provider,
                model=model_id,
                api_key=key,
                dataset=dataset,
                data_dir=data_dir,
                delay=args.delay,
                thinking_budget=thinking,
            )
            results.append(summary)

        # ── Final suite summary ────────────────────────────────────
        total_ok  = sum(r["ok"] for r in results)
        total_err = sum(r["errors"] for r in results)
        total_syn = sum(r["syntax_errs"] for r in results)
        total_elapsed = sum(r["elapsed_s"] for r in results)
        h, rem = divmod(int(total_elapsed), 3600)
        m, s   = divmod(rem, 60)

        print(f"\n{'█' * 72}")
        print(f"  SUITE COMPLETE — {suite_label}")
        print(f"{'─' * 72}")
        for r in results:
            print(
                f"    {r['model']:<35s}  OK={r['ok']:<4d} "
                f"ERR={r['errors']:<3d} "
                f"SYN={r['syntax_errs']:<3d}  "
                f"→  {r['csv_path']}"
            )
        if skipped_models:
            print(f"{'─' * 72}")
            print(f"    Skipped (missing key): {', '.join(skipped_models)}")
        print(f"{'─' * 72}")
        print(
            f"    Totals   OK={total_ok}  ERR={total_err}  "
            f"SYNTAX={total_syn}  "
            f"Wall={h:02d}h {m:02d}m {s:02d}s"
        )
        print(f"{'█' * 72}\n")
        return

    # ── Single model mode ──────────────────────────────────────────

    # Auto-resolve provider from registry if not specified
    if not args.provider:
        if args.model in MODEL_REGISTRY:
            args.provider = MODEL_REGISTRY[args.model]["provider"]
        else:
            sys.exit(
                f"ERROR: --provider is required for model '{args.model}' "
                f"(not found in registry)."
            )

    key = _resolve_api_key(args.provider, args.api_key)
    if not key:
        sys.exit(
            f"ERROR: No API key for provider '{args.provider}'. "
            "Pass --api-key or set a key in backend/.env."
        )

    # If model is in registry, auto-apply thinking_budget when not set via CLI
    reg = MODEL_REGISTRY.get(args.model, {})
    thinking = args.thinking_budget or reg.get("thinking_budget")

    run_single_benchmark(
        provider=args.provider,
        model=args.model,
        api_key=key,
        dataset=dataset,
        data_dir=data_dir,
        delay=args.delay,
        thinking_budget=thinking,
    )


if __name__ == "__main__":
    main()
