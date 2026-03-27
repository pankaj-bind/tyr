# Benchmark Pipeline

End-to-end guide for running the Tyr benchmark: generating LLM solutions (Stage 1) and formally verifying them (Stage 2).

```
tyr_benchmark_250.json                      # 250 hand-curated O(N^2) problems
        |
        v
[Stage 1] generate_llm_benchmark.py        --> data/raw/<provider>_<model>.csv
        |
        v
[Stage 2] verify_llm_results.py            --> data/verified/<provider>_<model>.csv
        |
        v
[Analysis] analyze_metrics.py / find_bug.py / generate_graphs.py
```

---

## Prerequisites

```bash
# Activate venv first (see README for setup)
# Windows (PowerShell): .\.venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate

pip install -r backend/requirements.txt
pip install tqdm python-dotenv openai google-genai
```

Configure your `.env` in the project root:

```env
# Powers 9 of the 11 benchmark models (via Azure AI Inference)
GITHUB_TOKEN=your-github-pat-here

# Powers gemini-2.5-pro and gemini-2.5-flash
GEMINI_API_KEY=your-gemini-key-here
```

---

## The Dataset

The benchmark consists of **250 hand-curated problems** in `data/benchmarks/tyr_benchmark_250.json`. Each provides a deliberately suboptimal O(N^2) Python function and asks the LLM to produce a semantically equivalent version at a lower complexity class.

```json
{
  "id": "TYR-042",
  "name": "longest_unique_substring_length",
  "category": "sliding-window",
  "difficulty": "medium",
  "description": "Return the length of the longest substring without repeating characters.",
  "original_code": "def longest_unique_substring_length(s):\n    ...",
  "original_complexity": "O(N^2)",
  "target_complexity": "O(N)"
}
```

**Distribution:** 100 easy, 100 medium, 50 hard across 18 categories.

To regenerate the dataset from source definitions:

```bash
python scripts/build_dataset.py
# Validates all 250 problems via ast.parse()
# Outputs -> data/benchmarks/tyr_benchmark_250.json
```

---

## Stage 1: LLM Code Generation

`src/generators/generate_llm_benchmark.py` sends each problem to an LLM API and records the response.

### 11 Registered Models

| Suite | Model | Provider | Mode |
|:------|:------|:---------|:-----|
| System 1 | gpt-4.1 | github | Standard |
| System 1 | DeepSeek-V3-0324 | github | Standard |
| System 1 | Meta-Llama-3.1-405B-Instruct | github | Standard |
| System 1 | Codestral-2501 | github | Standard |
| System 1 | grok-3 | github | Standard |
| System 1 | gemini-2.5-pro | gemini | Standard |
| System 2 | gpt-5 | github | Reasoning |
| System 2 | o3 | github | Reasoning |
| System 2 | o4-mini | github | Reasoning |
| System 2 | DeepSeek-R1-0528 | github | Reasoning |
| System 2 | gemini-2.5-flash | gemini | Thinking |

### Usage

**Run a single model:**

```bash
python src/generators/generate_llm_benchmark.py \
    --provider github --model gpt-4.1 --api-key ghp_XXX
```

**Run an entire suite:**

```bash
# System 1 (6 standard models)
python src/generators/generate_llm_benchmark.py --suite system1

# System 2 (5 reasoning models)
python src/generators/generate_llm_benchmark.py --suite system2

# All 11 models
python src/generators/generate_llm_benchmark.py --suite all
```

**List available models and key status:**

```bash
python src/generators/generate_llm_benchmark.py --list-models
```

**Interactive mode** (no flags -- presents a numbered picker):

```bash
python src/generators/generate_llm_benchmark.py
```

### CLI Flags

| Flag | Default | Description |
|:-----|:--------|:------------|
| `--provider` | auto | `github`, `openai`, `gemini`, `deepseek` |
| `--model` | -- | Model identifier from registry |
| `--suite` | -- | `system1`, `system2`, or `all` |
| `--api-key` | from `.env` | API key override |
| `--delay` | `4.5` | Seconds between API calls |
| `--thinking-budget` | per-model | Token budget for Gemini thinking models |
| `--dataset` | `data/benchmarks/tyr_benchmark_250.json` | Path to benchmark JSON |
| `--output-dir` | `data/raw/` | Output directory for CSVs |

### Output

Each run produces a CSV in `data/raw/` named `<provider>_<model>.csv`:

```
data/raw/
  github_gpt_4_1.csv
  github_gpt_5.csv
  github_o3.csv
  github_o4_mini.csv
  github_grok_3.csv
  github_codestral_2501.csv
  github_deepseek_v3_0324.csv
  github_deepseek_r1_0528.csv
  github_meta_llama_3_1_405b_instruct.csv
  gemini_gemini_2_5_pro.csv
  gemini_gemini_2_5_flash.csv
```

**Stage 1 CSV columns:**

| Column | Description |
|:-------|:------------|
| `id` | Problem ID (e.g., TYR-042) |
| `name` | Problem name |
| `model_name` | Model used |
| `category` | Problem category |
| `difficulty` | easy / medium / hard |
| `original_complexity` | Source Big-O (e.g., O(N^2)) |
| `target_complexity` | Requested Big-O (e.g., O(N)) |
| `original_code` | Input function |
| `generated_code` | LLM output |
| `latency_ms` | API response time |
| `prompt_tokens` | Prompt token count |
| `reasoning_tokens` | Reasoning/thinking tokens (if applicable) |
| `completion_tokens` | Output token count |
| `total_tokens` | Total tokens consumed |
| `api_status` | `OK`, `ERROR`, or `SYNTAX_ERROR` |
| `error_detail` | Error traceback (if any) |

### Resume Support

The script is **crash-safe**. Each row is appended immediately after the API call. On restart, previously processed problem IDs are skipped automatically.

---

## Stage 2: Formal Verification

`src/evaluators/verify_llm_results.py` reads Stage 1 CSVs and runs each `(original_code, generated_code)` pair through Tyr's Z3-based verification engine.

### Usage

**Verify all models at once:**

```bash
python src/evaluators/verify_llm_results.py --all
```

**Verify a single CSV:**

```bash
python src/evaluators/verify_llm_results.py --input data/raw/github_gpt_4_1.csv
```

**Verify a specific row range:**

```bash
python src/evaluators/verify_llm_results.py --input data/raw/github_gpt_4_1.csv --range 1-50
```

**Interactive mode** (no flags -- presents a file picker):

```bash
python src/evaluators/verify_llm_results.py
```

### Verification Modes

By default, the verifier imports the backend engine directly (zero network overhead). To use a running Tyr server instead:

```bash
python src/evaluators/verify_llm_results.py --all --url http://localhost:8000/verify-pair
```

### Output

Verified CSVs are written to `data/verified/` with the same filename as input. Four columns are appended to the Stage 1 schema:

| Column | Description |
|:-------|:------------|
| `verdict` | `UNSAT`, `SAT`, `WARNING`, `TIMEOUT`, or `ERROR` |
| `optimized_complexity_time` | Estimated Big-O of the generated code |
| `complexity_improved` | `True` / `False` |
| `verify_latency_ms` | Verification wall-clock time (ms) |

### Verdict Definitions

| Verdict | Meaning |
|:--------|:--------|
| **UNSAT** | Z3 proved the functions are equivalent within BMC bounds. No input exists (in the bounded domain) that produces different outputs. |
| **SAT** | Z3 found a concrete counterexample -- the LLM's optimization changed the function's behavior. This is a **confirmed hallucination**. |
| **WARNING** | Z3 timed out or hit unsupported constructs. Concrete fallback testing found no divergence, but this is empirical, not a proof. |
| **TIMEOUT** | Verification exceeded the time budget. |
| **ERROR** | Internal error during verification (parse failure, type mismatch, etc.). |

---

## Full Pipeline Example

```bash
# 1. Generate the dataset (if not already present)
python scripts/build_dataset.py

# 2. Run Stage 1 for all 11 models
python src/generators/generate_llm_benchmark.py --suite all

# 3. Run Stage 2 verification
python src/evaluators/verify_llm_results.py --all

# 4. Analyze results (from inside data/verified/)
cd data/verified
python analyze_metrics.py        # per-model statistics
python find_bug.py               # inspect a SAT case
python generate_graphs.py        # comparison chart
```
