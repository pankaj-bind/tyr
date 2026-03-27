# Analysis & Graph Generation

This guide covers the three analysis tools bundled in `data/verified/`. They are designed to run **from that directory**, directly alongside the verified CSVs they operate on.

**Prerequisites:** Verified CSVs must exist in `data/verified/`. See the [Benchmark Pipeline](BENCHMARK_GENERATION.md) for how to produce them.

```bash
pip install pandas matplotlib numpy
```

---

## Overview

All three scripts live in `data/verified/` and operate on the CSVs in the same folder:

| Script | Purpose | Output |
|:-------|:--------|:-------|
| `analyze_metrics.py` | Per-model summary statistics | Console output |
| `find_bug.py` | Extract and display a confirmed SAT hallucination | Console output |
| `generate_graphs.py` | Publication-ready stacked bar chart | `stacked_bar_chart.png` |

```bash
cd data/verified
```

---

## 1. Per-Model Metrics -- `analyze_metrics.py`

Prints optimization success rate, verdict distribution, and latency statistics for each model.

### Usage

```bash
python analyze_metrics.py
```

You'll see a picker listing all CSVs in the directory. Select a model or type `all` to run every CSV.

### Output

```
--------------------------------------------------
Model Name: github_gpt_5
--------------------------------------------------
  1. OPTIMIZATION SUCCESS RATE (Big-O Shift)
--------------------------------------------------
Successful O(N) Reductions: 145 / 250 (58.0%)
Failed to Optimize / Regression: 105 / 250 (42.0%)

--------------------------------------------------
  2. FORMAL VERIFICATION VERDICTS (Equivalence)
--------------------------------------------------
WARNING   : 97 cases (38.8%)
TIMEOUT   : 52 cases (20.8%)
UNSAT     : 56 cases (22.4%)
SAT       : 26 cases (10.4%)
ERROR     : 19 cases (7.6%)

--------------------------------------------------
  3. PERFORMANCE & LATENCY
--------------------------------------------------
Average Verification Latency : 4.21 seconds
Maximum Verification Latency : 10.02 seconds
Average Reasoning Tokens Used: 8,431 tokens
```

### What It Measures

| Metric | Description |
|:-------|:------------|
| **Optimization Success Rate** | Fraction of problems where the LLM actually improved time complexity |
| **Verdict Distribution** | Breakdown of UNSAT / SAT / WARNING / TIMEOUT / ERROR |
| **Avg Verification Latency** | Mean time for Z3 + concrete fallback per problem |
| **Max Verification Latency** | Worst-case verification time (usually near Z3 timeout) |
| **Avg Reasoning Tokens** | Mean reasoning/thinking tokens per problem (relevant for System 2 models) |

---

## 2. SAT Case Inspector -- `find_bug.py`

Loads all verified CSVs, finds a confirmed hallucination (SAT verdict), and prints the original vs. generated code side-by-side. Prioritizes bugs from Llama or Gemini models for maximum impact.

### Usage

```bash
python find_bug.py
```

### Output

```
  Master Verified Data Ready: 2750 rows.

  HUNTING FOR A PREMIUM 'SAT' BUG...

  FATAL LOGIC BUG CAUGHT!
  Model: github_meta_llama_3_1_405b_instruct
  Problem ID: TYR-087

  ==================================================
  ORIGINAL CODE O(N^2) [TRUSTED]
  ==================================================
  def find_majority_element(nums):
      for i in range(len(nums)):
          count = 0
          for j in range(len(nums)):
              if nums[j] == nums[i]:
                  count += 1
          if count > len(nums) // 2:
              return nums[i]
      return -1

  ==================================================
  GENERATED CODE O(N) [HALLUCINATION]
  ==================================================
  def find_majority_element(nums):
      candidate = nums[0]
      count = 1
      ...
```

---

## 3. Comparison Chart -- `generate_graphs.py`

Generates a stacked bar chart comparing all 11 models across 4 verdict categories (UNSAT, WARNING, SAT, ERROR).

### Usage

```bash
python generate_graphs.py
```

### Output

Saves `stacked_bar_chart.png` (300 DPI) in the current directory. A pre-generated version and PDF export are also included:

| File | Description |
|:-----|:------------|
| `stacked_bar_chart.png` | Stacked bar chart (11 models, 4 verdicts) |
| `Tyr_Multi_Model_Comparison_Fixed.pdf` | Publication-ready PDF version |

### Chart Details

- **X-axis:** 11 frontier models sorted by overall safety (left = safest)
- **Y-axis:** Percentage of 250 problems per verdict
- **Colors:**
  - Green (`#2ca02c`): UNSAT -- Provably Safe
  - Orange (`#f5b041`): WARNING -- Empirically Safe
  - Red (`#d62728`): SAT -- Bug Detected
  - Purple (`#9467bd`): ERROR -- Sandbox / Timeout
- All bars stack to exactly 100%

---

## Working with Raw Data

All CSVs follow the same schema. Quick pandas example:

```python
import pandas as pd
import glob

# Load all verified results
frames = [pd.read_csv(f) for f in sorted(glob.glob("*.csv"))]
all_results = pd.concat(frames, ignore_index=True)

# Verdict distribution across all models
print(all_results["verdict"].value_counts())

# Per-model failure rate (SAT = hallucination caught)
for model, group in all_results.groupby("model_name"):
    sat = (group["verdict"] == "SAT").sum()
    total = len(group)
    print(f"{model:<45s}  SAT={sat}/{total}  ({sat/total*100:.1f}%)")
```

### Key Columns

| Column | Type | Description |
|:-------|:-----|:------------|
| `verdict` | str | UNSAT / SAT / WARNING / TIMEOUT / ERROR |
| `complexity_improved` | str | "True" or "False" |
| `verify_latency_ms` | float | Verification wall-clock time |
| `latency_ms` | float | LLM API response time |
| `reasoning_tokens` | int | Thinking/reasoning tokens (System 2 models) |
| `category` | str | Problem category (pair-finding, subarray, etc.) |
| `difficulty` | str | easy / medium / hard |
