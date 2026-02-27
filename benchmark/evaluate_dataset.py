"""
Tyr — Empirical Evaluation Benchmark
=====================================
Sends a curated set of O(N²) algorithmic problems to the local Tyr backend
(/verify endpoint), records the verification verdict, wall-clock latency,
and the AI-generated optimized code, then persists everything to CSV for
use in a Formal Methods research paper.

Usage:
    python evaluate_dataset.py            # default: http://localhost:8000
    python evaluate_dataset.py --url http://host:port

Requires:
    pip install requests
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import sys
import threading
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_URL = "http://localhost:8000/verify"
TIMEOUT_S = 60  # 60s per problem (LLM + CGSC + Z3)
MAX_WORKERS = 6  # parallel problems in flight

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = OUTPUT_DIR / "paper_results.csv"

CSV_COLUMNS = [
    "Problem_Name",
    "Status",
    "Time_Taken_ms",
    "Correction_Rounds",
    "Original_Complexity",
    "Optimized_Complexity",
    "Complexity_Improved",
    "Original_Code",
    "Refactored_Code",
]

# ---------------------------------------------------------------------------
# Dataset — 30 genuine O(N²) brute-force solutions across algorithm categories
#
# Categories covered:
#   - Searching & Counting (7)
#   - Sorting & Ordering (5)
#   - Array Manipulation (6)
#   - Subarray / Subsequence (5)
#   - Pair / Triplet Problems (4)
#   - Mathematical / Numerical (3)
# ---------------------------------------------------------------------------

DATASET: list[dict[str, str]] = [
    # ── Searching & Counting ───────────────────────────────────────────
    {
        "name": "Two Sum",
        "original_code": (
            "def two_sum(nums, target):\n"
            "    n = len(nums)\n"
            "    for i in range(n):\n"
            "        for j in range(i + 1, n):\n"
            "            if nums[i] + nums[j] == target:\n"
            "                return i\n"
            "    return -1\n"
        ),
    },
    {
        "name": "Contains Duplicate",
        "original_code": (
            "def contains_duplicate(nums):\n"
            "    n = len(nums)\n"
            "    for i in range(n):\n"
            "        for j in range(i + 1, n):\n"
            "            if nums[i] == nums[j]:\n"
            "                return 1\n"
            "    return 0\n"
        ),
    },
    {
        "name": "Majority Element",
        "original_code": (
            "def majority_element(nums):\n"
            "    n = len(nums)\n"
            "    for i in range(n):\n"
            "        count = 0\n"
            "        for j in range(n):\n"
            "            if nums[j] == nums[i]:\n"
            "                count = count + 1\n"
            "        if count > n // 2:\n"
            "            return nums[i]\n"
            "    return -1\n"
        ),
    },
    {
        "name": "First Unique Character",
        "original_code": (
            "def first_unique(nums):\n"
            "    n = len(nums)\n"
            "    for i in range(n):\n"
            "        count = 0\n"
            "        for j in range(n):\n"
            "            if nums[j] == nums[i]:\n"
            "                count = count + 1\n"
            "        if count == 1:\n"
            "            return i\n"
            "    return -1\n"
        ),
    },
    {
        "name": "Find Duplicates",
        "original_code": (
            "def find_duplicates(nums):\n"
            "    result = []\n"
            "    n = len(nums)\n"
            "    for i in range(n):\n"
            "        for j in range(i + 1, n):\n"
            "            if nums[i] == nums[j]:\n"
            "                if nums[i] not in result:\n"
            "                    result.append(nums[i])\n"
            "    return result\n"
        ),
    },
    {
        "name": "Count Element Frequency",
        "original_code": (
            "def count_frequency(nums):\n"
            "    n = len(nums)\n"
            "    max_count = 0\n"
            "    for i in range(n):\n"
            "        count = 0\n"
            "        for j in range(n):\n"
            "            if nums[j] == nums[i]:\n"
            "                count = count + 1\n"
            "        if count > max_count:\n"
            "            max_count = count\n"
            "    return max_count\n"
        ),
    },
    {
        "name": "Element Appears K Times",
        "original_code": (
            "def appears_k_times(nums, k):\n"
            "    n = len(nums)\n"
            "    for i in range(n):\n"
            "        count = 0\n"
            "        for j in range(n):\n"
            "            if nums[j] == nums[i]:\n"
            "                count = count + 1\n"
            "        if count == k:\n"
            "            return nums[i]\n"
            "    return -1\n"
        ),
    },
    # ── Sorting & Ordering ─────────────────────────────────────────────
    {
        "name": "Selection Sort Min Index",
        "original_code": (
            "def selection_sort_passes(nums):\n"
            "    n = len(nums)\n"
            "    total = 0\n"
            "    for i in range(n):\n"
            "        min_idx = i\n"
            "        for j in range(i + 1, n):\n"
            "            if nums[j] < nums[min_idx]:\n"
            "                min_idx = j\n"
            "        total = total + min_idx\n"
            "    return total\n"
        ),
    },
    {
        "name": "Count Inversions",
        "original_code": (
            "def count_inversions(nums):\n"
            "    n = len(nums)\n"
            "    count = 0\n"
            "    for i in range(n):\n"
            "        for j in range(i + 1, n):\n"
            "            if nums[i] > nums[j]:\n"
            "                count = count + 1\n"
            "    return count\n"
        ),
    },
    {
        "name": "Is Sorted Check",
        "original_code": (
            "def is_sorted(nums):\n"
            "    n = len(nums)\n"
            "    for i in range(n):\n"
            "        for j in range(i + 1, n):\n"
            "            if nums[i] > nums[j]:\n"
            "                return 0\n"
            "    return 1\n"
        ),
    },
    {
        "name": "Kth Smallest Element",
        "original_code": (
            "def kth_smallest(nums, k):\n"
            "    n = len(nums)\n"
            "    for i in range(n):\n"
            "        count = 0\n"
            "        for j in range(n):\n"
            "            if nums[j] < nums[i]:\n"
            "                count = count + 1\n"
            "        if count == k:\n"
            "            return nums[i]\n"
            "    return -1\n"
        ),
    },
    {
        "name": "Second Largest",
        "original_code": (
            "def second_largest(nums):\n"
            "    n = len(nums)\n"
            "    largest = nums[0]\n"
            "    for i in range(1, n):\n"
            "        if nums[i] > largest:\n"
            "            largest = nums[i]\n"
            "    second = nums[0]\n"
            "    for i in range(1, n):\n"
            "        if nums[i] > second and nums[i] != largest:\n"
            "            second = nums[i]\n"
            "    return second\n"
        ),
    },
    # ── Array Manipulation ─────────────────────────────────────────────
    {
        "name": "Move Zeroes Count",
        "original_code": (
            "def move_zeroes_swaps(nums):\n"
            "    n = len(nums)\n"
            "    swaps = 0\n"
            "    for i in range(n):\n"
            "        for j in range(i + 1, n):\n"
            "            if nums[i] == 0 and nums[j] != 0:\n"
            "                swaps = swaps + 1\n"
            "    return swaps\n"
        ),
    },
    {
        "name": "Sum of All Pairs",
        "original_code": (
            "def sum_all_pairs(nums):\n"
            "    n = len(nums)\n"
            "    total = 0\n"
            "    for i in range(n):\n"
            "        for j in range(i + 1, n):\n"
            "            total = total + nums[i] + nums[j]\n"
            "    return total\n"
        ),
    },
    {
        "name": "Product Except Self Sum",
        "original_code": (
            "def product_except_self_sum(nums):\n"
            "    n = len(nums)\n"
            "    total = 0\n"
            "    for i in range(n):\n"
            "        product = 1\n"
            "        for j in range(n):\n"
            "            if j != i:\n"
            "                product = product * nums[j]\n"
            "        total = total + product\n"
            "    return total\n"
        ),
    },
    {
        "name": "Running Sum Brute",
        "original_code": (
            "def running_sum_total(nums):\n"
            "    n = len(nums)\n"
            "    total = 0\n"
            "    for i in range(n):\n"
            "        prefix = 0\n"
            "        for j in range(i + 1):\n"
            "            prefix = prefix + nums[j]\n"
            "        total = total + prefix\n"
            "    return total\n"
        ),
    },
    {
        "name": "Max Difference",
        "original_code": (
            "def max_difference(nums):\n"
            "    n = len(nums)\n"
            "    max_diff = 0\n"
            "    for i in range(n):\n"
            "        for j in range(i + 1, n):\n"
            "            diff = nums[j] - nums[i]\n"
            "            if diff > max_diff:\n"
            "                max_diff = diff\n"
            "    return max_diff\n"
        ),
    },
    {
        "name": "Equilibrium Index",
        "original_code": (
            "def equilibrium_index(nums):\n"
            "    n = len(nums)\n"
            "    for i in range(n):\n"
            "        left_sum = 0\n"
            "        for j in range(i):\n"
            "            left_sum = left_sum + nums[j]\n"
            "        right_sum = 0\n"
            "        for j in range(i + 1, n):\n"
            "            right_sum = right_sum + nums[j]\n"
            "        if left_sum == right_sum:\n"
            "            return i\n"
            "    return -1\n"
        ),
    },
    # ── Subarray / Subsequence ─────────────────────────────────────────
    {
        "name": "Max Subarray Sum",
        "original_code": (
            "def max_subarray_sum(nums):\n"
            "    n = len(nums)\n"
            "    max_sum = nums[0]\n"
            "    for i in range(n):\n"
            "        current = 0\n"
            "        for j in range(i, n):\n"
            "            current = current + nums[j]\n"
            "            if current > max_sum:\n"
            "                max_sum = current\n"
            "    return max_sum\n"
        ),
    },
    {
        "name": "Subarray Sum Equals K",
        "original_code": (
            "def subarray_sum_k(nums, k):\n"
            "    n = len(nums)\n"
            "    count = 0\n"
            "    for i in range(n):\n"
            "        total = 0\n"
            "        for j in range(i, n):\n"
            "            total = total + nums[j]\n"
            "            if total == k:\n"
            "                count = count + 1\n"
            "    return count\n"
        ),
    },
    {
        "name": "Longest Increasing Subsequence Length",
        "original_code": (
            "def lis_length(nums):\n"
            "    n = len(nums)\n"
            "    max_len = 1\n"
            "    for i in range(n):\n"
            "        length = 1\n"
            "        prev = nums[i]\n"
            "        for j in range(i + 1, n):\n"
            "            if nums[j] > prev:\n"
            "                length = length + 1\n"
            "                prev = nums[j]\n"
            "        if length > max_len:\n"
            "            max_len = length\n"
            "    return max_len\n"
        ),
    },
    {
        "name": "Count Distinct Subarrays",
        "original_code": (
            "def count_distinct_subarrays(nums):\n"
            "    n = len(nums)\n"
            "    count = 0\n"
            "    for i in range(n):\n"
            "        seen = []\n"
            "        all_distinct = 1\n"
            "        for j in range(i, n):\n"
            "            if nums[j] in seen:\n"
            "                all_distinct = 0\n"
            "            else:\n"
            "                seen.append(nums[j])\n"
            "            if all_distinct == 1:\n"
            "                count = count + 1\n"
            "    return count\n"
        ),
    },
    {
        "name": "Min Subarray Length",
        "original_code": (
            "def min_subarray_len(nums, target):\n"
            "    n = len(nums)\n"
            "    min_len = n + 1\n"
            "    for i in range(n):\n"
            "        total = 0\n"
            "        for j in range(i, n):\n"
            "            total = total + nums[j]\n"
            "            if total >= target:\n"
            "                length = j - i + 1\n"
            "                if length < min_len:\n"
            "                    min_len = length\n"
            "                break\n"
            "    if min_len == n + 1:\n"
            "        return 0\n"
            "    return min_len\n"
        ),
    },
    # ── Pair / Triplet Problems ────────────────────────────────────────
    {
        "name": "Count Pairs With Sum",
        "original_code": (
            "def count_pairs_sum(nums, target):\n"
            "    n = len(nums)\n"
            "    count = 0\n"
            "    for i in range(n):\n"
            "        for j in range(i + 1, n):\n"
            "            if nums[i] + nums[j] == target:\n"
            "                count = count + 1\n"
            "    return count\n"
        ),
    },
    {
        "name": "Three Sum Count",
        "original_code": (
            "def three_sum_count(nums, target):\n"
            "    n = len(nums)\n"
            "    count = 0\n"
            "    for i in range(n):\n"
            "        for j in range(i + 1, n):\n"
            "            remaining = target - nums[i] - nums[j]\n"
            "            for k in range(j + 1, n):\n"
            "                if nums[k] == remaining:\n"
            "                    count = count + 1\n"
            "    return count\n"
        ),
    },
    {
        "name": "Closest Pair Sum",
        "original_code": (
            "def closest_pair_sum(nums, target):\n"
            "    n = len(nums)\n"
            "    best = nums[0] + nums[1]\n"
            "    for i in range(n):\n"
            "        for j in range(i + 1, n):\n"
            "            s = nums[i] + nums[j]\n"
            "            if abs(s - target) < abs(best - target):\n"
            "                best = s\n"
            "    return best\n"
        ),
    },
    {
        "name": "Max Product Pair",
        "original_code": (
            "def max_product_pair(nums):\n"
            "    n = len(nums)\n"
            "    max_prod = nums[0] * nums[1]\n"
            "    for i in range(n):\n"
            "        for j in range(i + 1, n):\n"
            "            prod = nums[i] * nums[j]\n"
            "            if prod > max_prod:\n"
            "                max_prod = prod\n"
            "    return max_prod\n"
        ),
    },
    # ── Mathematical / Numerical ───────────────────────────────────────
    {
        "name": "Find Missing Number",
        "original_code": (
            "def find_missing(nums):\n"
            "    n = len(nums)\n"
            "    for i in range(n + 1):\n"
            "        found = 0\n"
            "        for j in range(n):\n"
            "            if nums[j] == i:\n"
            "                found = 1\n"
            "        if found == 0:\n"
            "            return i\n"
            "    return -1\n"
        ),
    },
    {
        "name": "Single Number (XOR)",
        "original_code": (
            "def single_number(nums):\n"
            "    n = len(nums)\n"
            "    for i in range(n):\n"
            "        count = 0\n"
            "        for j in range(n):\n"
            "            if nums[j] == nums[i]:\n"
            "                count = count + 1\n"
            "        if count == 1:\n"
            "            return nums[i]\n"
            "    return -1\n"
        ),
    },
    {
        "name": "Max Stock Profit",
        "original_code": (
            "def max_profit(prices):\n"
            "    n = len(prices)\n"
            "    max_p = 0\n"
            "    for i in range(n):\n"
            "        for j in range(i + 1, n):\n"
            "            profit = prices[j] - prices[i]\n"
            "            if profit > max_p:\n"
            "                max_p = profit\n"
            "    return max_p\n"
        ),
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _send_verify(url: str, code: str) -> tuple[dict, int]:
    """
    POST to /verify and return (response_json, elapsed_ms).

    Raises on HTTP or connection errors.
    """
    payload = {"code": code, "language": "python"}
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=TIMEOUT_S)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    resp.raise_for_status()
    return resp.json(), elapsed_ms


def _sanitise(text: str) -> str:
    """Collapse newlines for CSV readability."""
    return text.replace("\r\n", "\n").strip()


def _write_csv(rows: list[dict[str, str]]) -> None:
    """Write all accumulated rows to CSV (called after each problem)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Single-problem worker (runs in thread pool)
# ---------------------------------------------------------------------------

def _run_one(idx: int, total: int, problem: dict, url: str) -> dict[str, str]:
    """Verify one problem and return a CSV row dict.  Thread-safe."""
    name = problem["name"]
    code = problem["original_code"]
    tag = f"[{idx}/{total}]"

    try:
        data, elapsed_ms = _send_verify(url, code)
    except requests.ConnectionError:
        print(f"  {tag} {name} ... CONNECT_ERROR", flush=True)
        return _error_row(name, code, "CONNECT_ERROR")
    except requests.HTTPError as exc:
        st = f"HTTP_{exc.response.status_code}"
        print(f"  {tag} {name} ... {st}", flush=True)
        return _error_row(name, code, st)
    except Exception as exc:
        print(f"  {tag} {name} ... ERROR ({exc})", flush=True)
        return _error_row(name, code, "ERROR")

    status = data.get("status", "UNKNOWN")
    optimized = data.get("optimized_code", "")
    rounds = data.get("total_rounds", 0)
    orig_cx = data.get("original_complexity", {}) or {}
    opt_cx = data.get("optimized_complexity", {}) or {}
    cx_improved = data.get("complexity_improved")
    orig_time = orig_cx.get("time", "")
    opt_time = opt_cx.get("time", "")

    label = f"[{status}]"
    if cx_improved is True:
        label += f" {orig_time}->{opt_time}"
    print(f"  {tag} {name} ... {label} in {elapsed_ms}ms ({rounds}r)", flush=True)

    return {
        "Problem_Name": name,
        "Status": status,
        "Time_Taken_ms": str(elapsed_ms),
        "Correction_Rounds": str(rounds),
        "Original_Complexity": orig_time,
        "Optimized_Complexity": opt_time,
        "Complexity_Improved": str(cx_improved) if cx_improved is not None else "",
        "Original_Code": _sanitise(code),
        "Refactored_Code": _sanitise(optimized),
    }


def _error_row(name: str, code: str, status: str) -> dict[str, str]:
    return {
        "Problem_Name": name,
        "Status": status,
        "Time_Taken_ms": "0",
        "Correction_Rounds": "0",
        "Original_Complexity": "",
        "Optimized_Complexity": "",
        "Complexity_Improved": "",
        "Original_Code": _sanitise(code),
        "Refactored_Code": "",
    }


# ---------------------------------------------------------------------------
# Main - parallel execution
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tyr Empirical Evaluation")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Verify endpoint URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Parallel workers (default: {MAX_WORKERS})",
    )
    args = parser.parse_args()
    url: str = args.url
    workers: int = args.workers

    print(f"\n{'=' * 60}")
    print("  Tyr - Empirical Evaluation Benchmark (PARALLEL)")
    print(f"  Endpoint : {url}")
    print(f"  Problems : {len(DATASET)}")
    print(f"  Workers  : {workers}")
    print(f"  Output   : {OUTPUT_CSV}")
    print(f"{'=' * 60}\n")

    # Quick connectivity check
    try:
        requests.get(url.rsplit("/", 1)[0] + "/health", timeout=5)
    except Exception:
        print("[WARN] Could not reach /health - make sure the backend is running.\n")

    total = len(DATASET)
    results: list[dict[str, str] | None] = [None] * total
    t_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx: dict[concurrent.futures.Future, int] = {}
        for i, problem in enumerate(DATASET):
            fut = pool.submit(_run_one, i + 1, total, problem, url)
            future_to_idx[fut] = i

        done_count = 0
        for fut in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                name = DATASET[idx]["name"]
                code = DATASET[idx]["original_code"]
                print(f"  [{idx+1}/{total}] {name} ... WORKER_ERROR ({exc})", flush=True)
                results[idx] = _error_row(name, code, "WORKER_ERROR")
            done_count += 1

            # Incremental CSV write with whatever we have so far
            completed_rows = [r for r in results if r is not None]
            _write_csv(completed_rows)

    rows = [r for r in results if r is not None]
    _write_csv(rows)  # final ordered write

    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"Results saved to {OUTPUT_CSV}")
    print(f"  Total wall-clock : {elapsed_total:.1f}s")

    # Summary table
    unsat = sum(1 for r in rows if r["Status"] == "UNSAT")
    sat = sum(1 for r in rows if r["Status"] == "SAT")
    warn = sum(1 for r in rows if r["Status"] == "WARNING")
    errs = len(rows) - unsat - sat - warn
    avg_ms = (
        int(sum(int(r["Time_Taken_ms"]) for r in rows) / len(rows))
        if rows else 0
    )
    print(f"  UNSAT (proven equivalent) : {unsat}/{len(rows)}")
    print(f"  SAT   (semantics differ)  : {sat}/{len(rows)}")
    print(f"  WARNING (empirical only)  : {warn}/{len(rows)}")
    print(f"  ERROR / other             : {errs}/{len(rows)}")
    print(f"  Avg latency               : {avg_ms}ms")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
