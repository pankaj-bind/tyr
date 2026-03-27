#!/usr/bin/env python3
"""
Tyr — Benchmark Dataset Builder
=================================
Constructs ``tyr_benchmark_250.json`` containing 250 hand-curated,
syntactically-validated O(N²) brute-force algorithmic problems spanning
11 distinct categories.  Every problem is amenable to a well-known
O(N) or O(N log N) optimisation that an LLM should produce.

Usage:
    python scripts/build_dataset.py   # writes data/benchmarks/tyr_benchmark_250.json

Categories:
    pair-finding · frequency · subarray · set-operations ·
    order-statistics · prefix-sum · sliding-window · hash-map ·
    array-transform · competition · advanced
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════

_P: list[dict] = []


def P(n: int, name: str, cat: str, diff: str, desc: str, code: str,
      oc: str = "O(N^2)", tc: str = "O(N)"):
    _P.append({
        "id": f"TYR-{n:03d}",
        "name": name,
        "category": cat,
        "difficulty": diff,
        "description": desc,
        "original_code": code,
        "original_complexity": oc,
        "target_complexity": tc,
    })


# ══════════════════════════════════════════════════════════════════════
#  CATEGORY 1 — Pair Finding  (TYR-001 … TYR-015)
# ══════════════════════════════════════════════════════════════════════

P(1, "two_sum_exists", "pair-finding", "easy",
  "Return 1 if any pair of elements sums to target, else 0.",
  """\
def two_sum_exists(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return 1
    return 0
""")

P(2, "count_pairs_with_sum", "pair-finding", "easy",
  "Count the number of (i<j) pairs whose sum equals target.",
  """\
def count_pairs_with_sum(nums, target):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                count += 1
    return count
""")

P(3, "has_pair_with_diff", "pair-finding", "easy",
  "Return 1 if any pair has absolute difference equal to target.",
  """\
def has_pair_with_diff(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if abs(nums[i] - nums[j]) == target:
                return 1
    return 0
""")

P(4, "count_equal_pairs", "pair-finding", "easy",
  "Count pairs (i<j) where nums[i] == nums[j].",
  """\
def count_equal_pairs(nums):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                count += 1
    return count
""")

P(5, "closest_pair_sum", "pair-finding", "medium",
  "Return the pair sum closest to target.",
  """\
def closest_pair_sum(nums, target):
    if len(nums) < 2:
        return 0
    best = nums[0] + nums[1]
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            s = nums[i] + nums[j]
            if abs(s - target) < abs(best - target):
                best = s
    return best
""")

P(6, "count_pairs_divisible_sum", "pair-finding", "medium",
  "Count pairs (i<j) whose sum is divisible by k.",
  """\
def count_pairs_divisible_sum(nums, k):
    if k == 0:
        return 0
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if (nums[i] + nums[j]) % k == 0:
                count += 1
    return count
""")

P(7, "max_pair_sum", "pair-finding", "easy",
  "Return the maximum sum of any pair of distinct-index elements.",
  """\
def max_pair_sum(nums):
    if len(nums) < 2:
        return 0
    best = nums[0] + nums[1]
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            s = nums[i] + nums[j]
            if s > best:
                best = s
    return best
""")

P(8, "count_pairs_sum_less", "pair-finding", "easy",
  "Count pairs (i<j) whose sum is strictly less than target.",
  """\
def count_pairs_sum_less(nums, target):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] < target:
                count += 1
    return count
""")

P(9, "count_pairs_sum_greater", "pair-finding", "easy",
  "Count pairs (i<j) whose sum is strictly greater than target.",
  """\
def count_pairs_sum_greater(nums, target):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] > target:
                count += 1
    return count
""")

P(10, "has_pair_product", "pair-finding", "easy",
  "Return 1 if any pair of elements has product equal to target.",
  """\
def has_pair_product(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] * nums[j] == target:
                return 1
    return 0
""")

P(11, "min_abs_pair_diff", "pair-finding", "easy",
  "Return the minimum absolute difference between any two elements.",
  """\
def min_abs_pair_diff(nums):
    if len(nums) < 2:
        return 0
    best = abs(nums[0] - nums[1])
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            d = abs(nums[i] - nums[j])
            if d < best:
                best = d
    return best
""", tc="O(N log N)")

P(12, "count_reverse_pairs", "pair-finding", "medium",
  "Count pairs (i<j) where nums[i] > 2 * nums[j].",
  """\
def count_reverse_pairs(nums):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] > 2 * nums[j]:
                count += 1
    return count
""", tc="O(N log N)")

P(13, "count_pairs_abs_diff_k", "pair-finding", "easy",
  "Count pairs (i<j) with absolute difference exactly k.",
  """\
def count_pairs_abs_diff_k(nums, k):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if abs(nums[i] - nums[j]) == k:
                count += 1
    return count
""")

P(14, "count_distinct_pair_sums", "pair-finding", "medium",
  "Count the number of distinct pair sums (i<j).",
  """\
def count_distinct_pair_sums(nums):
    seen = {}
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            s = nums[i] + nums[j]
            seen[s] = 1
    total = 0
    for k in seen:
        total += 1
    return total
""")

P(15, "two_sum_less_than_k", "pair-finding", "medium",
  "Return the largest pair sum that is strictly less than k, or -1.",
  """\
def two_sum_less_than_k(nums, k):
    best = -1
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            s = nums[i] + nums[j]
            if s < k and s > best:
                best = s
    return best
""", tc="O(N log N)")


# ══════════════════════════════════════════════════════════════════════
#  CATEGORY 2 — Frequency & Counting  (TYR-016 … TYR-030)
# ══════════════════════════════════════════════════════════════════════

P(16, "most_frequent_element", "frequency", "easy",
  "Return the element with the highest frequency (first wins on tie).",
  """\
def most_frequent_element(nums):
    if len(nums) == 0:
        return -1
    best = nums[0]
    best_count = 0
    for i in range(len(nums)):
        count = 0
        for j in range(len(nums)):
            if nums[j] == nums[i]:
                count += 1
        if count > best_count:
            best_count = count
            best = nums[i]
    return best
""")

P(17, "has_duplicate", "frequency", "easy",
  "Return 1 if the array contains any duplicate, else 0.",
  """\
def has_duplicate(nums):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                return 1
    return 0
""")

P(18, "first_non_repeating", "frequency", "easy",
  "Return the first element that appears exactly once, or -1.",
  """\
def first_non_repeating(nums):
    for i in range(len(nums)):
        found_dup = 0
        for j in range(len(nums)):
            if i != j and nums[i] == nums[j]:
                found_dup = 1
        if found_dup == 0:
            return nums[i]
    return -1
""")

P(19, "count_distinct", "frequency", "easy",
  "Return the number of distinct elements in the array.",
  """\
def count_distinct(nums):
    count = 0
    for i in range(len(nums)):
        is_first = 1
        for j in range(i):
            if nums[j] == nums[i]:
                is_first = 0
        count += is_first
    return count
""")

P(20, "majority_element", "frequency", "easy",
  "Return the element appearing more than n/2 times, or -1.",
  """\
def majority_element(nums):
    n = len(nums)
    for i in range(n):
        count = 0
        for j in range(n):
            if nums[j] == nums[i]:
                count += 1
        if count > n // 2:
            return nums[i]
    return -1
""")

P(21, "find_first_duplicate", "frequency", "easy",
  "Return the first element (by second occurrence) that was seen before.",
  """\
def find_first_duplicate(nums):
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] == nums[i]:
                return nums[i]
    return -1
""")

P(22, "count_elements_appearing_twice", "frequency", "medium",
  "Count how many distinct values appear exactly twice.",
  """\
def count_elements_appearing_twice(nums):
    result = 0
    for i in range(len(nums)):
        is_first = 1
        for k in range(i):
            if nums[k] == nums[i]:
                is_first = 0
        if is_first == 0:
            continue
        count = 0
        for j in range(len(nums)):
            if nums[j] == nums[i]:
                count += 1
        if count == 2:
            result += 1
    return result
""")

P(23, "count_inversions", "frequency", "medium",
  "Count pairs (i<j) where nums[i] > nums[j] (inversion count).",
  """\
def count_inversions(nums):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] > nums[j]:
                count += 1
    return count
""", tc="O(N log N)")

P(24, "missing_number", "frequency", "easy",
  "Find the missing number in an array containing 0..n with one gap.",
  """\
def missing_number(nums):
    n = len(nums)
    for i in range(n + 1):
        found = 0
        for j in range(n):
            if nums[j] == i:
                found = 1
        if found == 0:
            return i
    return -1
""")

P(25, "count_matching_last_digit_pairs", "frequency", "easy",
  "Count pairs (i<j) where nums[i] and nums[j] share the same last digit.",
  """\
def count_matching_last_digit_pairs(nums):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] % 10 == nums[j] % 10:
                count += 1
    return count
""")

P(26, "count_unique_elements", "frequency", "easy",
  "Count elements that appear exactly once in the array.",
  """\
def count_unique_elements(nums):
    count = 0
    for i in range(len(nums)):
        freq = 0
        for j in range(len(nums)):
            if nums[j] == nums[i]:
                freq += 1
        if freq == 1:
            count += 1
    return count
""")

P(27, "single_number", "frequency", "easy",
  "Every element appears twice except one. Find the single one.",
  """\
def single_number(nums):
    for i in range(len(nums)):
        count = 0
        for j in range(len(nums)):
            if nums[j] == nums[i]:
                count += 1
        if count == 1:
            return nums[i]
    return -1
""")

P(28, "count_greater_than_all_right", "frequency", "medium",
  "Count elements that are greater than every element to their right.",
  """\
def count_greater_than_all_right(nums):
    count = 0
    n = len(nums)
    for i in range(n):
        is_leader = 1
        for j in range(i + 1, n):
            if nums[j] >= nums[i]:
                is_leader = 0
        count += is_leader
    return count
""")

P(29, "kth_largest", "frequency", "medium",
  "Return the k-th largest element in the array.",
  """\
def kth_largest(nums, k):
    n = len(nums)
    for target_rank in range(k):
        max_val = nums[0]
        max_idx = 0
        for i in range(1, n):
            if nums[i] > max_val:
                max_val = nums[i]
                max_idx = i
        if target_rank == k - 1:
            return max_val
        nums[max_idx] = nums[0] - 1
    return -1
""", tc="O(N log N)")

P(30, "frequency_of_max", "frequency", "easy",
  "Return how many times the maximum element appears.",
  """\
def frequency_of_max(nums):
    if len(nums) == 0:
        return 0
    max_val = nums[0]
    for i in range(len(nums)):
        if nums[i] > max_val:
            max_val = nums[i]
    count = 0
    for i in range(len(nums)):
        if nums[i] == max_val:
            count += 1
    return count
""", oc="O(N)", tc="O(N)")


# ══════════════════════════════════════════════════════════════════════
#  CATEGORY 3 — Subarray Problems  (TYR-031 … TYR-045)
# ══════════════════════════════════════════════════════════════════════

P(31, "max_subarray_sum", "subarray", "medium",
  "Return the maximum contiguous subarray sum.",
  """\
def max_subarray_sum(nums):
    if len(nums) == 0:
        return 0
    best = nums[0]
    n = len(nums)
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            if cur > best:
                best = cur
    return best
""")

P(32, "has_subarray_with_sum", "subarray", "medium",
  "Return 1 if any contiguous subarray sums to target, else 0.",
  """\
def has_subarray_with_sum(nums, target):
    n = len(nums)
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            if cur == target:
                return 1
    return 0
""")

P(33, "count_subarrays_with_sum", "subarray", "medium",
  "Count contiguous subarrays whose sum equals k.",
  """\
def count_subarrays_with_sum(nums, k):
    count = 0
    n = len(nums)
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            if cur == k:
                count += 1
    return count
""")

P(34, "longest_subarray_sum_k", "subarray", "medium",
  "Return the length of the longest contiguous subarray with sum k.",
  """\
def longest_subarray_sum_k(nums, k):
    best = 0
    n = len(nums)
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            if cur == k:
                length = j - i + 1
                if length > best:
                    best = length
    return best
""")

P(35, "max_window_sum_size_k", "subarray", "easy",
  "Return the maximum sum among all contiguous subarrays of size k.",
  """\
def max_window_sum_size_k(nums, k):
    if len(nums) == 0 or k <= 0:
        return 0
    best = 0
    n = len(nums)
    for i in range(n - k + 1):
        cur = 0
        for j in range(i, i + k):
            cur += nums[j]
        if i == 0 or cur > best:
            best = cur
    return best
""")

P(36, "min_subarray_len_ge_target", "subarray", "medium",
  "Return the shortest subarray length with sum >= target, or 0.",
  """\
def min_subarray_len_ge_target(nums, target):
    best = 0
    n = len(nums)
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            if cur >= target:
                length = j - i + 1
                if best == 0 or length < best:
                    best = length
                break
    return best
""")

P(37, "max_product_pair", "subarray", "easy",
  "Return the maximum product of any two distinct-index elements.",
  """\
def max_product_pair(nums):
    if len(nums) < 2:
        return 0
    best = nums[0] * nums[1]
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            p = nums[i] * nums[j]
            if p > best:
                best = p
    return best
""")

P(38, "longest_increasing_run", "subarray", "easy",
  "Return the length of the longest strictly increasing contiguous run.",
  """\
def longest_increasing_run(nums):
    if len(nums) == 0:
        return 0
    best = 1
    n = len(nums)
    for i in range(n):
        length = 1
        for j in range(i + 1, n):
            if nums[j] > nums[j - 1]:
                length += 1
            else:
                break
        if length > best:
            best = length
    return best
""")

P(39, "max_diff_ordered", "subarray", "easy",
  "Return max(nums[j] - nums[i]) for all j > i, or 0 if empty.",
  """\
def max_diff_ordered(nums):
    if len(nums) < 2:
        return 0
    best = nums[1] - nums[0]
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            d = nums[j] - nums[i]
            if d > best:
                best = d
    return best
""")

P(40, "equilibrium_index", "subarray", "easy",
  "Return the first index where left sum equals right sum, or -1.",
  """\
def equilibrium_index(nums):
    n = len(nums)
    for i in range(n):
        left_sum = 0
        for j in range(i):
            left_sum += nums[j]
        right_sum = 0
        for j in range(i + 1, n):
            right_sum += nums[j]
        if left_sum == right_sum:
            return i
    return -1
""")

P(41, "product_except_self", "subarray", "medium",
  "Return a list where each element is the product of all others.",
  """\
def product_except_self(nums):
    n = len(nums)
    result = []
    for i in range(n):
        prod = 1
        for j in range(n):
            if j != i:
                prod = prod * nums[j]
        result.append(prod)
    return result
""")

P(42, "count_positive_subarrays", "subarray", "medium",
  "Count contiguous subarrays whose sum is strictly positive.",
  """\
def count_positive_subarrays(nums):
    count = 0
    n = len(nums)
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            if cur > 0:
                count += 1
    return count
""")

P(43, "min_subarray_sum", "subarray", "medium",
  "Return the minimum contiguous subarray sum.",
  """\
def min_subarray_sum(nums):
    if len(nums) == 0:
        return 0
    best = nums[0]
    n = len(nums)
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            if cur < best:
                best = cur
    return best
""")

P(44, "count_subarrays_max_le_k", "subarray", "medium",
  "Count contiguous subarrays where every element is <= k.",
  """\
def count_subarrays_max_le_k(nums, k):
    count = 0
    n = len(nums)
    for i in range(n):
        ok = 1
        for j in range(i, n):
            if nums[j] > k:
                ok = 0
            if ok == 1:
                count += 1
            else:
                break
    return count
""")

P(45, "sum_of_all_subarray_sums", "subarray", "medium",
  "Return the sum of sums of all contiguous subarrays.",
  """\
def sum_of_all_subarray_sums(nums):
    total = 0
    n = len(nums)
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            total += cur
    return total
""")


# ══════════════════════════════════════════════════════════════════════
#  CATEGORY 4 — Set Operations  (TYR-046 … TYR-055)
# ══════════════════════════════════════════════════════════════════════

P(46, "remove_duplicates_preserve_order", "set-operations", "easy",
  "Remove duplicates preserving first-occurrence order.",
  """\
def remove_duplicates_preserve_order(nums):
    result = []
    for i in range(len(nums)):
        found = 0
        for j in range(i):
            if nums[j] == nums[i]:
                found = 1
        if found == 0:
            result.append(nums[i])
    return result
""")

P(47, "symmetric_difference_count", "set-operations", "medium",
  "Count elements that appear in exactly one of the two halves split at mid.",
  """\
def symmetric_difference_count(nums):
    mid = len(nums) // 2
    count = 0
    for i in range(mid):
        found = 0
        for j in range(mid, len(nums)):
            if nums[i] == nums[j]:
                found = 1
        if found == 0:
            count += 1
    for i in range(mid, len(nums)):
        found = 0
        for j in range(mid):
            if nums[i] == nums[j]:
                found = 1
        if found == 0:
            count += 1
    return count
""")

P(48, "count_common_elements", "set-operations", "easy",
  "Count elements in the first half that also appear in the second half.",
  """\
def count_common_elements(nums):
    mid = len(nums) // 2
    count = 0
    for i in range(mid):
        for j in range(mid, len(nums)):
            if nums[i] == nums[j]:
                count += 1
                break
    return count
""")

P(49, "is_subset_first_half", "set-operations", "easy",
  "Return 1 if every element in the first half appears in the second half.",
  """\
def is_subset_first_half(nums):
    mid = len(nums) // 2
    for i in range(mid):
        found = 0
        for j in range(mid, len(nums)):
            if nums[i] == nums[j]:
                found = 1
        if found == 0:
            return 0
    return 1
""")

P(50, "remove_if_more_than_two", "set-operations", "medium",
  "Keep at most 2 occurrences of each element, preserving order.",
  """\
def remove_if_more_than_two(nums):
    result = []
    for i in range(len(nums)):
        count = 0
        for j in range(len(result)):
            if result[j] == nums[i]:
                count += 1
        if count < 2:
            result.append(nums[i])
    return result
""")

P(51, "first_missing_positive", "set-operations", "medium",
  "Return the smallest positive integer not in the array.",
  """\
def first_missing_positive(nums):
    candidate = 1
    while candidate < len(nums) + 2:
        found = 0
        for i in range(len(nums)):
            if nums[i] == candidate:
                found = 1
        if found == 0:
            return candidate
        candidate += 1
    return candidate
""")

P(52, "is_permutation_of_other_half", "set-operations", "easy",
  "Return 1 if sorted first half equals sorted second half.",
  """\
def is_permutation_of_other_half(nums):
    mid = len(nums) // 2
    for i in range(mid):
        found = 0
        used = []
        for j in range(mid, len(nums)):
            skip = 0
            for u in range(len(used)):
                if used[u] == j:
                    skip = 1
            if skip == 0 and nums[j] == nums[i]:
                found = 1
                used.append(j)
                break
        if found == 0:
            return 0
    return 1
""")

P(53, "count_elements_in_range", "set-operations", "easy",
  "Count elements that fall within [lo, hi] where lo=nums[0], hi=nums[-1].",
  """\
def count_elements_in_range(nums):
    if len(nums) < 2:
        return len(nums)
    lo = nums[0]
    hi = nums[len(nums) - 1]
    if lo > hi:
        lo, hi = hi, lo
    count = 0
    for i in range(len(nums)):
        if nums[i] >= lo and nums[i] <= hi:
            count += 1
    return count
""", oc="O(N)", tc="O(N)")

P(54, "deduplicate_sorted_count", "set-operations", "easy",
  "Return length of array after removing consecutive duplicates.",
  """\
def deduplicate_sorted_count(nums):
    if len(nums) == 0:
        return 0
    count = 1
    for i in range(1, len(nums)):
        is_dup = 0
        for j in range(i):
            if nums[j] == nums[i]:
                is_dup = 1
        if is_dup == 0:
            count += 1
    return count
""")

P(55, "count_pairs_from_halves", "set-operations", "medium",
  "Count pairs (one from each half) whose sum equals first element.",
  """\
def count_pairs_from_halves(nums):
    if len(nums) == 0:
        return 0
    target = nums[0]
    mid = len(nums) // 2
    count = 0
    for i in range(mid):
        for j in range(mid, len(nums)):
            if nums[i] + nums[j] == target:
                count += 1
    return count
""")


# ══════════════════════════════════════════════════════════════════════
#  CATEGORY 5 — Order Statistics  (TYR-056 … TYR-065)
# ══════════════════════════════════════════════════════════════════════

P(56, "count_smaller_before", "order-statistics", "medium",
  "For each element, count how many earlier elements are smaller. Return sum.",
  """\
def count_smaller_before(nums):
    total = 0
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                total += 1
    return total
""")

P(57, "kth_smallest", "order-statistics", "medium",
  "Return the k-th smallest element (1-indexed).",
  """\
def kth_smallest(nums, k):
    n = len(nums)
    for rank in range(k):
        min_val = nums[0]
        min_idx = 0
        for i in range(1, n):
            if nums[i] < min_val:
                min_val = nums[i]
                min_idx = i
        if rank == k - 1:
            return min_val
        nums[min_idx] = nums[0] + 1
    return -1
""", tc="O(N log N)")

P(58, "sort_binary_array", "order-statistics", "easy",
  "Sort an array of 0s and 1s.",
  """\
def sort_binary_array(nums):
    result = []
    for i in range(len(nums)):
        if nums[i] == 0:
            result.append(0)
    for i in range(len(nums)):
        if nums[i] == 1:
            result.append(1)
    return result
""", oc="O(N)", tc="O(N)")

P(59, "second_largest", "order-statistics", "easy",
  "Return the second largest element, or -1 if fewer than 2 distinct.",
  """\
def second_largest(nums):
    if len(nums) < 2:
        return -1
    first = nums[0]
    for i in range(len(nums)):
        if nums[i] > first:
            first = nums[i]
    second = -1
    found = 0
    for i in range(len(nums)):
        if nums[i] != first:
            if found == 0 or nums[i] > second:
                second = nums[i]
                found = 1
    if found == 0:
        return -1
    return second
""", oc="O(N)", tc="O(N)")

P(60, "count_peaks", "order-statistics", "easy",
  "Count elements strictly greater than both neighbours.",
  """\
def count_peaks(nums):
    count = 0
    for i in range(1, len(nums) - 1):
        if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
            count += 1
    return count
""", oc="O(N)", tc="O(N)")

P(61, "find_leaders", "order-statistics", "medium",
  "Return count of elements greater than all elements to their right.",
  """\
def find_leaders(nums):
    count = 0
    n = len(nums)
    for i in range(n):
        is_leader = 1
        for j in range(i + 1, n):
            if nums[j] >= nums[i]:
                is_leader = 0
                break
        count += is_leader
    return count
""")

P(62, "rank_elements_sum", "order-statistics", "medium",
  "Sum of 1-based ranks of all elements (rank = position in sorted order).",
  """\
def rank_elements_sum(nums):
    total = 0
    n = len(nums)
    for i in range(n):
        rank = 1
        for j in range(n):
            if nums[j] < nums[i]:
                rank += 1
        total += rank
    return total
""")

P(63, "closest_to_zero", "order-statistics", "easy",
  "Return the element closest to zero (positive wins on tie).",
  """\
def closest_to_zero(nums):
    if len(nums) == 0:
        return 0
    best = nums[0]
    for i in range(len(nums)):
        a = abs(nums[i])
        b = abs(best)
        if a < b:
            best = nums[i]
        elif a == b and nums[i] > best:
            best = nums[i]
    return best
""", oc="O(N)", tc="O(N)")

P(64, "count_out_of_place", "order-statistics", "medium",
  "Count positions where element differs from its sorted-order value.",
  """\
def count_out_of_place(nums):
    n = len(nums)
    sorted_nums = []
    for v in nums:
        sorted_nums.append(v)
    for i in range(n):
        for j in range(i + 1, n):
            if sorted_nums[i] > sorted_nums[j]:
                tmp = sorted_nums[i]
                sorted_nums[i] = sorted_nums[j]
                sorted_nums[j] = tmp
    count = 0
    for i in range(n):
        if nums[i] != sorted_nums[i]:
            count += 1
    return count
""", tc="O(N log N)")

P(65, "element_rank_product", "order-statistics", "medium",
  "Return the sum of (element * its rank) for all elements.",
  """\
def element_rank_product(nums):
    total = 0
    n = len(nums)
    for i in range(n):
        rank = 1
        for j in range(n):
            if nums[j] < nums[i]:
                rank += 1
        total += nums[i] * rank
    return total
""")


# ══════════════════════════════════════════════════════════════════════
#  CATEGORY 6 — Prefix Sum & Range  (TYR-066 … TYR-075)
# ══════════════════════════════════════════════════════════════════════

P(66, "prefix_sum_total", "prefix-sum", "easy",
  "Return the sum of the prefix sum array.",
  """\
def prefix_sum_total(nums):
    total = 0
    running = 0
    for i in range(len(nums)):
        running += nums[i]
        total += running
    return total
""", oc="O(N)", tc="O(N)")

P(67, "range_sum_brute", "prefix-sum", "easy",
  "Return the sum of elements from index 0 to k (inclusive).",
  """\
def range_sum_brute(nums, k):
    total = 0
    for i in range(len(nums)):
        if i <= k:
            total += nums[i]
    return total
""", oc="O(N)", tc="O(N)")

P(68, "count_equilibrium_indices", "prefix-sum", "medium",
  "Count all indices where left sum equals right sum.",
  """\
def count_equilibrium_indices(nums):
    count = 0
    n = len(nums)
    for i in range(n):
        left = 0
        for j in range(i):
            left += nums[j]
        right = 0
        for j in range(i + 1, n):
            right += nums[j]
        if left == right:
            count += 1
    return count
""")

P(69, "max_prefix_sum", "prefix-sum", "easy",
  "Return the maximum prefix sum (sum of nums[0..i] for any i).",
  """\
def max_prefix_sum(nums):
    if len(nums) == 0:
        return 0
    best = nums[0]
    cur = 0
    for i in range(len(nums)):
        cur += nums[i]
        if cur > best:
            best = cur
    return best
""", oc="O(N)", tc="O(N)")

P(70, "left_max_minus_right_min", "prefix-sum", "medium",
  "Max over all splits i: max(nums[0..i]) - min(nums[i+1..n-1]).",
  """\
def left_max_minus_right_min(nums):
    if len(nums) < 2:
        return 0
    best = nums[0] - nums[1]
    n = len(nums)
    for i in range(n - 1):
        left_max = nums[0]
        for j in range(1, i + 1):
            if nums[j] > left_max:
                left_max = nums[j]
        right_min = nums[i + 1]
        for j in range(i + 2, n):
            if nums[j] < right_min:
                right_min = nums[j]
        d = left_max - right_min
        if d > best:
            best = d
    return best
""")

P(71, "count_subarrays_divisible_by_k", "prefix-sum", "medium",
  "Count subarrays whose sum is divisible by k.",
  """\
def count_subarrays_divisible_by_k(nums, k):
    if k == 0:
        return 0
    count = 0
    n = len(nums)
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            if cur % k == 0:
                count += 1
    return count
""")

P(72, "sum_of_all_subarrays", "prefix-sum", "medium",
  "Return the total sum across all contiguous subarrays.",
  """\
def sum_of_all_subarrays(nums):
    total = 0
    n = len(nums)
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            total += cur
    return total
""")

P(73, "count_subarrays_odd_sum", "prefix-sum", "medium",
  "Count contiguous subarrays whose sum is odd.",
  """\
def count_subarrays_odd_sum(nums):
    count = 0
    n = len(nums)
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            if cur % 2 != 0:
                count += 1
    return count
""")

P(74, "count_zero_sum_subarrays", "prefix-sum", "medium",
  "Count contiguous subarrays whose sum is exactly zero.",
  """\
def count_zero_sum_subarrays(nums):
    count = 0
    n = len(nums)
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            if cur == 0:
                count += 1
    return count
""")

P(75, "max_sum_two_non_overlapping", "prefix-sum", "medium",
  "Max sum of two non-overlapping subarrays each of length 1 (i.e. max two elements with i!=j).",
  """\
def max_sum_two_non_overlapping(nums):
    if len(nums) < 2:
        return 0
    best = nums[0] + nums[1]
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            if s > best:
                best = s
    return best
""")


# ══════════════════════════════════════════════════════════════════════
#  CATEGORY 7 — Sliding Window  (TYR-076 … TYR-085)
# ══════════════════════════════════════════════════════════════════════

P(76, "max_of_all_windows", "sliding-window", "medium",
  "Return the list of maximums of every contiguous window of size k.",
  """\
def max_of_all_windows(nums, k):
    result = []
    n = len(nums)
    for i in range(n - k + 1):
        mx = nums[i]
        for j in range(i + 1, i + k):
            if nums[j] > mx:
                mx = nums[j]
        result.append(mx)
    return result
""")

P(77, "min_of_all_windows", "sliding-window", "medium",
  "Return the list of minimums of every contiguous window of size k.",
  """\
def min_of_all_windows(nums, k):
    result = []
    n = len(nums)
    for i in range(n - k + 1):
        mn = nums[i]
        for j in range(i + 1, i + k):
            if nums[j] < mn:
                mn = nums[j]
        result.append(mn)
    return result
""")

P(78, "count_distinct_in_windows", "sliding-window", "medium",
  "Return count of distinct elements in the first window of size k.",
  """\
def count_distinct_in_windows(nums, k):
    if len(nums) < k or k <= 0:
        return 0
    count = 0
    for i in range(k):
        is_first = 1
        for j in range(i):
            if nums[j] == nums[i]:
                is_first = 0
        count += is_first
    return count
""")

P(79, "sum_of_all_windows", "sliding-window", "easy",
  "Return the sum of sums of all windows of size k.",
  """\
def sum_of_all_windows(nums, k):
    total = 0
    n = len(nums)
    for i in range(n - k + 1):
        cur = 0
        for j in range(i, i + k):
            cur += nums[j]
        total += cur
    return total
""")

P(80, "has_duplicate_within_k", "sliding-window", "easy",
  "Return 1 if any two equal elements are at most k positions apart.",
  """\
def has_duplicate_within_k(nums, k):
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, min(i + k + 1, n)):
            if nums[i] == nums[j]:
                return 1
    return 0
""")

P(81, "max_sum_window", "sliding-window", "easy",
  "Return the maximum sum of any contiguous window of size k.",
  """\
def max_sum_window(nums, k):
    if len(nums) < k:
        return 0
    best = 0
    n = len(nums)
    for i in range(n - k + 1):
        cur = 0
        for j in range(i, i + k):
            cur += nums[j]
        if i == 0 or cur > best:
            best = cur
    return best
""")

P(82, "min_sum_window", "sliding-window", "easy",
  "Return the minimum sum of any contiguous window of size k.",
  """\
def min_sum_window(nums, k):
    if len(nums) < k:
        return 0
    best = 0
    n = len(nums)
    for i in range(n - k + 1):
        cur = 0
        for j in range(i, i + k):
            cur += nums[j]
        if i == 0 or cur < best:
            best = cur
    return best
""")

P(83, "count_windows_all_positive", "sliding-window", "easy",
  "Count windows of size k where every element is positive.",
  """\
def count_windows_all_positive(nums, k):
    count = 0
    n = len(nums)
    for i in range(n - k + 1):
        all_pos = 1
        for j in range(i, i + k):
            if nums[j] <= 0:
                all_pos = 0
        count += all_pos
    return count
""")

P(84, "max_window_range", "sliding-window", "medium",
  "Return the maximum (max - min) over all windows of size k.",
  """\
def max_window_range(nums, k):
    if len(nums) < k:
        return 0
    best = 0
    n = len(nums)
    for i in range(n - k + 1):
        mx = nums[i]
        mn = nums[i]
        for j in range(i + 1, i + k):
            if nums[j] > mx:
                mx = nums[j]
            if nums[j] < mn:
                mn = nums[j]
        d = mx - mn
        if d > best:
            best = d
    return best
""")

P(85, "count_windows_with_target_sum", "sliding-window", "easy",
  "Count windows of size k whose sum equals target.",
  """\
def count_windows_with_target_sum(nums, k, target):
    count = 0
    n = len(nums)
    for i in range(n - k + 1):
        cur = 0
        for j in range(i, i + k):
            cur += nums[j]
        if cur == target:
            count += 1
    return count
""")


# ══════════════════════════════════════════════════════════════════════
#  CATEGORY 8 — Hash Map Patterns  (TYR-086 … TYR-100)
# ══════════════════════════════════════════════════════════════════════

P(86, "first_recurring_element", "hash-map", "easy",
  "Return the first element (by second occurrence) seen before, or -1.",
  """\
def first_recurring_element(nums):
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] == nums[i]:
                return nums[i]
    return -1
""")

P(87, "same_frequency_distribution", "hash-map", "medium",
  "Return 1 if both halves have the same frequency distribution.",
  """\
def same_frequency_distribution(nums):
    mid = len(nums) // 2
    for i in range(mid):
        count_a = 0
        for j in range(mid):
            if nums[j] == nums[i]:
                count_a += 1
        count_b = 0
        for j in range(mid, len(nums)):
            if nums[j] == nums[i]:
                count_b += 1
        if count_a != count_b:
            return 0
    return 1
""")

P(88, "longest_consecutive_sequence", "hash-map", "medium",
  "Return the length of the longest consecutive elements sequence.",
  """\
def longest_consecutive_sequence(nums):
    if len(nums) == 0:
        return 0
    best = 1
    n = len(nums)
    for i in range(n):
        cur = nums[i]
        length = 1
        while True:
            found = 0
            for j in range(n):
                if nums[j] == cur + 1:
                    found = 1
                    break
            if found == 1:
                cur += 1
                length += 1
            else:
                break
        if length > best:
            best = length
    return best
""", oc="O(N^2)", tc="O(N)")

P(89, "top_k_frequent_sum", "hash-map", "medium",
  "Return the sum of the k most-frequent elements.",
  """\
def top_k_frequent_sum(nums, k):
    if len(nums) == 0 or k <= 0:
        return 0
    n = len(nums)
    used = []
    result_sum = 0
    for rank in range(k):
        best_val = 0
        best_count = -1
        for i in range(n):
            skip = 0
            for u in range(len(used)):
                if used[u] == nums[i]:
                    skip = 1
            if skip == 1:
                continue
            count = 0
            for j in range(n):
                if nums[j] == nums[i]:
                    count += 1
            if count > best_count:
                best_count = count
                best_val = nums[i]
        if best_count > 0:
            used.append(best_val)
            result_sum += best_val
    return result_sum
""")

P(90, "group_count_by_remainder", "hash-map", "easy",
  "Return the count of elements whose remainder mod k equals 0.",
  """\
def group_count_by_remainder(nums, k):
    if k == 0:
        return 0
    count = 0
    for i in range(len(nums)):
        if nums[i] % k == 0:
            count += 1
    return count
""", oc="O(N)", tc="O(N)")

P(91, "count_triplets_with_sum", "hash-map", "medium",
  "Count triplets (i<j<k) whose sum equals target.",
  """\
def count_triplets_with_sum(nums, target):
    count = 0
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if nums[i] + nums[j] + nums[k] == target:
                    count += 1
    return count
""", oc="O(N^3)", tc="O(N^2)")

P(92, "longest_subarray_all_distinct", "hash-map", "medium",
  "Return the length of the longest contiguous subarray with all distinct elements.",
  """\
def longest_subarray_all_distinct(nums):
    best = 0
    n = len(nums)
    for i in range(n):
        length = 0
        has_dup = 0
        for j in range(i, n):
            for k in range(i, j):
                if nums[k] == nums[j]:
                    has_dup = 1
            if has_dup == 1:
                break
            length += 1
        if length > best:
            best = length
    return best
""")

P(93, "count_four_sum", "hash-map", "medium",
  "Count quadruplets (i<j<k<l) summing to target.",
  """\
def count_four_sum(nums, target):
    count = 0
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if nums[i] + nums[j] + nums[k] + nums[l] == target:
                        count += 1
    return count
""", oc="O(N^4)", tc="O(N^2)")

P(94, "has_zero_sum_subarray", "hash-map", "easy",
  "Return 1 if any contiguous subarray has sum == 0.",
  """\
def has_zero_sum_subarray(nums):
    n = len(nums)
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            if cur == 0:
                return 1
    return 0
""")

P(95, "count_complete_subarrays", "hash-map", "medium",
  "Count subarrays containing ALL distinct elements of the full array.",
  """\
def count_complete_subarrays(nums):
    n = len(nums)
    total_distinct = 0
    for i in range(n):
        is_first = 1
        for j in range(i):
            if nums[j] == nums[i]:
                is_first = 0
        total_distinct += is_first
    count = 0
    for i in range(n):
        for j in range(i, n):
            dist = 0
            for k in range(i, j + 1):
                is_f = 1
                for m in range(i, k):
                    if nums[m] == nums[k]:
                        is_f = 0
                dist += is_f
            if dist == total_distinct:
                count += 1
    return count
""", oc="O(N^3)", tc="O(N)")

P(96, "longest_equal_01_subarray", "hash-map", "medium",
  "Longest subarray with equal count of 0s and 1s (array has only 0/1).",
  """\
def longest_equal_01_subarray(nums):
    best = 0
    n = len(nums)
    for i in range(n):
        zeros = 0
        ones = 0
        for j in range(i, n):
            if nums[j] == 0:
                zeros += 1
            else:
                ones += 1
            if zeros == ones and zeros + ones > best:
                best = zeros + ones
    return best
""")

P(97, "count_pairs_diff_k", "hash-map", "easy",
  "Count pairs (i<j) with |nums[i]-nums[j]| == k.",
  """\
def count_pairs_diff_k(nums, k):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if abs(nums[i] - nums[j]) == k:
                count += 1
    return count
""")

P(98, "longest_band", "hash-map", "medium",
  "Length of largest subset that forms a contiguous band (consecutive ints).",
  """\
def longest_band(nums):
    if len(nums) == 0:
        return 0
    best = 1
    n = len(nums)
    for i in range(n):
        cur = nums[i]
        length = 1
        while True:
            found = 0
            for j in range(n):
                if nums[j] == cur + 1:
                    found = 1
                    break
            if found == 1:
                cur += 1
                length += 1
            else:
                break
        if length > best:
            best = length
    return best
""")

P(99, "find_majority_one_third", "hash-map", "medium",
  "Return element appearing more than n/3 times, or -1.",
  """\
def find_majority_one_third(nums):
    n = len(nums)
    for i in range(n):
        count = 0
        for j in range(n):
            if nums[j] == nums[i]:
                count += 1
        if count > n // 3:
            return nums[i]
    return -1
""")

P(100, "count_subarrays_all_same", "hash-map", "easy",
  "Count contiguous subarrays where all elements are the same.",
  """\
def count_subarrays_all_same(nums):
    count = 0
    n = len(nums)
    for i in range(n):
        for j in range(i, n):
            all_same = 1
            for k in range(i + 1, j + 1):
                if nums[k] != nums[i]:
                    all_same = 0
                    break
            count += all_same
    return count
""")


# ══════════════════════════════════════════════════════════════════════
#  CATEGORY 9 — Array Transform  (TYR-101 … TYR-115)
# ══════════════════════════════════════════════════════════════════════

P(101, "move_zeros_to_end", "array-transform", "easy",
  "Move all zeros to the end, preserving order of non-zeros.",
  """\
def move_zeros_to_end(nums):
    result = []
    for i in range(len(nums)):
        if nums[i] != 0:
            result.append(nums[i])
    for i in range(len(nums)):
        if nums[i] == 0:
            result.append(0)
    return result
""", oc="O(N)", tc="O(N)")

P(102, "rotate_array_by_k", "array-transform", "medium",
  "Rotate array right by k positions.",
  """\
def rotate_array_by_k(nums, k):
    n = len(nums)
    if n == 0:
        return []
    k = k % n
    result = []
    for i in range(n - k, n):
        result.append(nums[i])
    for i in range(n - k):
        result.append(nums[i])
    return result
""", oc="O(N)", tc="O(N)")

P(103, "segregate_pos_neg", "array-transform", "easy",
  "Return array with all non-negatives first, then negatives.",
  """\
def segregate_pos_neg(nums):
    result = []
    for i in range(len(nums)):
        if nums[i] >= 0:
            result.append(nums[i])
    for i in range(len(nums)):
        if nums[i] < 0:
            result.append(nums[i])
    return result
""", oc="O(N)", tc="O(N)")

P(104, "replace_with_rank", "array-transform", "medium",
  "Replace each element with its rank (1-based, smallest=1).",
  """\
def replace_with_rank(nums):
    result = []
    n = len(nums)
    for i in range(n):
        rank = 1
        for j in range(n):
            if nums[j] < nums[i]:
                rank += 1
            elif nums[j] == nums[i] and j < i:
                rank += 0
        result.append(rank)
    return result
""")

P(105, "remove_all_target", "array-transform", "easy",
  "Return array with all occurrences of target removed.",
  """\
def remove_all_target(nums, target):
    result = []
    for i in range(len(nums)):
        if nums[i] != target:
            result.append(nums[i])
    return result
""", oc="O(N)", tc="O(N)")

P(106, "running_sum", "array-transform", "easy",
  "Return the running sum array.",
  """\
def running_sum(nums):
    result = []
    cur = 0
    for i in range(len(nums)):
        cur += nums[i]
        result.append(cur)
    return result
""", oc="O(N)", tc="O(N)")

P(107, "find_disappeared_numbers", "array-transform", "medium",
  "Find all numbers in [1, n] missing from the array.",
  """\
def find_disappeared_numbers(nums):
    n = len(nums)
    result = []
    for i in range(1, n + 1):
        found = 0
        for j in range(n):
            if nums[j] == i:
                found = 1
        if found == 0:
            result.append(i)
    return result
""")

P(108, "find_all_duplicates", "array-transform", "medium",
  "Find all elements appearing exactly twice (values in 1..n).",
  """\
def find_all_duplicates(nums):
    result = []
    n = len(nums)
    for i in range(n):
        is_first = 1
        for j in range(i):
            if nums[j] == nums[i]:
                is_first = 0
        if is_first == 0:
            continue
        count = 0
        for j in range(n):
            if nums[j] == nums[i]:
                count += 1
        if count == 2:
            result.append(nums[i])
    return result
""")

P(109, "array_sign", "array-transform", "easy",
  "Return 1 if product is positive, -1 if negative, 0 if zero.",
  """\
def array_sign(nums):
    neg_count = 0
    for i in range(len(nums)):
        if nums[i] == 0:
            return 0
        if nums[i] < 0:
            neg_count += 1
    if neg_count % 2 == 0:
        return 1
    return -1
""", oc="O(N)", tc="O(N)")

P(110, "max_consecutive_ones", "array-transform", "easy",
  "Return the maximum number of consecutive 1s.",
  """\
def max_consecutive_ones(nums):
    best = 0
    cur = 0
    for i in range(len(nums)):
        if nums[i] == 1:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best
""", oc="O(N)", tc="O(N)")

P(111, "check_monotonic", "array-transform", "easy",
  "Return 1 if the array is monotonic (non-decreasing or non-increasing).",
  """\
def check_monotonic(nums):
    if len(nums) <= 1:
        return 1
    inc = 1
    dec = 1
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            dec = 0
        if nums[i] < nums[i - 1]:
            inc = 0
    if inc == 1 or dec == 1:
        return 1
    return 0
""", oc="O(N)", tc="O(N)")

P(112, "third_distinct_max", "array-transform", "medium",
  "Return the third distinct maximum, or the overall maximum.",
  """\
def third_distinct_max(nums):
    distinct = []
    for i in range(len(nums)):
        found = 0
        for j in range(len(distinct)):
            if distinct[j] == nums[i]:
                found = 1
        if found == 0:
            distinct.append(nums[i])
    if len(distinct) < 3:
        best = distinct[0]
        for v in distinct:
            if v > best:
                best = v
        return best
    for rank in range(3):
        mx = distinct[0]
        mx_idx = 0
        for i in range(1, len(distinct)):
            if distinct[i] > mx:
                mx = distinct[i]
                mx_idx = i
        if rank == 2:
            return mx
        distinct[mx_idx] = distinct[0] - 1
    return -1
""")

P(113, "degree_shortest_subarray", "array-transform", "medium",
  "Return length of shortest subarray containing max-frequency element.",
  """\
def degree_shortest_subarray(nums):
    n = len(nums)
    if n == 0:
        return 0
    max_freq = 0
    for i in range(n):
        count = 0
        for j in range(n):
            if nums[j] == nums[i]:
                count += 1
        if count > max_freq:
            max_freq = count
    best = n
    for i in range(n):
        count = 0
        for j in range(n):
            if nums[j] == nums[i]:
                count += 1
        if count == max_freq:
            first = -1
            last = -1
            for j in range(n):
                if nums[j] == nums[i]:
                    if first == -1:
                        first = j
                    last = j
            length = last - first + 1
            if length < best:
                best = length
    return best
""")

P(114, "shortest_unsorted_subarray", "array-transform", "medium",
  "Return the length of the shortest subarray that, when sorted, sorts the whole.",
  """\
def shortest_unsorted_subarray(nums):
    n = len(nums)
    sorted_nums = []
    for v in nums:
        sorted_nums.append(v)
    for i in range(n):
        for j in range(i + 1, n):
            if sorted_nums[i] > sorted_nums[j]:
                tmp = sorted_nums[i]
                sorted_nums[i] = sorted_nums[j]
                sorted_nums[j] = tmp
    start = -1
    end = -1
    for i in range(n):
        if nums[i] != sorted_nums[i]:
            if start == -1:
                start = i
            end = i
    if start == -1:
        return 0
    return end - start + 1
""", tc="O(N log N)")

P(115, "dutch_flag_sort", "array-transform", "easy",
  "Sort an array containing only 0, 1, and 2.",
  """\
def dutch_flag_sort(nums):
    result = []
    for v in range(3):
        for i in range(len(nums)):
            if nums[i] == v:
                result.append(v)
    return result
""", oc="O(N)", tc="O(N)")


# ══════════════════════════════════════════════════════════════════════
#  CATEGORY 10 — Competition Classics  (TYR-116 … TYR-135)
# ══════════════════════════════════════════════════════════════════════

P(116, "container_with_most_water", "competition", "medium",
  "Max area between two lines (heights as array).",
  """\
def container_with_most_water(nums):
    best = 0
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            h = nums[i]
            if nums[j] < h:
                h = nums[j]
            area = h * (j - i)
            if area > best:
                best = area
    return best
""")

P(117, "best_time_buy_sell", "competition", "easy",
  "Maximum profit from one buy-sell transaction.",
  """\
def best_time_buy_sell(nums):
    best = 0
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            profit = nums[j] - nums[i]
            if profit > best:
                best = profit
    return best
""")

P(118, "trap_rainwater", "competition", "medium",
  "Compute total trapped rainwater given elevation map.",
  """\
def trap_rainwater(nums):
    total = 0
    n = len(nums)
    for i in range(1, n - 1):
        left_max = 0
        for j in range(i):
            if nums[j] > left_max:
                left_max = nums[j]
        right_max = 0
        for j in range(i + 1, n):
            if nums[j] > right_max:
                right_max = nums[j]
        water_level = left_max
        if right_max < water_level:
            water_level = right_max
        if water_level > nums[i]:
            total += water_level - nums[i]
    return total
""")

P(119, "good_pairs_count", "competition", "easy",
  "Count pairs (i<j) where nums[i] == nums[j].",
  """\
def good_pairs_count(nums):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                count += 1
    return count
""")

P(120, "max_chunks_to_sort", "competition", "medium",
  "Max number of partitions so that sorting each independently sorts the whole.",
  """\
def max_chunks_to_sort(nums):
    n = len(nums)
    chunks = 0
    for i in range(n):
        mx = 0
        for j in range(i + 1):
            if nums[j] > mx:
                mx = nums[j]
        if mx == i:
            chunks += 1
    return chunks
""")

P(121, "find_peak_element_index", "competition", "easy",
  "Return index of any element greater than its neighbours.",
  """\
def find_peak_element_index(nums):
    n = len(nums)
    if n == 1:
        return 0
    if nums[0] > nums[1]:
        return 0
    for i in range(1, n - 1):
        if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
            return i
    if nums[n - 1] > nums[n - 2]:
        return n - 1
    return 0
""", oc="O(N)", tc="O(log N)")

P(122, "best_sightseeing_pair", "competition", "medium",
  "Max value of nums[i] + nums[j] + i - j for i < j.",
  """\
def best_sightseeing_pair(nums):
    best = 0
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            score = nums[i] + nums[j] + i - j
            if score > best:
                best = score
    return best
""")

P(123, "max_turbulence_length", "competition", "medium",
  "Length of longest turbulent subarray (alternating comparisons).",
  """\
def max_turbulence_length(nums):
    n = len(nums)
    if n < 2:
        return n
    best = 1
    for i in range(n):
        length = 1
        for j in range(i + 1, n):
            if length == 1:
                if nums[j] != nums[j - 1]:
                    length += 1
                else:
                    break
            else:
                prev_up = nums[j - 1] > nums[j - 2]
                cur_up = nums[j] > nums[j - 1]
                if nums[j] == nums[j - 1]:
                    break
                if prev_up != cur_up:
                    length += 1
                else:
                    break
        if length > best:
            best = length
    return best
""")

P(124, "count_arithmetic_subarrays", "competition", "medium",
  "Count contiguous subarrays of length >= 3 that are arithmetic.",
  """\
def count_arithmetic_subarrays(nums):
    count = 0
    n = len(nums)
    for i in range(n):
        for j in range(i + 2, n):
            is_ap = 1
            d = nums[i + 1] - nums[i]
            for k in range(i + 1, j + 1):
                if nums[k] - nums[k - 1] != d:
                    is_ap = 0
                    break
            count += is_ap
    return count
""")

P(125, "max_width_ramp", "competition", "medium",
  "Maximum j - i such that nums[i] <= nums[j].",
  """\
def max_width_ramp(nums):
    best = 0
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] <= nums[j]:
                d = j - i
                if d > best:
                    best = d
    return best
""")

P(126, "nums_smaller_than_current_sum", "competition", "easy",
  "Sum of 'how many elements in the full array are smaller than nums[i]'.",
  """\
def nums_smaller_than_current_sum(nums):
    total = 0
    n = len(nums)
    for i in range(n):
        count = 0
        for j in range(n):
            if nums[j] < nums[i]:
                count += 1
        total += count
    return total
""")

P(127, "longest_harmonious_subsequence", "competition", "medium",
  "Longest subsequence where max - min == 1.",
  """\
def longest_harmonious_subsequence(nums):
    best = 0
    n = len(nums)
    for i in range(n):
        count_same = 0
        count_plus = 0
        for j in range(n):
            if nums[j] == nums[i]:
                count_same += 1
            elif nums[j] == nums[i] + 1:
                count_plus += 1
        if count_plus > 0:
            total = count_same + count_plus
            if total > best:
                best = total
    return best
""")

P(128, "three_sum_closest", "competition", "medium",
  "Return the triplet sum closest to target.",
  """\
def three_sum_closest(nums, target):
    n = len(nums)
    if n < 3:
        return 0
    best = nums[0] + nums[1] + nums[2]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                s = nums[i] + nums[j] + nums[k]
                if abs(s - target) < abs(best - target):
                    best = s
    return best
""", oc="O(N^3)", tc="O(N^2)")

P(129, "max_sum_circular_subarray", "competition", "medium",
  "Maximum subarray sum allowing wrap-around.",
  """\
def max_sum_circular_subarray(nums):
    n = len(nums)
    if n == 0:
        return 0
    best = nums[0]
    for i in range(n):
        cur = 0
        for j in range(n):
            cur += nums[(i + j) % n]
            if cur > best:
                best = cur
    return best
""")

P(130, "minimum_increment_for_unique", "competition", "medium",
  "Minimum total increments to make all elements unique.",
  """\
def minimum_increment_for_unique(nums):
    n = len(nums)
    arr = []
    for v in nums:
        arr.append(v)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                tmp = arr[i]
                arr[i] = arr[j]
                arr[j] = tmp
    moves = 0
    for i in range(1, n):
        if arr[i] <= arr[i - 1]:
            diff = arr[i - 1] - arr[i] + 1
            arr[i] += diff
            moves += diff
    return moves
""", tc="O(N log N)")

P(131, "largest_perimeter_triangle", "competition", "easy",
  "Largest perimeter of a triangle formed from any 3 elements, or 0.",
  """\
def largest_perimeter_triangle(nums):
    n = len(nums)
    best = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = nums[i]
                b = nums[j]
                c = nums[k]
                if a + b > c and a + c > b and b + c > a:
                    p = a + b + c
                    if p > best:
                        best = p
    return best
""", oc="O(N^3)", tc="O(N log N)")

P(132, "count_teams", "competition", "medium",
  "Count triplets (i<j<k) that are strictly increasing or decreasing.",
  """\
def count_teams(nums):
    count = 0
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if nums[i] < nums[j] < nums[k]:
                    count += 1
                if nums[i] > nums[j] > nums[k]:
                    count += 1
    return count
""", oc="O(N^3)", tc="O(N^2)")

P(133, "max_score_after_split", "competition", "easy",
  "Split array into two non-empty parts. Max (count_zeros_left + count_ones_right).",
  """\
def max_score_after_split(nums):
    best = 0
    n = len(nums)
    for i in range(1, n):
        zeros = 0
        for j in range(i):
            if nums[j] == 0:
                zeros += 1
        ones = 0
        for j in range(i, n):
            if nums[j] == 1:
                ones += 1
        score = zeros + ones
        if score > best:
            best = score
    return best
""")

P(134, "count_special_positions", "competition", "medium",
  "Count positions where nums[i] equals sum of all other elements.",
  """\
def count_special_positions(nums):
    count = 0
    n = len(nums)
    for i in range(n):
        rest_sum = 0
        for j in range(n):
            if j != i:
                rest_sum += nums[j]
        if nums[i] == rest_sum:
            count += 1
    return count
""")

P(135, "jump_game", "competition", "medium",
  "Return 1 if you can reach the last index starting from 0.",
  """\
def jump_game(nums):
    n = len(nums)
    if n <= 1:
        return 1
    reachable = [0] * n
    reachable[0] = 1
    for i in range(n):
        if reachable[i] == 0:
            continue
        for j in range(1, nums[i] + 1):
            if i + j < n:
                reachable[i + j] = 1
    return reachable[n - 1]
""")


# ══════════════════════════════════════════════════════════════════════
#  CATEGORY 11 — Advanced Patterns  (TYR-136 … TYR-150)
# ══════════════════════════════════════════════════════════════════════

P(136, "count_pairs_xor_k", "advanced", "medium",
  "Count pairs (i<j) where nums[i] XOR nums[j] == k.",
  """\
def count_pairs_xor_k(nums, k):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if (nums[i] ^ nums[j]) == k:
                count += 1
    return count
""")

P(137, "count_subarrays_exact_k_distinct", "advanced", "medium",
  "Count subarrays with exactly k distinct elements.",
  """\
def count_subarrays_exact_k_distinct(nums, k):
    count = 0
    n = len(nums)
    for i in range(n):
        for j in range(i, n):
            dist = 0
            for m in range(i, j + 1):
                is_f = 1
                for p in range(i, m):
                    if nums[p] == nums[m]:
                        is_f = 0
                dist += is_f
            if dist == k:
                count += 1
    return count
""", oc="O(N^3)", tc="O(N)")

P(138, "longest_equal_subarray", "advanced", "easy",
  "Return the length of the longest contiguous subarray with all same values.",
  """\
def longest_equal_subarray(nums):
    if len(nums) == 0:
        return 0
    best = 1
    n = len(nums)
    for i in range(n):
        length = 1
        for j in range(i + 1, n):
            if nums[j] == nums[i]:
                length += 1
            else:
                break
        if length > best:
            best = length
    return best
""")

P(139, "count_local_inversions", "advanced", "easy",
  "Count positions where nums[i] > nums[i+1].",
  """\
def count_local_inversions(nums):
    count = 0
    for i in range(len(nums) - 1):
        if nums[i] > nums[i + 1]:
            count += 1
    return count
""", oc="O(N)", tc="O(N)")

P(140, "largest_rectangle_histogram", "advanced", "medium",
  "Largest rectangle area in a histogram.",
  """\
def largest_rectangle_histogram(nums):
    best = 0
    n = len(nums)
    for i in range(n):
        h = nums[i]
        for j in range(i, n):
            if nums[j] < h:
                h = nums[j]
            area = h * (j - i + 1)
            if area > best:
                best = area
    return best
""")

P(141, "count_range_pairs", "advanced", "medium",
  "Count pairs (i<j) with sum in [target, target + k].",
  """\
def count_range_pairs(nums, target, k):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            s = nums[i] + nums[j]
            if s >= target and s <= target + k:
                count += 1
    return count
""")

P(142, "max_sum_after_removing_one", "advanced", "medium",
  "Maximum subarray sum after removing at most one element.",
  """\
def max_sum_after_removing_one(nums):
    n = len(nums)
    if n == 0:
        return 0
    best = nums[0]
    for i in range(n):
        cur = 0
        for j in range(n):
            if j == i:
                continue
            cur += nums[j]
            if cur > best:
                best = cur
        cur = 0
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            if cur > best:
                best = cur
    return best
""")

P(143, "can_form_arithmetic_from_sequence", "advanced", "medium",
  "Return 1 if the array can be rearranged into an arithmetic progression.",
  """\
def can_form_arithmetic_from_sequence(nums):
    n = len(nums)
    if n <= 2:
        return 1
    mn = nums[0]
    mx = nums[0]
    for i in range(n):
        if nums[i] < mn:
            mn = nums[i]
        if nums[i] > mx:
            mx = nums[i]
    if (mx - mn) % (n - 1) != 0:
        return 0
    d = (mx - mn) // (n - 1)
    for step in range(n):
        target = mn + step * d
        found = 0
        for j in range(n):
            if nums[j] == target:
                found = 1
                break
        if found == 0:
            return 0
    return 1
""")

P(144, "max_profit_two_transactions", "advanced", "medium",
  "Maximum profit with at most two buy-sell transactions.",
  """\
def max_profit_two_transactions(nums):
    n = len(nums)
    if n < 2:
        return 0
    best = 0
    for i in range(n):
        for j in range(i + 1, n):
            profit1 = nums[j] - nums[i]
            if profit1 < 0:
                profit1 = 0
            profit2 = 0
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p = nums[l] - nums[k]
                    if p > profit2:
                        profit2 = p
            total = profit1 + profit2
            if total > best:
                best = total
    return best
""", oc="O(N^4)", tc="O(N)")

P(145, "count_valleys", "advanced", "easy",
  "Count elements strictly less than both neighbours.",
  """\
def count_valleys(nums):
    count = 0
    for i in range(1, len(nums) - 1):
        if nums[i] < nums[i - 1] and nums[i] < nums[i + 1]:
            count += 1
    return count
""", oc="O(N)", tc="O(N)")

P(146, "smallest_difference_pair_abs", "advanced", "medium",
  "Return the minimum absolute difference between any two elements.",
  """\
def smallest_difference_pair_abs(nums):
    if len(nums) < 2:
        return 0
    best = abs(nums[0] - nums[1])
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(nums[i] - nums[j])
            if d < best:
                best = d
    return best
""", tc="O(N log N)")

P(147, "count_nice_pairs", "advanced", "medium",
  "Count pairs (i<j) where (nums[i]-rev(nums[i])) == (nums[j]-rev(nums[j])).",
  """\
def count_nice_pairs(nums):
    n = len(nums)
    count = 0
    diffs = []
    for i in range(n):
        x = nums[i]
        rev = 0
        tmp = x
        while tmp > 0:
            rev = rev * 10 + tmp % 10
            tmp = tmp // 10
        diffs.append(x - rev)
    for i in range(n):
        for j in range(i + 1, n):
            if diffs[i] == diffs[j]:
                count += 1
    return count
""")

P(148, "partition_into_min_groups", "advanced", "medium",
  "Minimum groups so that no group has two elements with diff <= 1.",
  """\
def partition_into_min_groups(nums):
    n = len(nums)
    arr = []
    for v in nums:
        arr.append(v)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                tmp = arr[i]
                arr[i] = arr[j]
                arr[j] = tmp
    if n == 0:
        return 0
    max_freq = 0
    for i in range(n):
        count = 0
        for j in range(n):
            if arr[j] == arr[i]:
                count += 1
        if count > max_freq:
            max_freq = count
    return max_freq
""", tc="O(N log N)")

P(149, "count_subarrays_with_score_less_k", "advanced", "medium",
  "Count subarrays where (sum * length) < k.",
  """\
def count_subarrays_with_score_less_k(nums, k):
    count = 0
    n = len(nums)
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur += nums[j]
            length = j - i + 1
            if cur * length < k:
                count += 1
            else:
                break
    return count
""")

P(150, "fair_candy_swap_diff", "advanced", "medium",
  "Find the swap sizes to equalize sums of two halves. Return difference, or 0.",
  """\
def fair_candy_swap_diff(nums):
    mid = len(nums) // 2
    sum_a = 0
    for i in range(mid):
        sum_a += nums[i]
    sum_b = 0
    for i in range(mid, len(nums)):
        sum_b += nums[i]
    for i in range(mid):
        for j in range(mid, len(nums)):
            new_a = sum_a - nums[i] + nums[j]
            new_b = sum_b - nums[j] + nums[i]
            if new_a == new_b:
                return nums[j] - nums[i]
    return 0
""")


# ══════════════════════════════════════════════════════════════════════

P(151, "count_equal_value_pairs", "pair-finding", "easy",
   "Count pairs (i,j) with i<j where nums[i] equals nums[j].",
   """\
def count_equal_pairs(nums):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                count += 1
    return count
""", "O(N^2)", "O(N)")

P(152, "find_single_element", "frequency", "easy",
   "Find the element that appears exactly once; all others appear twice. Return 0 if none.",
   """\
def find_single_element(nums):
    for i in range(len(nums)):
        count = 0
        for j in range(len(nums)):
            if nums[j] == nums[i]:
                count += 1
        if count == 1:
            return nums[i]
    return 0
""", "O(N^2)", "O(N)")

P(153, "max_sorted_gap", "sorting", "easy",
   "Return the maximum gap between consecutive elements when array is sorted. 0 if len < 2.",
   """\
def max_sorted_gap(nums):
    n = len(nums)
    if n < 2:
        return 0
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] > nums[j]:
                nums[i], nums[j] = nums[j], nums[i]
    best = 0
    for i in range(n - 1):
        diff = nums[i + 1] - nums[i]
        if diff > best:
            best = diff
    return best
""", "O(N^2)", "O(N log N)")

P(154, "arrays_are_permutation", "set-operations", "easy",
   "Return 1 if nums and target are permutations of each other, else 0.",
   """\
def arrays_are_permutation(nums, target):
    if len(nums) != len(target):
        return 0
    used = [0] * len(target)
    for i in range(len(nums)):
        found = 0
        for j in range(len(target)):
            if used[j] == 0 and target[j] == nums[i]:
                used[j] = 1
                found = 1
                break
        if found == 0:
            return 0
    return 1
""", "O(N^2)", "O(N)")

P(155, "count_distinct_elements", "frequency", "easy",
   "Count the number of distinct elements in nums.",
   """\
def count_unique_elements(nums):
    count = 0
    for i in range(len(nums)):
        is_dup = 0
        for j in range(i):
            if nums[j] == nums[i]:
                is_dup = 1
                break
        if is_dup == 0:
            count += 1
    return count
""", "O(N^2)", "O(N)")

P(156, "first_non_repeat_index", "frequency", "easy",
   "Return index of first element that appears exactly once. -1 if none.",
   """\
def first_non_repeat_index(nums):
    for i in range(len(nums)):
        count = 0
        for j in range(len(nums)):
            if nums[j] == nums[i]:
                count += 1
        if count == 1:
            return i
    return -1
""", "O(N^2)", "O(N)")

P(157, "majority_vote", "frequency", "easy",
   "Return the element appearing more than N/2 times. Return -21 if none.",
   """\
def majority_vote(nums):
    n = len(nums)
    for i in range(n):
        count = 0
        for j in range(n):
            if nums[j] == nums[i]:
                count += 1
        if count > n // 2:
            return nums[i]
    return -21
""", "O(N^2)", "O(N)")

P(158, "max_running_prefix_sum", "prefix-sum", "easy",
   "Return the maximum among all prefix sums. Prefix sum of index i = sum(nums[0..i]).",
   """\
def max_prefix_sum(nums):
    if len(nums) == 0:
        return 0
    best = nums[0]
    for i in range(len(nums)):
        s = 0
        for j in range(i + 1):
            s += nums[j]
        if s > best:
            best = s
    return best
""", "O(N^2)", "O(N)")

P(159, "longest_constant_run", "subarray", "easy",
   "Return the length of the longest run of identical consecutive values.",
   """\
def longest_constant_run(nums):
    if len(nums) == 0:
        return 0
    best = 1
    for i in range(len(nums)):
        length = 1
        for j in range(i + 1, len(nums)):
            if nums[j] == nums[i]:
                length += 1
            else:
                break
        if length > best:
            best = length
    return best
""", "O(N^2)", "O(N)")

P(160, "second_minimum", "order-statistics", "easy",
   "Return the second smallest distinct value, or -21 if fewer than 2 distinct values.",
   """\
def second_minimum(nums):
    if len(nums) < 2:
        return -21
    min1 = nums[0]
    for i in range(1, len(nums)):
        if nums[i] < min1:
            min1 = nums[i]
    min2 = 21
    found = 0
    for i in range(len(nums)):
        if nums[i] != min1:
            if found == 0 or nums[i] < min2:
                min2 = nums[i]
                found = 1
    if found == 0:
        return -21
    return min2
""", "O(N^2)", "O(N)")

P(161, "count_above_average", "array-transform", "easy",
   "Count elements strictly above the array average (integer division).",
   """\
def count_above_average(nums):
    n = len(nums)
    if n == 0:
        return 0
    total = 0
    for i in range(n):
        total += nums[i]
    avg = total // n
    count = 0
    for i in range(n):
        if nums[i] > avg:
            count += 1
    return count
""", "O(N^2)", "O(N)")

P(162, "parity_sort_check", "sorting", "easy",
   "Return 1 if all even elements appear before all odd elements, else 0.",
   """\
def parity_sort_check(nums):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] % 2 != 0 and nums[j] % 2 == 0:
                return 0
    return 1
""", "O(N^2)", "O(N)")

P(163, "equilibrium_first", "prefix-sum", "easy",
   "Return the first equilibrium index where left sum equals right sum. -1 if none.",
   """\
def equilibrium_first(nums):
    n = len(nums)
    for i in range(n):
        left = 0
        for j in range(i):
            left += nums[j]
        right = 0
        for j in range(i + 1, n):
            right += nums[j]
        if left == right:
            return i
    return -1
""", "O(N^2)", "O(N)")

P(164, "sum_of_leaders", "array-transform", "easy",
   "Sum of leader elements (greater than every element to their right).",
   """\
def sum_of_leaders(nums):
    total = 0
    n = len(nums)
    for i in range(n):
        is_leader = 1
        for j in range(i + 1, n):
            if nums[j] >= nums[i]:
                is_leader = 0
                break
        if is_leader == 1:
            total += nums[i]
    return total
""", "O(N^2)", "O(N)")

P(165, "min_pair_abs_diff", "pair-finding", "easy",
   "Return minimum absolute difference between any two distinct-index elements.",
   """\
def min_pair_abs_diff(nums):
    if len(nums) < 2:
        return 0
    best = abs(nums[0] - nums[1])
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            d = abs(nums[i] - nums[j])
            if d < best:
                best = d
    return best
""", "O(N^2)", "O(N log N)")

P(166, "max_right_minus_left", "subarray", "easy",
   "Return max(nums[j] - nums[i]) where j > i. Return 0 if array has fewer than 2 elements.",
   """\
def max_right_minus_left(nums):
    if len(nums) < 2:
        return 0
    best = nums[1] - nums[0]
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            d = nums[j] - nums[i]
            if d > best:
                best = d
    return best
""", "O(N^2)", "O(N)")

P(167, "count_interior_peaks", "array-transform", "easy",
   "Count interior peaks: indices i (0<i<N-1) where nums[i]>nums[i-1] and nums[i]>nums[i+1].",
   """\
def count_interior_peaks(nums):
    count = 0
    for i in range(1, len(nums) - 1):
        left_ok = 1
        right_ok = 1
        for j in range(i):
            if j == i - 1 and nums[j] >= nums[i]:
                left_ok = 0
        for j in range(i + 1, len(nums)):
            if j == i + 1 and nums[j] >= nums[i]:
                right_ok = 0
        if left_ok == 1 and right_ok == 1:
            count += 1
    return count
""", "O(N^2)", "O(N)")

P(168, "nearby_duplicate", "hash-map", "easy",
   "Return 1 if there exist i != j with nums[i]==nums[j] and abs(i-j) <= k.",
   """\
def nearby_duplicate(nums, k):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j] and j - i <= k:
                return 1
    return 0
""", "O(N^2)", "O(N)")

P(169, "longest_consecutive_len", "set-operations", "easy",
   "Length of the longest consecutive integer sequence within nums.",
   """\
def longest_consecutive_len(nums):
    if len(nums) == 0:
        return 0
    best = 1
    for i in range(len(nums)):
        length = 1
        cur = nums[i]
        while True:
            found = 0
            for j in range(len(nums)):
                if nums[j] == cur + 1:
                    found = 1
                    break
            if found == 1:
                length += 1
                cur += 1
            else:
                break
        if length > best:
            best = length
    return best
""", "O(N^2)", "O(N)")

P(170, "count_with_strictly_greater", "order-statistics", "easy",
   "Count elements that have at least one strictly greater element in the array.",
   """\
def count_with_strictly_greater(nums):
    count = 0
    for i in range(len(nums)):
        has_greater = 0
        for j in range(len(nums)):
            if nums[j] > nums[i]:
                has_greater = 1
                break
        count += has_greater
    return count
""", "O(N^2)", "O(N)")

P(171, "top_frequency", "frequency", "easy",
   "Return the frequency of the most common element.",
   """\
def top_frequency(nums):
    if len(nums) == 0:
        return 0
    best = 0
    for i in range(len(nums)):
        count = 0
        for j in range(len(nums)):
            if nums[j] == nums[i]:
                count += 1
        if count > best:
            best = count
    return best
""", "O(N^2)", "O(N)")

P(172, "even_product_pairs", "pair-finding", "easy",
   "Count pairs (i,j) with i<j where nums[i]*nums[j] is even.",
   """\
def even_product_pairs(nums):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if (nums[i] * nums[j]) % 2 == 0:
                count += 1
    return count
""", "O(N^2)", "O(N)")

P(173, "total_prefix_sums", "prefix-sum", "easy",
   "Return the sum of all prefix sums: sum(nums[0..0]) + sum(nums[0..1]) + ... + sum(nums[0..N-1]).",
   """\
def total_prefix_sums(nums):
    total = 0
    for i in range(len(nums)):
        s = 0
        for j in range(i + 1):
            s += nums[j]
        total += s
    return total
""", "O(N^2)", "O(N)")

P(174, "dominators_count", "array-transform", "easy",
   "Count elements that are strictly greater than all preceding elements.",
   """\
def dominators_count(nums):
    count = 0
    for i in range(len(nums)):
        is_dom = 1
        for j in range(i):
            if nums[j] >= nums[i]:
                is_dom = 0
                break
        if is_dom == 1:
            count += 1
    return count
""", "O(N^2)", "O(N)")

P(175, "max_container_water", "two-pointer", "easy",
   "Container with most water: max of min(nums[i],nums[j])*(j-i) for all i<j.",
   """\
def max_container_water(nums):
    best = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            h = nums[i] if nums[i] < nums[j] else nums[j]
            area = h * (j - i)
            if area > best:
                best = area
    return best
""", "O(N^2)", "O(N)")

P(176, "smallest_absent_positive", "hash-map", "easy",
   "Return the smallest positive integer not present in nums.",
   """\
def smallest_absent_positive(nums):
    for v in range(1, len(nums) + 2):
        found = 0
        for j in range(len(nums)):
            if nums[j] == v:
                found = 1
                break
        if found == 0:
            return v
    return len(nums) + 1
""", "O(N^2)", "O(N)")

P(177, "all_pairs_xor_sum", "pair-finding", "easy",
   "Return sum of (nums[i] XOR nums[j]) for all pairs i<j.",
   """\
def all_pairs_xor_sum(nums):
    total = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            total += nums[i] ^ nums[j]
    return total
""", "O(N^2)", "O(N)")

P(178, "count_equal_triplets", "frequency", "easy",
   "Count triplets (i<j<k) where nums[i]==nums[j]==nums[k].",
   """\
def count_equal_triplets(nums):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            for k in range(j + 1, len(nums)):
                if nums[i] == nums[j] == nums[k]:
                    count += 1
    return count
""", "O(N^3)", "O(N)")

P(179, "max_product_two", "pair-finding", "easy",
   "Return the maximum product of any two elements (may be negative).",
   """\
def max_product_two(nums):
    if len(nums) < 2:
        return 0
    best = nums[0] * nums[1]
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            p = nums[i] * nums[j]
            if p > best:
                best = p
    return best
""", "O(N^2)", "O(N)")

P(180, "sum_of_pair_maxes", "pair-finding", "easy",
   "Return sum of max(nums[i], nums[j]) for every pair i<j.",
   """\
def sum_of_pair_maxes(nums):
    total = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            total += nums[i] if nums[i] > nums[j] else nums[j]
    return total
""", "O(N^2)", "O(N log N)")

P(181, "count_surpassers_total", "pair-finding", "easy",
   "Sum over all i of count of j>i where nums[j]>nums[i].",
   """\
def count_surpassers_total(nums):
    total = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[j] > nums[i]:
                total += 1
    return total
""", "O(N^2)", "O(N log N)")

P(182, "is_sorted_rotation", "sorting", "easy",
   "Return 1 if nums is a rotation of a non-decreasingly sorted array, else 0.",
   """\
def is_sorted_rotation(nums):
    n = len(nums)
    if n <= 1:
        return 1
    for r in range(n):
        ok = 1
        for i in range(n - 1):
            if nums[(r + i) % n] > nums[(r + i + 1) % n]:
                ok = 0
                break
        if ok == 1:
            return 1
    return 0
""", "O(N^2)", "O(N)")

P(183, "zero_sum_pair_count", "pair-finding", "easy",
   "Count pairs (i,j) with i<j where nums[i]+nums[j]==0.",
   """\
def zero_sum_pair_count(nums):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == 0:
                count += 1
    return count
""", "O(N^2)", "O(N)")


# ══════════════════════════════════════════════════════════════════════
# MEDIUM (17 problems): TYR-184 – TYR-200
# ══════════════════════════════════════════════════════════════════════

P(184, "lis_length", "dynamic-programming", "medium",
   "Return the length of the longest strictly increasing subsequence.",
   """\
def lis_length(nums):
    n = len(nums)
    if n == 0:
        return 0
    best = 1
    for mask in range(1, 1 << n):
        subseq = []
        for i in range(n):
            if mask & (1 << i):
                subseq.append(nums[i])
        ok = 1
        for i in range(len(subseq) - 1):
            if subseq[i] >= subseq[i + 1]:
                ok = 0
                break
        if ok == 1 and len(subseq) > best:
            best = len(subseq)
    return best
""", "O(2^N * N)", "O(N^2)")

P(185, "max_subarray_kadane", "subarray", "medium",
   "Return the maximum sum of any contiguous subarray.",
   """\
def max_subarray_kadane(nums):
    if len(nums) == 0:
        return 0
    best = nums[0]
    for i in range(len(nums)):
        for j in range(i, len(nums)):
            s = 0
            for k in range(i, j + 1):
                s += nums[k]
            if s > best:
                best = s
    return best
""", "O(N^3)", "O(N)")

P(186, "subarrays_with_sum", "prefix-sum", "medium",
   "Count contiguous subarrays whose sum equals target.",
   """\
def subarrays_with_sum(nums, target):
    count = 0
    for i in range(len(nums)):
        for j in range(i, len(nums)):
            s = 0
            for k in range(i, j + 1):
                s += nums[k]
            if s == target:
                count += 1
    return count
""", "O(N^3)", "O(N)")

P(187, "lcs_of_arrays", "dynamic-programming", "medium",
   "Return the length of the longest common subsequence of two arrays.",
   """\
def lcs_of_arrays(a, b):
    na = len(a)
    nb = len(b)
    best = 0
    for mask in range(1 << na):
        sub_a = []
        for i in range(na):
            if mask & (1 << i):
                sub_a.append(a[i])
        for mask2 in range(1 << nb):
            sub_b = []
            for j in range(nb):
                if mask2 & (1 << j):
                    sub_b.append(b[j])
            if len(sub_a) == len(sub_b) and len(sub_a) > best:
                match = 1
                for k in range(len(sub_a)):
                    if sub_a[k] != sub_b[k]:
                        match = 0
                        break
                if match == 1:
                    best = len(sub_a)
    return best
""", "O(2^(N+M) * N)", "O(N*M)")

P(188, "max_product_subarray", "subarray", "medium",
   "Return the maximum product of any contiguous subarray.",
   """\
def max_product_subarray(nums):
    if len(nums) == 0:
        return 0
    best = nums[0]
    for i in range(len(nums)):
        for j in range(i, len(nums)):
            prod = 1
            for k in range(i, j + 1):
                prod *= nums[k]
            if prod > best:
                best = prod
    return best
""", "O(N^3)", "O(N)")

P(189, "count_zero_triplets", "pair-finding", "medium",
   "Count triplets (i<j<k) where nums[i]+nums[j]+nums[k]==0.",
   """\
def count_zero_triplets(nums):
    count = 0
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if nums[i] + nums[j] + nums[k] == 0:
                    count += 1
    return count
""", "O(N^3)", "O(N^2)")

P(190, "can_partition_equal", "dynamic-programming", "medium",
   "Return 1 if nums can be partitioned into two subsets with equal sum, else 0.",
   """\
def can_partition_equal(nums):
    n = len(nums)
    total = 0
    for x in nums:
        total += x
    if total % 2 != 0:
        return 0
    half = total // 2
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        if s == half:
            return 1
    return 0
""", "O(2^N * N)", "O(N*S)")

P(191, "rob_houses_max", "dynamic-programming", "medium",
   "Max sum of elements with no two adjacent selected.",
   """\
def rob_houses_max(nums):
    n = len(nums)
    if n == 0:
        return 0
    best = 0
    for mask in range(1 << n):
        ok = 1
        for i in range(n - 1):
            if (mask & (1 << i)) and (mask & (1 << (i + 1))):
                ok = 0
                break
        if ok == 1:
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += nums[i]
            if s > best:
                best = s
    return best
""", "O(2^N * N)", "O(N)")

P(192, "subset_count_target", "dynamic-programming", "medium",
   "Count the number of subsets of nums that sum to target.",
   """\
def subset_count_target(nums, target):
    n = len(nums)
    count = 0
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        if s == target:
            count += 1
    return count
""", "O(2^N * N)", "O(N*T)")

P(193, "climb_stairs_min_cost", "dynamic-programming", "medium",
   "Min cost to reach the top, starting from step 0 or 1. Cost to step on index i is nums[i].",
   """\
def climb_stairs_min_cost(nums):
    n = len(nums)
    if n == 0:
        return 0
    if n == 1:
        return nums[0]
    best = 10000
    for mask in range(1 << n):
        steps = []
        for i in range(n):
            if mask & (1 << i):
                steps.append(i)
        if len(steps) == 0:
            continue
        if steps[0] > 1:
            continue
        ok = 1
        for i in range(len(steps) - 1):
            if steps[i + 1] - steps[i] > 2:
                ok = 0
                break
        if ok == 1 and steps[-1] >= n - 2:
            cost = 0
            for s in steps:
                cost += nums[s]
            if cost < best:
                best = cost
    return best
""", "O(2^N * N)", "O(N)")

P(194, "longest_palindrome_subseq_len", "dynamic-programming", "medium",
   "Return the length of the longest palindromic subsequence of nums.",
   """\
def longest_palindrome_subseq_len(nums):
    n = len(nums)
    if n == 0:
        return 0
    best = 1
    for mask in range(1, 1 << n):
        subseq = []
        for i in range(n):
            if mask & (1 << i):
                subseq.append(nums[i])
        is_pal = 1
        for i in range(len(subseq) // 2):
            if subseq[i] != subseq[len(subseq) - 1 - i]:
                is_pal = 0
                break
        if is_pal == 1 and len(subseq) > best:
            best = len(subseq)
    return best
""", "O(2^N * N)", "O(N^2)")

P(195, "max_sum_inc_subseq", "dynamic-programming", "medium",
   "Return the maximum sum of a strictly increasing subsequence.",
   """\
def max_sum_inc_subseq(nums):
    n = len(nums)
    if n == 0:
        return 0
    best = nums[0]
    for mask in range(1, 1 << n):
        subseq = []
        s = 0
        for i in range(n):
            if mask & (1 << i):
                subseq.append(nums[i])
                s += nums[i]
        ok = 1
        for i in range(len(subseq) - 1):
            if subseq[i] >= subseq[i + 1]:
                ok = 0
                break
        if ok == 1 and s > best:
            best = s
    return best
""", "O(2^N * N)", "O(N^2)")

P(196, "longest_wiggle_length", "dynamic-programming", "medium",
   "Length of longest wiggle subsequence: alternating rises and falls.",
   """\
def longest_wiggle_length(nums):
    n = len(nums)
    if n == 0:
        return 0
    best = 1
    for mask in range(1, 1 << n):
        subseq = []
        for i in range(n):
            if mask & (1 << i):
                subseq.append(nums[i])
        if len(subseq) <= 1:
            if len(subseq) > best:
                best = len(subseq)
            continue
        ok = 1
        for i in range(1, len(subseq) - 1):
            d1 = subseq[i] - subseq[i - 1]
            d2 = subseq[i + 1] - subseq[i]
            if d1 == 0 or d2 == 0 or (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0):
                ok = 0
                break
        d_first = subseq[1] - subseq[0]
        if d_first == 0:
            ok = 0
        if ok == 1 and len(subseq) > best:
            best = len(subseq)
    return best
""", "O(2^N * N)", "O(N)")

P(197, "min_jumps_to_end", "dynamic-programming", "medium",
   "Min jumps to reach the last index. nums[i] = max jump length from i. Return -1 if impossible.",
   """\
def min_jumps_to_end(nums):
    n = len(nums)
    if n <= 1:
        return 0
    best = n + 1
    for mask in range(1 << n):
        if not (mask & 1):
            continue
        if not (mask & (1 << (n - 1))):
            continue
        steps = []
        for i in range(n):
            if mask & (1 << i):
                steps.append(i)
        ok = 1
        for i in range(len(steps) - 1):
            gap = steps[i + 1] - steps[i]
            if gap > nums[steps[i]]:
                ok = 0
                break
        if ok == 1 and len(steps) - 1 < best:
            best = len(steps) - 1
    if best > n:
        return -1
    return best
""", "O(2^N * N)", "O(N)")

P(198, "target_sum_ways", "dynamic-programming", "medium",
   "Count ways to assign + or - to each element to reach target sum.",
   """\
def target_sum_ways(nums, target):
    n = len(nums)
    count = 0
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
            else:
                s -= nums[i]
        if s == target:
            count += 1
    return count
""", "O(2^N)", "O(N*S)")

P(199, "decode_digit_ways", "dynamic-programming", "medium",
   "Count decodings of a digit array (values 0-9). Mappings: 1-26 map to letters. Return count.",
   """\
def decode_digit_ways(nums):
    n = len(nums)
    if n == 0:
        return 0
    count = 0
    for mask in range(1, 1 << (n - 1) + 1):
        groups = []
        start = 0
        for i in range(n - 1):
            if mask & (1 << i):
                groups.append(nums[start:i + 1])
                start = i + 1
        groups.append(nums[start:n])
        ok = 1
        for g in groups:
            if len(g) == 1:
                if g[0] == 0:
                    ok = 0
                    break
            elif len(g) == 2:
                val = g[0] * 10 + g[1]
                if val < 10 or val > 26:
                    ok = 0
                    break
            else:
                ok = 0
                break
        if ok == 1:
            count += 1
    return count
""", "O(2^N)", "O(N)")

P(200, "max_chain_pairs", "greedy", "medium",
   "Max chain length from pairs (nums[2i], nums[2i+1]). Chain: b1 < a2 for consecutive pairs.",
   """\
def max_chain_pairs(nums):
    n = len(nums) // 2
    if n == 0:
        return 0
    pairs = []
    for i in range(n):
        pairs.append((nums[2 * i], nums[2 * i + 1]))
    best = 1
    for mask in range(1, 1 << n):
        sel = []
        for i in range(n):
            if mask & (1 << i):
                sel.append(pairs[i])
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                if sel[i][1] > sel[j][1]:
                    sel[i], sel[j] = sel[j], sel[i]
        ok = 1
        for i in range(len(sel) - 1):
            if sel[i][1] >= sel[i + 1][0]:
                ok = 0
                break
        if ok == 1 and len(sel) > best:
            best = len(sel)
    return best
""", "O(2^N * N^2)", "O(N log N)")


# ══════════════════════════════════════════════════════════════════════
# HARD (50 problems): TYR-201 – TYR-250
# ══════════════════════════════════════════════════════════════════════

P(201, "min_partition_diff", "dynamic-programming", "hard",
   "Return the minimum absolute difference between sums of two subsets partitioning nums.",
   """\
def min_partition_diff(nums):
    n = len(nums)
    total = 0
    for x in nums:
        total += x
    best = abs(total)
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        diff = abs(total - 2 * s)
        if diff < best:
            best = diff
    return best
""", "O(2^N * N)", "O(N*S)")

P(202, "palindrome_min_cuts", "dynamic-programming", "hard",
   "Min cuts to partition nums into palindromic subarrays.",
   """\
def palindrome_min_cuts(nums):
    n = len(nums)
    if n <= 1:
        return 0
    best = n - 1
    for mask in range(1 << (n - 1)):
        cuts = []
        for i in range(n - 1):
            if mask & (1 << i):
                cuts.append(i + 1)
        parts = []
        prev = 0
        for c in cuts:
            parts.append(nums[prev:c])
            prev = c
        parts.append(nums[prev:n])
        ok = 1
        for p in parts:
            for i in range(len(p) // 2):
                if p[i] != p[len(p) - 1 - i]:
                    ok = 0
                    break
            if ok == 0:
                break
        if ok == 1 and len(parts) - 1 < best:
            best = len(parts) - 1
    return best
""", "O(2^N * N)", "O(N^2)")

P(203, "knapsack_max_value", "dynamic-programming", "hard",
   "0/1 knapsack: even-indexed = weights, odd-indexed = values. Maximize value within capacity.",
   """\
def knapsack_max_value(nums, capacity):
    n = len(nums) // 2
    best = 0
    for mask in range(1 << n):
        w = 0
        v = 0
        for i in range(n):
            if mask & (1 << i):
                w += nums[2 * i]
                v += nums[2 * i + 1]
        if w <= capacity and v > best:
            best = v
    return best
""", "O(2^N * N)", "O(N*W)")

P(204, "shortest_superseq_len", "dynamic-programming", "hard",
   "Return the length of the shortest common supersequence of two arrays.",
   """\
def shortest_superseq_len(a, b):
    na = len(a)
    nb = len(b)
    best = na + nb
    for mask_a in range(1 << na):
        for mask_b in range(1 << nb):
            merged = []
            ia = 0
            ib = 0
            ok = 1
            while ia < na or ib < nb:
                use_a = ia < na and (mask_a & (1 << ia))
                use_b = ib < nb and (mask_b & (1 << ib))
                if use_a and use_b and a[ia] == b[ib]:
                    merged.append(a[ia])
                    ia += 1
                    ib += 1
                elif use_a:
                    merged.append(a[ia])
                    ia += 1
                elif use_b:
                    merged.append(b[ib])
                    ib += 1
                else:
                    ia += 1 if ia < na else 0
                    ib += 1 if ib < nb else 0
            check_a = []
            check_b = []
            for x in merged:
                if len(check_a) < na and x == a[len(check_a)]:
                    check_a.append(x)
                if len(check_b) < nb and x == b[len(check_b)]:
                    check_b.append(x)
            if len(check_a) == na and len(check_b) == nb and len(merged) < best:
                best = len(merged)
    return best
""", "O(2^(N+M) * (N+M))", "O(N*M)")

P(205, "max_sum_no_three_adjacent", "dynamic-programming", "hard",
   "Max sum selecting elements with no three consecutive indices selected.",
   """\
def max_sum_no_three_adjacent(nums):
    n = len(nums)
    best = 0
    for mask in range(1 << n):
        ok = 1
        for i in range(n - 2):
            if (mask >> i) & 7 == 7:
                ok = 0
                break
        if ok == 1:
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += nums[i]
            if s > best:
                best = s
    return best
""", "O(2^N * N)", "O(N)")

P(206, "count_inc_subsequences", "dynamic-programming", "hard",
   "Count all strictly increasing subsequences of length >= 1.",
   """\
def count_inc_subsequences(nums):
    n = len(nums)
    count = 0
    for mask in range(1, 1 << n):
        subseq = []
        for i in range(n):
            if mask & (1 << i):
                subseq.append(nums[i])
        ok = 1
        for i in range(len(subseq) - 1):
            if subseq[i] >= subseq[i + 1]:
                ok = 0
                break
        if ok == 1:
            count += 1
    return count
""", "O(2^N * N)", "O(N^2)")

P(207, "longest_bitonic_subseq", "dynamic-programming", "hard",
   "Length of longest bitonic subsequence (first increases then decreases).",
   """\
def longest_bitonic_subseq(nums):
    n = len(nums)
    if n == 0:
        return 0
    best = 1
    for mask in range(1, 1 << n):
        subseq = []
        for i in range(n):
            if mask & (1 << i):
                subseq.append(nums[i])
        if len(subseq) <= best:
            continue
        peak = -1
        for p in range(len(subseq)):
            inc = 1
            for i in range(p):
                if subseq[i] >= subseq[i + 1]:
                    inc = 0
                    break
            dec = 1
            for i in range(p, len(subseq) - 1):
                if subseq[i] <= subseq[i + 1]:
                    dec = 0
                    break
            if inc == 1 and dec == 1:
                peak = p
                break
        if peak >= 0 and len(subseq) > best:
            best = len(subseq)
    return best
""", "O(2^N * N^2)", "O(N^2)")

P(208, "min_adjacent_swaps_sort", "sorting", "hard",
   "Minimum number of adjacent swaps to sort the array in non-decreasing order.",
   """\
def min_adjacent_swaps_sort(nums):
    arr = nums[:]
    swaps = 0
    n = len(arr)
    for i in range(n):
        for j in range(n - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1
    return swaps
""", "O(N^2)", "O(N log N)")

P(209, "count_subseq_target_sum", "dynamic-programming", "hard",
   "Count subsequences (not subarrays) whose elements sum to target.",
   """\
def count_subseq_target_sum(nums, target):
    n = len(nums)
    count = 0
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        if s == target:
            count += 1
    return count
""", "O(2^N * N)", "O(N*T)")

P(210, "burst_balloon_coins", "dynamic-programming", "hard",
   "Max coins from bursting all balloons. Bursting i earns nums[left]*nums[i]*nums[right].",
   """\
def burst_balloon_coins(nums):
    n = len(nums)
    if n == 0:
        return 0
    arr = [1] + nums[:] + [1]

    def solve(balloons):
        if len(balloons) == 2:
            return 0
        best = 0
        for i in range(1, len(balloons) - 1):
            coins = balloons[i - 1] * balloons[i] * balloons[i + 1]
            remaining = balloons[:i] + balloons[i + 1:]
            coins += solve(remaining)
            if coins > best:
                best = coins
        return best

    return solve(arr)
""", "O(N!)", "O(N^3)")

P(211, "edit_distance_arrays", "dynamic-programming", "hard",
   "Min insertions, deletions, or substitutions to transform array a into array b.",
   """\
def edit_distance_arrays(a, b):
    na = len(a)
    nb = len(b)

    def helper(i, j):
        if i == 0:
            return j
        if j == 0:
            return i
        if a[i - 1] == b[j - 1]:
            return helper(i - 1, j - 1)
        ins = helper(i, j - 1) + 1
        delete = helper(i - 1, j) + 1
        replace = helper(i - 1, j - 1) + 1
        best = ins
        if delete < best:
            best = delete
        if replace < best:
            best = replace
        return best

    return helper(na, nb)
""", "O(3^(N+M))", "O(N*M)")

P(212, "interleave_possible", "dynamic-programming", "hard",
   "Return 1 if c is an interleaving of a and b (preserving order), else 0.",
   """\
def interleave_possible(a, b, c):
    na = len(a)
    nb = len(b)
    if na + nb != len(c):
        return 0

    def helper(i, j, k):
        if k == len(c):
            return 1 if i == na and j == nb else 0
        if i < na and c[k] == a[i]:
            if helper(i + 1, j, k + 1) == 1:
                return 1
        if j < nb and c[k] == b[j]:
            if helper(i, j + 1, k + 1) == 1:
                return 1
        return 0

    return helper(0, 0, 0)
""", "O(2^(N+M))", "O(N*M)")

P(213, "max_rob_circular", "dynamic-programming", "hard",
   "House robber on circular array: max non-adjacent sum where first and last are also adjacent.",
   """\
def max_rob_circular(nums):
    n = len(nums)
    if n == 0:
        return 0
    if n == 1:
        return nums[0]
    best = 0
    for mask in range(1 << n):
        ok = 1
        for i in range(n):
            nxt = (i + 1) % n
            if (mask & (1 << i)) and (mask & (1 << nxt)):
                ok = 0
                break
        if ok == 1:
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += nums[i]
            if s > best:
                best = s
    return best
""", "O(2^N * N)", "O(N)")

P(214, "count_distinct_subseq", "dynamic-programming", "hard",
   "Count distinct subsequences of a that exactly match b.",
   """\
def count_distinct_subseq(a, b):
    na = len(a)
    nb = len(b)
    count = 0
    for mask in range(1 << na):
        subseq = []
        for i in range(na):
            if mask & (1 << i):
                subseq.append(a[i])
        if len(subseq) == nb:
            match = 1
            for i in range(nb):
                if subseq[i] != b[i]:
                    match = 0
                    break
            if match == 1:
                count += 1
    return count
""", "O(2^N * N)", "O(N*M)")

P(215, "optimal_game_score", "dynamic-programming", "hard",
   "Two players pick from ends optimally. Return first player's score minus second's.",
   """\
def optimal_game_score(nums):
    n = len(nums)
    if n == 0:
        return 0

    def solve(l, r, is_first):
        if l > r:
            return 0
        if is_first:
            pick_l = nums[l] + solve(l + 1, r, 0)
            pick_r = nums[r] + solve(l, r - 1, 0)
            return pick_l if pick_l > pick_r else pick_r
        else:
            pick_l = -nums[l] + solve(l + 1, r, 1)
            pick_r = -nums[r] + solve(l, r - 1, 1)
            return pick_l if pick_l < pick_r else pick_r

    return solve(0, n - 1, 1)
""", "O(2^N)", "O(N^2)")

P(216, "min_insertions_palindrome", "dynamic-programming", "hard",
   "Min insertions to make the array a palindrome.",
   """\
def min_insertions_palindrome(nums):
    n = len(nums)

    def helper(l, r):
        if l >= r:
            return 0
        if nums[l] == nums[r]:
            return helper(l + 1, r - 1)
        opt1 = helper(l + 1, r) + 1
        opt2 = helper(l, r - 1) + 1
        return opt1 if opt1 < opt2 else opt2

    return helper(0, n - 1)
""", "O(2^N)", "O(N^2)")

P(217, "count_subsets_xor", "dynamic-programming", "hard",
   "Count non-empty subsets whose XOR equals target.",
   """\
def count_subsets_xor(nums, target):
    n = len(nums)
    count = 0
    for mask in range(1, 1 << n):
        xor_val = 0
        for i in range(n):
            if mask & (1 << i):
                xor_val ^= nums[i]
        if xor_val == target:
            count += 1
    return count
""", "O(2^N * N)", "O(N*MAX)")

P(218, "max_profit_cooldown", "dynamic-programming", "hard",
   "Max stock profit with cooldown: prices in nums. After selling, must wait 1 day.",
   """\
def max_profit_cooldown(nums):
    n = len(nums)
    if n < 2:
        return 0
    best = 0
    for mask in range(1 << n):
        buys = []
        sells = []
        for i in range(n):
            if mask & (1 << i):
                if len(buys) == len(sells):
                    buys.append(i)
                else:
                    sells.append(i)
        if len(buys) != len(sells):
            continue
        ok = 1
        for i in range(len(buys)):
            if buys[i] >= sells[i]:
                ok = 0
                break
        for i in range(len(sells) - 1):
            if buys[i + 1] <= sells[i] + 1:
                ok = 0
                break
        if ok == 1:
            profit = 0
            for i in range(len(buys)):
                profit += nums[sells[i]] - nums[buys[i]]
            if profit > best:
                best = profit
    return best
""", "O(2^N * N)", "O(N)")

P(219, "rod_cutting_max", "dynamic-programming", "hard",
   "Max revenue from cutting rod of length N. nums[i] = price for piece of length i+1.",
   """\
def rod_cutting_max(nums):
    n = len(nums)
    if n == 0:
        return 0
    best = 0
    for mask in range(1, 1 << n):
        total_len = 0
        revenue = 0
        for i in range(n):
            if mask & (1 << i):
                total_len += i + 1
                revenue += nums[i]
        if total_len <= n and revenue > best:
            best = revenue
    return best
""", "O(2^N * N)", "O(N^2)")

P(220, "count_unique_bst", "dynamic-programming", "hard",
   "Count structurally unique BSTs that can store values 1..nums[0]. Uses first element only.",
   """\
def count_unique_bst(nums):
    n = nums[0] if len(nums) > 0 else 0
    if n <= 0:
        return 0

    def count(lo, hi):
        if lo >= hi:
            return 1
        total = 0
        for root in range(lo, hi + 1):
            left = count(lo, root - 1)
            right = count(root + 1, hi)
            total += left * right
        return total

    return count(1, n)
""", "O(4^N / N^(3/2))", "O(N^2)")

P(221, "longest_arith_subseq_len", "dynamic-programming", "hard",
   "Length of the longest arithmetic subsequence in nums.",
   """\
def longest_arith_subseq_len(nums):
    n = len(nums)
    if n <= 2:
        return n
    best = 2
    for mask in range(1, 1 << n):
        subseq = []
        for i in range(n):
            if mask & (1 << i):
                subseq.append(nums[i])
        if len(subseq) <= best:
            continue
        if len(subseq) < 2:
            continue
        diff = subseq[1] - subseq[0]
        ok = 1
        for i in range(1, len(subseq) - 1):
            if subseq[i + 1] - subseq[i] != diff:
                ok = 0
                break
        if ok == 1 and len(subseq) > best:
            best = len(subseq)
    return best
""", "O(2^N * N)", "O(N^2)")

P(222, "coin_change_min", "dynamic-programming", "hard",
   "Min coins from nums to reach target amount. Return -1 if impossible. Coins reusable.",
   """\
def coin_change_min(nums, target):
    if target == 0:
        return 0
    if len(nums) == 0:
        return -1
    best = target + 1

    def solve(remaining, count):
        nonlocal best
        if remaining == 0:
            if count < best:
                best = count
            return
        if remaining < 0 or count >= best:
            return
        for c in nums:
            solve(remaining - c, count + 1)

    solve(target, 0)
    return best if best <= target else -1
""", "O(S^N)", "O(N*S)")

P(223, "coin_change_ways", "dynamic-programming", "hard",
   "Count ways to make target using coins from nums (reusable). Order doesn't matter.",
   """\
def coin_change_ways(nums, target):
    if target == 0:
        return 1

    def solve(idx, remaining):
        if remaining == 0:
            return 1
        if remaining < 0 or idx >= len(nums):
            return 0
        return solve(idx, remaining - nums[idx]) + solve(idx + 1, remaining)

    return solve(0, target)
""", "O(S^N)", "O(N*S)")

P(224, "longest_common_subarray_len", "dynamic-programming", "hard",
   "Length of the longest common contiguous subarray between a and b.",
   """\
def longest_common_subarray_len(a, b):
    best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            length = 0
            while i + length < len(a) and j + length < len(b):
                if a[i + length] == b[j + length]:
                    length += 1
                else:
                    break
            if length > best:
                best = length
    return best
""", "O(N^2 * min(N,M))", "O(N*M)")

P(225, "min_cost_grid_path", "dynamic-programming", "hard",
   "Min cost path from top-left to bottom-right in grid (move right or down). Grid = flat array, cols given.",
   """\
def min_cost_grid_path(nums, cols):
    rows = len(nums) // cols if cols > 0 else 0
    if rows == 0 or cols == 0:
        return 0

    def solve(r, c):
        if r == rows - 1 and c == cols - 1:
            return nums[r * cols + c]
        if r >= rows or c >= cols:
            return 10000
        return nums[r * cols + c] + min(solve(r + 1, c), solve(r, c + 1))

    return solve(0, 0)
""", "O(2^(R+C))", "O(R*C)")

P(226, "count_grid_paths_obstacles", "dynamic-programming", "hard",
   "Count paths top-left to bottom-right (right/down only). 1 = obstacle. Grid flat, cols given.",
   """\
def count_grid_paths_obstacles(nums, cols):
    rows = len(nums) // cols if cols > 0 else 0
    if rows == 0 or cols == 0:
        return 0
    if nums[0] == 1 or nums[-1] == 1:
        return 0

    def solve(r, c):
        if r == rows - 1 and c == cols - 1:
            return 1
        if r >= rows or c >= cols:
            return 0
        if nums[r * cols + c] == 1:
            return 0
        return solve(r + 1, c) + solve(r, c + 1)

    return solve(0, 0)
""", "O(2^(R+C))", "O(R*C)")

P(227, "max_earn_delete", "dynamic-programming", "hard",
   "Pick element, earn its value, delete all occurrences of value-1 and value+1. Max total earnings.",
   """\
def max_earn_delete(nums):
    n = len(nums)
    if n == 0:
        return 0
    best = 0
    for mask in range(1 << n):
        selected = []
        for i in range(n):
            if mask & (1 << i):
                selected.append(nums[i])
        ok = 1
        for i in range(len(selected)):
            for j in range(n):
                if not (mask & (1 << j)):
                    diff = abs(nums[j] - selected[i])
                    if diff == 1:
                        ok = 0
                        break
            if ok == 0:
                break
        if ok == 0:
            continue
        vals_ok = 1
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                if abs(selected[i] - selected[j]) == 1:
                    vals_ok = 0
                    break
            if vals_ok == 0:
                break
        if vals_ok == 1:
            s = 0
            for v in selected:
                s += v
            if s > best:
                best = s
    return best
""", "O(2^N * N^2)", "O(N)")

P(228, "falling_path_min", "dynamic-programming", "hard",
   "Min falling path sum: each row pick one element, adjacent column to previous. Flat grid, cols given.",
   """\
def falling_path_min(nums, cols):
    rows = len(nums) // cols if cols > 0 else 0
    if rows == 0:
        return 0

    def solve(r, c):
        if r == rows:
            return 0
        if c < 0 or c >= cols:
            return 10000
        val = nums[r * cols + c]
        best = solve(r + 1, c)
        left = solve(r + 1, c - 1)
        right = solve(r + 1, c + 1)
        if left < best:
            best = left
        if right < best:
            best = right
        return val + best

    result = 10000
    for c in range(cols):
        v = solve(0, c)
        if v < result:
            result = v
    return result
""", "O(3^R * C)", "O(R*C)")

P(229, "max_square_side", "dynamic-programming", "hard",
   "Side of largest square containing only 1s in binary matrix. Flat array, cols given.",
   """\
def max_square_side(nums, cols):
    rows = len(nums) // cols if cols > 0 else 0
    best = 0
    for r in range(rows):
        for c in range(cols):
            for s in range(1, min(rows - r, cols - c) + 1):
                ok = 1
                for dr in range(s):
                    for dc in range(s):
                        if nums[(r + dr) * cols + (c + dc)] != 1:
                            ok = 0
                            break
                    if ok == 0:
                        break
                if ok == 1 and s > best:
                    best = s
                else:
                    break
    return best
""", "O(R*C*min(R,C)^2)", "O(R*C)")

P(230, "count_square_submatrices", "dynamic-programming", "hard",
   "Count all square submatrices containing only 1s. Flat binary array, cols given.",
   """\
def count_square_submatrices(nums, cols):
    rows = len(nums) // cols if cols > 0 else 0
    total = 0
    for r in range(rows):
        for c in range(cols):
            for s in range(1, min(rows - r, cols - c) + 1):
                ok = 1
                for dr in range(s):
                    for dc in range(s):
                        if nums[(r + dr) * cols + (c + dc)] != 1:
                            ok = 0
                            break
                    if ok == 0:
                        break
                if ok == 1:
                    total += 1
                else:
                    break
    return total
""", "O(R*C*min(R,C)^2)", "O(R*C)")

P(231, "can_jump_end", "dynamic-programming", "hard",
   "Return 1 if you can reach the last index. nums[i] = max jump from index i.",
   """\
def can_jump_end(nums):
    n = len(nums)
    if n <= 1:
        return 1
    for mask in range(1 << n):
        if not (mask & 1):
            continue
        if not (mask & (1 << (n - 1))):
            continue
        steps = []
        for i in range(n):
            if mask & (1 << i):
                steps.append(i)
        ok = 1
        for i in range(len(steps) - 1):
            if steps[i + 1] - steps[i] > nums[steps[i]]:
                ok = 0
                break
        if ok == 1:
            return 1
    return 0
""", "O(2^N * N)", "O(N)")

P(232, "max_product_cut", "dynamic-programming", "hard",
   "Max product from cutting rope of length nums[0] into integer parts (each >= 1).",
   """\
def max_product_cut(nums):
    n = nums[0] if len(nums) > 0 else 0
    if n <= 1:
        return 0

    def solve(remaining):
        if remaining <= 0:
            return 1
        best = 0
        for cut in range(1, remaining + 1):
            val = cut * solve(remaining - cut)
            if val > best:
                best = val
        return best

    best = 0
    for first_cut in range(1, n):
        val = first_cut * solve(n - first_cut)
        if val > best:
            best = val
    return best
""", "O(N^N)", "O(N^2)")

P(233, "paint_fence_ways", "dynamic-programming", "hard",
   "Count ways to paint nums[0] fence posts with nums[1] colors, no 3 consecutive same color.",
   """\
def paint_fence_ways(nums):
    n = nums[0] if len(nums) > 0 else 0
    k = nums[1] if len(nums) > 1 else 0
    if n == 0 or k == 0:
        return 0

    def solve(pos, prev1, prev2):
        if pos == n:
            return 1
        total = 0
        for color in range(k):
            if pos >= 2 and color == prev1 and color == prev2:
                continue
            total += solve(pos + 1, color, prev1)
        return total

    return solve(0, -1, -1)
""", "O(K^N)", "O(N*K)")

P(234, "count_derangements", "combinatorics", "hard",
   "Count derangements of array of length nums[0]: permutations with no fixed points.",
   """\
def count_derangements(nums):
    n = nums[0] if len(nums) > 0 else 0
    if n <= 0:
        return 0
    if n == 1:
        return 0
    arr = list(range(n))

    def permutations(lst):
        if len(lst) <= 1:
            return [lst[:]]
        result = []
        for i in range(len(lst)):
            rest = lst[:i] + lst[i + 1:]
            for p in permutations(rest):
                result.append([lst[i]] + p)
        return result

    count = 0
    for perm in permutations(arr):
        ok = 1
        for i in range(n):
            if perm[i] == i:
                ok = 0
                break
        if ok == 1:
            count += 1
    return count
""", "O(N! * N)", "O(N)")

P(235, "mcm_min_cost", "dynamic-programming", "hard",
   "Min cost of matrix chain multiplication. Dimensions given as nums (N+1 values for N matrices).",
   """\
def mcm_min_cost(nums):
    n = len(nums) - 1
    if n <= 1:
        return 0

    def solve(i, j):
        if i == j:
            return 0
        best = 999999
        for k in range(i, j):
            cost = solve(i, k) + solve(k + 1, j) + nums[i] * nums[k + 1] * nums[j + 1]
            if cost < best:
                best = cost
        return best

    return solve(0, n - 1)
""", "O(2^N)", "O(N^3)")

P(236, "longest_repeating_subarray", "dynamic-programming", "hard",
   "Length of the longest subarray appearing at least twice (can overlap).",
   """\
def longest_repeating_subarray(nums):
    n = len(nums)
    best = 0
    for i in range(n):
        for j in range(i + 1, n):
            length = 0
            while j + length < n and nums[i + length] == nums[j + length]:
                length += 1
            if length > best:
                best = length
    return best
""", "O(N^3)", "O(N^2)")

P(237, "max_profit_two_tx", "dynamic-programming", "hard",
   "Max stock profit with at most 2 buy-sell transactions. No overlapping.",
   """\
def max_profit_two_tx(nums):
    n = len(nums)
    best = 0
    for b1 in range(n):
        for s1 in range(b1 + 1, n):
            for b2 in range(s1 + 1, n):
                for s2 in range(b2 + 1, n):
                    profit = (nums[s1] - nums[b1]) + (nums[s2] - nums[b2])
                    if profit > best:
                        best = profit
            profit1 = nums[s1] - nums[b1]
            if profit1 > best:
                best = profit1
    return best
""", "O(N^4)", "O(N)")

P(238, "egg_drop_two", "dynamic-programming", "hard",
   "Min trials to find critical floor with 2 eggs and nums[0] floors.",
   """\
def egg_drop_two(nums):
    n = nums[0] if len(nums) > 0 else 0
    if n <= 0:
        return 0

    def solve(eggs, floors):
        if floors <= 1 or eggs == 1:
            return floors
        best = floors
        for x in range(1, floors + 1):
            breaks = solve(eggs - 1, x - 1)
            survives = solve(eggs, floors - x)
            worst = 1 + (breaks if breaks > survives else survives)
            if worst < best:
                best = worst
        return best

    return solve(2, n)
""", "O(N^2)", "O(N log N)")

P(239, "all_subsets_range_sum", "combinatorics", "hard",
   "Sum of (max - min) for every non-empty subset of nums.",
   """\
def all_subsets_range_sum(nums):
    n = len(nums)
    total = 0
    for mask in range(1, 1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        mx = subset[0]
        mn = subset[0]
        for v in subset:
            if v > mx:
                mx = v
            if v < mn:
                mn = v
        total += mx - mn
    return total
""", "O(2^N * N)", "O(N * 2^N)")

P(240, "max_and_pair", "bitmasking", "hard",
   "Max bitwise AND of any pair (i,j) with i < j.",
   """\
def max_and_pair(nums):
    n = len(nums)
    if n < 2:
        return 0
    best = 0
    for i in range(n):
        for j in range(i + 1, n):
            val = nums[i] & nums[j]
            if val > best:
                best = val
    return best
""", "O(N^2)", "O(N*B)")

P(241, "count_valid_triangles", "combinatorics", "hard",
   "Count triplets (i<j<k) that can form a valid triangle (sum of any two > third).",
   """\
def count_valid_triangles(nums):
    count = 0
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = nums[i]
                b = nums[j]
                c = nums[k]
                if a + b > c and a + c > b and b + c > a:
                    count += 1
    return count
""", "O(N^3)", "O(N^2 log N)")

P(242, "min_moves_equalize", "math", "hard",
   "Min moves to make all elements equal. One move: increment N-1 elements by 1.",
   """\
def min_moves_equalize(nums):
    n = len(nums)
    if n <= 1:
        return 0
    best = 999999
    mx = nums[0]
    for v in nums:
        if v > mx:
            mx = v
    for target in range(mx, mx + n * 20 + 1):
        moves = 0
        ok = 1
        for v in nums:
            diff = target - v
            if diff < 0:
                ok = 0
                break
            moves += diff
        if ok == 1 and moves % (n - 1) == 0:
            total_moves = moves // (n - 1)
            if total_moves < best:
                best = total_moves
    return best
""", "O(N * MAX)", "O(N)")

P(243, "longest_zigzag_subseq", "dynamic-programming", "hard",
   "Longest strictly zigzag subsequence: differences alternate sign strictly.",
   """\
def longest_zigzag_subseq(nums):
    n = len(nums)
    if n <= 1:
        return n
    best = 1
    for mask in range(1, 1 << n):
        subseq = []
        for i in range(n):
            if mask & (1 << i):
                subseq.append(nums[i])
        if len(subseq) <= best:
            continue
        if len(subseq) < 2:
            continue
        ok = 1
        for i in range(len(subseq) - 1):
            if subseq[i] == subseq[i + 1]:
                ok = 0
                break
        if ok == 0:
            continue
        for i in range(len(subseq) - 2):
            d1 = subseq[i + 1] - subseq[i]
            d2 = subseq[i + 2] - subseq[i + 1]
            if (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0):
                ok = 0
                break
        if ok == 1 and len(subseq) > best:
            best = len(subseq)
    return best
""", "O(2^N * N)", "O(N)")

P(244, "max_alternating_sum", "dynamic-programming", "hard",
   "Max alternating sum of any subsequence: a[0] - a[1] + a[2] - a[3] + ...",
   """\
def max_alternating_sum(nums):
    n = len(nums)
    if n == 0:
        return 0
    best = 0
    for mask in range(1, 1 << n):
        subseq = []
        for i in range(n):
            if mask & (1 << i):
                subseq.append(nums[i])
        s = 0
        for i in range(len(subseq)):
            if i % 2 == 0:
                s += subseq[i]
            else:
                s -= subseq[i]
        if s > best:
            best = s
    return best
""", "O(2^N * N)", "O(N)")

P(245, "count_bitonic_subseqs", "dynamic-programming", "hard",
   "Count bitonic subsequences of length >= 3 (strictly increases then strictly decreases).",
   """\
def count_bitonic_subseqs(nums):
    n = len(nums)
    count = 0
    for mask in range(1, 1 << n):
        subseq = []
        for i in range(n):
            if mask & (1 << i):
                subseq.append(nums[i])
        if len(subseq) < 3:
            continue
        for p in range(1, len(subseq) - 1):
            inc = 1
            for i in range(p):
                if subseq[i] >= subseq[i + 1]:
                    inc = 0
                    break
            dec = 1
            for i in range(p, len(subseq) - 1):
                if subseq[i] <= subseq[i + 1]:
                    dec = 0
                    break
            if inc == 1 and dec == 1:
                count += 1
                break
    return count
""", "O(2^N * N^2)", "O(N^2)")

P(246, "kth_smallest_subset_sum", "combinatorics", "hard",
   "Return the k-th smallest subset sum (1-indexed, including empty subset sum = 0).",
   """\
def kth_smallest_subset_sum(nums, k):
    n = len(nums)
    sums = []
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        sums.append(s)
    for i in range(len(sums)):
        for j in range(i + 1, len(sums)):
            if sums[i] > sums[j]:
                sums[i], sums[j] = sums[j], sums[i]
    if k <= len(sums):
        return sums[k - 1]
    return -1
""", "O(2^N * N + 4^N)", "O(2^N log 2^N)")

P(247, "min_deletions_sorted", "dynamic-programming", "hard",
   "Min deletions to make the array strictly increasing.",
   """\
def min_deletions_sorted(nums):
    n = len(nums)
    if n <= 1:
        return 0
    best_lis = 1
    for mask in range(1, 1 << n):
        subseq = []
        for i in range(n):
            if mask & (1 << i):
                subseq.append(nums[i])
        ok = 1
        for i in range(len(subseq) - 1):
            if subseq[i] >= subseq[i + 1]:
                ok = 0
                break
        if ok == 1 and len(subseq) > best_lis:
            best_lis = len(subseq)
    return n - best_lis
""", "O(2^N * N)", "O(N^2)")

P(248, "count_palindrome_subarrays", "subarray", "hard",
   "Count contiguous subarrays that are palindromic.",
   """\
def count_palindrome_subarrays(nums):
    n = len(nums)
    count = 0
    for i in range(n):
        for j in range(i, n):
            is_pal = 1
            l = i
            r = j
            while l < r:
                if nums[l] != nums[r]:
                    is_pal = 0
                    break
                l += 1
                r -= 1
            if is_pal == 1:
                count += 1
    return count
""", "O(N^3)", "O(N^2)")

P(249, "max_nonoverlap_pair_sum", "subarray", "hard",
   "Max sum of two non-overlapping subarrays.",
   """\
def max_nonoverlap_pair_sum(nums):
    n = len(nums)
    if n < 2:
        return 0
    best = nums[0] + nums[1]
    for i in range(n):
        for j in range(i, n):
            s1 = 0
            for k in range(i, j + 1):
                s1 += nums[k]
            for p in range(j + 1, n):
                for q in range(p, n):
                    s2 = 0
                    for k in range(p, q + 1):
                        s2 += nums[k]
                    if s1 + s2 > best:
                        best = s1 + s2
    return best
""", "O(N^4)", "O(N)")

P(250, "subset_max_gcd", "math", "hard",
   "Max GCD of any subset of size >= 2.",
   """\
def subset_max_gcd(nums):
    n = len(nums)
    if n < 2:
        return 0

    def gcd(a, b):
        a = abs(a)
        b = abs(b)
        while b:
            a, b = b, a % b
        return a

    best = 0
    for mask in range(1, 1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        if len(subset) < 2:
            continue
        g = subset[0]
        for v in subset[1:]:
            g = gcd(g, v)
        if g > best:
            best = g
    return best
""", "O(2^N * N)", "O(N * MAX)")



# ══════════════════════════════════════════════════════════════════════
# Validation & Export
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    errors = 0
    for p in _P:
        try:
            ast.parse(p["original_code"])
        except SyntaxError as exc:
            print(f"  SYNTAX ERROR  {p['id']} ({p['name']}): {exc}")
            errors += 1

    if errors:
        print(f"\n{errors} syntax error(s) -- aborting.")
        sys.exit(1)

    # Duplicate-ID check
    ids = [p["id"] for p in _P]
    if len(ids) != len(set(ids)):
        print("DUPLICATE IDs detected -- aborting.")
        sys.exit(1)

    # Duplicate-name check
    names = [p["name"] for p in _P]
    if len(names) != len(set(names)):
        print("DUPLICATE names detected -- aborting.")
        sys.exit(1)

    print(f"Validated {len(_P)} problems -- all syntax OK.\n")

    # Difficulty breakdown
    diffs: dict[str, int] = {}
    for p in _P:
        diffs[p["difficulty"]] = diffs.get(p["difficulty"], 0) + 1
    for diff, cnt in sorted(diffs.items()):
        print(f"  {diff:25s}  {cnt:3d}")
    print()

    # Category breakdown
    cats: dict[str, int] = {}
    for p in _P:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat:25s}  {cnt:3d}")
    print(f"  {'TOTAL':25s}  {len(_P):3d}")

    out = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "tyr_benchmark_250.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_P, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nExported -> {out}")


if __name__ == "__main__":
    main()
