#!/usr/bin/env python3
"""
Tyr — Benchmark Dataset Builder
=================================
Constructs ``tyr_benchmark_150.json`` containing 150 hand-curated,
syntactically-validated O(N²) brute-force algorithmic problems spanning
11 distinct categories.  Every problem is amenable to a well-known
O(N) or O(N log N) optimisation that an LLM should produce.

Usage:
    python build_dataset.py          # writes tyr_benchmark_150.json

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
        print(f"\n{errors} syntax error(s) — aborting.")
        sys.exit(1)

    # Duplicate-ID check
    ids = [p["id"] for p in _P]
    if len(ids) != len(set(ids)):
        print("DUPLICATE IDs detected — aborting.")
        sys.exit(1)

    print(f"Validated {len(_P)} problems — all syntax OK.\n")

    # Category breakdown
    cats: dict[str, int] = {}
    for p in _P:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat:25s}  {cnt:3d}")
    print(f"  {'TOTAL':25s}  {len(_P):3d}")

    out = Path(__file__).resolve().parent / "tyr_benchmark_150.json"
    out.write_text(json.dumps(_P, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nExported → {out}")


if __name__ == "__main__":
    main()
