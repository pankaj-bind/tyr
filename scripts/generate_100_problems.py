#!/usr/bin/env python3
"""
generate_100_problems.py
========================
Append 100 NEW benchmark problems (TYR-151 through TYR-250) to the dataset.

Distribution:
    33  easy    (TYR-151 – TYR-183)
    17  medium  (TYR-184 – TYR-200)
    50  hard    (TYR-201 – TYR-250)

SMT constraints enforced:
    • Arrays max length 6, integers in [-20, 20]
    • Pure Python built-ins only (no imports)
    • original_code = worst brute-force
    • Clear Big-O notation

Run:
    python scripts/generate_100_problems.py
"""
import ast
import json
import sys
from pathlib import Path

DATASET = Path("dataset/tyr_benchmark_150.json")
_probs: list[dict] = []
_nid = 151


def _a(nm: str, cat: str, diff: str, desc: str, code: str, oc: str, tc: str):
    """Add a validated problem entry."""
    global _nid
    try:
        ast.parse(code)
    except SyntaxError as e:
        sys.exit(f"SYNTAX ERROR in TYR-{_nid:03d} ({nm}): {e}")
    _probs.append({
        "id": f"TYR-{_nid:03d}",
        "name": nm,
        "category": cat,
        "difficulty": diff,
        "description": desc,
        "original_code": code,
        "original_complexity": oc,
        "target_complexity": tc,
    })
    _nid += 1


# ══════════════════════════════════════════════════════════════════════
# EASY (33 problems): TYR-151 – TYR-183
# ══════════════════════════════════════════════════════════════════════

_a("count_equal_pairs", "pair-finding", "easy",
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

_a("find_single_element", "frequency", "easy",
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

_a("max_sorted_gap", "sorting", "easy",
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

_a("arrays_are_permutation", "set-operations", "easy",
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

_a("count_unique_elements", "frequency", "easy",
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

_a("first_non_repeat_index", "frequency", "easy",
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

_a("majority_vote", "frequency", "easy",
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

_a("max_prefix_sum", "prefix-sum", "easy",
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

_a("longest_constant_run", "subarray", "easy",
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

_a("second_minimum", "order-statistics", "easy",
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

_a("count_above_average", "array-transform", "easy",
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

_a("parity_sort_check", "sorting", "easy",
   "Return 1 if all even elements appear before all odd elements, else 0.",
   """\
def parity_sort_check(nums):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] % 2 != 0 and nums[j] % 2 == 0:
                return 0
    return 1
""", "O(N^2)", "O(N)")

_a("equilibrium_first", "prefix-sum", "easy",
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

_a("sum_of_leaders", "array-transform", "easy",
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

_a("min_pair_abs_diff", "pair-finding", "easy",
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

_a("max_right_minus_left", "subarray", "easy",
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

_a("count_interior_peaks", "array-transform", "easy",
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

_a("nearby_duplicate", "hash-map", "easy",
   "Return 1 if there exist i != j with nums[i]==nums[j] and abs(i-j) <= k.",
   """\
def nearby_duplicate(nums, k):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j] and j - i <= k:
                return 1
    return 0
""", "O(N^2)", "O(N)")

_a("longest_consecutive_len", "set-operations", "easy",
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

_a("count_with_strictly_greater", "order-statistics", "easy",
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

_a("top_frequency", "frequency", "easy",
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

_a("even_product_pairs", "pair-finding", "easy",
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

_a("total_prefix_sums", "prefix-sum", "easy",
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

_a("dominators_count", "array-transform", "easy",
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

_a("max_container_water", "two-pointer", "easy",
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

_a("smallest_absent_positive", "hash-map", "easy",
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

_a("all_pairs_xor_sum", "pair-finding", "easy",
   "Return sum of (nums[i] XOR nums[j]) for all pairs i<j.",
   """\
def all_pairs_xor_sum(nums):
    total = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            total += nums[i] ^ nums[j]
    return total
""", "O(N^2)", "O(N)")

_a("count_equal_triplets", "frequency", "easy",
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

_a("max_product_two", "pair-finding", "easy",
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

_a("sum_of_pair_maxes", "pair-finding", "easy",
   "Return sum of max(nums[i], nums[j]) for every pair i<j.",
   """\
def sum_of_pair_maxes(nums):
    total = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            total += nums[i] if nums[i] > nums[j] else nums[j]
    return total
""", "O(N^2)", "O(N log N)")

_a("count_surpassers_total", "pair-finding", "easy",
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

_a("is_sorted_rotation", "sorting", "easy",
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

_a("zero_sum_pair_count", "pair-finding", "easy",
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

_a("lis_length", "dynamic-programming", "medium",
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

_a("max_subarray_kadane", "subarray", "medium",
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

_a("subarrays_with_sum", "prefix-sum", "medium",
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

_a("lcs_of_arrays", "dynamic-programming", "medium",
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

_a("max_product_subarray", "subarray", "medium",
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

_a("count_zero_triplets", "pair-finding", "medium",
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

_a("can_partition_equal", "dynamic-programming", "medium",
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

_a("rob_houses_max", "dynamic-programming", "medium",
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

_a("subset_count_target", "dynamic-programming", "medium",
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

_a("climb_stairs_min_cost", "dynamic-programming", "medium",
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

_a("longest_palindrome_subseq_len", "dynamic-programming", "medium",
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

_a("max_sum_inc_subseq", "dynamic-programming", "medium",
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

_a("longest_wiggle_length", "dynamic-programming", "medium",
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

_a("min_jumps_to_end", "dynamic-programming", "medium",
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

_a("target_sum_ways", "dynamic-programming", "medium",
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

_a("decode_digit_ways", "dynamic-programming", "medium",
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

_a("max_chain_pairs", "greedy", "medium",
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

_a("min_partition_diff", "dynamic-programming", "hard",
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

_a("palindrome_min_cuts", "dynamic-programming", "hard",
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

_a("knapsack_max_value", "dynamic-programming", "hard",
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

_a("shortest_superseq_len", "dynamic-programming", "hard",
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

_a("max_sum_no_three_adjacent", "dynamic-programming", "hard",
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

_a("count_inc_subsequences", "dynamic-programming", "hard",
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

_a("longest_bitonic_subseq", "dynamic-programming", "hard",
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

_a("min_adjacent_swaps_sort", "sorting", "hard",
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

_a("count_subseq_target_sum", "dynamic-programming", "hard",
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

_a("burst_balloon_coins", "dynamic-programming", "hard",
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

_a("edit_distance_arrays", "dynamic-programming", "hard",
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

_a("interleave_possible", "dynamic-programming", "hard",
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

_a("max_rob_circular", "dynamic-programming", "hard",
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

_a("count_distinct_subseq", "dynamic-programming", "hard",
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

_a("optimal_game_score", "dynamic-programming", "hard",
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

_a("min_insertions_palindrome", "dynamic-programming", "hard",
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

_a("count_subsets_xor", "dynamic-programming", "hard",
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

_a("max_profit_cooldown", "dynamic-programming", "hard",
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

_a("rod_cutting_max", "dynamic-programming", "hard",
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

_a("count_unique_bst", "dynamic-programming", "hard",
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

_a("longest_arith_subseq_len", "dynamic-programming", "hard",
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

_a("coin_change_min", "dynamic-programming", "hard",
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

_a("coin_change_ways", "dynamic-programming", "hard",
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

_a("longest_common_subarray_len", "dynamic-programming", "hard",
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

_a("min_cost_grid_path", "dynamic-programming", "hard",
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

_a("count_grid_paths_obstacles", "dynamic-programming", "hard",
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

_a("max_earn_delete", "dynamic-programming", "hard",
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

_a("falling_path_min", "dynamic-programming", "hard",
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

_a("max_square_side", "dynamic-programming", "hard",
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

_a("count_square_submatrices", "dynamic-programming", "hard",
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

_a("can_jump_end", "dynamic-programming", "hard",
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

_a("max_product_cut", "dynamic-programming", "hard",
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

_a("paint_fence_ways", "dynamic-programming", "hard",
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

_a("count_derangements", "combinatorics", "hard",
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

_a("mcm_min_cost", "dynamic-programming", "hard",
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

_a("longest_repeating_subarray", "dynamic-programming", "hard",
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

_a("max_profit_two_tx", "dynamic-programming", "hard",
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

_a("egg_drop_two", "dynamic-programming", "hard",
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

_a("all_subsets_range_sum", "combinatorics", "hard",
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

_a("max_and_pair", "bitmasking", "hard",
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

_a("count_valid_triangles", "combinatorics", "hard",
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

_a("min_moves_equalize", "math", "hard",
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

_a("longest_zigzag_subseq", "dynamic-programming", "hard",
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

_a("max_alternating_sum", "dynamic-programming", "hard",
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

_a("count_bitonic_subseqs", "dynamic-programming", "hard",
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

_a("kth_smallest_subset_sum", "combinatorics", "hard",
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

_a("min_deletions_sorted", "dynamic-programming", "hard",
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

_a("count_palindrome_subarrays", "subarray", "hard",
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

_a("max_nonoverlap_pair_sum", "subarray", "hard",
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

_a("subset_max_gcd", "math", "hard",
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
# Merge & Save
# ══════════════════════════════════════════════════════════════════════

def main():
    # ── Validate ───────────────────────────────────────────────────
    easy = sum(1 for p in _probs if p["difficulty"] == "easy")
    med  = sum(1 for p in _probs if p["difficulty"] == "medium")
    hard = sum(1 for p in _probs if p["difficulty"] == "hard")
    print(f"Generated: {len(_probs)} problems  (easy={easy}, medium={med}, hard={hard})")
    assert len(_probs) == 100, f"Expected 100, got {len(_probs)}"
    assert easy == 33, f"Expected 33 easy, got {easy}"
    assert med == 17, f"Expected 17 medium, got {med}"
    assert hard == 50, f"Expected 50 hard, got {hard}"

    # Check unique IDs and names
    ids = [p["id"] for p in _probs]
    names = [p["name"] for p in _probs]
    assert len(set(ids)) == 100, "Duplicate IDs found"
    assert len(set(names)) == 100, "Duplicate names found"

    # ── Load existing ──────────────────────────────────────────────
    with open(DATASET, "r", encoding="utf-8") as f:
        existing = json.load(f)
    print(f"Existing dataset: {len(existing)} problems")

    # Check no ID collision
    existing_ids = {p["id"] for p in existing}
    for p in _probs:
        assert p["id"] not in existing_ids, f"ID collision: {p['id']}"

    # ── Merge & Write ──────────────────────────────────────────────
    merged = existing + _probs
    with open(DATASET, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"✔ Dataset updated: {len(merged)} total problems → {DATASET}")


if __name__ == "__main__":
    main()
