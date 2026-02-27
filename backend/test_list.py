import sys
sys.path.insert(0, ".")
from z3_verifier import verify_equivalence

original = """
def find_duplicates(nums):
    result = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j] and nums[i] not in result:
                result.append(nums[i])
    return result
"""

optimized = """
def find_duplicates(nums):
    num_set = set()
    result = set()
    for num in nums:
        if num in num_set and num not in result:
            result.add(num)
        num_set.add(num)
    return list(result)
"""

r = verify_equivalence(original, optimized)
print("Status:", r["status"])
print("Message:", r["message"])
if r.get("counterexample"):
    print("Counterexample:", r["counterexample"])
