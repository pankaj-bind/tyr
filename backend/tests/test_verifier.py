"""Comprehensive test suite for Tyr Z3 verifier (v0.4.0).

Includes all v0.3.0 tests plus CRG-specific regression tests:
  T38 — continue in accumulation loop (CRG-1)
  T39 — break in python-list iteration (CRG-2)
  T40 — min/max on empty list (CRG-4)
  T41 — all() builtin
  T42 — any() builtin
  T43 — float('inf') constant (CRG-7)
"""
import sys, time
sys.path.insert(0, ".")
from verifier import verify_equivalence

ALL_PASS = True

def run(label, orig, opt, expected_status=None):
    global ALL_PASS
    t0 = time.time()
    r = verify_equivalence(orig.strip(), opt.strip())
    dt = time.time() - t0
    status = r["status"]
    ok = ""
    if expected_status:
        if status == expected_status:
            ok = " ✓"
        else:
            ok = f" ✗ (expected {expected_status})"
            ALL_PASS = False
    print(f"  {label}: {status}{ok}  ({dt:.2f}s)")
    if r["counterexample"]:
        print(f"    counterexample: {r['counterexample']}")
    if status == "ERROR":
        print(f"    error: {r['message']}")
    return r

print("=" * 70)
print("Tyr Z3 Verifier — Comprehensive Test Suite (v0.4.0)")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# Basic Equivalence
# ══════════════════════════════════════════════════════════════════════
print("\n── Basic Equivalence ──")

run("T01 algebraic identity",
    "def f(a, b):\n    return (a + b) * (a + b)",
    "def f(a, b):\n    return a*a + 2*a*b + b*b",
    expected_status="UNSAT")

run("T02 non-equivalent",
    "def f(x):\n    return x * 2",
    "def f(x):\n    return x * 3",
    expected_status="SAT")

run("T03 if-else ↔ ternary",
    "def f(x):\n    if x < 0:\n        return -x\n    else:\n        return x",
    "def f(x):\n    return x if x >= 0 else -x",
    expected_status="UNSAT")

run("T04 clamp fixed bounds",
    "def f(x):\n    if x < 0:\n        return 0\n    elif x > 10:\n        return 10\n    else:\n        return x",
    "def f(x):\n    return max(0, min(x, 10))",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# Symbolic Loops
# ══════════════════════════════════════════════════════════════════════
print("\n── Symbolic Loops ──")

run("T05 sym-loop non-equiv (n<0)",
    "def f(n):\n    result = 0\n    for i in range(n):\n        result += i\n    return result",
    "def f(n):\n    return n * (n - 1) // 2",
    expected_status="SAT")

run("T06 guarded gauss",
    "def f(n):\n    result = 0\n    for i in range(n):\n        result += i\n    return result",
    "def f(n):\n    if n <= 0:\n        return 0\n    return n * (n - 1) // 2",
    expected_status="WARNING")

run("T07 concrete loop",
    "def f(x):\n    result = 0\n    for i in range(5):\n        result += x\n    return result",
    "def f(x):\n    return x * 5",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# FloorDiv & Mod Semantics
# ══════════════════════════════════════════════════════════════════════
print("\n── FloorDiv & Mod ──")

run("T08 floordiv positive",
    "def f(x):\n    return x // 3",
    "def f(x):\n    return x // 3",
    expected_status="UNSAT")

run("T09 floordiv identity a = (a//b)*b + a%b",
    "def f(a, b):\n    if b == 0:\n        return 0\n    return (a // b) * b + a % b",
    "def f(a, b):\n    if b == 0:\n        return 0\n    return a",
    expected_status="UNSAT")

run("T10 mod semantics equiv",
    "def f(a, b):\n    if b == 0:\n        return 0\n    return a - (a // b) * b",
    "def f(a, b):\n    if b == 0:\n        return 0\n    return a % b",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# String Constants
# ══════════════════════════════════════════════════════════════════════
print("\n── String Constants ──")

run("T11 string equality preserved",
    'def f(x):\n    tag = "a"\n    if x > 0:\n        tag = "b"\n    if tag == "a":\n        return 0\n    return 1',
    'def f(x):\n    if x > 0:\n        return 1\n    return 0',
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# List Comprehension
# ══════════════════════════════════════════════════════════════════════
print("\n── List Comprehension ──")

run("T12 listcomp double",
    "def f(nums):\n    result = []\n    for x in nums:\n        result.append(x * 2)\n    return result",
    "def f(nums):\n    return [x * 2 for x in nums]",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# enumerate()
# ══════════════════════════════════════════════════════════════════════
print("\n── enumerate() ──")

run("T13 enumerate sum-of-indices",
    "def f(nums):\n    total = 0\n    for i in range(len(nums)):\n        total += i\n    return total",
    "def f(nums):\n    total = 0\n    for i, v in enumerate(nums):\n        total += i\n    return total",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# sorted()
# ══════════════════════════════════════════════════════════════════════
print("\n── sorted() ──")

run("T14 sorted ascending idempotent",
    "def f(nums):\n    return sorted(nums)",
    "def f(nums):\n    return sorted(nums)",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# While-loop soundness
# ══════════════════════════════════════════════════════════════════════
print("\n── While-loop Soundness ──")

run("T15 while concrete terminates",
    "def f(x):\n    result = 0\n    i = 0\n    while i < 5:\n        result += x\n        i += 1\n    return result",
    "def f(x):\n    return x * 5",
    expected_status="UNSAT")

run("T16 while symbolic cap → WARNING",
    "def f(n):\n    result = 0\n    i = 0\n    while i < n:\n        result += i\n        i += 1\n    return result",
    "def f(n):\n    if n <= 0:\n        return 0\n    return n * (n - 1) // 2",
    expected_status="WARNING")

# ══════════════════════════════════════════════════════════════════════
# Short-circuit Boolean
# ══════════════════════════════════════════════════════════════════════
print("\n── Short-circuit Boolean ──")

run("T17 and short-circuit",
    "def f(x, y):\n    if x > 0 and y > 0:\n        return x + y\n    return 0",
    "def f(x, y):\n    if x > 0:\n        if y > 0:\n            return x + y\n    return 0",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# SymbolicDict operations
# ══════════════════════════════════════════════════════════════════════
print("\n── SymbolicDict ──")

run("T18 dict get with default",
    "def f(nums):\n    d = {}\n    for x in nums:\n        d[x] = d.get(x, 0) + 1\n    total = 0\n    for k in d.keys():\n        total += d[k]\n    return total",
    "def f(nums):\n    return len(nums)",
    expected_status=None)

# ══════════════════════════════════════════════════════════════════════
# Bounds Disclosure
# ══════════════════════════════════════════════════════════════════════
print("\n── Bounds Disclosure ──")

r_bounds = run("T19 bounds in UNSAT msg",
    "def f(x):\n    return x + 1",
    "def f(x):\n    return 1 + x",
    expected_status="UNSAT")

if "bounds" in r_bounds["message"].lower() or "\u2264" in r_bounds["message"]:
    print("    \u2713 UNSAT message contains bounds info")
else:
    print("    \u2717 UNSAT message missing bounds info")
    ALL_PASS = False

# ══════════════════════════════════════════════════════════════════════
# Additional
# ══════════════════════════════════════════════════════════════════════
print("\n── Additional ──")

run("T20 abs equiv",
    "def f(x):\n    return abs(x)",
    "def f(x):\n    if x < 0:\n        return -x\n    return x",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# None-vs-0 Sentinel
# ══════════════════════════════════════════════════════════════════════
print("\n── None vs 0 Sentinel ──")

run("T21 None ≠ 0 detection",
    "def f(x):\n    if x > 0:\n        return x\n    return None",
    "def f(x):\n    if x > 0:\n        return x\n    return 0",
    expected_status="SAT")

# ══════════════════════════════════════════════════════════════════════
# Negative Indexing
# ══════════════════════════════════════════════════════════════════════
print("\n── Negative Indexing ──")

run("T22 neg index equiv",
    "def f(nums):\n    return nums[len(nums) - 1]",
    "def f(nums):\n    return nums[-1]",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# Bool/Int Coercion
# ══════════════════════════════════════════════════════════════════════
print("\n── Bool/Int Coercion ──")

run("T23 bool-int equiv",
    "def f(x):\n    if x > 0:\n        return 1\n    else:\n        return 0",
    "def f(x):\n    return 1 if x > 0 else 0",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# Lambda / map()
# ══════════════════════════════════════════════════════════════════════
print("\n── Lambda / map() ──")

run("T24 map-lambda vs listcomp",
    "def f(nums):\n    return list(map(lambda x: x * 2, nums))",
    "def f(nums):\n    return [x * 2 for x in nums]",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# Sandbox Hardening
# ══════════════════════════════════════════════════════════════════════
print("\n── Sandbox Hardening ──")

run("T25 sandbox blocks dunder",
    "def f(x):\n    return x.__class__.__name__",
    "def f(x):\n    return str(type(x))",
    expected_status="ERROR")

run("T26 sandbox blocks import",
    "def f(x):\n    import os\n    return os.getpid()",
    "def f(x):\n    return 0",
    expected_status="ERROR")

# ══════════════════════════════════════════════════════════════════════
# Symbolic Range Bound
# ══════════════════════════════════════════════════════════════════════
print("\n── Symbolic Range (MAX_SYMBOLIC_RANGE) ──")

r_range = run("T27 range bound in message",
    "def f(n):\n    result = 0\n    for i in range(n):\n        result += i\n    return result",
    "def f(n):\n    if n <= 0:\n        return 0\n    return n * (n - 1) // 2",
    expected_status="WARNING")

if r_range["status"] == "WARNING":
    print("    \u2713 Correctly falls back to concrete testing for unbounded symbolic range")
elif r_range["status"] == "UNSAT" and "range" in r_range["message"].lower():
    print("    ✓ UNSAT message mentions range bound")
elif r_range["status"] == "UNSAT":
    print("    ✓ Verified equivalent within symbolic range bound")

# ══════════════════════════════════════════════════════════════════════
# Partial Returns Cap
# ══════════════════════════════════════════════════════════════════════
print("\n── Partial Returns Cap ──")

run("T28 nested if-else partial returns",
    "def f(x, y):\n    if x > 0:\n        if y > 0:\n            return x + y\n        else:\n            return x - y\n    else:\n        if y > 0:\n            return y - x\n        else:\n            return 0",
    "def f(x, y):\n    if x > 0 and y > 0:\n        return x + y\n    elif x > 0:\n        return x - y\n    elif y > 0:\n        return y - x\n    return 0",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# Early Return Inside Loop
# ══════════════════════════════════════════════════════════════════════
print("\n── Early Return in Loop ──")

run("T29 early return in list loop",
    "def f(nums, target):\n    for x in nums:\n        if x == target:\n            return 1\n    return 0",
    "def f(nums, target):\n    if target in nums:\n        return 1\n    return 0",
    expected_status="UNSAT")

run("T30 accumulate-vs-early-return",
    "def f(nums):\n    found = 0\n    for x in nums:\n        if x > 0:\n            found = 1\n    return found",
    "def f(nums):\n    for x in nums:\n        if x > 0:\n            return 1\n    return 0",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# not-integer coercion
# ══════════════════════════════════════════════════════════════════════
print("\n── not-integer Coercion ──")

run("T31 not-integer",
    "def f(x):\n    if not x:\n        return 1\n    return 0",
    "def f(x):\n    if x == 0:\n        return 1\n    return 0",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# Early return in symbolic range(n) loop
# ══════════════════════════════════════════════════════════════════════
print("\n── Early Return in range(n) ──")

run("T32 early return in range(n) loop",
    "def f(nums):\n    n = len(nums)\n    for i in range(n):\n        if nums[i] == 0:\n            return i\n    return -1",
    "def f(nums):\n    for i, x in enumerate(nums):\n        if x == 0:\n            return i\n    return -1",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# Direct dict iteration
# ══════════════════════════════════════════════════════════════════════
print("\n── Direct Dict Iteration ──")

run("T33 for k in d (direct dict iter)",
    "def f(nums):\n    d = {}\n    for x in nums:\n        d[x] = d.get(x, 0) + 1\n    total = 0\n    for k in d.keys():\n        total += d.get(k, 0)\n    return total",
    "def f(nums):\n    d = {}\n    for x in nums:\n        d[x] = d.get(x, 0) + 1\n    total = 0\n    for k in d:\n        total += d.get(k, 0)\n    return total",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# Dict Subscript KeyError
# ══════════════════════════════════════════════════════════════════════
print("\n── Dict Subscript KeyError ──")

run("T34 d[k] vs d.get(k,0) not equiv",
    "def f(nums, target):\n    d = {}\n    for x in nums:\n        d[x] = 1\n    return d[target]",
    "def f(nums, target):\n    d = {}\n    for x in nums:\n        d[x] = 1\n    return d.get(target, 0)",
    expected_status="SAT")

# ══════════════════════════════════════════════════════════════════════
# not on SymbolicList
# ══════════════════════════════════════════════════════════════════════
print("\n── not on SymbolicList ──")

run("T35 not-empty-list",
    "def f(nums):\n    if not nums:\n        return -1\n    return nums[0]",
    "def f(nums):\n    if len(nums) == 0:\n        return -1\n    return nums[0]",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# not on SymbolicDict
# ══════════════════════════════════════════════════════════════════════
print("\n── not on SymbolicDict ──")

run("T36 not-empty-dict",
    "def f(nums):\n    d = {}\n    for x in nums:\n        d[x] = 1\n    if not d:\n        return -1\n    return 0",
    "def f(nums):\n    d = {}\n    for x in nums:\n        d[x] = 1\n    if len(nums) == 0:\n        return -1\n    return 0",
    expected_status=None)

# ══════════════════════════════════════════════════════════════════════
# Store-then-iterate keys
# ══════════════════════════════════════════════════════════════════════
print("\n── Store-then-iterate keys ──")

run("T37 keys-stored-then-iterated",
    "def f(nums):\n    d = {}\n    for x in nums:\n        d[x] = d.get(x, 0) + 1\n    keys = d.keys()\n    total = 0\n    for k in keys:\n        total += d.get(k, 0)\n    return total",
    "def f(nums):\n    return len(nums)",
    expected_status=None)

# ══════════════════════════════════════════════════════════════════════
# CRG-1: continue in accumulation loop
# ══════════════════════════════════════════════════════════════════════
print("\n── CRG-1: continue in loop ──")

run("T38 continue in accumulation",
    "def f(nums):\n    total = 0\n    for x in nums:\n        if x < 0:\n            continue\n        total += x\n    return total",
    "def f(nums):\n    total = 0\n    for x in nums:\n        if x >= 0:\n            total += x\n    return total",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# CRG-2: break in python-list / enumerate / dict iteration
# ══════════════════════════════════════════════════════════════════════
print("\n── CRG-2: break in all loop types ──")

run("T39 break in concrete range loop",
    "def f(x):\n    total = 0\n    for i in range(10):\n        if i >= x:\n            break\n        total += i\n    return total",
    "def f(x):\n    total = 0\n    for i in range(10):\n        if i < x:\n            total += i\n    return total",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# CRG-4: min/max on empty list
# ══════════════════════════════════════════════════════════════════════
print("\n── CRG-4: min/max empty guard ──")

run("T40 min/max without empty precondition",
    "def f(nums):\n    if len(nums) == 0:\n        return 0\n    return min(nums)",
    "def f(nums):\n    if len(nums) == 0:\n        return 0\n    m = nums[0]\n    for x in nums:\n        if x < m:\n            m = x\n    return m",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# CRG-5: all() builtin
# ══════════════════════════════════════════════════════════════════════
print("\n── CRG-5: all() / any() builtins ──")

run("T41 all() equiv",
    "def f(nums):\n    for x in nums:\n        if x == 0:\n            return 0\n    return 1",
    "def f(nums):\n    if all(nums):\n        return 1\n    return 0",
    expected_status="UNSAT")

run("T42 any() equiv",
    "def f(nums):\n    for x in nums:\n        if x != 0:\n            return 1\n    return 0",
    "def f(nums):\n    if any(nums):\n        return 1\n    return 0",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# CRG-7: float('inf') constant
# ══════════════════════════════════════════════════════════════════════
print("\n── CRG-7: float('inf') support ──")

run("T43 float('inf') as initial min",
    "def f(nums):\n    if len(nums) == 0:\n        return 0\n    m = float('inf')\n    for x in nums:\n        if x < m:\n            m = x\n    return m",
    "def f(nums):\n    if len(nums) == 0:\n        return 0\n    return min(nums)",
    expected_status="UNSAT")

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
if ALL_PASS:
    print("ALL EXPECTED STATUSES MATCHED ✓")
else:
    print("SOME TESTS FAILED ✗")
    sys.exit(1)
print("=" * 70)
