"""
demo/buggy_calculator.py
========================
A calculator with 3 intentional bugs that Kartavya should fix.

Run it first (it will fail) then use:
  python cli/main.py run "fix the bugs in demo/buggy_calculator.py" \
      --provider ollama --model qwen2.5-coder:7b
"""


def add(a, b):
    """Correct: returns sum."""
    total = a + b
    return total


def subtract(a, b):
    """Bug 1: returns sum instead of difference."""
    return a - b


def multiply(a, b):
    """Correct: returns product."""
    return a * b


def divide(a, b):
    """Bug 2: crashes on zero divisor — no guard."""
    if b == 0:
        return None
    return a / b


def factorial(n):
    """Bug 3: base case wrong, makes all results 0."""
    if n == 0:
        return 1
    return n * factorial(n - 1)


def power(base, exp):
    """Correct: returns base**exp."""
    return base ** exp


if __name__ == "__main__":
    print("=" * 50)
    print("buggy_calculator.py self-test")
    print("=" * 50)

    r = add(2, 3)
    assert r == 5, "add(2,3): expected 5, got " + str(r)
    print("  [PASS] add(2, 3) =", r)

    r = subtract(5, 3)
    assert r == 2, "subtract(5,3): expected 2, got " + str(r)
    print("  [PASS] subtract(5, 3) =", r)

    r = multiply(4, 3)
    assert r == 12, "multiply(4,3): expected 12, got " + str(r)
    print("  [PASS] multiply(4, 3) =", r)

    r = divide(10, 2)
    assert r == 5.0, "divide(10,2): expected 5.0, got " + str(r)
    print("  [PASS] divide(10, 2) =", r)

    try:
        r = divide(10, 0)
        assert r is None, "divide(10,0): expected None, got " + str(r)
        print("  [PASS] divide(10, 0) =", r)
    except ZeroDivisionError:
        raise AssertionError("divide(10,0) raised ZeroDivisionError — add: if b == 0: return None")

    r = factorial(5)
    assert r == 120, "factorial(5): expected 120, got " + str(r)
    print("  [PASS] factorial(5) =", r)

    r = factorial(0)
    assert r == 1, "factorial(0): expected 1, got " + str(r)
    print("  [PASS] factorial(0) =", r)

    r = power(2, 8)
    assert r == 256, "power(2,8): expected 256, got " + str(r)
    print("  [PASS] power(2, 8) =", r)

    print("=" * 50)
    print("All tests PASSED!")
    print("=" * 50)
