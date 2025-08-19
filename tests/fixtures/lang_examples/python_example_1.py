#!/usr/bin/env python3
"""
Test Python file for GraphCodeBERT embedding verification.
This should use microsoft/graphcodebert-base model.
"""


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


class MathUtils:
    """Utility class for mathematical operations."""

    @staticmethod
    def is_prime(num: int) -> bool:
        """Check if a number is prime."""
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True


if __name__ == "__main__":
    print(f"Fibonacci(10): {fibonacci(10)}")
    print(f"Is 17 prime? {MathUtils.is_prime(17)}")
