#!/usr/bin/env python3
"""Python example with 1-2 syntax errors for testing error recovery"""

# Valid function


def add(x, y):
    """Add two numbers"""
    return x + y

# Valid class


class Calculator:
    """Simple calculator class"""

    def __init__(self):
        self.history = []

    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

# Error 1: Invalid indentation


def broken_function():
    x = 5
  y = 10  # Error: inconsistent indentation
    return x + y

# Valid function
def subtract(a, b):
    return a - b

# Error 2: Invalid syntax in dictionary literal
config = {
    "debug": True,
    "timeout": 30,
    "retries": 3
    "max_size": 100  # Error: missing comma before this line
}

# Valid function continues
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
