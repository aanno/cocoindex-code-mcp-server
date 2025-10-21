#!/usr/bin/env python3
"""
Test runner script for haskell-tree-sitter project.
Provides multiple ways to run tests with different levels of verbosity.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_unittest():
    """Run tests using unittest discovery."""
    print("Running tests with unittest...")
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "tests/", "-v"],
        cwd=Path(__file__).parent,
    )
    return result.returncode


def run_pytest():
    """Run tests using pytest (if available)."""
    print("Running tests with pytest...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v"], cwd=Path(__file__).parent
        )
        return result.returncode
    except FileNotFoundError:
        print("pytest not found. Install with: pip install pytest")
        return 1


def run_coverage():
    """Run tests with coverage (if available)."""
    print("Running tests with coverage...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "--cov=haskell_tree_sitter",
                "--cov-report=term-missing",
            ],
            cwd=Path(__file__).parent,
        )
        return result.returncode
    except FileNotFoundError:
        print("pytest-cov not found. Install with: pip install pytest-cov")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run tests for haskell-tree-sitter")
    parser.add_argument(
        "--runner",
        choices=["unittest", "pytest", "coverage"],
        default="unittest",
        help="Test runner to use",
    )

    args = parser.parse_args()

    if args.runner == "unittest":
        return run_unittest()
    elif args.runner == "pytest":
        return run_pytest()
    elif args.runner == "coverage":
        return run_coverage()


if __name__ == "__main__":
    sys.exit(main())
