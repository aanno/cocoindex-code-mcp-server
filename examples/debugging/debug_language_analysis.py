#!/usr/bin/env python3
"""
Example script for debugging language analysis issues.
Combines useful parts of debug_kotlin_test.py, debug_haskell_*.py scripts.

This script demonstrates how to:
1. Test Kotlin and Haskell analysis directly
2. Debug chunk extraction issues
3. Compare AST vs fallback parsing methods
"""

import sys
from pathlib import Path

import cocoindex_code_mcp_server._haskell_tree_sitter as hts
from cocoindex_code_mcp_server.language_handlers.haskell_visitor import (
    analyze_haskell_code,
)
from cocoindex_code_mcp_server.language_handlers.kotlin_visitor import (
    analyze_kotlin_code,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_kotlin_analysis():
    """Test Kotlin analysis with sample code."""
    print("🔍 Testing Kotlin Analysis")
    print("=" * 50)

    kotlin_code = """
data class Person(val name: String, val age: Int) {
    fun isAdult(): Boolean = age >= 18
    fun greet(): String = "Hello, I'm $name and I'm $age years old"
}

fun fibonacci(n: Int): Int {
    return when (n) {
        0 -> 0
        1 -> 1
        else -> fibonacci(n - 1) + fibonacci(n - 2)
    }
}

fun main() {
    val person = Person("Alice", 25)
    println(person.greet())
    println("Fibonacci 10: ${fibonacci(10)}")
}
"""

    result = analyze_kotlin_code(kotlin_code, "example.kt")

    print(f"Success: {result.get('success', False)}")
    print(f"Analysis Method: {result.get('analysis_method', 'unknown')}")
    print(f"Functions: {result.get('functions', [])}")
    print(f"Classes: {result.get('classes', [])}")
    print(f"Data Classes: {result.get('data_classes', [])}")

    if "error" in result:
        print(f"Error: {result['error']}")

    print()


def test_haskell_analysis():
    """Test Haskell analysis with sample code."""
    print("🔍 Testing Haskell Analysis")
    print("=" * 50)

    # Test simple code that should work
    simple_haskell = """
fibonacci :: Int -> Int
fibonacci n
    | n <= 1    = n
    | otherwise = fibonacci (n - 1) + fibonacci (n - 2)

add :: Int -> Int -> Int
add x y = x + y
"""

    print("1. Simple Haskell code:")

    # Test chunk extraction
    chunks = hts.get_haskell_ast_chunks(simple_haskell)
    print(f"   AST chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"   {i + 1}. {chunk.node_type()}")

    # Test full analysis
    result = analyze_haskell_code(simple_haskell, "simple.hs")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Analysis Method: {result.get('analysis_method', 'unknown')}")
    print(f"   Functions: {result.get('functions', [])}")
    print(f"   Imports: {result.get('imports', [])}")

    print()

    # Test complex code (demonstrates the issue)
    print("2. Complex Haskell code (demonstrates parsing issue):")

    complex_haskell = """
module Example where

import Data.List

data Person = Person
    { personName :: String
    , personAge  :: Int
    } deriving (Show, Eq)

data Tree a = Leaf a | Branch (Tree a) (Tree a)
    deriving (Show, Eq)

fibonacci :: Int -> Int
fibonacci n
    | n <= 1    = n
    | otherwise = fibonacci (n - 1) + fibonacci (n - 2)

sumList :: [Int] -> Int
sumList []     = 0
sumList (x:xs) = x + sumList xs

treeMap :: (a -> b) -> Tree a -> Tree b
treeMap f (Leaf x)     = Leaf (f x)
treeMap f (Branch l r) = Branch (treeMap f l) (treeMap f r)
"""

    # Compare AST vs fallback parsing
    ast_chunks = hts.get_haskell_ast_chunks(complex_haskell)
    fallback_chunks = hts.get_haskell_ast_chunks_with_fallback(complex_haskell)

    print(f"   AST chunks: {len(ast_chunks)}")
    print(f"   Fallback chunks: {len(fallback_chunks)}")

    if len(fallback_chunks) > 0:
        chunk_types = [chunk.node_type() for chunk in fallback_chunks[:5]]
        print(f"   Fallback chunk types (first 5): {chunk_types}")

    # Test full analysis
    result = analyze_haskell_code(complex_haskell, "complex.hs")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Analysis Method: {result.get('analysis_method', 'unknown')}")
    print(f"   Functions: {result.get('functions', [])}")
    print(f"   Imports: {result.get('imports', [])}")

    print()


def compare_haskell_parsing_methods():
    """Compare different Haskell parsing methods."""
    print("🔍 Comparing Haskell Parsing Methods")
    print("=" * 50)

    test_code = """
factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)
"""

    print(f"Test code:\n{test_code}")
    print()

    methods = [
        ("get_haskell_ast_chunks", hts.get_haskell_ast_chunks),
        ("get_haskell_ast_chunks_with_fallback", hts.get_haskell_ast_chunks_with_fallback),
    ]

    for name, method in methods:
        try:
            chunks = method(test_code)
            print(f"{name}:")
            print(f"  Chunks: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                chunk_type = chunk.node_type()
                print(f"  {i + 1}. {chunk_type}")
                if hasattr(chunk, "code"):
                    code_sample = chunk.code()[:50].replace("\n", "\\n")
                    print(f"     Sample: {code_sample}")
            print()
        except Exception as e:
            print(f"{name}: Error - {e}")
            print()


def main():
    """Run all debugging tests."""
    try:
        test_kotlin_analysis()
        test_haskell_analysis()
        compare_haskell_parsing_methods()

        print("🎯 Summary:")
        print("- Kotlin analysis should show detected functions and classes")
        print("- Simple Haskell code should work with AST chunks")
        print("- Complex Haskell code demonstrates the AST parsing issue")
        print("- Fallback parsing produces 'regex_chunk' types that handler can't process")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all language handlers are available.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
