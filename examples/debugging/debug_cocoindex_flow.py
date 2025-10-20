#!/usr/bin/env python3
"""
Example script for debugging CocoIndex flow operations.
Combines useful parts of debug_cocoindex_chunks.py, test_minimal_flow.py, trace_content_loss.py.

This script demonstrates how to:
1. Test CocoIndex flow with a single file
2. Trace content through the pipeline
3. Debug chunk operations and transformations
"""

import os
import sys
import tempfile
from pathlib import Path

from cocoindex_code_mcp_server.ast_chunking import (
    CocoIndexASTChunker,
    ensure_unique_chunk_locations,
)
from cocoindex_code_mcp_server.cocoindex_config import update_flow_config

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_single_file_flow():
    """Test CocoIndex flow with a single file."""
    print("🔍 Testing Single File CocoIndex Flow")
    print("=" * 50)

    # Create test code
    test_code = '''
def fibonacci(n):
    """Calculate fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    """Calculate factorial."""
    if n <= 1:
        return 1
    return n * factorial(n-1)

class Calculator:
    """A simple calculator class."""

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

def main():
    calc = Calculator()
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"4 * 6 = {calc.multiply(4, 6)}")
    print(f"Fibonacci(10) = {fibonacci(10)}")
    print(f"Factorial(5) = {factorial(5)}")

if __name__ == "__main__":
    main()
'''

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_code)
        temp_file = f.name

    try:
        # Update flow config for single file
        print(f"Processing file: {temp_file}")
        update_flow_config(temp_file)

        # Run the flow
        print("Running CocoIndex flow...")

        # Here you would normally run the flow
        # This is a placeholder for the actual flow execution
        print("✅ Flow configuration updated successfully")
        print("Note: Actual flow execution would happen here")

    finally:
        # Cleanup
        os.unlink(temp_file)

    print()


def test_ast_chunking():
    """Test AST chunking operations."""
    print("🔍 Testing AST Chunking")
    print("=" * 30)

    test_code = '''
import os
import sys

def function_one():
    """First function."""
    return "one"

def function_two():
    """Second function."""
    return "two"

class MyClass:
    """Example class."""

    def __init__(self):
        self.value = 0

    def method_one(self):
        return self.value + 1

    def method_two(self):
        return self.value * 2
'''

    # Test chunking
    chunker = CocoIndexASTChunker()
    chunks = chunker.chunk_code(test_code, language="python", filename="test.py")

    print(f"Generated {len(chunks)} chunks:")

    total_content_length = 0
    for i, chunk in enumerate(chunks):
        content_preview = chunk.content[:80].replace("\n", "\\n")
        print(f"  {i + 1}. Length: {len(chunk.content)}, Preview: {content_preview}")
        total_content_length += len(chunk.content)

        # Show metadata if available
        if hasattr(chunk, "metadata") and chunk.metadata:
            relevant_metadata = {
                k: v for k, v in chunk.metadata.items() if k in ["functions", "classes", "imports", "analysis_method"]
            }
            if relevant_metadata:
                print(f"      Metadata: {relevant_metadata}")

    print(f"\nOriginal code length: {len(test_code)}")
    print(f"Total chunk content length: {total_content_length}")
    print(f"Coverage: {total_content_length / len(test_code):.1%}")

    # Test ensure_unique_chunk_locations
    print("\nTesting unique chunk locations...")
    try:
        unique_chunks = ensure_unique_chunk_locations(chunks)
        print(f"✅ Processed {len(chunks)} chunks -> {len(unique_chunks)} unique chunks")
    except Exception as e:
        print(f"❌ Error in ensure_unique_chunk_locations: {e}")

    print()


def trace_content_preservation():
    """Trace how content is preserved through the pipeline."""
    print("🔍 Tracing Content Preservation")
    print("=" * 35)

    original_code = """
def hello_world():
    print("Hello, World!")
    return "success"

def goodbye_world():
    print("Goodbye, World!")
    return "farewell"
"""

    print(f"Original code ({len(original_code)} chars):")
    print(original_code)
    print("-" * 40)

    # Step 1: AST Chunking
    chunker = CocoIndexASTChunker()
    chunks = chunker.chunk_code(original_code, language="python", filename="trace.py")

    print(f"After AST chunking: {len(chunks)} chunks")
    reconstructed = ""
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i + 1}: {len(chunk.content)} chars")
        reconstructed += chunk.content

    print(f"Reconstructed length: {len(reconstructed)} chars")

    # Check preservation
    original_functions = original_code.count("def ")
    reconstructed_functions = reconstructed.count("def ")

    print(f"Functions preserved: {reconstructed_functions}/{original_functions}")

    if "hello_world" in reconstructed and "goodbye_world" in reconstructed:
        print("✅ Key functions preserved")
    else:
        print("❌ Key functions lost")

    print()


def test_metadata_extraction():
    """Test metadata extraction from chunks."""
    print("🔍 Testing Metadata Extraction")
    print("=" * 35)

    test_cases = [
        (
            "Python",
            "python",
            """
def calculate(a, b):
    return a + b

class Math:
    def power(self, x, y):
        return x ** y
""",
        ),
        (
            "Kotlin",
            "kotlin",
            """
data class Person(val name: String, val age: Int)

fun greet(person: Person): String {
    return "Hello, ${person.name}!"
}
""",
        ),
        (
            "Haskell",
            "haskell",
            """
fibonacci :: Int -> Int
fibonacci n = if n <= 1 then n else fibonacci (n-1) + fibonacci (n-2)
""",
        ),
    ]

    for name, language, code in test_cases:
        print(f"{name} code:")

        chunker = CocoIndexASTChunker()
        chunks = chunker.chunk_code(code, language=language, filename=f"test.{language}")

        print(f"  Chunks generated: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            if hasattr(chunk, "metadata") and chunk.metadata:
                # Show relevant metadata
                metadata = chunk.metadata
                functions = metadata.get("functions", [])
                classes = metadata.get("classes", [])
                analysis_method = metadata.get("analysis_method", "unknown")

                if functions or classes or analysis_method != "unknown":
                    print(f"    Chunk {i + 1} metadata:")
                    if functions:
                        print(f"      Functions: {functions}")
                    if classes:
                        print(f"      Classes: {classes}")
                    print(f"      Analysis: {analysis_method}")

        print()


def main():
    """Run all debugging tests."""
    try:
        test_single_file_flow()
        test_ast_chunking()
        trace_content_preservation()
        test_metadata_extraction()

        print("🎯 Summary:")
        print("- Single file flow configuration should work")
        print("- AST chunking should preserve content and generate metadata")
        print("- Content preservation can be tracked through the pipeline")
        print("- Different languages should produce appropriate metadata")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
