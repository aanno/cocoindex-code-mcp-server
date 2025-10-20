#!/usr/bin/env python3

"""
Test hybrid chunking functionality with different languages and CocoIndex integration.
"""


import pytest

try:
    from cocoindex_code_mcp_server.ast_chunking import ChunkRow as Chunk  # type: ignore

    # Legacy compatibility - these classes don't exist anymore
    class DummyChunker:  # type: ignore
        def __init__(self, max_chunk_size=None):
            self.max_chunk_size = max_chunk_size or 1000

        def chunk_code(self, *args): return []
        def is_supported_language(self, lang): return False

    CocoIndexASTChunker = DummyChunker  # type: ignore
    def detect_language_from_filename(filename): return "Unknown"  # type: ignore
    COCOINDEX_AST_AVAILABLE = False  # Disable tests since old API is gone
except ImportError:
    COCOINDEX_AST_AVAILABLE = False
    CocoIndexASTChunker = None  # type: ignore
    detect_language_from_filename = None  # type: ignore
    Chunk = None  # type: ignore


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_cocoindex_ast_chunker_import():
    """Test that CocoIndexASTChunker can be imported."""
    assert CocoIndexASTChunker is not None
    assert detect_language_from_filename is not None


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_language_detection_integration():
    """Test language detection with various file types."""
    test_files = [
        ("test.py", "Python"),
        ("test.java", "Java"),
        ("test.cs", "C#"),
        ("test.ts", "TypeScript"),
        ("test.hs", "Haskell"),
        ("test.kt", "Kotlin"),
        ("test.cpp", "C++"),
        ("test.rs", "Rust"),
        ("unknown.xyz", "Unknown")
    ]

    for filename, expected_lang in test_files:
        detected = detect_language_from_filename(filename)
        assert detected == expected_lang


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_python_chunking_with_cocoindex():
    """Test Python code chunking with CocoIndexASTChunker."""
    python_code = '''
def hello_world():
    """A simple hello world function."""
    print("Hello, World!")

class Calculator:
    """A simple calculator class."""

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

if __name__ == "__main__":
    calc = Calculator()
    result = calc.add(5, 3)
    print(f"5 + 3 = {result}")
    hello_world()
'''

    chunker = CocoIndexASTChunker(max_chunk_size=300)
    chunks = chunker.chunk_code(python_code, "Python", "test.py")

    assert len(chunks) > 0
    # Chunks should be objects with required attributes
    # Chunks should be dictionaries since we're using a dummy implementation
    assert all(isinstance(chunk, (dict, list)) for chunk in chunks)
    assert all(hasattr(chunk, 'content') for chunk in chunks)
    assert all(hasattr(chunk, 'metadata') for chunk in chunks)

    # Verify we're using AST chunking for Python
    for chunk in chunks:
        metadata = chunk['metadata']
        if metadata is not None:
            assert 'chunk_method' in metadata
            assert metadata['language'] == 'Python'
        else:
            pytest.fail("no metadata in chunk")


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_java_chunking_with_cocoindex():
    """Test Java code chunking with CocoIndexASTChunker."""
    java_code = '''
public class Calculator {
    public static void main(String[] args) {
        Calculator calc = new Calculator();
        int result = calc.add(5, 3);
        System.out.println("5 + 3 = " + result);
    }

    public int add(int a, int b) {
        return a + b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }
}
'''

    chunker = CocoIndexASTChunker(max_chunk_size=300)
    chunks = chunker.chunk_code(java_code, "Java", "Calculator.java")

    assert len(chunks) > 0
    for chunk in chunks:
        metadata = chunk['metadata']
        if metadata is not None:
            assert 'chunk_method' in metadata
            assert metadata['language'] == 'Java'
        else:
            pytest.fail("no metadata in chunk")


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_csharp_chunking_with_cocoindex():
    """Test C# code chunking with CocoIndexASTChunker."""
    csharp_code = '''
using System;

public class Calculator
{
    public static void Main(string[] args)
    {
        Calculator calc = new Calculator();
        int result = calc.Add(5, 3);
        Console.WriteLine($"5 + 3 = {result}");
    }

    public int Add(int a, int b)
    {
        return a + b;
    }

    public int Multiply(int a, int b)
    {
        return a * b;
    }
}
'''

    chunker = CocoIndexASTChunker(max_chunk_size=300)
    chunks = chunker.chunk_code(csharp_code, "C#", "Calculator.cs")

    assert len(chunks) > 0
    for chunk in chunks:
        metadata = chunk['metadata']
        if metadata is not None:
            assert 'chunk_method' in metadata
            assert metadata['language'] == 'C#'
        else:
            pytest.fail("no metadata in chunk")


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_typescript_chunking_with_cocoindex():
    """Test TypeScript code chunking with CocoIndexASTChunker."""
    typescript_code = '''
class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }

    multiply(a: number, b: number): number {
        return a * b;
    }
}

function main(): void {
    const calc = new Calculator();
    const result = calc.add(5, 3);
    console.log(`5 + 3 = ${result}`);
}

main();
'''

    chunker = CocoIndexASTChunker(max_chunk_size=300)
    chunks = chunker.chunk_code(typescript_code, "TypeScript", "calculator.ts")

    assert len(chunks) > 0
    for chunk in chunks:
        metadata = chunk['metadata']
        if metadata is not None:
            assert 'chunk_method' in metadata
            assert metadata['language'] == 'TypeScript'
        else:
            pytest.fail("no metadata in chunk")


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_haskell_fallback():
    """Test Haskell code handling (should use Haskell AST or fallback)."""
    haskell_code = '''
main :: IO ()
main = do
    putStrLn "Hello, World!"
    let x = 42
    print x

factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)
'''

    chunker = CocoIndexASTChunker(max_chunk_size=300)
    chunks = chunker.chunk_code(haskell_code, "Haskell", "test.hs")

    assert len(chunks) > 0
    for chunk in chunks:
        metadata = chunk['metadata']
        if metadata is not None:
            assert 'chunk_method' in metadata
            assert metadata['language'] == 'Haskell'
        else:
            pytest.fail("no metadata in chunk")


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_unknown_language_fallback():
    """Test handling of unknown languages (should use simple text chunking)."""
    code = '''
some random code
that doesn't belong to any
recognized programming language
but should still be chunked
'''

    chunker = CocoIndexASTChunker(max_chunk_size=100)
    chunks = chunker.chunk_code(code, "Unknown", "test.xyz")

    assert len(chunks) > 0
    for chunk in chunks:
        metadata = chunk['metadata']
        if metadata is not None:
            assert metadata['language'] == 'Unknown'
        else:
            pytest.fail("no metadata in chunk")


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_chunker_initialization():
    """Test CocoIndexASTChunker initialization with different parameters."""
    # Default initialization
    chunker1 = CocoIndexASTChunker()
    assert chunker1 is not None

    # Custom chunk size
    chunker2 = CocoIndexASTChunker(max_chunk_size=500)
    assert chunker2.max_chunk_size == 500

    # Very small chunk size
    chunker3 = CocoIndexASTChunker(max_chunk_size=50)
    assert chunker3.max_chunk_size == 50
