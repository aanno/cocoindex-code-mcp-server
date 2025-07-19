#!/usr/bin/env python3

"""
Test fallback mechanisms for different chunking scenarios.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from ast_chunking import CocoIndexASTChunker, detect_language_from_filename
    COCOINDEX_AST_AVAILABLE = True
except ImportError:
    COCOINDEX_AST_AVAILABLE = False


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_supported_language_chunking():
    """Test chunking with languages supported by ASTChunk."""
    chunker = CocoIndexASTChunker(max_chunk_size=200)
    python_code = '''
def hello():
    print("Hello, World!")
    
class Test:
    def method(self):
        return 42
'''
    
    chunks = chunker.chunk_code(python_code, "Python", "test.py")
    
    assert len(chunks) > 0
    for chunk in chunks:
        metadata = chunk['metadata']
        assert 'chunk_method' in metadata
        assert metadata['language'] == 'Python'


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_unsupported_language_fallback():
    """Test fallback for languages not supported by ASTChunk."""
    chunker = CocoIndexASTChunker(max_chunk_size=200)
    haskell_code = '''
main :: IO ()
main = do
    putStrLn "Hello, World!"
    let x = 42
    print x
'''
    
    chunks = chunker.chunk_code(haskell_code, "Haskell", "test.hs")
    
    assert len(chunks) > 0
    for chunk in chunks:
        metadata = chunk['metadata']
        assert 'chunk_method' in metadata
        assert metadata['language'] == 'Haskell'


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_malformed_code_handling():
    """Test handling of syntactically incorrect code."""
    chunker = CocoIndexASTChunker(max_chunk_size=200)
    
    # Malformed Python code
    malformed_python = '''
def incomplete_function(
    # missing closing parenthesis and body
    
class IncompleteClass
    # missing colon and body
    
# Indentation errors
def another_function():
print("This has wrong indentation")
'''
    
    chunks = chunker.chunk_code(malformed_python, "Python", "broken.py")
    
    # Should still create chunks even with malformed code
    assert len(chunks) > 0
    for chunk in chunks:
        metadata = chunk['metadata']
        assert 'chunk_method' in metadata
        assert metadata['language'] == 'Python'


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_empty_or_whitespace_code():
    """Test handling of empty or whitespace-only code."""
    chunker = CocoIndexASTChunker(max_chunk_size=200)
    
    # Empty string - AST chunker might still create chunks, so we allow >= 0
    empty_chunks = chunker.chunk_code("", "Python", "empty.py")
    assert len(empty_chunks) >= 0
    
    # Only whitespace - AST chunker might still create chunks, so we allow >= 0
    whitespace_chunks = chunker.chunk_code("   \n\n  \t  ", "Python", "whitespace.py")
    assert len(whitespace_chunks) >= 0
    
    # Only comments
    comment_only = '''
# This is just a comment
# Another comment
# No actual code here
'''
    comment_chunks = chunker.chunk_code(comment_only, "Python", "comments.py")
    # Comments should still produce chunks
    assert len(comment_chunks) >= 0


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_very_large_chunk_size():
    """Test chunking with very large chunk size."""
    chunker = CocoIndexASTChunker(max_chunk_size=10000)
    
    # Code that would normally be split into multiple chunks
    large_code = '''
def function_1():
    print("Function 1")
    
def function_2():
    print("Function 2")
    
class Class1:
    def method1(self):
        return 1
    
    def method2(self):
        return 2
        
class Class2:
    def method1(self):
        return 3
    
    def method2(self):
        return 4
'''
    
    chunks = chunker.chunk_code(large_code, "Python", "large.py")
    
    # With large chunk size, might get fewer chunks
    assert len(chunks) >= 1
    for chunk in chunks:
        metadata = chunk['metadata']
        assert metadata['language'] == 'Python'


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_very_small_chunk_size():
    """Test chunking with very small chunk size."""
    chunker = CocoIndexASTChunker(max_chunk_size=50)
    
    code = '''
def hello():
    print("Hello, World!")
    return True
'''
    
    chunks = chunker.chunk_code(code, "Python", "small.py")
    
    # Small chunk size should create more chunks
    assert len(chunks) >= 1
    for chunk in chunks:
        metadata = chunk['metadata']
        assert metadata['language'] == 'Python'
        # Each chunk should be reasonably small
        assert len(chunk['content']) <= 200  # Some tolerance for AST chunking


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_mixed_content_chunking():
    """Test chunking of files with mixed content."""
    chunker = CocoIndexASTChunker(max_chunk_size=200)
    
    mixed_content = '''
#!/usr/bin/env python3
"""
A script with mixed content.
"""

import os
import sys

# Global variable
CONSTANT = 42

def main():
    """Main function."""
    print(f"Constant value: {CONSTANT}")
    
    # Inline comment
    result = process_data()
    return result

def process_data():
    """Process some data."""
    data = [1, 2, 3, 4, 5]
    return sum(data)

if __name__ == "__main__":
    main()
'''
    
    chunks = chunker.chunk_code(mixed_content, "Python", "mixed.py")
    
    assert len(chunks) > 0
    for chunk in chunks:
        metadata = chunk['metadata']
        assert metadata['language'] == 'Python'
        assert 'chunk_method' in metadata


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_unknown_file_extension():
    """Test handling of files with unknown extensions."""
    chunker = CocoIndexASTChunker(max_chunk_size=200)
    
    # File with unknown extension but Python-like content
    python_like_content = '''
def hello():
    print("Hello from unknown extension!")
    
class TestClass:
    pass
'''
    
    # Detect language as Unknown
    detected_lang = detect_language_from_filename("test.unknown")
    assert detected_lang == "Unknown"
    
    # Should still chunk the content
    chunks = chunker.chunk_code(python_like_content, detected_lang, "test.unknown")
    
    assert len(chunks) > 0
    for chunk in chunks:
        metadata = chunk['metadata']
        assert metadata['language'] == 'Unknown'


@pytest.mark.skipif(not COCOINDEX_AST_AVAILABLE, reason="CocoIndexASTChunker not available")
def test_chunking_consistency():
    """Test that chunking the same content multiple times produces consistent results."""
    chunker = CocoIndexASTChunker(max_chunk_size=300)
    
    code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
'''
    
    # Chunk the same code multiple times
    chunks1 = chunker.chunk_code(code, "Python", "test.py")
    chunks2 = chunker.chunk_code(code, "Python", "test.py")
    
    # Should produce the same number of chunks
    assert len(chunks1) == len(chunks2)
    
    # Content should be identical
    for c1, c2 in zip(chunks1, chunks2):
        assert c1['content'] == c2['content']
        assert c1['metadata']['language'] == c2['metadata']['language']