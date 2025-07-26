#!/usr/bin/env python3

"""
Integration test for AST chunking functionality.
Moved from src/ast_chunking.py to tests/
"""

import sys
import os
import logging

# Set up logger for tests
LOGGER = logging.getLogger(__name__)

try:
    import ast_chunking
    from ast_chunking import CocoIndexASTChunker
    AST_CHUNKING_AVAILABLE = True
except ImportError as e:
    LOGGER.warning(f"AST chunking not available: {e}")
    AST_CHUNKING_AVAILABLE = False


def test_ast_chunking():
    """Test the AST chunking functionality."""
    if not AST_CHUNKING_AVAILABLE:
        print("⚠️ Skipping AST chunking test - ast_chunking not available")
        return
    
    # Test with Python code
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
    
    LOGGER.info(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        LOGGER.info(f"\nChunk {i + 1}:")
        LOGGER.info(f"Method: {chunk['metadata']['chunk_method']}")
        LOGGER.info(f"Size: {chunk['metadata']['chunk_size']} chars")
        LOGGER.info(f"Lines: {chunk['metadata']['line_count']}")
        LOGGER.info("Content:")
        LOGGER.info(chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'])
    
    # Basic assertions
    assert len(chunks) > 0, "No chunks created"
    for chunk in chunks:
        assert 'content' in chunk
        assert 'metadata' in chunk
        assert chunk['metadata']['chunk_size'] > 0
    
    print("✅ AST chunking test passed!")
    return chunks


def test_ast_chunking_languages():
    """Test AST chunking with different supported languages."""
    if not AST_CHUNKING_AVAILABLE:
        print("⚠️ Skipping AST language test - ast_chunking not available")
        return
    
    chunker = CocoIndexASTChunker(max_chunk_size=500)
    
    # Test language support detection
    supported_languages = ["Python", "Java", "C#", "TypeScript", "JavaScript", "TSX"]
    unsupported_languages = ["Haskell", "Rust", "Go"]
    
    for lang in supported_languages:
        assert chunker.is_supported_language(lang), f"Language {lang} should be supported"
    
    for lang in unsupported_languages:
        assert not chunker.is_supported_language(lang), f"Language {lang} should not be supported"
    
    print("✅ AST chunking language support test passed!")


def test_ast_chunking_fallback():
    """Test fallback chunking for unsupported languages."""
    if not AST_CHUNKING_AVAILABLE:
        print("⚠️ Skipping AST fallback test - ast_chunking not available")
        return
    
    # Test with Haskell (should use fallback)
    haskell_code = '''
module Main where

import Data.List

-- A simple function
factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)

main :: IO ()
main = print (factorial 5)
'''
    
    chunker = CocoIndexASTChunker(max_chunk_size=200)
    chunks = chunker.chunk_code(haskell_code, "Haskell", "test.hs")
    
    # Should fallback to Haskell AST chunking or simple text chunking
    assert len(chunks) > 0, "No chunks created in fallback"
    
    # Check that fallback method is indicated
    for chunk in chunks:
        method = chunk['metadata']['chunk_method']
        assert method in ['haskell_ast_chunking', 'simple_text_chunking'], f"Unexpected method: {method}"
    
    print("✅ AST chunking fallback test passed!")


if __name__ == "__main__":
    print("🧪 Running AST Chunking Integration Tests")
    print("=" * 50)
    
    try:
        test_ast_chunking_languages()
        test_ast_chunking_fallback()
        test_ast_chunking()
        
        print("\n🎉 All AST chunking integration tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
