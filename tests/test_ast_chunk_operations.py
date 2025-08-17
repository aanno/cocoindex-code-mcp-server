#!/usr/bin/env python3
"""
Test AST chunk operations used in CocoIndex flow.
Converted from test_ast_chunk_operation.py and test_cocoindex_chunking.py
"""

import pytest
from pathlib import Path
import sys
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cocoindex_code_mcp_server.ast_chunking import CocoIndexASTChunker, Chunk, create_ast_chunking_operation


class TestASTChunkOperation:
    """Test the ASTChunkOperation used in CocoIndex flow."""
    
    def test_ast_chunk_operation_with_python_code(self):
        """Test AST chunking with Python code."""
        python_code = '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
'''
        
        # Create chunker
        chunker = CocoIndexASTChunker()
        
        # Process the code
        result = chunker.chunk_code(python_code, language="python", file_path="test.py")
        
        # Should return chunks
        assert isinstance(result, list), "Should return list of chunks"
        assert len(result) > 0, "Should produce at least one chunk"
        
        # Check chunk structure
        for chunk in result:
            assert isinstance(chunk, Chunk), "Should return Chunk objects"
            assert hasattr(chunk, 'content'), "Chunk should have content"
            assert hasattr(chunk, 'metadata'), "Chunk should have metadata"
            assert len(chunk.content.strip()) > 0, "Chunk content should not be empty"
    
    def test_ast_chunk_operation_with_kotlin_code(self):
        """Test AST chunking with Kotlin code."""
        kotlin_code = '''
data class Person(val name: String, val age: Int) {
    fun isAdult(): Boolean = age >= 18
}

fun fibonacci(n: Int): Int {
    return when (n) {
        0 -> 0
        1 -> 1
        else -> fibonacci(n - 1) + fibonacci(n - 2)
    }
}
'''
        
        chunker = CocoIndexASTChunker()
        result = chunker.chunk_code(kotlin_code, language="kotlin", file_path="test.kt")
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Check that chunks contain the expected code elements
        all_content = " ".join(chunk.content for chunk in result)
        assert "data class Person" in all_content or any("Person" in chunk.content for chunk in result)
        assert "fibonacci" in all_content or any("fibonacci" in chunk.content for chunk in result)


class TestCocoIndexASTChunker:
    """Test the CocoIndexASTChunker directly."""
    
    def test_cocoindex_ast_chunker_python(self):
        """Test CocoIndexASTChunker with Python code."""
        python_code = '''
import os
import sys

def main():
    print("Hello, World!")
    
class Example:
    def method(self):
        return 42
'''
        
        chunker = CocoIndexASTChunker()
        chunks = chunker.chunk_code(python_code, language="python", file_path="example.py")
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Check chunk properties
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.content is not None
            assert len(chunk.content.strip()) > 0
            assert chunk.metadata is not None
            
    def test_content_preservation(self):
        """Test that chunking preserves all code content."""
        test_code = '''
def function_one():
    return "one"

def function_two():
    return "two"
    
class MyClass:
    def method(self):
        return "method"
'''
        
        chunker = CocoIndexASTChunker()
        chunks = chunker.chunk_code(test_code, language="python", file_path="test.py")
        
        # Reconstruct content from chunks
        reconstructed = ""
        for chunk in chunks:
            reconstructed += chunk.content
        
        # Should preserve all important elements
        assert "function_one" in reconstructed
        assert "function_two" in reconstructed
        assert "MyClass" in reconstructed
        assert "method" in reconstructed
        
        # Total length should be similar (allowing for some formatting differences)
        original_significant = len([c for c in test_code if c.isalnum() or c in "(){}"])
        reconstructed_significant = len([c for c in reconstructed if c.isalnum() or c in "(){}"])
        
        # Should preserve at least 80% of significant characters
        preservation_ratio = reconstructed_significant / original_significant
        assert preservation_ratio >= 0.8, f"Should preserve most content, got {preservation_ratio:.2%}"


class TestASTChunkLibraryUsage:
    """Test to verify if ASTChunk library is actually being used."""
    
    def test_astchunk_library_direct_usage(self):
        """Test direct usage of ASTChunk library through ASTChunkExecutor."""
        # Import the executor directly
        from cocoindex_code_mcp_server.ast_chunking import ASTChunkExecutor, ASTChunkSpec
        
        # Test Python code that should use ASTChunk library
        python_code = '''
def test_function():
    """A simple test function."""
    x = 1
    y = 2
    return x + y

class TestClass:
    def method(self):
        return "hello"
'''
        
        # Create executor with spec
        spec = ASTChunkSpec(max_chunk_size=1800)
        executor = ASTChunkExecutor()
        executor.spec = spec
        
        # Call the executor directly
        result = executor(python_code, "Python")
        
        # Verify we got results
        assert isinstance(result, list), "Should return list of ASTChunkRow objects"
        assert len(result) > 0, "Should produce at least one chunk"
        
        # Check the chunking_method values
        chunking_methods = [chunk.chunking_method for chunk in result]
        print(f"\n🔍 Direct ASTChunk test - chunking_methods found: {chunking_methods}")
        
        # If ASTChunk library is working, we should see "astchunk_library"
        assert any(method == "astchunk_library" for method in chunking_methods), \
            f"Expected 'astchunk_library' in chunking methods, but got: {chunking_methods}"
    
    def test_astchunk_library_availability(self):
        """Test if ASTChunk library is available and can be imported."""
        from cocoindex_code_mcp_server.ast_chunking import ASTCHUNK_AVAILABLE, ASTChunkBuilder
        
        print(f"\n🔍 ASTChunk availability test - ASTCHUNK_AVAILABLE: {ASTCHUNK_AVAILABLE}")
        print(f"🔍 ASTChunkBuilder: {ASTChunkBuilder}")
        
        # Check if ASTChunk is available
        if not ASTCHUNK_AVAILABLE:
            pytest.skip("ASTChunk library is not available - this explains why astchunk_library doesn't appear")
        
        # If available, try to create a builder
        assert ASTChunkBuilder is not None, "ASTChunkBuilder should be available"
        
        # Try to create a Python builder
        try:
            builder = ASTChunkBuilder(
                max_chunk_size=1800,
                language="python",
                metadata_template="default",
                chunk_expansion=False
            )
            assert builder is not None, "Should be able to create ASTChunkBuilder for Python"
            print("✅ Successfully created ASTChunkBuilder for Python")
        except Exception as e:
            pytest.fail(f"Failed to create ASTChunkBuilder: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
