#!/usr/bin/env python3
"""
Comprehensive test suite for AST chunking functionality.
Converted from test_multiple_languages.py and test_ast_chunking.py
"""

import pytest
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cocoindex_code_mcp_server.ast_chunking import CocoIndexASTChunker
from cocoindex_code_mcp_server.cocoindex_config import extract_code_metadata


class TestASTChunkingComprehensive:
    """Comprehensive tests for AST chunking across multiple languages."""
    
    @pytest.fixture
    def chunker(self):
        """Create a standard AST chunker."""
        return CocoIndexASTChunker(
            max_chunk_size=1800,
            chunk_overlap=0,
            chunk_expansion=False,
            metadata_template="default"
        )
    
    @pytest.mark.parametrize("language,code,expected_chunking_method", [
        ("Python", '''def hello():
    """Say hello."""
    print("Hello, World!")

class TestClass:
    def method(self):
        return "test"
''', "ast_tree_sitter"),
        ("Java", '''public class Test {
    public void hello() {
        System.out.println("Hello");
    }
}''', "ast_tree_sitter"),
        ("JavaScript", '''function hello() {
    console.log("Hello");
}

class TestClass {
    method() {
        return "test";
    }
}''', "ast_tree_sitter"),
        ("TypeScript", '''interface Person {
    name: string;
    age: number;
}

function greet(person: Person): void {
    console.log(`Hello, ${person.name}!`);
}''', "ast_tree_sitter"),
        ("C#", '''using System;

public class HelloWorld {
    public static void Main(string[] args) {
        Console.WriteLine("Hello, World!");
    }
}''', "ast_tree_sitter"),
    ])
    def test_astchunk_supported_languages(self, chunker, language, code, expected_chunking_method):
        """Test that ASTChunk-supported languages produce ast_tree_sitter chunking method."""
        # Verify language is supported
        assert chunker.is_supported_language(language), f"{language} should be supported by ASTChunk"
        
        # Chunk the code
        chunks = chunker.chunk_code(code, language, f"test.{language.lower()}")
        
        # Should produce at least one chunk
        assert len(chunks) > 0, f"Should produce chunks for {language}"
        
        # Check first chunk
        chunk = chunks[0]
        assert hasattr(chunk, 'metadata'), "Chunk should have metadata"
        assert chunk.metadata.get('chunking_method') == expected_chunking_method, \
            f"Expected {expected_chunking_method}, got {chunk.metadata.get('chunking_method')}"
        
        # Verify tree-sitter error flags
        assert chunk.metadata.get('tree_sitter_chunking_error') is False, \
            "tree_sitter_chunking_error should be False for successful chunking"
        assert chunk.metadata.get('tree_sitter_analyze_error') is False, \
            "tree_sitter_analyze_error should be False for successful chunking"
    
    def test_haskell_custom_chunking(self, chunker):
        """Test that Haskell uses custom Rust chunking."""
        haskell_code = '''main :: IO ()
main = putStrLn "Hello, World!"

data Person = Person String Int

fibonacci :: Integer -> Integer
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n-1) + fibonacci (n-2)
'''
        
        # Verify Haskell is not supported by ASTChunk
        assert not chunker.is_supported_language("Haskell"), "Haskell should not be supported by ASTChunk"
        
        # Chunk the code
        chunks = chunker.chunk_code(haskell_code, "Haskell", "test.hs")
        
        # Should produce chunks
        assert len(chunks) > 0, "Should produce chunks for Haskell"
        
        # Check chunks have proper Rust chunking method names
        for chunk in chunks:
            chunking_method = chunk.metadata.get('chunking_method')
            assert chunking_method is not None, "Chunk should have chunking_method"
            assert chunking_method.startswith('rust_haskell_'), \
                f"Haskell chunks should use rust_haskell_* method, got: {chunking_method}"
    
    def test_unsupported_language_fallback(self, chunker):
        """Test that unsupported languages fall back to ast_fallback_unavailable."""
        unsupported_code = '''
This is some random text
that doesn't belong to any
programming language we support.
'''
        
        # Use an unknown language
        language = "UnknownLanguage"
        
        # Verify language is not supported
        assert not chunker.is_supported_language(language), "Unknown language should not be supported"
        
        # Chunk the code
        chunks = chunker.chunk_code(unsupported_code, language, "test.unknown")
        
        # Should produce at least one chunk
        assert len(chunks) > 0, "Should produce fallback chunks"
        
        # Check fallback chunking method
        chunk = chunks[0]
        assert chunk.metadata.get('chunking_method') == "ast_fallback_unavailable", \
            f"Expected ast_fallback_unavailable, got {chunk.metadata.get('chunking_method')}"
        
        # Verify tree-sitter error flags
        assert chunk.metadata.get('tree_sitter_chunking_error') is True, \
            "tree_sitter_chunking_error should be True for fallback"
    
    def test_metadata_preservation_through_extraction(self, chunker):
        """Test that chunking_method is preserved through metadata extraction."""
        python_code = '''def hello():
    """Say hello."""
    print("Hello, World!")

class TestClass:
    def method(self):
        return "test"
'''
        
        # Chunk the code
        chunks = chunker.chunk_code(python_code, "Python", "test.py")
        assert len(chunks) > 0, "Should produce chunks"
        
        chunk = chunks[0]
        original_chunking_method = chunk.metadata.get('chunking_method')
        assert original_chunking_method == "ast_tree_sitter", \
            f"Expected ast_tree_sitter, got {original_chunking_method}"
        
        # Test metadata extraction with existing metadata
        existing_metadata_json = json.dumps(chunk.metadata)
        
        # Extract metadata (simulating the flow process)
        extracted_metadata_json = extract_code_metadata(
            text=chunk.content,
            language="Python",
            filename="test.py",
            existing_metadata_json=existing_metadata_json
        )
        
        extracted_metadata = json.loads(extracted_metadata_json)
        final_chunking_method = extracted_metadata.get('chunking_method')
        
        # Chunking method should be preserved
        assert final_chunking_method == original_chunking_method, \
            f"Chunking method should be preserved: expected {original_chunking_method}, got {final_chunking_method}"
    
    @pytest.mark.parametrize("language,extension", [
        ("Python", "py"),
        ("Java", "java"),
        ("JavaScript", "js"),
        ("TypeScript", "ts"),
        ("C#", "cs"),
        ("Haskell", "hs"),
    ])
    def test_chunk_structure_completeness(self, chunker, language, extension):
        """Test that chunks have complete structure regardless of language."""
        # Simple test code
        test_code = "// Simple test code\nfunction test() { return true; }"
        
        # Chunk the code
        chunks = chunker.chunk_code(test_code, language, f"test.{extension}")
        
        # Should produce at least one chunk
        assert len(chunks) > 0, f"Should produce chunks for {language}"
        
        for i, chunk in enumerate(chunks):
            # Check required attributes
            assert hasattr(chunk, 'content'), f"Chunk {i} should have content attribute"
            assert hasattr(chunk, 'metadata'), f"Chunk {i} should have metadata attribute"
            assert hasattr(chunk, 'location'), f"Chunk {i} should have location attribute"
            assert hasattr(chunk, 'start'), f"Chunk {i} should have start attribute"
            assert hasattr(chunk, 'end'), f"Chunk {i} should have end attribute"
            
            # Check content is not empty
            assert len(chunk.content.strip()) > 0, f"Chunk {i} content should not be empty"
            
            # Check metadata has required fields
            metadata = chunk.metadata
            assert isinstance(metadata, dict), f"Chunk {i} metadata should be dict"
            assert 'chunking_method' in metadata, f"Chunk {i} should have chunking_method in metadata"
            assert 'language' in metadata, f"Chunk {i} should have language in metadata"
            assert metadata['language'] == language, f"Chunk {i} language should match input language"
    
    def test_chunk_dictionary_access(self, chunker):
        """Test that chunks support dictionary-style access for CocoIndex compatibility."""
        python_code = '''def test():
    return "Hello, World!"
'''
        
        chunks = chunker.chunk_code(python_code, "Python", "test.py")
        assert len(chunks) > 0, "Should produce chunks"
        
        chunk = chunks[0]
        
        # Test dictionary-style access
        assert chunk["content"] == chunk.content, "Dictionary access to content should work"
        assert chunk["metadata"] == chunk.metadata, "Dictionary access to metadata should work"
        
        # Test get method
        assert chunk.get("content") == chunk.content, "get() method should work for content"
        assert chunk.get("nonexistent", "default") == "default", "get() method should return default for missing keys"
        
        # Test contains
        assert "content" in chunk, "Should support 'in' operator for content"
        assert "metadata" in chunk, "Should support 'in' operator for metadata"
        assert "nonexistent" not in chunk, "Should support 'in' operator for missing keys"
        
        # Test keys
        keys = chunk.keys()
        assert "content" in keys, "keys() should include content"
        assert "metadata" in keys, "keys() should include metadata"


if __name__ == "__main__":
    pytest.main([__file__])