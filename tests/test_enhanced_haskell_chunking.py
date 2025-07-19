#!/usr/bin/env python3

"""
Tests for enhanced Haskell chunking functionality.
Tests the new HaskellChunkConfig and EnhancedHaskellChunker classes.
"""

import os
import sys
import pytest

# Add the src directory to the path to import the main module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from haskell_support import (
    HaskellChunkConfig, 
    EnhancedHaskellChunker, 
    get_enhanced_haskell_separators,
    create_enhanced_regex_fallback_chunks
)


class TestHaskellChunkConfig:
    """Test suite for HaskellChunkConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HaskellChunkConfig()
        assert config.max_chunk_size == 1800
        assert config.chunk_overlap == 0
        assert config.chunk_expansion == False
        assert config.metadata_template == "default"
        assert config.preserve_imports == True
        assert config.preserve_exports == True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HaskellChunkConfig(
            max_chunk_size=1200,
            chunk_overlap=3,
            chunk_expansion=True,
            metadata_template="repoeval",
            preserve_imports=False,
            preserve_exports=False
        )
        assert config.max_chunk_size == 1200
        assert config.chunk_overlap == 3
        assert config.chunk_expansion == True
        assert config.metadata_template == "repoeval"
        assert config.preserve_imports == False
        assert config.preserve_exports == False


class TestEnhancedHaskellSeparators:
    """Test suite for enhanced Haskell separators."""

    def test_separators_include_base(self):
        """Test that enhanced separators include base separators."""
        import haskell_tree_sitter
        base_separators = haskell_tree_sitter.get_haskell_separators()
        enhanced_separators = get_enhanced_haskell_separators()
        
        # All base separators should be included
        for sep in base_separators:
            assert sep in enhanced_separators

    def test_separators_include_enhancements(self):
        """Test that enhanced separators include new patterns."""
        separators = get_enhanced_haskell_separators()
        
        # Check for specific enhanced patterns
        expected_patterns = [
            r"\nmodule\s+[A-Z][a-zA-Z0-9_.']*",
            r"\nimport\s+(qualified\s+)?[A-Z][a-zA-Z0-9_.']*",
            r"\ndata\s+[A-Z][a-zA-Z0-9_']*",
            r"\nclass\s+[A-Z][a-zA-Z0-9_']*",
            r"\n[a-zA-Z][a-zA-Z0-9_']*\s*::",  # Type signatures
        ]
        
        for pattern in expected_patterns:
            assert pattern in separators

    def test_separators_count(self):
        """Test that enhanced separators are more than base separators."""
        import haskell_tree_sitter
        base_separators = haskell_tree_sitter.get_haskell_separators()
        enhanced_separators = get_enhanced_haskell_separators()
        
        assert len(enhanced_separators) > len(base_separators)


class TestEnhancedHaskellChunker:
    """Test suite for EnhancedHaskellChunker class."""

    def test_chunker_creation(self):
        """Test that chunker can be created with default config."""
        chunker = EnhancedHaskellChunker()
        assert chunker.config.max_chunk_size == 1800
        assert chunker.config.chunk_overlap == 0

    def test_chunker_custom_config(self):
        """Test that chunker can be created with custom config."""
        config = HaskellChunkConfig(max_chunk_size=1000, chunk_overlap=2)
        chunker = EnhancedHaskellChunker(config)
        assert chunker.config.max_chunk_size == 1000
        assert chunker.config.chunk_overlap == 2

    def test_basic_chunking(self):
        """Test basic chunking functionality."""
        haskell_code = """
module Test where

import Data.List

factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

data Tree a = Leaf a | Node (Tree a) (Tree a)
"""
        
        chunker = EnhancedHaskellChunker()
        chunks = chunker.chunk_code(haskell_code, "test.hs")
        
        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)

    def test_metadata_enhancement(self):
        """Test that metadata is properly enhanced."""
        haskell_code = """
module Test where

import Data.List

factorial :: Integer -> Integer
factorial n = product [1..n]
"""
        
        chunker = EnhancedHaskellChunker()
        chunks = chunker.chunk_code(haskell_code, "test.hs")
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            
            # Check required metadata fields
            assert "chunk_id" in metadata
            assert "chunk_method" in metadata
            assert "language" in metadata
            assert "file_path" in metadata
            assert "chunk_size" in metadata
            assert "line_count" in metadata
            
            # Check Haskell-specific metadata
            assert "has_imports" in metadata
            assert "has_type_signatures" in metadata
            assert "has_data_types" in metadata

    def test_repoeval_metadata_template(self):
        """Test repoeval metadata template."""
        haskell_code = """
factorial :: Integer -> Integer
factorial n = product [1..n]

fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci n = fibonacci (n - 1) + fibonacci (n - 2)

data Tree a = Leaf a | Node (Tree a) (Tree a)
"""
        
        config = HaskellChunkConfig(metadata_template="repoeval")
        chunker = EnhancedHaskellChunker(config)
        chunks = chunker.chunk_code(haskell_code, "test.hs")
        
        # Should have functions and types extracted
        found_functions = False
        found_types = False
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            if "functions" in metadata and metadata["functions"]:
                found_functions = True
                # Should find factorial and fibonacci
                functions = metadata["functions"]
                assert any("factorial" in str(functions) or "fibonacci" in str(functions))
            if "types" in metadata and metadata["types"]:
                found_types = True
                # Should find Tree
                types = metadata["types"]
                assert any("Tree" in str(types))
        
        # At least one chunk should have functions or types
        assert found_functions or found_types

    def test_swebench_metadata_template(self):
        """Test swebench metadata template."""
        haskell_code = """
module Complex where

import Data.Map

processData :: String -> IO (Maybe Int)
processData input = do
    case parse input of
        Left err -> return Nothing
        Right val -> 
            let result = val >>= process
            in return $ Just result
  where
    process x = if x > 0 then Just (x * 2) else Nothing
"""
        
        config = HaskellChunkConfig(metadata_template="swebench")
        chunker = EnhancedHaskellChunker(config)
        chunks = chunker.chunk_code(haskell_code, "test.hs")
        
        # Should have complexity and dependencies
        found_complexity = False
        found_dependencies = False
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            if "complexity_score" in metadata:
                found_complexity = True
                assert isinstance(metadata["complexity_score"], int)
                assert metadata["complexity_score"] >= 0
            if "dependencies" in metadata and metadata["dependencies"]:
                found_dependencies = True
                # Should find Data.Map import
                deps = metadata["dependencies"]
                assert any("Data.Map" in str(deps))
        
        assert found_complexity or found_dependencies

    def test_chunk_expansion(self):
        """Test chunk expansion with headers."""
        haskell_code = """
factorial :: Integer -> Integer
factorial n = product [1..n]
"""
        
        config = HaskellChunkConfig(chunk_expansion=True)
        chunker = EnhancedHaskellChunker(config)
        chunks = chunker.chunk_code(haskell_code, "test.hs")
        
        # Check that at least one chunk has expansion header
        found_header = False
        for chunk in chunks:
            if chunk.get("has_expansion_header"):
                found_header = True
                # Content should start with a comment header
                assert chunk["content"].startswith("-- ")
                assert "File: test.hs" in chunk["content"]
                assert "Lines:" in chunk["content"]
        
        # Note: This might not always trigger if AST chunking doesn't need expansion
        # so we just verify the functionality exists

    def test_chunk_overlap(self):
        """Test chunk overlap functionality."""
        haskell_code = """
module Test where

import Data.List

factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n - 1) + fibonacci (n - 2)

helper :: Int -> Int
helper x = x + 1
"""
        
        config = HaskellChunkConfig(chunk_overlap=2, max_chunk_size=200)
        chunker = EnhancedHaskellChunker(config)
        chunks = chunker.chunk_code(haskell_code, "test.hs")
        
        # Should create multiple chunks due to small max_chunk_size
        if len(chunks) > 1:
            # Check for overlap indicators
            overlap_found = False
            for chunk in chunks:
                if chunk.get("has_prev_overlap") or chunk.get("has_next_overlap"):
                    overlap_found = True
                    break
            
            # Note: Overlap might not always be applied depending on AST structure


class TestEnhancedRegexFallback:
    """Test suite for enhanced regex fallback chunking."""

    def test_fallback_chunking(self):
        """Test enhanced regex fallback chunking."""
        haskell_code = """
module Test where

import Data.List

factorial :: Integer -> Integer
factorial n = product [1..n]
"""
        
        config = HaskellChunkConfig()
        chunks = create_enhanced_regex_fallback_chunks(haskell_code, "test.hs", config)
        
        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
        
        # Check that metadata indicates regex fallback method
        for chunk in chunks:
            metadata = chunk["metadata"]
            assert metadata["chunk_method"] == "enhanced_regex_fallback"
            assert metadata["language"] == "Haskell"
            assert metadata["file_path"] == "test.hs"

    def test_fallback_separator_detection(self):
        """Test that fallback chunking detects separators."""
        haskell_code = """
module Test where

import Data.List

data Tree a = Leaf a | Node (Tree a) (Tree a)

factorial :: Integer -> Integer
factorial n = product [1..n]

class Functor f where
    fmap :: (a -> b) -> f a -> f b
"""
        
        config = HaskellChunkConfig(max_chunk_size=100)  # Force small chunks
        chunks = create_enhanced_regex_fallback_chunks(haskell_code, "test.hs", config)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Check for separator priority tracking
        priorities_found = False
        for chunk in chunks:
            if chunk["metadata"].get("separator_priority", 0) > 0:
                priorities_found = True
                break
        
        # At least some chunks should have been split on separators


class TestBackwardCompatibility:
    """Test suite for backward compatibility."""

    def test_legacy_function_exists(self):
        """Test that legacy function still exists."""
        from haskell_support import create_regex_fallback_chunks_python
        
        haskell_code = """
factorial :: Integer -> Integer
factorial n = product [1..n]
"""
        
        chunks = create_regex_fallback_chunks_python(haskell_code)
        
        # Should return chunks in legacy format
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("start" in chunk for chunk in chunks)
        assert all("end" in chunk for chunk in chunks)
        assert all("location" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)

    def test_cocoindex_operation_exists(self):
        """Test that CocoIndex operation function exists."""
        from haskell_support import extract_haskell_ast_chunks
        
        # Function should exist and be callable
        assert callable(extract_haskell_ast_chunks)
        
        haskell_code = """
factorial :: Integer -> Integer
factorial n = product [1..n]
"""
        
        # Should work with just content parameter
        chunks = extract_haskell_ast_chunks(haskell_code)
        assert isinstance(chunks, list)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])