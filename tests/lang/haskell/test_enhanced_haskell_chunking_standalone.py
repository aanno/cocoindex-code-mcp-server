#!/usr/bin/env python3

"""
Standalone tests for enhanced Haskell chunking functionality.
Tests the new HaskellChunkConfig and EnhancedHaskellChunker classes without CocoIndex imports.
"""

from typing import Any, Dict, List
import sys

import pytest

# Set up mock early to avoid circular import - this runs at module load time
@pytest.fixture(scope="session", autouse=True)
def setup_cocoindex_mock():
    """Set up cocoindex mock before any imports."""
    # Create a simple mock class that mimics unittest.mock.Mock behavior
    class SimpleMock:
        def __init__(self):
            self._attrs = {}
        def __call__(self, *args, **kwargs):
            return lambda f: f
        def __getattr__(self, name):
            if name not in self._attrs:
                self._attrs[name] = SimpleMock()
            return self._attrs[name]
        def __setattr__(self, name, value):
            if name == '_attrs':
                object.__setattr__(self, name, value)
            else:
                if not hasattr(self, '_attrs'):
                    object.__setattr__(self, '_attrs', {})
                self._attrs[name] = value  # type: ignore[attr-defined]

    mock_cocoindex = SimpleMock()
    mock_cocoindex.op = SimpleMock()
    mock_cocoindex.op.function = SimpleMock()
    sys.modules['cocoindex'] = mock_cocoindex  # type: ignore[assignment]

    yield

    # Cleanup
    if 'cocoindex' in sys.modules:
        del sys.modules['cocoindex']

# Import modules after cocoindex mock is available
import haskell_tree_sitter

from cocoindex_code_mcp_server.lang.haskell.haskell_ast_chunker import (
    HaskellChunkConfig,
    create_enhanced_regex_fallback_chunks,
    get_enhanced_haskell_separators,
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
        base_separators = haskell_tree_sitter.get_haskell_separators()
        enhanced_separators = get_enhanced_haskell_separators()

        assert len(enhanced_separators) > len(base_separators)


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
            assert metadata["chunk_method"] == "rust_haskell_regex_fallback_python"
            assert metadata["language"] == "Haskell"
            assert metadata["file_path"] == "test.hs"

    def test_fallback_metadata_fields(self):
        """Test that fallback chunking includes all expected metadata fields."""
        haskell_code = """
module Test where

import Data.List

factorial :: Integer -> Integer
factorial n = product [1..n]

data Tree a = Leaf a | Node (Tree a) (Tree a)
"""

        config = HaskellChunkConfig()
        chunks = create_enhanced_regex_fallback_chunks(haskell_code, "test.hs", config)

        for chunk in chunks:
            metadata = chunk["metadata"]

            # Check required metadata fields
            assert "chunk_id" in metadata
            assert "chunk_method" in metadata
            assert "language" in metadata
            assert "file_path" in metadata
            assert "chunk_size" in metadata
            assert "line_count" in metadata
            assert "start_line" in metadata
            assert "end_line" in metadata

            # Check Haskell-specific metadata
            assert "has_imports" in metadata
            assert "has_type_signatures" in metadata
            assert "has_data_types" in metadata
            assert "has_instances" in metadata
            assert "has_classes" in metadata

            # Check boolean values
            assert isinstance(metadata["has_imports"], bool)
            assert isinstance(metadata["has_type_signatures"], bool)
            assert isinstance(metadata["has_data_types"], bool)

    def test_fallback_content_analysis(self):
        """Test that fallback chunking correctly analyzes Haskell content."""
        haskell_code = """
module TestModule (factorial, Tree(..)) where

import qualified Data.List as L
import Control.Monad

factorial :: Integer -> Integer
factorial n = product [1..n]

data Tree a = Leaf a | Node (Tree a) (Tree a)
    deriving (Show, Eq)

class Functor f where
    fmap :: (a -> b) -> f a -> f b

instance Functor Tree where
    fmap f (Leaf x) = Leaf (f x)
    fmap f (Node l r) = Node (fmap f l) (fmap f r)
"""

        config = HaskellChunkConfig()
        chunks: List[Dict[str, Any]] = create_enhanced_regex_fallback_chunks(haskell_code, "test.hs", config)

        # Combine all chunks to check overall content analysis
        " ".join(chunk["content"] for chunk in chunks)
        all_metadata: Dict[str, Any] = {}
        for chunk in chunks:
            for key, value in chunk["metadata"].items():
                if key.startswith("has_"):
                    all_metadata[key] = all_metadata.get(key, False) or value

        # Verify content analysis
        assert all_metadata.get("has_imports", False) == True
        assert all_metadata.get("has_exports", False) == True  # module with exports
        assert all_metadata.get("has_type_signatures", False) == True
        assert all_metadata.get("has_data_types", False) == True
        assert all_metadata.get("has_classes", False) == True
        assert all_metadata.get("has_instances", False) == True

    def test_fallback_separator_detection(self):
        """Test that fallback chunking detects separators correctly."""
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

        # Should create multiple chunks due to small max_chunk_size
        assert len(chunks) > 1

        # Check for separator priority tracking
        priorities_found = []
        force_splits_found = []

        for chunk in chunks:
            metadata = chunk["metadata"]
            priorities_found.append(metadata.get("separator_priority", 0))
            force_splits_found.append(metadata.get("was_force_split", False))

        # At least some chunks should have been split on separators (priority > 0)
        assert max(priorities_found) > 0


class TestConfigurationOptions:
    """Test suite for new configuration options."""

    def test_max_chunk_size_respected(self):
        """Test that max_chunk_size configuration is respected."""
        # Create a long piece of code
        haskell_code = """
module LongTest where

import Data.List
import Data.Map
import Control.Monad
import System.IO

""" + "\n".join([f"function{i} :: Int -> Int\nfunction{i} x = x + {i}" for i in range(20)])

        config = HaskellChunkConfig(max_chunk_size=200)
        chunks = create_enhanced_regex_fallback_chunks(haskell_code, "test.hs", config)

        # Check that chunks respect the size limit (non-whitespace chars)
        for chunk in chunks:
            non_whitespace_size = chunk["metadata"]["non_whitespace_size"]
            # Allow some tolerance for splitting logic
            assert non_whitespace_size <= config.max_chunk_size * 1.1

    def test_different_metadata_templates(self):
        """Test that different metadata templates can be configured."""
        haskell_code = """
factorial :: Integer -> Integer
factorial n = product [1..n]
"""

        # Test default template
        config_default = HaskellChunkConfig(metadata_template="default")

        # Test repoeval template
        config_repoeval = HaskellChunkConfig(metadata_template="repoeval")

        # Test swebench template
        config_swebench = HaskellChunkConfig(metadata_template="swebench")

        # All should be valid configurations
        assert config_default.metadata_template == "default"
        assert config_repoeval.metadata_template == "repoeval"
        assert config_swebench.metadata_template == "swebench"

    def test_preserve_options(self):
        """Test preserve_imports and preserve_exports options."""
        config_preserve = HaskellChunkConfig(
            preserve_imports=True,
            preserve_exports=True
        )

        config_no_preserve = HaskellChunkConfig(
            preserve_imports=False,
            preserve_exports=False
        )

        assert config_preserve.preserve_imports == True
        assert config_preserve.preserve_exports == True
        assert config_no_preserve.preserve_imports == False
        assert config_no_preserve.preserve_exports == False


class TestBackwardCompatibility:
    """Test suite for backward compatibility."""

    def test_legacy_function_exists(self):
        """Test that legacy function still exists."""
        from cocoindex_code_mcp_server.lang.haskell.haskell_ast_chunker import (
            create_regex_fallback_chunks_python,
        )

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

    def test_legacy_format_conversion(self):
        """Test that legacy format conversion works correctly."""
        from cocoindex_code_mcp_server.lang.haskell.haskell_ast_chunker import (
            create_regex_fallback_chunks_python,
        )

        haskell_code = """
module Test where

factorial :: Integer -> Integer
factorial n = product [1..n]
"""

        chunks = create_regex_fallback_chunks_python(haskell_code)

        for chunk in chunks:
            # Legacy format should have specific fields
            assert "text" in chunk
            assert isinstance(chunk["text"], str)

            assert "start" in chunk
            # Start and end are now integers (line numbers) instead of dicts
            assert isinstance(chunk["start"], int)

            assert "end" in chunk
            assert isinstance(chunk["end"], int)

            assert "location" in chunk
            assert isinstance(chunk["location"], str)
            assert ":" in chunk["location"]  # Should be "start:end" format

            assert "node_type" in chunk
            assert chunk["node_type"] == "regex_chunk"

            assert "metadata" in chunk
            assert "category" in chunk["metadata"]
            assert chunk["metadata"]["category"] == "regex_fallback"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
