#!/usr/bin/env python3

"""
Standalone tests for enhanced Haskell chunking components that don't require CocoIndex.
Tests HaskellChunkConfig, enhanced separators, and regex fallback functionality.
"""

import logging
import re
from typing import Any, Dict, List

import cocoindex_code_mcp_server._haskell_tree_sitter as hts
import pytest

LOGGER = logging.getLogger(__name__)

# Import only the specific components we need, avoiding CocoIndex import


class HaskellChunkConfig:
    """Standalone version of HaskellChunkConfig for testing."""

    def __init__(self,
                 max_chunk_size: int = 1800,
                 chunk_overlap: int = 0,
                 chunk_expansion: bool = False,
                 metadata_template: str = "default",
                 preserve_imports: bool = True,
                 preserve_exports: bool = True):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_expansion = chunk_expansion
        self.metadata_template = metadata_template
        self.preserve_imports = preserve_imports
        self.preserve_exports = preserve_exports


def get_enhanced_haskell_separators() -> List[str]:
    """Standalone version of enhanced separators."""
    base_separators = hts.get_haskell_separators()

    # Add additional AST-aware separators with priority ordering
    enhanced_separators = base_separators + [
        # High priority: Module and import boundaries (should rarely be split)
        r"\nmodule\s+[A-Z][a-zA-Z0-9_.']*",
        r"\nimport\s+(qualified\s+)?[A-Z][a-zA-Z0-9_.']*",

        # Medium priority: Type and data definitions
        r"\ndata\s+[A-Z][a-zA-Z0-9_']*",
        r"\nnewtype\s+[A-Z][a-zA-Z0-9_']*",
        r"\ntype\s+[A-Z][a-zA-Z0-9_']*",
        r"\nclass\s+[A-Z][a-zA-Z0-9_']*",
        r"\ninstance\s+[A-Z][a-zA-Z0-9_']*",

        # Medium priority: Function definitions with type signatures
        r"\n[a-zA-Z][a-zA-Z0-9_']*\s*::",  # Type signatures
        r"\n[a-zA-Z][a-zA-Z0-9_']*.*\s*=",  # Function definitions

        # Lower priority: Block structures
        r"\nwhere\s*$",
        r"\nlet\s+",
        r"\nin\s+",
        r"\ndo\s*$",

        # Language pragmas (usually at file top, high priority)
        r"\n{-#\s*[A-Z]+",

        # Comment blocks (can be good separation points)
        r"\n--\s*[=-]{3,}",  # Comment separators like "-- ==="
        r"\n{-\s*[=-]{3,}",  # Block comment separators
    ]

    return enhanced_separators


def test_safe_regex_matching():
    """Test that regex matching is safe and handles problematic patterns."""
    separators = get_enhanced_haskell_separators()
    test_lines = [
        "module Test where",
        "import Data.List",
        "factorial :: Int -> Int",
        "data Tree a = Leaf",
        "",  # Empty line
        "-- Some comment",
        "    indented line"
    ]

    problems_found = []

    for line in test_lines:
        for separator in separators:
            # Test the pattern processing logic
            if not isinstance(separator, str):
                continue
            pattern = separator
            if pattern.startswith('\n'):
                pattern = pattern[2:]  # Remove \n
                # Handle double newlines and other special cases
                if pattern.startswith('\n'):
                    if pattern == '\n+':
                        pattern = '^$'  # Match empty lines
                    else:
                        pattern = pattern[2:] + '$'  # Make it end-of-line match for empty lines

            try:
                re.match(pattern, line)
                # We don't care about the result, just that it doesn't crash
            except re.error as e:
                problems_found.append(f"Pattern {repr(separator)} -> {repr(pattern)} failed on line {repr(line)}: {e}")

    if problems_found:
        for problem in problems_found:
            LOGGER.debug(problem)

    assert len(problems_found) == 0, f"Found {len(problems_found)} regex problems"


def create_test_regex_fallback_chunks(content: str, file_path: str, config: HaskellChunkConfig) -> List[Dict[str, Any]]:
    """Standalone test version of regex fallback chunking."""
    separators = get_enhanced_haskell_separators()
    lines = content.split('\n')
    chunks: List[Dict[str, Any]] = []

    current_start = 0
    current_size = 0

    for i, line in enumerate(lines):
        line_size = len(line.replace(" ", "").replace("\t", ""))
        is_separator = False
        separator_priority = 0

        # Check for separator patterns with priority
        for priority, separator in enumerate(separators):
            # Remove leading \n but handle special cases like \n\n
            if not isinstance(separator, str):
                continue
            pattern = separator
            if pattern.startswith('\n'):
                pattern = pattern[2:]  # Remove \n
                # Handle double newlines and other special cases
                if pattern.startswith('\n'):
                    if pattern == '\n+':
                        pattern = '^$'  # Match empty lines
                    else:
                        pattern = pattern[2:] + '$'  # Make it end-of-line match for empty lines

            try:
                if re.match(pattern, line):
                    is_separator = True
                    separator_priority = len(separators) - priority
                    break
            except re.error:
                # Skip invalid regex patterns
                continue

        # Force split if chunk gets too large
        force_split = current_size + line_size > config.max_chunk_size

        if (is_separator or force_split) and current_start < i:
            chunk_lines = lines[current_start:i]
            chunk_text = '\n'.join(chunk_lines)

            if chunk_text.strip():
                metadata = {
                    "chunk_id": len(chunks),
                    "chunk_method": "enhanced_regex_fallback",
                    "language": "Haskell",
                    "file_path": file_path,
                    "chunk_size": len(chunk_text),
                    "non_whitespace_size": len(chunk_text.replace(" ", "").replace("\n", "").replace("\t", "")),
                    "line_count": len(chunk_lines),
                    "start_line": current_start + 1,
                    "end_line": i,
                    "separator_priority": separator_priority,
                    "was_force_split": force_split and not is_separator,

                    # Haskell-specific content analysis
                    "has_imports": "import " in chunk_text,
                    "has_exports": "module " in chunk_text and "(" in chunk_text,
                    "has_type_signatures": "::" in chunk_text,
                    "has_data_types": any(keyword in chunk_text for keyword in ["data ", "newtype ", "type "]),
                    "has_instances": "instance " in chunk_text,
                    "has_classes": "class " in chunk_text,
                }

                chunk_dict = {
                    "content": chunk_text,
                    "metadata": metadata
                }
                chunks.append(chunk_dict)

            current_start = i
            current_size = line_size
        else:
            current_size += line_size

    # Handle the last chunk
    if current_start < len(lines):
        chunk_lines = lines[current_start:]
        chunk_text = '\n'.join(chunk_lines)

        if chunk_text.strip():
            metadata = {
                "chunk_id": len(chunks),
                "chunk_method": "enhanced_regex_fallback",
                "language": "Haskell",
                "file_path": file_path,
                "chunk_size": len(chunk_text),
                "non_whitespace_size": len(chunk_text.replace(" ", "").replace("\n", "").replace("\t", "")),
                "line_count": len(chunk_lines),
                "start_line": current_start + 1,
                "end_line": len(lines),
                "separator_priority": 0,
                "was_force_split": False,
                "is_final_chunk": True,

                # Haskell-specific content analysis
                "has_imports": "import " in chunk_text,
                "has_exports": "module " in chunk_text and "(" in chunk_text,
                "has_type_signatures": "::" in chunk_text,
                "has_data_types": any(keyword in chunk_text for keyword in ["data ", "newtype ", "type "]),
                "has_instances": "instance " in chunk_text,
                "has_classes": "class " in chunk_text,
            }

            chunk_dict = {
                "content": chunk_text,
                "metadata": metadata
            }
            chunks.append(chunk_dict)

    return chunks


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
        base_separators = hts.get_haskell_separators()
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
        base_separators = hts.get_haskell_separators()
        enhanced_separators = get_enhanced_haskell_separators()

        assert len(enhanced_separators) > len(base_separators)

    def test_regex_safety(self):
        """Test that all separators can be safely used in regex matching."""
        test_safe_regex_matching()


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
        chunks = create_test_regex_fallback_chunks(haskell_code, "test.hs", config)

        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)

        # Check that metadata indicates regex fallback method
        for chunk in chunks:
            metadata = chunk["metadata"]
            assert metadata["chunk_method"] == "enhanced_regex_fallback"
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
        chunks = create_test_regex_fallback_chunks(haskell_code, "test.hs", config)

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

    def test_content_analysis_accuracy(self):
        """Test that content analysis correctly identifies Haskell constructs."""
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
        chunks = create_test_regex_fallback_chunks(haskell_code, "test.hs", config)

        # Combine all chunks to check overall content analysis
        has_imports = any(chunk["metadata"]["has_imports"] for chunk in chunks)
        has_exports = any(chunk["metadata"]["has_exports"] for chunk in chunks)
        has_type_signatures = any(chunk["metadata"]["has_type_signatures"] for chunk in chunks)
        has_data_types = any(chunk["metadata"]["has_data_types"] for chunk in chunks)
        has_classes = any(chunk["metadata"]["has_classes"] for chunk in chunks)
        has_instances = any(chunk["metadata"]["has_instances"] for chunk in chunks)

        # Verify content analysis
        assert has_imports == True
        assert has_exports == True  # module with exports
        assert has_type_signatures == True
        assert has_data_types == True
        assert has_classes == True
        assert has_instances == True

    def test_small_chunk_size_splitting(self):
        """Test that small chunk sizes force appropriate splitting."""
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
        chunks = create_test_regex_fallback_chunks(haskell_code, "test.hs", config)

        # Should create multiple chunks due to small max_chunk_size
        assert len(chunks) > 1

        # Check that chunks respect size limits (with some tolerance)
        for chunk in chunks:
            non_whitespace_size = chunk["metadata"]["non_whitespace_size"]
            # Type-safe comparison
            if isinstance(non_whitespace_size, int):
                # Allow some tolerance for splitting logic
                assert non_whitespace_size <= config.max_chunk_size * 1.2

    def test_separator_priority_tracking(self):
        """Test that separator priority is correctly tracked."""
        haskell_code = """
module Test where

import Data.List

data Tree a = Leaf a

factorial :: Integer -> Integer
factorial n = product [1..n]
"""

        config = HaskellChunkConfig(max_chunk_size=100)  # Force splitting
        chunks = create_test_regex_fallback_chunks(haskell_code, "test.hs", config)

        # Should have multiple chunks with different separator priorities
        assert len(chunks) > 1

        priorities = [chunk["metadata"]["separator_priority"] for chunk in chunks]
        force_splits = [chunk["metadata"]["was_force_split"] for chunk in chunks]

        # Type-safe operations
        int_priorities = [p for p in priorities if isinstance(p, int)]
        bool_force_splits = [f for f in force_splits if isinstance(f, bool)]

        # At least some chunks should have separator-based splits (priority > 0)
        if int_priorities:
            assert max(int_priorities) > 0
        # Should have a mix of separator and force splits for this test case
        assert any(bool_force_splits) or (int_priorities and max(int_priorities) > 0)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
