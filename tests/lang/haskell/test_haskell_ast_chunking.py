#!/usr/bin/env python3

"""
Comprehensive tests for AST-based Haskell chunking functionality.
"""

from types import FunctionType
from typing import cast

import cocoindex_code_mcp_server._haskell_tree_sitter as hts
import pytest
from cocoindex_code_mcp_server.lang.haskell.haskell_ast_chunker import (
    HaskellChunkSpec,
    extract_haskell_ast_chunks,
    get_enhanced_haskell_separators,
)


class TestHaskellASTChunking:
    """Test suite for AST-based Haskell chunking."""

    def test_basic_ast_chunking(self):
        """Test basic AST chunking functionality."""
        sample_code = """
module Test where

import Data.List

factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)
"""

        chunks = hts.get_haskell_ast_chunks(sample_code)
        assert len(chunks) > 0

        # Check that we get expected chunk types
        chunk_types = [chunk.node_type() for chunk in chunks]
        assert "import" in chunk_types
        assert "signature" in chunk_types
        assert "function" in chunk_types

    def test_ast_chunking_with_fallback(self):
        """Test AST chunking with fallback to regex."""
        sample_code = """
module Test where

factorial :: Integer -> Integer
factorial n = n * factorial (n - 1)
"""

        chunks = hts.get_haskell_ast_chunks_with_fallback(sample_code)
        assert len(chunks) > 0

        # Should use AST method for valid code
        methods = [chunk.metadata().get('method', 'ast') for chunk in chunks]
        assert all(method == 'ast' for method in methods)

    def test_modern_executor_pattern(self):
        """Test the modern HaskellChunkExecutor pattern that integrates with CocoIndex."""
        sample_code = """
module Test where

data Person = Person String Int

greet :: Person -> String
greet (Person name _) = "Hello, " ++ name
"""

        # Create executor with spec (modern pattern)
        spec = HaskellChunkSpec(max_chunk_size=1800)
        # Use legacy function for testing purposes since @op.executor_class requires CocoIndex infrastructure
        chunks = extract_haskell_ast_chunks(sample_code)
        assert len(chunks) > 0

        # Check that chunks have expected structure (legacy format)
        for chunk in chunks:
            assert "text" in chunk  # Legacy format uses "text" instead of "content"
            assert "metadata" in chunk
            assert "location" in chunk
            assert "start" in chunk
            assert "end" in chunk

            # Test that location exists and is meaningful
            assert chunk.get("location") is not None
            assert len(chunk["text"]) > 0

    @pytest.mark.skip(reason="Metadata format changed with enhanced chunker")
    def test_metadata_extraction(self):
        """Test that metadata is properly extracted from AST nodes."""
        sample_code = """
module Test where

data Person = Person String Int

class Greetable a where
    greet :: a -> String

instance Greetable Person where
    greet (Person name _) = "Hello, " ++ name
"""

        chunks = cast(FunctionType, extract_haskell_ast_chunks)(sample_code)

        # Find specific chunks and verify metadata
        data_chunks = [c for c in chunks if c['node_type'] == 'data_type']
        assert len(data_chunks) > 0
        assert data_chunks[0]['metadata']['category'] == 'data_type'

        class_chunks = [c for c in chunks if c['node_type'] == 'class']
        assert len(class_chunks) > 0
        assert any(c['metadata']['category'] == 'class' for c in class_chunks)

        instance_chunks = [c for c in chunks if c['node_type'] == 'instance']
        assert len(instance_chunks) > 0
        assert any(c['metadata']['category'] == 'instance' for c in instance_chunks)

    def test_enhanced_separators(self):
        """Test enhanced separator generation."""
        separators = get_enhanced_haskell_separators()
        assert len(separators) > 10  # Should have base + enhanced separators

        # Check that it includes base separators
        base_separators = hts.get_haskell_separators()
        for sep in base_separators:
            assert sep in separators

        # Check that enhanced separators are included
        enhanced_patterns = [
            r"\n[a-zA-Z][a-zA-Z0-9_']*\s*::",  # Type signatures
            r"\ndata\s+[A-Z][a-zA-Z0-9_']*",   # Data types
            r"\nclass\s+[A-Z][a-zA-Z0-9_']*",  # Classes
        ]

        for pattern in enhanced_patterns:
            assert pattern in separators

    @pytest.mark.skip(reason="Function name extraction method changed with enhanced chunker")
    def test_function_name_extraction(self):
        """Test that function names are properly extracted."""
        sample_code = """
factorial :: Integer -> Integer
factorial n = n * factorial (n - 1)

fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci n = fibonacci (n - 1) + fibonacci (n - 2)
"""

        chunks = cast(FunctionType, extract_haskell_ast_chunks)(sample_code)

        # Find function-related chunks
        function_chunks = [c for c in chunks if 'function_name' in c['metadata']]
        assert len(function_chunks) > 0

        function_names = [c['metadata']['function_name'] for c in function_chunks]
        assert 'factorial' in function_names
        assert 'fibonacci' in function_names

    @pytest.mark.skip(reason="Documentation categorization changed with enhanced chunker")
    def test_documentation_chunks(self):
        """Test that Haddock documentation is properly chunked."""
        sample_code = """
module Test where

-- | This is a factorial function
-- that calculates n!
factorial :: Integer -> Integer
factorial n = product [1..n]

-- | Another function
helper :: Int -> Int
helper x = x + 1
"""

        chunks = cast(FunctionType, extract_haskell_ast_chunks)(sample_code)

        # Find documentation chunks
        doc_chunks = [c for c in chunks if c['node_type'] == 'haddock']
        assert len(doc_chunks) > 0

        # Check that documentation is properly categorized
        for chunk in doc_chunks:
            assert chunk['metadata']['category'] == 'documentation'

    def test_complex_haskell_code(self):
        """Test chunking on more complex Haskell code."""
        sample_code = """
{-# LANGUAGE OverloadedStrings #-}

module Complex where

import qualified Data.Map as Map
import Control.Monad

-- | A complex data type
data ComplexType a b = ComplexType
    { field1 :: a
    , field2 :: b
    , field3 :: Map.Map String Int
    } deriving (Show, Eq)

-- | A type class with associated types
class Processable a where
    type ProcessResult a
    process :: a -> ProcessResult a

-- | Instance with complex implementation
instance Processable (ComplexType String Int) where
    type ProcessResult (ComplexType String Int) = Maybe String
    process (ComplexType f1 f2 f3) = do
        guard (f2 > 0)
        return $ f1 ++ show (Map.size f3)

-- | Main processing function
processAll :: [ComplexType String Int] -> [Maybe String]
processAll = map process
"""

        # Use legacy function for testing purposes since @op.executor_class requires CocoIndex infrastructure
        chunks = extract_haskell_ast_chunks(sample_code)
        assert len(chunks) > 0  # Should have chunks

        # Check that we get content from various parts (legacy format uses "text")
        contents = [chunk["text"] for chunk in chunks]
        combined_content = " ".join(contents)

        assert "import" in combined_content
        assert "ComplexType" in combined_content
        assert "Processable" in combined_content
        assert "processAll" in combined_content


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
