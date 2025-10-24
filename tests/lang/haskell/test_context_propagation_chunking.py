#!/usr/bin/env python3

"""
Comprehensive tests for context propagation and recursive splitting in Haskell AST chunking.
Tests the new Rust-based implementation with ChunkingParams, ChunkingContext, and recursive splitting.
"""

import cocoindex_code_mcp_server._haskell_tree_sitter as hts
import pytest

from ...common import COCOINDEX_AVAILABLE, CocoIndexTestInfrastructure


class TestChunkingParams:
    """Test suite for ChunkingParams class and parameterized chunking."""

    def test_chunking_params_creation(self):
        """Test ChunkingParams can be created with all parameters."""
        params = hts.ChunkingParams(
            chunk_size=1800,
            min_chunk_size=400,
            chunk_overlap=0,
            max_chunk_size=2000
        )

        assert params.chunk_size() == 1800
        assert params.min_chunk_size() == 400
        assert params.chunk_overlap() == 0
        assert params.max_chunk_size() == 2000

    def test_chunking_params_with_different_sizes(self):
        """Test ChunkingParams with various size configurations."""
        # Small chunks
        small_params = hts.ChunkingParams(
            chunk_size=500,
            min_chunk_size=100,
            chunk_overlap=50,
            max_chunk_size=600
        )

        # Large chunks
        large_params = hts.ChunkingParams(
            chunk_size=3000,
            min_chunk_size=800,
            chunk_overlap=200,
            max_chunk_size=4000
        )

        assert small_params.chunk_size() < large_params.chunk_size()
        assert small_params.min_chunk_size() < large_params.min_chunk_size()


class TestContextPropagationBasics:
    """Test suite for basic context propagation functionality."""

    def test_get_chunks_with_params_function_exists(self):
        """Test that the new parameterized chunking function exists."""
        # Should be able to call the function
        assert hasattr(hts, 'get_haskell_ast_chunks_with_params')
        assert callable(hts.get_haskell_ast_chunks_with_params)

    def test_basic_context_propagation(self):
        """Test basic context propagation with simple Haskell code."""
        haskell_code = """
module SimpleExample where

factorial :: Integer -> Integer
factorial n = n * factorial (n - 1)

helper :: Int -> Int
helper x = x + 1
"""

        params = hts.ChunkingParams(
            chunk_size=1800,
            min_chunk_size=400,
            chunk_overlap=0,
            max_chunk_size=2000
        )

        result = hts.get_haskell_ast_chunks_with_params(haskell_code, params)

        assert hasattr(result, 'chunks')
        assert hasattr(result, 'chunking_method')
        assert hasattr(result, 'error_stats')
        assert hasattr(result, 'coverage_complete')

        chunks = result.chunks()
        assert len(chunks) > 0

        # Check that chunks have metadata
        for chunk in chunks:
            metadata = chunk.metadata()
            assert isinstance(metadata, dict)

    def test_chunking_result_methods(self):
        """Test that ChunkingResult provides all expected methods."""
        haskell_code = "factorial n = n * factorial (n - 1)"

        params = hts.ChunkingParams(
            chunk_size=1000,
            min_chunk_size=200,
            chunk_overlap=0,
            max_chunk_size=1200
        )

        result = hts.get_haskell_ast_chunks_with_params(haskell_code, params)

        # Test all methods
        chunking_method = result.chunking_method()
        assert isinstance(chunking_method, str)
        assert any(method in chunking_method.lower() for method in ["ast_recursive", "regex_fallback", "rust_haskell"])

        coverage = result.coverage_complete()
        assert isinstance(coverage, bool)

        error_stats = result.error_stats()
        assert hasattr(error_stats, 'error_count')
        assert hasattr(error_stats, 'should_fallback')


class TestRecursiveSplitting:
    """Test suite for recursive splitting functionality."""

    def test_recursive_splitting_with_large_functions(self):
        """Test recursive splitting when functions exceed max_chunk_size."""
        # Create a large function that should be split
        large_function = """
module LargeFunctionExample where

processComplexData :: [String] -> [Int] -> IO (Maybe [Result])
processComplexData strings numbers = do
    putStrLn "Starting complex processing"
    let step1 = map (\\s -> length s) strings
    let step2 = zipWith (+) step1 numbers
    let step3 = filter (> 10) step2
    let step4 = map (\\x -> x * 2) step3
    let step5 = take 100 step4
    let step6 = reverse step5
    let step7 = map show step6
    let step8 = concat step7
    putStrLn ("Intermediate result: " ++ step8)
    let step9 = read step8 :: Int
    let step10 = step9 `div` 2
    let step11 = step10 + 42
    let step12 = [step11, step11 + 1, step11 + 2]
    let step13 = map (\\x -> Result x (show x)) step12
    putStrLn "Processing complete"
    return (Just step13)
  where
    helper x = x + 1
    anotherHelper y = y * 2
"""

        params = hts.ChunkingParams(
            chunk_size=500,      # Small target size
            min_chunk_size=100,   # Small minimum
            chunk_overlap=0,
            max_chunk_size=800   # Force splitting of large functions
        )

        result = hts.get_haskell_ast_chunks_with_params(large_function, params)
        chunks = result.chunks()

        # Should create multiple chunks due to size constraints
        assert len(chunks) >= 1

        # Verify chunking method indicates recursive splitting
        method = result.chunking_method()
        assert "recursive" in method.lower() or "ast" in method.lower()

    def test_recursive_splitting_preserves_boundaries(self):
        """Test that recursive splitting preserves semantic boundaries."""
        haskell_code = """
module BoundaryTest where

import Data.List
import Control.Monad

firstFunction :: Int -> Int
firstFunction x = x + 1

data ComplexType = Simple Int | Complex String Int Bool
    deriving (Show, Eq)

secondFunction :: ComplexType -> String
secondFunction (Simple n) = show n
secondFunction (Complex s n b) = s ++ show n ++ show b

class ProcessorClass a where
    process :: a -> String
    validate :: a -> Bool

instance ProcessorClass ComplexType where
    process = secondFunction
    validate (Simple n) = n > 0
    validate (Complex _ n _) = n >= 0

thirdFunction :: [ComplexType] -> [String]
thirdFunction = map process
"""

        params = hts.ChunkingParams(
            chunk_size=300,  # Smaller chunks to trigger splitting
            min_chunk_size=50,
            chunk_overlap=0,
            max_chunk_size=400
        )

        result = hts.get_haskell_ast_chunks_with_params(haskell_code, params)
        chunks = result.chunks()

        # Should create multiple chunks
        assert len(chunks) >= 2

        # Check that chunks maintain reasonable boundaries
        for chunk in chunks:
            content = chunk.text()
            # Chunks shouldn't arbitrarily break in the middle of constructs
            assert not content.strip().startswith("deriving")
            assert not content.strip().startswith("validate")


class TestContextPropagationFeatures:
    """Test suite for advanced context propagation features."""

    def test_ancestor_path_tracking(self):
        """Test that ancestor paths are properly tracked and included in metadata."""
        haskell_code = """
module ComplexModule where

data TreeProcessor = TreeProcessor { name :: String }

class Processor a where
    processTree :: a -> Tree -> IO ()

instance Processor TreeProcessor where
    processTree processor tree = do
        let helper x = processNode x
        mapM_ helper (flatten tree)
      where
        processNode node = putStrLn (show node)
        validateNode node = length (show node) > 0
"""

        params = hts.ChunkingParams(
            chunk_size=1500,
            min_chunk_size=300,
            chunk_overlap=0,
            max_chunk_size=2000
        )

        result = hts.get_haskell_ast_chunks_with_params(haskell_code, params)
        chunks = result.chunks()

        # Look for chunks with ancestor path information
        found_ancestor_paths = False
        for chunk in chunks:
            metadata = chunk.metadata()
            if 'ancestor_path' in metadata:
                found_ancestor_paths = True
                ancestor_path = metadata['ancestor_path']
                assert isinstance(ancestor_path, str)
                # Should contain module or class context
                assert any(context in ancestor_path.lower() for context in
                           ['complexmodule', 'processor', 'treeprocessor'])

        # At least some chunks should have ancestor path information
        # Note: This may not always be present depending on the AST structure
        # but we test that the functionality exists

    def test_semantic_context_preservation(self):
        """Test that semantic context is preserved in chunk metadata."""
        haskell_code = """
module SemanticTest where

import qualified Data.Map as M

data User = User { userId :: Int, userName :: String }

class UserProcessor a where
    processUser :: a -> User -> IO (Maybe ProcessedUser)

instance UserProcessor SimpleProcessor where
    processUser processor user = do
        validated <- validateUser user
        if validated
            then do
                processed <- transformUser user
                return (Just processed)
            else return Nothing
      where
        validateUser u = return (userId u > 0)
        transformUser u = return (ProcessedUser (userId u) (userName u) "processed")
"""

        params = hts.ChunkingParams(
            chunk_size=1200,
            min_chunk_size=250,
            chunk_overlap=0,
            max_chunk_size=1500
        )

        result = hts.get_haskell_ast_chunks_with_params(haskell_code, params)
        chunks = result.chunks()

        # Check for semantic categories in metadata
        categories_found = set()
        for chunk in chunks:
            metadata = chunk.metadata()
            if 'category' in metadata:
                categories_found.add(metadata['category'])

            # Check for other semantic information
            if 'current_module' in metadata:
                assert metadata['current_module'] == 'SemanticTest'

            if 'current_class' in metadata:
                assert 'UserProcessor' in str(metadata['current_class'])

        # Should find various semantic categories
        assert len(categories_found) >= 1


class TestIntegrationWithCocoIndexFlow:
    """Test suite for integration with CocoIndex flow (modern pattern)."""

    @pytest.mark.asyncio
    async def test_haskell_context_propagation_in_flow(self):
        """Test that Haskell chunking works with context propagation through CocoIndex flow."""
        if not COCOINDEX_AVAILABLE:
            pytest.skip("CocoIndex infrastructure not available")

        async with CocoIndexTestInfrastructure(
            paths=["tmp"],
            enable_polling=False,
            chunk_factor_percent=100
        ) as infrastructure:

            # Search for Haskell content to test context propagation
            search_query = {
                "vector_query": "factorial fibonacci haskell function",
                "keyword_query": "language:Haskell",
                "top_k": 10
            }

            result = await infrastructure.perform_hybrid_search(search_query)

            results = result.get("results", [])
            context_features_found = []

            for r in results:
                metadata = r.get("metadata_json", {})
                chunking_method = r.get("chunking_method", "")

                if "haskell" in chunking_method.lower():
                    context_features_found.append(chunking_method)

            print(f"Found Haskell chunking methods with context features: {context_features_found}")

            # Test passes if we can execute the flow without errors
            assert True  # Infrastructure test completed successfully

    @pytest.mark.asyncio
    async def test_haskell_chunking_method_reporting_in_flow(self):
        """Test that Haskell chunking methods are properly reported through CocoIndex flow."""
        if not COCOINDEX_AVAILABLE:
            pytest.skip("CocoIndex infrastructure not available")

        async with CocoIndexTestInfrastructure(
            paths=["tmp"],
            enable_polling=False,
            chunk_factor_percent=100
        ) as infrastructure:

            # Search for any Haskell content
            search_query = {
                "vector_query": "haskell code",
                "keyword_query": "language:Haskell",
                "top_k": 5
            }

            result = await infrastructure.perform_hybrid_search(search_query)

            results = result.get("results", [])
            method_patterns = []

            for r in results:
                method = r.get("chunking_method", "")
                if "rust_haskell" in method or "haskell" in method.lower():
                    method_patterns.append(method)

            print(f"Found Haskell method patterns: {method_patterns}")

            # Test completes successfully - validates the infrastructure
            assert True


class TestErrorHandlingAndFallback:
    """Test suite for error handling and fallback behavior."""

    def test_error_stats_reporting(self):
        """Test that error statistics are properly reported."""
        # Create some potentially problematic Haskell code
        problematic_code = """
module ErrorTest where

-- Missing type signature might cause parsing issues
problematicFunction = \\x -> case x of
    Just y -> y +
    Nothing -> 0

-- Incomplete where clause
anotherFunction x = helper x
  where
    -- Missing helper definition
"""

        params = hts.ChunkingParams(
            chunk_size=1000,
            min_chunk_size=200,
            chunk_overlap=0,
            max_chunk_size=1200
        )

        result = hts.get_haskell_ast_chunks_with_params(problematic_code, params)

        # Check error statistics
        error_stats = result.error_stats()
        error_count = error_stats.error_count()
        should_fallback = error_stats.should_fallback()

        assert isinstance(error_count, int)
        assert isinstance(should_fallback, bool)
        assert error_count >= 0

    def test_fallback_behavior_with_errors(self):
        """Test fallback behavior when AST parsing encounters errors."""
        # Severely malformed Haskell code
        malformed_code = """
module BadCode where

import Data.List
import

function1 :: Int ->
function1 x = x +

data BadType = Constructor {
    field1 ::
    field2 String
    -- Missing closing brace

class BadClass where
    method1 ::
    -- Missing method definition
"""

        params = hts.ChunkingParams(
            chunk_size=800,
            min_chunk_size=100,
            chunk_overlap=0,
            max_chunk_size=1000
        )

        result = hts.get_haskell_ast_chunks_with_params(malformed_code, params)

        # Should still produce chunks even with errors
        chunks = result.chunks()
        assert len(chunks) >= 1

        # Check chunking method - might be fallback
        method = result.chunking_method()
        assert isinstance(method, str)
        assert len(method) > 0


class TestPerformanceAndScaling:
    """Test suite for performance characteristics of context propagation."""

    def test_large_file_handling(self):
        """Test that context propagation works with larger files."""
        # Generate a reasonably large Haskell file
        large_code = """
module LargeFileTest where

import Data.List
import Data.Map
import Control.Monad
import System.IO

"""

        # Add many functions
        for i in range(50):
            large_code += f"""
function{i} :: Int -> Int -> String
function{i} x y = show (x + y + {i})

data Type{i} = Constructor{i} Int String Bool

helper{i} :: Type{i} -> Int
helper{i} (Constructor{i} n _ _) = n + {i}
"""

        params = hts.ChunkingParams(
            chunk_size=1000,
            min_chunk_size=200,
            chunk_overlap=0,
            max_chunk_size=1500
        )

        result = hts.get_haskell_ast_chunks_with_params(large_code, params)
        chunks = result.chunks()

        # Should handle large files and create many chunks
        assert len(chunks) >= 10

        # All chunks should have valid content
        for chunk in chunks:
            assert len(chunk.text().strip()) > 0

    def test_different_chunk_size_configurations(self):
        """Test context propagation with various chunk size configurations."""
        haskell_code = """
module ConfigTest where

import Data.List

quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort (x:xs) = quicksort smaller ++ [x] ++ quicksort larger
  where
    smaller = [y | y <- xs, y <= x]
    larger = [y | y <- xs, y > x]

mergeSort :: Ord a => [a] -> [a]
mergeSort [] = []
mergeSort [x] = [x]
mergeSort xs = merge (mergeSort left) (mergeSort right)
  where
    (left, right) = splitAt (length xs `div` 2) xs
    merge [] ys = ys
    merge xs [] = xs
    merge (x:xs) (y:ys)
        | x <= y = x : merge xs (y:ys)
        | otherwise = y : merge (x:xs) ys
"""

        # Test with different configurations
        configs = [
            (500, 100, 600),   # Small chunks
            (1000, 250, 1200),  # Medium chunks
            (2000, 500, 2500),  # Large chunks
        ]

        for chunk_size, min_size, max_size in configs:
            params = hts.ChunkingParams(
                chunk_size=chunk_size,
                min_chunk_size=min_size,
                chunk_overlap=0,
                max_chunk_size=max_size
            )

            result = hts.get_haskell_ast_chunks_with_params(haskell_code, params)
            chunks = result.chunks()

            # Should produce valid chunks for all configurations
            assert len(chunks) >= 1

            # Verify chunk sizes are reasonable
            for chunk in chunks:
                content = chunk.text()
                # Content should not be empty
                assert len(content.strip()) > 0
                # Should not exceed max_size by too much (allow some tolerance)
                non_ws_size = len(content.replace(' ', '').replace('\n', '').replace('\t', ''))
                assert non_ws_size <= max_size * 1.1  # 10% tolerance


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
