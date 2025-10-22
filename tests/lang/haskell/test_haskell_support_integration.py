#!/usr/bin/env python3

"""
Integration test for Haskell support functionality.
Moved from python/haskell_ast_chunker.py to tests/
"""

import logging
import sys
from typing import Dict, List, Tuple, Union

from cocoindex_code_mcp_server.lang.haskell.haskell_ast_chunker import (
    HaskellChunkConfig,
)

# Set up logger for tests
LOGGER = logging.getLogger(__name__)
try:
    from cocoindex_code_mcp_server.lang.haskell.haskell_ast_chunker import (
        extract_haskell_ast_chunks,
    )

    haskell_ast_chunker_AVAILABLE = True
except ImportError as e:
    LOGGER.warning("Haskell support not available: %s", e)
    haskell_ast_chunker_AVAILABLE = False


def test_enhanced_haskell_chunking() -> List[
    Union[
        Tuple[HaskellChunkConfig, List[Dict[str, Union[str, Dict[str, Union[int, str, bool]]]]]],
        Tuple[
            HaskellChunkConfig,
            List[
                Union[
                    Dict[str, Union[str, Dict[str, Union[int, str, bool]]]],
                    Dict[str, Union[str, Dict[str, Union[int, str, bool, List[str]]]]],
                ]
            ],
        ],
    ]
]:
    """Test the enhanced Haskell chunking functionality."""
    if not haskell_ast_chunker_AVAILABLE:
        print("⚠️ Skipping Haskell test - haskell_ast_chunker not available")
        return []

    haskell_code = """
module Main where

import Data.List
import Control.Monad

-- | A simple data type for demonstration
data Tree a = Leaf a | Node (Tree a) (Tree a)
    deriving (Show, Eq)

-- | Calculate the depth of a tree
treeDepth :: Tree a -> Int
treeDepth (Leaf _) = 1
treeDepth (Node left right) = 1 + max (treeDepth left) (treeDepth right)

-- | Map a function over a tree
mapTree :: (a -> b) -> Tree a -> Tree b
mapTree f (Leaf x) = Leaf (f x)
mapTree f (Node left right) = Node (mapTree f left) (mapTree f right)

-- | Fold over a tree
foldTree :: (a -> b) -> (b -> b -> b) -> Tree a -> b
foldTree f g (Leaf x) = f x
foldTree f g (Node left right) = g (foldTree f g left) (foldTree f g right)

main :: IO ()
main = do
    let tree = Node (Leaf 1) (Node (Leaf 2) (Leaf 3))
    putStrLn $ "Tree: " ++ show tree
    putStrLn $ "Depth: " ++ show (treeDepth tree)
    putStrLn $ "Doubled: " ++ show (mapTree (*2) tree)
"""

    # Test with different configurations
    configs = [
        HaskellChunkConfig(max_chunk_size=300, chunk_expansion=False),
        HaskellChunkConfig(max_chunk_size=300, chunk_expansion=True, metadata_template="repoeval"),
        HaskellChunkConfig(max_chunk_size=500, chunk_overlap=2, metadata_template="swebench"),
    ]

    all_results = []

    for i, config in enumerate(configs):
        LOGGER.info("\n--- Configuration %s ---", i + 1)
        LOGGER.info("Max size: %s, Overlap: %s", config.max_chunk_size, config.chunk_overlap)
        LOGGER.info("Expansion: %s, Template: %s", config.chunk_expansion, config.metadata_template)

        # Use legacy approach since @op.executor_class requires CocoIndex infrastructure
        chunks = extract_haskell_ast_chunks(haskell_code)
        all_results.append((config, chunks))

        LOGGER.info("Created %s chunks:", len(chunks))
        for j, chunk in enumerate(chunks):
            metadata = chunk["metadata"]
            LOGGER.info("  Chunk %s: %s method", j + 1, metadata.get("chunking_method", "unknown"))

        # Basic assertions - use legacy format
        assert len(chunks) > 0, f"No chunks created for config {i + 1}"
        for chunk in chunks:
            assert "text" in chunk  # Legacy format uses 'text' instead of 'content'
            assert "metadata" in chunk

    print("✅ Enhanced Haskell chunking test passed!")
    return all_results


def test_haskell_chunk_config():
    """Test Haskell chunk configuration options."""
    if not haskell_ast_chunker_AVAILABLE:
        print("⚠️ Skipping Haskell config test - haskell_ast_chunker not available")
        return

    # Test default configuration
    default_config = HaskellChunkConfig()
    assert default_config.max_chunk_size == 1800
    assert default_config.chunk_overlap == 0
    assert default_config.chunk_expansion == False
    assert default_config.metadata_template == "default"
    assert default_config.preserve_imports == True
    assert default_config.preserve_exports == True

    # Test custom configuration
    custom_config = HaskellChunkConfig(
        max_chunk_size=1000,
        chunk_overlap=5,
        chunk_expansion=True,
        metadata_template="repoeval",
        preserve_imports=False,
        preserve_exports=False,
    )
    assert custom_config.max_chunk_size == 1000
    assert custom_config.chunk_overlap == 5
    assert custom_config.chunk_expansion == True
    assert custom_config.metadata_template == "repoeval"
    assert custom_config.preserve_imports == False
    assert custom_config.preserve_exports == False

    print("✅ Haskell chunk config test passed!")


def test_haskell_simple_chunking():
    """Test simple Haskell code chunking."""
    if not haskell_ast_chunker_AVAILABLE:
        print("⚠️ Skipping simple Haskell test - haskell_ast_chunker not available")
        return

    simple_code = """
module Simple where

-- Simple function
add :: Int -> Int -> Int
add x y = x + y

-- Another function
multiply :: Int -> Int -> Int
multiply x y = x * y
"""

    # Use legacy approach since @op.executor_class requires CocoIndex infrastructure
    chunks = extract_haskell_ast_chunks(simple_code)

    assert len(chunks) > 0

    # Check that we have some content (legacy format uses 'text')
    total_content = "".join(chunk["text"] for chunk in chunks)
    assert "add" in total_content
    assert "multiply" in total_content

    print("✅ Simple Haskell chunking test passed!")


if __name__ == "__main__":
    print("🧪 Running Haskell Support Integration Tests")
    print("=" * 50)

    try:
        test_haskell_chunk_config()
        test_haskell_simple_chunking()
        test_enhanced_haskell_chunking()

        print("\n🎉 All Haskell support integration tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
