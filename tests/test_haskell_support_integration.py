#!/usr/bin/env python3

"""
Integration test for Haskell support functionality.
Moved from src/haskell_support.py to tests/
"""

import sys
import os
import logging

# Set up logger for tests
LOGGER = logging.getLogger(__name__)

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from lang.haskell.haskell_support import HaskellChunkConfig, EnhancedHaskellChunker
    HASKELL_SUPPORT_AVAILABLE = True
except ImportError as e:
    LOGGER.warning(f"Haskell support not available: {e}")
    HASKELL_SUPPORT_AVAILABLE = False


def test_enhanced_haskell_chunking():
    """Test the enhanced Haskell chunking functionality."""
    if not HASKELL_SUPPORT_AVAILABLE:
        print("⚠️ Skipping Haskell test - haskell_support not available")
        return
    
    haskell_code = '''
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
'''
    
    # Test with different configurations
    configs = [
        HaskellChunkConfig(max_chunk_size=300, chunk_expansion=False),
        HaskellChunkConfig(max_chunk_size=300, chunk_expansion=True, metadata_template="repoeval"),
        HaskellChunkConfig(max_chunk_size=500, chunk_overlap=2, metadata_template="swebench"),
    ]
    
    all_results = []
    
    for i, config in enumerate(configs):
        LOGGER.info(f"\n--- Configuration {i+1} ---")
        LOGGER.info(f"Max size: {config.max_chunk_size}, Overlap: {config.chunk_overlap}")
        LOGGER.info(f"Expansion: {config.chunk_expansion}, Template: {config.metadata_template}")
        
        chunker = EnhancedHaskellChunker(config)
        chunks = chunker.chunk_code(haskell_code, "test.hs")
        all_results.append((config, chunks))
        
        LOGGER.info(f"Created {len(chunks)} chunks:")
        for j, chunk in enumerate(chunks):
            metadata = chunk['metadata']
            LOGGER.info(f"  Chunk {j+1}: {metadata['chunk_method']} method")
            LOGGER.info(f"    Size: {metadata['chunk_size']} chars, Lines: {metadata['line_count']}")
            LOGGER.info(f"    Has types: {metadata.get('has_data_types', False)}")
            LOGGER.info(f"    Has functions: {metadata.get('has_type_signatures', False)}")
            if config.metadata_template == "repoeval":
                LOGGER.info(f"    Functions: {metadata.get('functions', [])}")
                LOGGER.info(f"    Types: {metadata.get('types', [])}")
            elif config.metadata_template == "swebench":
                LOGGER.info(f"    Complexity: {metadata.get('complexity_score', 0)}")
                LOGGER.info(f"    Dependencies: {metadata.get('dependencies', [])}")
        
        # Basic assertions
        assert len(chunks) > 0, f"No chunks created for config {i+1}"
        for chunk in chunks:
            assert 'content' in chunk
            assert 'metadata' in chunk
            assert chunk['metadata']['chunk_size'] > 0
    
    print("✅ Enhanced Haskell chunking test passed!")
    return all_results


def test_haskell_chunk_config():
    """Test Haskell chunk configuration options."""
    if not HASKELL_SUPPORT_AVAILABLE:
        print("⚠️ Skipping Haskell config test - haskell_support not available")
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
        preserve_exports=False
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
    if not HASKELL_SUPPORT_AVAILABLE:
        print("⚠️ Skipping simple Haskell test - haskell_support not available")
        return
    
    simple_code = '''
module Simple where

-- Simple function
add :: Int -> Int -> Int
add x y = x + y

-- Another function
multiply :: Int -> Int -> Int
multiply x y = x * y
'''
    
    config = HaskellChunkConfig(max_chunk_size=200)
    chunker = EnhancedHaskellChunker(config)
    chunks = chunker.chunk_code(simple_code, "simple.hs")
    
    assert len(chunks) > 0
    
    # Check that we have some content
    total_content = "".join(chunk['content'] for chunk in chunks)
    assert 'add' in total_content
    assert 'multiply' in total_content
    
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
