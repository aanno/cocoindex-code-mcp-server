#!/usr/bin/env python3

"""
Direct test of Rust Haskell chunker to check chunking method names.
This bypasses all the CocoIndex flow and directly calls the Rust functions.
"""


import pytest

# Add src to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import cocoindex_code_mcp_server._haskell_tree_sitter as hts
    HASKELL_AVAILABLE = True
except ImportError as e:
    print(f"Haskell tree sitter not available: {e}")
    HASKELL_AVAILABLE = False


@pytest.mark.skipif(not HASKELL_AVAILABLE, reason="haskell_tree_sitter not available")
def test_rust_haskell_chunker_method_names():
    """Test direct Rust Haskell chunker to see what chunking method names it returns."""

    # Simple Haskell code for testing
    haskell_code = '''
module Test where

fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n-1) + fibonacci (n-2)

main :: IO ()
main = print (fibonacci 10)
'''

    print("\n🧪 Testing direct Rust Haskell chunker...")
    print(f"   Code length: {len(haskell_code)} chars")

    # Call the Rust chunker directly
    try:
        chunks = hts.get_haskell_ast_chunks(haskell_code)
        print(f"   Generated {len(chunks)} chunks")

        # Analyze the chunks and their metadata
        chunking_methods = set()
        for i, chunk in enumerate(chunks):
            print(f"\n   Chunk {i + 1}:")
            print(f"     Text length: {len(chunk.text())}")
            print(f"     Node type: {chunk.node_type()}")
            print(f"     Start line: {chunk.start_line()}")
            print(f"     End line: {chunk.end_line()}")

            # Get metadata dictionary
            metadata = chunk.metadata()
            print(f"     Metadata keys: {list(metadata.keys())}")

            chunking_method = metadata.get('chunking_method', 'MISSING')
            chunking_methods.add(chunking_method)
            print(f"     Chunking method: '{chunking_method}'")

            # Print all metadata for debugging
            print(f"     Full metadata: {metadata}")

        print("\n📊 Summary:")
        print(f"   All chunking methods found: {sorted(chunking_methods)}")

        # Check if we see the updated Rust names
        expected_rust_methods = {'rust_haskell_ast_recursive', 'rust_haskell_ast_recursive_with_errors'}
        found_rust_methods = chunking_methods.intersection(expected_rust_methods)
        old_methods = {'ast_recursive', 'ast_recursive_with_errors'}
        found_old_methods = chunking_methods.intersection(old_methods)

        print(f"   Expected rust methods found: {found_rust_methods}")
        print(f"   Old methods still found: {found_old_methods}")

        # Assertions
        assert len(chunks) > 0, "Should generate at least one chunk"

        # This will tell us what we're actually getting vs what we expect
        if found_rust_methods:
            print("✅ Found new Rust method names!")
        elif found_old_methods:
            print("❌ Still finding old method names - Rust update didn't take effect")
            # Don't fail the test yet, just report what we're seeing
        else:
            print(f"⚠️  Found unexpected methods: {chunking_methods}")

        return chunks, chunking_methods

    except Exception as e:
        print(f"❌ Error calling Rust chunker: {e}")
        raise


@pytest.mark.skipif(not HASKELL_AVAILABLE, reason="haskell_tree_sitter not available")
def test_rust_haskell_chunker_with_errors():
    """Test Rust Haskell chunker with code that has syntax errors."""

    # Haskell code with syntax errors
    buggy_haskell_code = '''
module Test where

fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n-1) + fibonacci (n-2  -- missing closing paren

main :: IO (
main = print (fibonacci 10)
'''

    print("\n🧪 Testing Rust Haskell chunker with errors...")

    try:
        chunks = hts.get_haskell_ast_chunks(buggy_haskell_code)
        print(f"   Generated {len(chunks)} chunks")

        # Look for error-related chunking methods
        error_methods = set()
        for chunk in chunks:
            metadata = chunk.metadata()
            method = metadata.get('chunking_method', '')
            if 'error' in method.lower():
                error_methods.add(method)

        print(f"   Error-related methods: {error_methods}")

        # Check if we get the updated error method names
        expected_error_methods = {'rust_haskell_ast_recursive_with_errors'}
        old_error_methods = {'ast_recursive_with_errors'}

        found_new_error = error_methods.intersection(expected_error_methods)
        found_old_error = error_methods.intersection(old_error_methods)

        print(f"   New error methods found: {found_new_error}")
        print(f"   Old error methods found: {found_old_error}")

        return chunks, error_methods

    except Exception as e:
        print(f"❌ Error calling Rust chunker with errors: {e}")
        raise


if __name__ == "__main__":
    if HASKELL_AVAILABLE:
        print("🚀 Running direct Rust Haskell tests...")

        try:
            chunks1, methods1 = test_rust_haskell_chunker_method_names()
            chunks2, methods2 = test_rust_haskell_chunker_with_errors()

            print("\n🎯 Final Results:")
            print(f"   Normal code methods: {sorted(methods1)}")
            print(f"   Error code methods: {sorted(methods2)}")
            print(f"   Total unique methods: {sorted(methods1.union(methods2))}")

        except Exception as e:
            print(f"❌ Test failed: {e}")
            raise
    else:
        print("❌ Haskell tree sitter not available - skipping tests")
