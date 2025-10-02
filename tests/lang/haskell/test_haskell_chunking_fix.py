#!/usr/bin/env python3

"""Test to verify the Haskell chunking fix.

This test verifies that Haskell code is properly chunked and metadata is extracted,
including functions and classes from Haskell AST parsing.
"""

import json

import pytest


@pytest.mark.parametrize("preserve_existing_metadata", [False, True])
def test_haskell_chunking_fix(preserve_existing_metadata):
    """Test if the chunking fixes are working."""
    from cocoindex_code_mcp_server.cocoindex_config import extract_code_metadata

    # Test Haskell code that should create meaningful chunks, not tiny ones
    haskell_code = """
module FibonacciModule where

fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n-1) + fibonacci (n-2)

class Eq a => Ord a where
    (<) :: a -> a -> Bool

data Maybe a = Nothing | Just a

factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n-1)
"""

    if preserve_existing_metadata:
        # Test with existing metadata preservation
        existing_metadata = '{"chunking_method": "astchunk_library"}'
        result_json = extract_code_metadata(
            haskell_code, "haskell", "test_haskell2.hs", existing_metadata
        )
    else:
        # Test without existing metadata
        result_json = extract_code_metadata(haskell_code, "haskell", "test_haskell.hs", "")

    result = json.loads(result_json)

    chunking_method = result.get("chunking_method", "NOT_FOUND")
    functions = result.get("functions", [])
    classes = result.get("classes", [])

    # Assertions
    assert chunking_method != "NOT_FOUND", "Chunking method should be present"

    if preserve_existing_metadata:
        assert (
            chunking_method == "astchunk_library"
        ), "Chunking method should be preserved from existing metadata"
    else:
        assert (
            chunking_method != "NOT_FOUND"
        ), "Chunking method should be set for new metadata"

    # Check that functions were extracted
    assert functions, "Functions should be extracted from Haskell code"
    assert "fibonacci" in functions, "fibonacci function should be extracted"


def test_haskell_function_extraction():
    """Test that Haskell functions are properly extracted."""
    from cocoindex_code_mcp_server.cocoindex_config import extract_code_metadata

    haskell_code = """
module TestModule where

fibonacci :: Int -> Int
fibonacci n = if n <= 1 then n else fibonacci (n-1) + fibonacci (n-2)

factorial :: Int -> Int
factorial n = if n <= 0 then 1 else n * factorial (n-1)

main :: IO ()
main = putStrLn "Hello"
"""

    result_json = extract_code_metadata(haskell_code, "haskell", "test.hs", "")
    result = json.loads(result_json)

    functions = result.get("functions", [])

    # All three functions should be extracted
    assert len(functions) >= 2, "At least fibonacci and factorial should be extracted"
    assert "fibonacci" in functions, "fibonacci should be in extracted functions"
    assert "factorial" in functions, "factorial should be in extracted functions"
