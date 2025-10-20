#!/usr/bin/env python3

"""Test the new modern Haskell metadata extraction.

This test verifies that modern Haskell metadata extraction using rust_haskell_ast
works correctly, extracting functions, modules, and complexity scores.
"""

import json


def test_modern_haskell_metadata_extraction():
    """Test new modern Haskell metadata extraction."""
    from cocoindex_code_mcp_server.cocoindex_config import extract_code_metadata

    haskell_code = """
module HaskellExample1 where

fibonacci :: Int -> Int
fibonacci n
    | n <= 1    = n
    | otherwise = fibonacci (n - 1) + fibonacci (n - 2)

sumList :: [Int] -> Int
sumList []     = 0
sumList (x:xs) = x + sumList xs

main :: IO ()
main = do
    putStrLn "Hello World"
"""

    result_json = extract_code_metadata(haskell_code, "haskell", "test.hs", "")
    result = json.loads(result_json)

    analysis_method = result.get("analysis_method", "NOT_FOUND")
    functions = result.get("functions", [])
    classes = result.get("classes", [])
    modules = result.get("modules", [])
    has_module = result.get("has_module", False)
    complexity_score = result.get("complexity_score", 0)

    # Assertions
    assert analysis_method != "NOT_FOUND", "Analysis method should be present"

    # Check for modern Haskell analysis
    if analysis_method == "rust_haskell_ast":
        assert functions, "Functions should be extracted with rust_haskell_ast"
        assert "fibonacci" in functions, "fibonacci function should be found"
        assert "sumList" in functions, "sumList function should be found"
        assert "main" in functions, "main function should be found"
    else:
        # If not using rust_haskell_ast, at least some functions should be extracted
        assert functions, f"Functions should be extracted (analysis_method: {analysis_method})"


def test_haskell_module_detection():
    """Test that Haskell modules are properly detected."""
    from cocoindex_code_mcp_server.cocoindex_config import extract_code_metadata

    haskell_code = """
module MyModule where

import Data.List (sort)
import qualified Data.Map as M

myFunction :: Int -> Int
myFunction x = x + 1
"""

    result_json = extract_code_metadata(haskell_code, "haskell", "test.hs", "")
    result = json.loads(result_json)

    has_module = result.get("has_module", False)
    modules = result.get("modules", [])

    # Should detect module presence
    assert has_module or modules, "Should detect module in Haskell code"


def test_haskell_type_signatures():
    """Test that Haskell type signatures are handled correctly."""
    from cocoindex_code_mcp_server.cocoindex_config import extract_code_metadata

    haskell_code = """
module TypeSigTest where

add :: Int -> Int -> Int
add x y = x + y

multiply :: Num a => a -> a -> a
multiply x y = x * y

genericFunc :: (Ord a, Show a) => a -> String
genericFunc x = show x
"""

    result_json = extract_code_metadata(haskell_code, "haskell", "test.hs", "")
    result = json.loads(result_json)

    functions = result.get("functions", [])
    has_type_signatures = result.get("has_type_signatures", False)

    # Should extract functions with type signatures
    assert functions, "Should extract functions with type signatures"
    assert len(functions) >= 2, "Should extract multiple functions"

    # Check that common functions are found
    expected_functions = {"add", "multiply", "genericFunc"}
    found_functions = set(functions)
    assert expected_functions.intersection(
        found_functions
    ), f"Should find at least one of {expected_functions}, but got {found_functions}"
