#!/usr/bin/env python3
"""Haskell baseline comparison test to verify implementation works."""

from typing import Set

import pytest

from cocoindex_code_mcp_server.ast_visitor import analyze_code
from cocoindex_code_mcp_server.language_handlers.haskell_handler import (
    analyze_haskell_code,
)


@pytest.mark.haskell
@pytest.mark.unit
class TestHaskellBaseline:
    """Test Haskell analysis with baseline comparison."""

    @pytest.fixture
    def haskell_code(self):
        """Test Haskell code fixture."""
        return '''-- Test Haskell file for fallback embedding verification.
-- This should use sentence-transformers/all-MiniLM-L6-v2 model.

module TestHaskell where

-- Simple data types
data Person = Person
    { personName :: String
    , personAge  :: Int
    } deriving (Show, Eq)

data Tree a = Leaf a | Branch (Tree a) (Tree a)
    deriving (Show, Eq)

-- Higher-order functions
fibonacci :: Int -> Int
fibonacci n
    | n <= 1    = n
    | otherwise = fibonacci (n - 1) + fibonacci (n - 2)

-- List processing with pattern matching
sumList :: [Int] -> Int
sumList []     = 0
sumList (x:xs) = x + sumList xs

-- Tree operations
treeMap :: (a -> b) -> Tree a -> Tree b
treeMap f (Leaf x)     = Leaf (f x)
treeMap f (Branch l r) = Branch (treeMap f l) (treeMap f r)

-- Function composition and currying
compose :: (b -> c) -> (a -> b) -> (a -> c)
compose f g x = f (g x)

addTen :: Int -> Int
addTen = (+) 10

multiplyByTwo :: Int -> Int
multiplyByTwo = (*) 2

-- Example usage
main :: IO ()
main = do
    let person = Person "Alice" 30
    putStrLn $ "Person: " ++ show person

    let numbers = [1, 2, 3, 4, 5]
    putStrLn $ "Sum of " ++ show numbers ++ " = " ++ show (sumList numbers)

    let tree = Branch (Leaf 1) (Branch (Leaf 2) (Leaf 3))
    let doubledTree = treeMap multiplyByTwo tree
    putStrLn $ "Original tree: " ++ show tree
    putStrLn $ "Doubled tree: " ++ show doubledTree

    let processNumber = compose addTen multiplyByTwo
    putStrLn $ "Process 5: " ++ show (processNumber 5)

    putStrLn $ "Fibonacci 10: " ++ show (fibonacci 10)'''

    @pytest.fixture
    def expected_functions(self):
        """Expected functions to find."""
        return {'fibonacci', 'sumList', 'treeMap', 'compose', 'addTen', 'multiplyByTwo', 'main'}

    @pytest.fixture
    def expected_data_types(self):
        """Expected data types to find."""
        return {'Person', 'Tree'}

    def test_text_baseline(self, haskell_code: str, expected_functions: Set[str], expected_data_types: Set[str]):
        """Test simple text-based baseline parsing."""
        lines = haskell_code.split('\n')

        functions = set()
        data_types = set()
        modules = set()

        for line in lines:
            line = line.strip()

            if line.startswith('--'):
                continue

            if line.startswith('data '):
                parts = line.split()
                if len(parts) > 1:
                    type_name = parts[1].split('=')[0].strip()
                    if type_name:
                        data_types.add(type_name)

            elif '::' in line and not line.startswith('--'):
                func_name = line.split('::')[0].strip()
                if func_name and func_name.isidentifier():
                    functions.add(func_name)

            elif line.startswith('module '):
                parts = line.split()
                if len(parts) > 1:
                    module_name = parts[1].split()[0]
                    if module_name != 'where':
                        modules.add(module_name)

        # Verify we found the expected items
        found_functions = functions & expected_functions
        found_data_types = data_types & expected_data_types

        assert len(found_functions) >= 4, f"Expected at least 4 functions, found {found_functions}"
        assert len(found_data_types) >= 1, f"Expected at least 1 data type, found {found_data_types}"
        assert 'TestHaskell' in modules, f"Expected TestHaskell module, found {modules}"

    def test_specialized_haskell_visitor(
            self, haskell_code: str, expected_functions: Set[str], expected_data_types: Set[str]):
        """Test our specialized Haskell visitor."""
        result = analyze_haskell_code(haskell_code, "HaskellExample1.hs")

        assert result.get('success', False), f"Analysis failed: {result}"
        assert 'analysis_method' in result, "Analysis method should be reported"

        # Check functions
        functions_found = set(result.get('functions', []))
        set(result.get('data_types', []))

        # Should find at least some functions and data types
        assert len(functions_found) > 0, f"Should find some functions, got {functions_found}"

        # Check for specific expected items
        found_expected_functions = functions_found & expected_functions
        assert len(
            found_expected_functions) >= 2, f"Should find at least 2 expected functions, got {found_expected_functions}"

    def test_generic_ast_visitor(self, haskell_code: str):
        """Test generic AST visitor fallback."""
        result = analyze_code(haskell_code, 'haskell', "HaskellExample1.hs")

        assert result.get('success', False), f"Analysis failed: {result}"
        assert 'analysis_method' in result, "Analysis method should be reported"

        # Should at least not crash and return some result
        functions = result.get('functions', [])
        assert isinstance(functions, list), "Functions should be a list"

    def test_visitor_comparison(self, haskell_code: str):
        """Compare different analysis methods."""
        # Test both visitors
        specialized_result = analyze_haskell_code(haskell_code, "test.hs")
        generic_result = analyze_code(haskell_code, 'haskell', "test.hs")

        # Both should succeed
        assert specialized_result.get('success', False), "Specialized visitor should succeed"
        assert generic_result.get('success', False), "Generic visitor should succeed"

        # Compare function counts
        specialized_functions = len(specialized_result.get('functions', []))
        generic_functions = len(generic_result.get('functions', []))

        # At least one should find functions
        assert specialized_functions > 0 or generic_functions > 0, "At least one visitor should find functions"
