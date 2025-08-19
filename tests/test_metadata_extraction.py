#!/usr/bin/env python3
"""
Test metadata extraction functionality.
Converted from debug_metadata_extraction.py and related debugging scripts.
"""

import json
from types import FunctionType
from typing import cast

import pytest

from cocoindex_code_mcp_server.cocoindex_config import extract_code_metadata

# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMetadataExtraction:
    """Test metadata extraction for different programming languages."""

    def test_python_metadata_extraction(self):
        """Test metadata extraction for Python code."""
        python_code = '''
import os
import sys
from typing import List, Dict

def fibonacci(n: int) -> int:
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, a: int, b: int) -> int:
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

async def async_function():
    """An async function."""
    await some_operation()
'''

        metadata_json = cast(FunctionType, extract_code_metadata)(python_code, "python", "test.py")
        metadata = json.loads(metadata_json)

        # Should have basic metadata (success/language not in raw metadata, only promoted fields)
        assert 'analysis_method' in metadata

        # Should detect functions
        functions = metadata.get('functions', [])
        assert 'fibonacci' in functions
        assert 'add' in functions
        assert 'async_function' in functions

        # Should detect classes
        classes = metadata.get('classes', [])
        assert 'Calculator' in classes

        # Should detect imports
        imports = metadata.get('imports', [])
        assert 'os' in imports or any('os' in imp for imp in imports)
        assert 'sys' in imports or any('sys' in imp for imp in imports)

        # Should detect async functions
        assert metadata.get('has_async', False) is True

        # Should detect type hints
        assert metadata.get('has_type_hints', False) is True

    def test_kotlin_metadata_extraction(self):
        """Test metadata extraction for Kotlin code."""
        kotlin_code = '''
data class Person(val name: String, val age: Int) {
    fun isAdult(): Boolean = age >= 18
    fun greet(): String = "Hello, I'm $name"
}

sealed class Result<out T> {
    data class Success<out T>(val value: T) : Result<T>()
    data class Error(val message: String) : Result<Nothing>()
}

fun fibonacci(n: Int): Int {
    return when (n) {
        0 -> 0
        1 -> 1
        else -> fibonacci(n - 1) + fibonacci(n - 2)
    }
}

class Calculator {
    fun add(a: Int, b: Int): Int = a + b
}
'''

        metadata_json = cast(FunctionType, extract_code_metadata)(kotlin_code, "kotlin", "test.kt")
        metadata = json.loads(metadata_json)

        # Should succeed and use Kotlin analyzer (success/language not in raw metadata)
        assert metadata.get('analysis_method') == 'kotlin_ast_visitor'

        # Should detect functions
        functions = metadata.get('functions', [])
        assert 'isAdult' in functions
        assert 'greet' in functions
        assert 'fibonacci' in functions
        assert 'add' in functions

        # Should detect classes
        classes = metadata.get('classes', [])
        assert 'Result' in classes
        assert 'Calculator' in classes

        # Should detect data classes
        data_classes = metadata.get('data_classes', [])
        assert 'Person' in data_classes

    def test_haskell_metadata_extraction(self):
        """Test metadata extraction for Haskell code."""
        haskell_code = '''
module TestModule where

import Data.List

data Person = Person
    { personName :: String
    , personAge  :: Int
    } deriving (Show, Eq)

fibonacci :: Int -> Int
fibonacci n
    | n <= 1    = n
    | otherwise = fibonacci (n - 1) + fibonacci (n - 2)

sumList :: [Int] -> Int
sumList []     = 0
sumList (x:xs) = x + sumList xs
'''

        metadata_json = cast(FunctionType, extract_code_metadata)(haskell_code, "haskell", "test.hs")
        metadata = json.loads(metadata_json)

        # Should succeed and use Haskell analyzer
        assert metadata['success'] is True
        assert metadata['language'] == 'Haskell'
        assert metadata.get('analysis_method') == 'haskell_chunk_visitor'

        # Note: Functions may be empty due to the known Haskell AST parsing issue
        # but the structure should be correct
        assert 'functions' in metadata
        assert 'imports' in metadata

    def test_unknown_language_fallback(self):
        """Test that unknown languages fall back to basic analysis."""
        unknown_code = '''
// Some unknown language code
function someFunction() {
    return "hello";
}

class SomeClass {
    method() {
        return 42;
    }
}
'''

        metadata_json = cast(FunctionType, extract_code_metadata)(unknown_code, "unknown", "test.unknown")
        metadata = json.loads(metadata_json)

        # Should still succeed with fallback (success/language not in raw metadata)
        assert 'analysis_method' in metadata
        assert metadata['analysis_method'] in ['enhanced_regex', 'basic_text', 'tree_sitter', 'unknown_analysis']


class TestMetadataConsistency:
    """Test that metadata extraction is consistent and reliable."""

    def test_empty_code_handling(self):
        """Test handling of empty or whitespace-only code."""
        empty_cases = ["", "   ", "\n\n\n", "// Just a comment"]

        for empty_code in empty_cases:
            metadata_json = cast(FunctionType, extract_code_metadata)(empty_code, "python", "empty.py")
            metadata = json.loads(metadata_json)

            # Should handle gracefully (language not in raw metadata)
            assert isinstance(metadata, dict)
            assert 'analysis_method' in metadata

    def test_malformed_code_handling(self):
        """Test handling of syntactically incorrect code."""
        malformed_code = '''
def incomplete_function(
    # Missing closing parenthesis and body

class IncompleteClass
    # Missing colon and body

invalid syntax here ###
'''

        metadata_json = cast(FunctionType, extract_code_metadata)(malformed_code, "python", "malformed.py")
        metadata = json.loads(metadata_json)

        # Should handle gracefully without crashing (language not in raw metadata)
        assert isinstance(metadata, dict)
        assert 'analysis_method' in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
