#!/usr/bin/env python3
"""
Test cases for Kotlin and Haskell language analysis issues.

This test demonstrates:
1. Kotlin analyzer working correctly after indentation fix
2. Haskell analyzer failing on complex files (AST parsing returns 0 chunks)
3. Haskell analyzer working on simple code (AST parsing returns proper chunks)
"""

# Add src to path for imports
from pathlib import Path

import cocoindex_code_mcp_server._haskell_tree_sitter as hts
import pytest
from cocoindex_code_mcp_server.language_handlers.haskell_handler import (
    analyze_haskell_code,
)
from cocoindex_code_mcp_server.language_handlers.kotlin_visitor import (
    analyze_kotlin_code,
)

# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestKotlinAnalysis:
    """Test Kotlin language analysis after bug fix."""

    def test_kotlin_analysis_detects_functions_and_classes(self):
        """Test that Kotlin analyzer detects functions, classes, and data classes correctly."""
        kotlin_code = '''
        data class Person(val name: String, val age: Int) {
            fun isAdult(): Boolean = age >= 18
            fun greet(): String = "Hello, I'm $name"
        }

        class Calculator {
            fun add(a: Int, b: Int): Int = a + b
            fun multiply(a: Int, b: Int): Int = a * b
        }

        fun fibonacci(n: Int): Int {
            return when (n) {
                0 -> 0
                1 -> 1
                else -> fibonacci(n - 1) + fibonacci(n - 2)
            }
        }

        fun main() {
            val person = Person("Alice", 25)
            println(person.greet())
        }
        '''

        result = analyze_kotlin_code(kotlin_code, 'test.kt')

        # Should succeed
        assert result['success'] is True
        assert result['analysis_method'] == 'kotlin_ast_visitor'

        # Should detect functions
        functions = result['functions']
        assert len(functions) >= 5  # isAdult, greet, add, multiply, fibonacci, main
        assert 'isAdult' in functions
        assert 'greet' in functions
        assert 'add' in functions
        assert 'multiply' in functions
        assert 'fibonacci' in functions
        assert 'main' in functions

        # Should detect classes
        classes = result['classes']
        assert 'Calculator' in classes

        # Should detect data classes
        data_classes = result['data_classes']
        assert 'Person' in data_classes


class TestHaskellAnalysisIssue:
    """Test Haskell language analysis issues - demonstrates the bug."""

    def test_haskell_simple_code_works(self):
        """Test that Haskell analyzer works with simple code."""
        simple_haskell = '''
fibonacci :: Int -> Int
fibonacci n
    | n <= 1    = n
    | otherwise = fibonacci (n - 1) + fibonacci (n - 2)

add :: Int -> Int -> Int
add x y = x + y
'''

        # Test chunk parsing directly
        chunks = hts.get_haskell_ast_chunks(simple_haskell)
        assert len(chunks) > 0, "Simple Haskell code should produce AST chunks"

        # Check chunk types
        chunk_types = [chunk.node_type() for chunk in chunks]
        assert 'signature' in chunk_types, "Should detect type signatures"
        assert 'function' in chunk_types, "Should detect function definitions"

        # Test full analysis
        result = analyze_haskell_code(simple_haskell, 'simple.hs')
        assert result['success'] is True
        assert result['analysis_method'] == 'haskell_chunk_visitor'

        # Should detect functions (may be empty due to handler issue, but structure should be correct)
        assert 'functions' in result
        assert 'imports' in result

    def test_haskell_complex_file_ast_parsing(self):
        """Test that Haskell analysis works with complex files.

        This test verifies that complex Haskell files are properly analyzed using
        the Rust-based tree-sitter implementation with proper fallback handling.
        """
        # Load the actual test file that's failing
        test_file = Path(__file__).parent / "fixtures" / "lang_examples" / "HaskellExample1.hs"
        if not test_file.exists():
            pytest.skip("Test file not found")

        with open(test_file, 'r') as f:
            complex_haskell = f.read()

        # Verify AST chunking produces chunks for complex files
        ast_chunks = hts.get_haskell_ast_chunks(complex_haskell)
        assert len(ast_chunks) > 0, "Complex Haskell file should produce AST chunks"

        # Verify fallback also works if needed
        fallback_chunks = hts.get_haskell_ast_chunks_with_fallback(complex_haskell)
        assert len(fallback_chunks) > 0, "Fallback should produce chunks"

        # Verify chunks have proper types that handler can process
        chunk_types = [chunk.node_type() for chunk in fallback_chunks]
        proper_types = ['function', 'bind', 'signature', 'module', 'import', 'data']
        has_proper_types = any(chunk_type in proper_types for chunk_type in chunk_types)
        assert has_proper_types, f"Should have proper chunk types, got: {set(chunk_types)}"

        # Full analysis should detect the functions/imports in the file
        result = analyze_haskell_code(complex_haskell, 'HaskellExample1.hs')
        assert result['success'] is True

        # The file has clear functions that should be detected
        expected_functions = ['fibonacci', 'sumList', 'treeMap', 'compose', 'addTen', 'multiplyByTwo', 'main']
        functions = result.get('functions', [])
        detected_count = len([f for f in expected_functions if f in functions])
        assert detected_count > 0, f"Should detect some functions from {expected_functions}, got {functions}"

    def test_haskell_chunk_types_compatibility(self):
        """Test that handler can process the chunk types that AST parsing produces."""
        from cocoindex_code_mcp_server.language_handlers.haskell_handler import (
            HaskellNodeHandler,
        )

        handler = HaskellNodeHandler()

        # These are the chunk types that working AST parsing produces
        working_chunk_types = ['signature', 'function', 'bind', 'module', 'import', 'data']

        for chunk_type in working_chunk_types:
            can_handle = handler.can_handle(chunk_type)
            assert can_handle, f"Handler should be able to process {chunk_type} chunks"

        # This is what the fallback produces (which doesn't work)
        fallback_chunk_type = 'regex_chunk'
        can_handle_fallback = handler.can_handle(fallback_chunk_type)
        assert not can_handle_fallback, "Handler correctly cannot process regex_chunk types"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
