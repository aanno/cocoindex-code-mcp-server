#!/usr/bin/env python3

from typing import Optional

import haskell_tree_sitter
import pytest


class TestHaskellParsing:
    """Test the haskell-tree-sitter extension functionality."""

    def setup_method(self):
        """Setup test fixtures for each test method."""
        self.sample_haskell_code = '''
module Main where

import Data.List

factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

main :: IO ()
main = do
    putStrLn "Hello, Haskell!"
    print (factorial 5)
'''

    def test_basic_parsing(self):
        """Test that basic Haskell code can be parsed."""
        tree: Optional[haskell_tree_sitter.HaskellTree] = haskell_tree_sitter.parse_haskell(self.sample_haskell_code)
        assert tree is not None, "Parsing should return a tree"

        if tree is not None:
            root = tree.root_node()
            assert root.kind() == "haskell"
            assert root.start_position() == (0, 0)
            assert root.child_count() > 0
            assert root.is_named() == True
            assert root.is_error() == False
        else:
            pytest.fail("no metadata in chunk")

    def test_empty_code_parsing(self) -> None:
        """Test parsing empty code."""
        tree: Optional[haskell_tree_sitter.HaskellTree] = haskell_tree_sitter.parse_haskell("")
        assert tree is not None

        if tree is not None:
            root = tree.root_node()
            assert root.kind() == "haskell"
            assert root.child_count() == 0
        else:
            pytest.fail("no metadata in chunk")

    def test_invalid_code_parsing(self) -> None:
        """Test parsing invalid Haskell code."""
        invalid_code = "module Main where\n  invalid syntax here @#$%"
        tree: Optional[haskell_tree_sitter.HaskellTree] = haskell_tree_sitter.parse_haskell(invalid_code)
        assert tree is not None, "Should still return a tree even for invalid code"

        if tree is not None:
            # Tree-sitter should handle errors gracefully
            root = tree.root_node()
            assert root.kind() == "haskell"
        else:
            pytest.fail("no metadata in chunk")

    def test_separator_patterns(self) -> None:
        """Test that expected separator patterns are returned."""
        separators = haskell_tree_sitter.get_haskell_separators()
        assert len(separators) == 11

        # Check for specific important separators
        assert r"\n\w+\s*::\s*" in separators  # Type signatures
        assert r"\n\w+.*=\s*" in separators    # Function definitions
        assert r"\nmodule\s+" in separators    # Module declarations
        assert r"\nimport\s+" in separators    # Import statements
        assert r"\ndata\s+" in separators      # Data declarations
        assert r"\nnewtype\s+" in separators   # Newtype declarations
        assert r"\ntype\s+" in separators      # Type aliases
        assert r"\nclass\s+" in separators     # Type classes
        assert r"\ninstance\s+" in separators  # Type class instances
        assert r"\n\n+" in separators          # Paragraph breaks
        assert r"\n" in separators             # Line breaks

    def test_parser_creation(self) -> None:
        """Test that HaskellParser can be created and used."""
        parser = haskell_tree_sitter.HaskellParser()
        assert parser is not None

        tree: Optional[haskell_tree_sitter.HaskellTree] = parser.parse(self.sample_haskell_code)
        assert tree is not None

        if tree is not None:
            root = tree.root_node()
            assert root.kind() == "haskell"
        else:
            pytest.fail("no metadata in chunk")

    def test_node_properties(self) -> None:
        """Test that parsed nodes have expected properties."""
        tree: Optional[haskell_tree_sitter.HaskellTree] = haskell_tree_sitter.parse_haskell(self.sample_haskell_code)
        if tree is not None:
            root = tree.root_node()

            # Test node position and byte information
            assert isinstance(root.start_byte(), int)
            assert isinstance(root.end_byte(), int)
            assert root.start_byte() <= root.end_byte()

            # Test position tuples
            start_pos = root.start_position()
            end_pos = root.end_position()
            assert isinstance(start_pos, tuple)
            assert isinstance(end_pos, tuple)
            assert len(start_pos) == 2
            assert len(end_pos) == 2

            # Test boolean properties
            assert isinstance(root.is_named(), bool)
            assert isinstance(root.is_error(), bool)
            assert isinstance(root.child_count(), int)
        else:
            pytest.fail("no metadata in chunk")


if __name__ == "__main__":
    pytest.main()
