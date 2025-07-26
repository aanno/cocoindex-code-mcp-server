#!/usr/bin/env python3

import unittest
import haskell_tree_sitter


class TestHaskellParsing(unittest.TestCase):
    """Test the haskell-tree-sitter extension functionality."""

    def setUp(self):
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
        tree = haskell_tree_sitter.parse_haskell(self.sample_haskell_code)
        self.assertIsNotNone(tree, "Parsing should return a tree")

        root = tree.root_node()
        self.assertEqual(root.kind(), "haskell")
        self.assertEqual(root.start_position(), (0, 0))
        self.assertGreater(root.child_count(), 0)
        self.assertTrue(root.is_named())
        self.assertFalse(root.is_error())

    def test_empty_code_parsing(self):
        """Test parsing empty code."""
        tree = haskell_tree_sitter.parse_haskell("")
        self.assertIsNotNone(tree)

        root = tree.root_node()
        self.assertEqual(root.kind(), "haskell")
        self.assertEqual(root.child_count(), 0)

    def test_invalid_code_parsing(self):
        """Test parsing invalid Haskell code."""
        invalid_code = "module Main where\n  invalid syntax here @#$%"
        tree = haskell_tree_sitter.parse_haskell(invalid_code)
        self.assertIsNotNone(tree, "Should still return a tree even for invalid code")

        # Tree-sitter should handle errors gracefully
        root = tree.root_node()
        self.assertEqual(root.kind(), "haskell")

    def test_separator_patterns(self):
        """Test that expected separator patterns are returned."""
        separators = haskell_tree_sitter.get_haskell_separators()
        self.assertEqual(len(separators), 11)

        # Check for specific important separators
        self.assertIn(r"\n\w+\s*::\s*", separators)  # Type signatures
        self.assertIn(r"\n\w+.*=\s*", separators)    # Function definitions
        self.assertIn(r"\nmodule\s+", separators)    # Module declarations
        self.assertIn(r"\nimport\s+", separators)    # Import statements
        self.assertIn(r"\ndata\s+", separators)      # Data declarations
        self.assertIn(r"\nnewtype\s+", separators)   # Newtype declarations
        self.assertIn(r"\ntype\s+", separators)      # Type aliases
        self.assertIn(r"\nclass\s+", separators)     # Type classes
        self.assertIn(r"\ninstance\s+", separators)  # Type class instances
        self.assertIn(r"\n\n+", separators)          # Paragraph breaks
        self.assertIn(r"\n", separators)             # Line breaks

    def test_parser_creation(self):
        """Test that HaskellParser can be created and used."""
        parser = haskell_tree_sitter.HaskellParser()
        self.assertIsNotNone(parser)

        tree = parser.parse(self.sample_haskell_code)
        self.assertIsNotNone(tree)

        root = tree.root_node()
        self.assertEqual(root.kind(), "haskell")

    def test_node_properties(self):
        """Test that parsed nodes have expected properties."""
        tree = haskell_tree_sitter.parse_haskell(self.sample_haskell_code)
        root = tree.root_node()

        # Test node position and byte information
        self.assertIsInstance(root.start_byte(), int)
        self.assertIsInstance(root.end_byte(), int)
        self.assertLessEqual(root.start_byte(), root.end_byte())

        # Test position tuples
        start_pos = root.start_position()
        end_pos = root.end_position()
        self.assertIsInstance(start_pos, tuple)
        self.assertIsInstance(end_pos, tuple)
        self.assertEqual(len(start_pos), 2)
        self.assertEqual(len(end_pos), 2)

        # Test boolean properties
        self.assertIsInstance(root.is_named(), bool)
        self.assertIsInstance(root.is_error(), bool)
        self.assertIsInstance(root.child_count(), int)


if __name__ == "__main__":
    unittest.main()
