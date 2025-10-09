#!/usr/bin/env python3

"""
Test suite for Haskell AST visitor implementation.
Tests the integration of HaskellNodeHandler with the AST visitor framework.
"""

from pathlib import Path

import pytest

try:
    from cocoindex_code_mcp_server.ast_visitor import MultiLevelAnalyzer, analyze_code
    from cocoindex_code_mcp_server.language_handlers import (
        get_handler_for_language,
        list_supported_languages,
    )
    IMPORTS_AVAILABLE = True
    import_error = None
except ImportError as e:
    IMPORTS_AVAILABLE = False
    import_error = str(e)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports not available: {import_error}")
class TestHaskellASTVisitor:
    """Test Haskell AST visitor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MultiLevelAnalyzer()

        # Sample Haskell code for testing
        self.sample_haskell_code = '''-- Test Haskell module
module TestModule (
    Person(..),
    TreeType(..),
    fibonacci,
    processData
) where

import Data.List (sort, nub)
import qualified Data.Map as Map
import Control.Monad (when)

-- Data type definitions
data Person = Person
    { personName :: String
    , personAge  :: Int
    } deriving (Show, Eq, Ord)

data TreeType a = Leaf a | Branch (TreeType a) (TreeType a)
    deriving (Show, Eq)

newtype UserId = UserId Int
    deriving (Show, Eq)

type PersonMap = Map.Map String Person

-- Type class definition
class Processable a where
    process :: a -> String
    validate :: a -> Bool

-- Instance declaration
instance Processable Person where
    process p = "Person: " ++ personName p
    validate p = personAge p > 0

-- Function with type signature
fibonacci :: Int -> Int
fibonacci n
    | n <= 1    = n
    | otherwise = fibonacci (n - 1) + fibonacci (n - 2)

-- Function with complex pattern matching
processData :: [Person] -> TreeType Person -> IO ()
processData people tree = do
    let sorted = sort people
    when (not $ null sorted) $ do
        putStrLn "Processing people..."
        mapM_ (putStrLn . process) sorted
    case tree of
        Leaf p -> putStrLn $ "Found leaf: " ++ show p
        Branch l r -> do
            putStrLn "Processing branch..."
            processData people l
            processData people r

-- Private helper function
_helperFunction :: String -> String
_helperFunction s = "Helper: " ++ s

-- Infix operator
(&&&) :: Bool -> Bool -> Bool
(&&&) = (&&)
'''

    def test_language_handler_registration(self):
        """Test that Haskell handler is properly registered."""
        supported_languages = list_supported_languages()
        assert 'haskell' in supported_languages, f"Haskell not in supported languages: {supported_languages}"

        handler = get_handler_for_language('haskell')
        assert handler is not None, "Haskell handler should be available"
        assert hasattr(handler, 'can_handle'), "Handler should have can_handle method"
        assert hasattr(handler, 'extract_metadata'), "Handler should have extract_metadata method"

    def test_haskell_code_analysis(self):
        """Test basic Haskell code analysis using the AST visitor."""
        metadata = self.analyzer.analyze_code(
            self.sample_haskell_code,
            language='haskell',
            filename='test.hs'
        )

        # Basic metadata should be present
        assert metadata['language'] == 'Haskell'
        assert metadata['filename'] == 'test.hs'
        assert metadata['line_count'] > 0
        assert metadata['char_count'] > 0

        # Should use tree-sitter analysis if available
        assert 'tree_sitter' in metadata.get('analysis_method', '')

        print(f"Analysis method: {metadata.get('analysis_method')}")
        print(f"Available metadata keys: {list(metadata.keys())}")

    def test_haskell_handler_functionality(self):
        """Test Haskell-specific handler functionality directly."""
        handler = get_handler_for_language('haskell')
        if not handler:
            pytest.skip("Haskell handler not available")

        # Test that handler can process sample node types
        assert handler.can_handle('module'), "Should handle module declarations"
        assert handler.can_handle('import'), "Should handle import statements"
        assert handler.can_handle('function'), "Should handle function definitions"
        assert handler.can_handle('data_type'), "Should handle data type declarations"
        assert handler.can_handle('class'), "Should handle type class declarations"
        assert handler.can_handle('instance'), "Should handle instance declarations"

        # Test summary functionality
        summary = handler.get_summary()
        assert isinstance(summary, dict), "Summary should be a dictionary"

        expected_summary_keys = [
            'module', 'functions', 'data_types', 'type_classes',
            'instances', 'imports', 'complexity_score'
        ]
        for key in expected_summary_keys:
            assert key in summary, f"Summary should contain '{key}' key"

    def test_analyze_code_convenience_function(self):
        """Test the convenience analyze_code function with Haskell."""
        metadata = analyze_code(
            self.sample_haskell_code,
            language='haskell',
            filename='test.hs'
        )

        assert isinstance(metadata, dict), "Should return metadata dictionary"
        assert metadata['language'] == 'Haskell'

        # Should contain analysis results
        assert 'analysis_method' in metadata
        print(f"Convenience function analysis method: {metadata['analysis_method']}")

    def test_haskell_file_extension_detection(self):
        """Test that Haskell files are properly detected by extension."""
        # Test .hs extension
        metadata_hs = self.analyzer.analyze_code(
            self.sample_haskell_code,
            filename='module.hs'
        )
        assert metadata_hs['language'] == 'Haskell'

        # Test .lhs extension (literate Haskell)
        metadata_lhs = self.analyzer.analyze_code(
            self.sample_haskell_code,
            filename='module.lhs'
        )
        assert metadata_lhs['language'] == 'Haskell'

    def test_fallback_behavior(self):
        """Test fallback behavior when tree-sitter parsing fails."""
        # Test with malformed Haskell code
        malformed_code = "module Broken where\n import [\n data Incomplete"

        metadata = self.analyzer.analyze_code(
            malformed_code,
            language='haskell',
            filename='broken.hs'
        )

        # Should still return metadata, possibly using fallback methods
        assert isinstance(metadata, dict)
        assert metadata['language'] == 'Haskell'
        assert 'analysis_method' in metadata

        print(f"Fallback analysis method: {metadata['analysis_method']}")

    @pytest.mark.integration
    def test_real_haskell_file_analysis(self):
        """Test analysis of a real Haskell file if available."""
        # Look for the test fixture
        test_file_path = Path(__file__).parent.parent.parent / 'fixtures' / 'lang_examples' / 'HaskellExample1.hs'

        if test_file_path.exists():
            with open(test_file_path, 'r') as f:
                haskell_code = f.read()

            metadata = self.analyzer.analyze_code(
                haskell_code,
                filename=str(test_file_path)
            )

            assert metadata['language'] == 'Haskell'
            print(f"Real file analysis method: {metadata['analysis_method']}")
            print(f"Real file metadata keys: {list(metadata.keys())}")
        else:
            pytest.skip(f"Test Haskell file not found at {test_file_path}")


if __name__ == "__main__":
    # Run with verbose output for debugging
    pytest.main([__file__, "-v", "-s", "--tb=short"])
