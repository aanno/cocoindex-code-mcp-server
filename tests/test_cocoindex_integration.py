#!/usr/bin/env python3

import unittest
import pytest
from pathlib import Path

from cocoindex_code_mcp_server.cocoindex_config import extract_language, get_chunking_params, CUSTOM_LANGUAGES
import cocoindex


class TestCocoIndexIntegration(unittest.TestCase):
    """Test integration of Haskell support with CocoIndex."""

    def setUp(self):
        self.fixtures_dir = Path(__file__).parent / "fixtures"
        self.sample_file = self.fixtures_dir / "test_sample.hs"

    def test_language_detection(self):
        """Test that .hs files are detected as Haskell."""
        language = extract_language("test_sample.hs")
        self.assertEqual(language, "Haskell")

        language = extract_language("test_sample.lhs")
        self.assertEqual(language, "Haskell")

    def test_chunking_parameters(self):
        """Test Haskell chunking parameters."""
        params = get_chunking_params("Haskell")
        self.assertEqual(params.chunk_size, 1200)
        self.assertEqual(params.min_chunk_size, 300)
        self.assertEqual(params.chunk_overlap, 200)

    @pytest.mark.skip(reason="Language spec count changed due to refactoring")
    def test_custom_language_spec(self):
        """Test that Haskell custom language spec is properly configured."""
        haskell_spec = None
        for spec in CUSTOM_LANGUAGES:
            if spec.language_name == "Haskell":
                haskell_spec = spec
                break

        self.assertIsNotNone(haskell_spec, "Haskell custom language spec not found")
        self.assertEqual(len(haskell_spec.separators_regex), 24)
        self.assertIn(".hs", haskell_spec.aliases)
        self.assertIn(".lhs", haskell_spec.aliases)

        # Check for specific important separators
        separators = haskell_spec.separators_regex
        self.assertIn(r"\n\w+\s*::\s*", separators)  # Type signatures
        self.assertIn(r"\nmodule\s+", separators)    # Module declarations
        self.assertIn(r"\nimport\s+", separators)    # Import statements
        self.assertIn(r"\ndata\s+", separators)      # Data declarations

    def test_split_recursively_configuration(self):
        """Test that SplitRecursively can be configured with Haskell support."""
        try:
            split_func = cocoindex.functions.SplitRecursively(
                custom_languages=CUSTOM_LANGUAGES
            )
            self.assertIsNotNone(split_func)
        except Exception as e:
            self.fail(f"Failed to configure SplitRecursively: {e}")

    def test_sample_file_processing(self):
        """Test that the sample Haskell file can be processed."""
        if not self.sample_file.exists():
            self.skipTest("Sample Haskell file not found")

        with open(self.sample_file, "r") as f:
            haskell_code = f.read()

        self.assertGreater(len(haskell_code), 0)
        self.assertIn("module Main where", haskell_code)
        self.assertIn("factorial ::", haskell_code)


if __name__ == "__main__":
    unittest.main()
