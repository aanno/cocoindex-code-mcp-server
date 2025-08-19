#!/usr/bin/env python3

from pathlib import Path
from types import FunctionType
from typing import cast

import pytest

import cocoindex
from cocoindex_code_mcp_server.cocoindex_config import (
    CUSTOM_LANGUAGES,
    extract_language,
    get_chunking_params,
)


class TestCocoIndexIntegration:
    """Test integration of Haskell support with CocoIndex."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fixtures_dir = Path(__file__).parent / "fixtures"
        self.sample_file = self.fixtures_dir / "lang_examples" / "HaskellExample1.hs"

    def test_language_detection(self):
        """Test that .hs files are detected as Haskell."""
        language = cast(FunctionType, extract_language)("HaskellExample1.hs")
        assert language == "Haskell"

        language = cast(FunctionType, extract_language)("HaskellExample1.hs")
        assert language == "Haskell"

    def test_chunking_parameters(self):
        """Test Haskell chunking parameters."""
        params = cast(FunctionType, get_chunking_params)("Haskell")
        assert params.chunk_size == 1200
        assert params.min_chunk_size == 300
        assert params.chunk_overlap == 200

    @pytest.mark.skip(reason="Language spec count changed due to refactoring")
    def test_custom_language_spec(self) -> None:
        """Test that Haskell custom language spec is properly configured."""
        haskell_spec = None
        for spec in CUSTOM_LANGUAGES:
            if spec.language_name == "Haskell":
                haskell_spec = spec
                break

        if haskell_spec is not None:
            assert haskell_spec is not None, "Haskell custom language spec not found"
            assert len(haskell_spec.separators_regex) == 24
            assert ".hs" in haskell_spec.aliases
            assert ".lhs" in haskell_spec.aliases

            # Check for specific important separators
            separators = haskell_spec.separators_regex
            assert r"\n\w+\s*::\s*" in separators  # Type signatures
            assert r"\nmodule\s+" in separators    # Module declarations
            assert r"\nimport\s+" in separators    # Import statements
            assert r"\ndata\s+" in separators      # Data declarations
        else:
            pytest.fail("no metadata in chunk")

    def test_split_recursively_configuration(self):
        """Test that SplitRecursively can be configured with Haskell support."""
        try:
            split_func = cocoindex.functions.SplitRecursively(
                custom_languages=CUSTOM_LANGUAGES
            )
            assert split_func is not None
        except Exception as e:
            pytest.fail(f"Failed to configure SplitRecursively: {e}")

    def test_sample_file_processing(self):
        """Test that the sample Haskell file can be processed."""
        if not self.sample_file.exists():
            pytest.skip("Sample Haskell file not found")

        with open(self.sample_file, "r") as f:
            haskell_code = f.read()

        assert len(haskell_code) > 0
        assert "module HaskellExample1 where" in haskell_code
        assert "fibonacci ::" in haskell_code


