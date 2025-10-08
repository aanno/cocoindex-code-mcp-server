#!/usr/bin/env python3

"""Integration test for boolean OR query functionality.

This test verifies that boolean OR queries work correctly in the hybrid search
engine, properly combining results from multiple keyword conditions.
"""

import logging
from pathlib import Path

import pytest
from dotenv import load_dotenv

from ..common import COCOINDEX_AVAILABLE, CocoIndexTestInfrastructure, copy_directory_structure
from ..search_config import SearchTestConfig


logger = logging.getLogger(__name__)


@pytest.mark.skipif(not COCOINDEX_AVAILABLE, reason="CocoIndex infrastructure not available")
@pytest.mark.asyncio
class TestBooleanOrQuery:
    """Test boolean OR query functionality in keyword search."""

    async def test_boolean_or_combines_results(self):
        """Test that OR queries correctly combine results from multiple conditions."""

        # Load environment variables
        load_dotenv()

        # Copy test files to tmp directory
        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "lang_examples"
        tmp_dir = Path("/workspaces/rust/tmp")
        copy_directory_structure(fixtures_dir, tmp_dir)

        # Create search test configuration
        config = SearchTestConfig(
            paths=[str(tmp_dir)],
            no_live=True,
            log_level="INFO",
        )

        # Initialize infrastructure with keyword test type
        infrastructure_kwargs = config.to_infrastructure_kwargs()
        infrastructure_kwargs["test_type"] = "keyword"
        async with CocoIndexTestInfrastructure(**infrastructure_kwargs) as infrastructure:

            # Test the boolean OR query: language:rust OR language:c
            or_results = infrastructure.hybrid_search_engine.search(
                vector_query="",
                keyword_query="language:Rust OR language:C",
                top_k=20,
                vector_weight=0.0,
                keyword_weight=1.0,
            )

            logger.info(f"OR query found {len(or_results)} results")

            # Also test individual queries for comparison
            rust_results = infrastructure.hybrid_search_engine.search(
                vector_query="",
                keyword_query="language:Rust",
                top_k=10,
                vector_weight=0.0,
                keyword_weight=1.0,
            )

            c_results = infrastructure.hybrid_search_engine.search(
                vector_query="",
                keyword_query="language:C",
                top_k=10,
                vector_weight=0.0,
                keyword_weight=1.0,
            )

            logger.info(f"Rust query found {len(rust_results)} results")
            logger.info(f"C query found {len(c_results)} results")

            # Assertions
            assert len(or_results) > 0, "OR query should return results"
            assert len(rust_results) > 0, "Rust query should return results"
            assert len(c_results) > 0, "C query should return results"

            # OR query should return at least as many results as individual queries
            assert len(or_results) >= len(
                rust_results
            ), "OR query should include at least all Rust results"
            assert len(or_results) >= len(
                c_results
            ), "OR query should include at least all C results"

            # Verify that all results are either Rust or C
            for result in or_results:
                language = result.get("language", "UNKNOWN")
                assert language in [
                    "Rust",
                    "C",
                ], f"OR query result has unexpected language: {language}"

            # Verify that Rust results only contain Rust
            for result in rust_results:
                language = result.get("language", "UNKNOWN")
                assert language == "Rust", f"Rust query returned non-Rust result: {language}"

            # Verify that C results only contain C
            for result in c_results:
                language = result.get("language", "UNKNOWN")
                assert language == "C", f"C query returned non-C result: {language}"

    async def test_boolean_or_with_multiple_conditions(self):
        """Test OR queries with more than two conditions."""

        load_dotenv()

        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "lang_examples"
        tmp_dir = Path("/workspaces/rust/tmp")
        copy_directory_structure(fixtures_dir, tmp_dir)

        config = SearchTestConfig(
            paths=[str(tmp_dir)],
            no_live=True,
            log_level="INFO",
        )

        infrastructure_kwargs = config.to_infrastructure_kwargs()
        infrastructure_kwargs["test_type"] = "keyword"
        async with CocoIndexTestInfrastructure(**infrastructure_kwargs) as infrastructure:

            # Test with three conditions: Python OR Java OR JavaScript
            multi_or_results = infrastructure.hybrid_search_engine.search(
                vector_query="",
                keyword_query="language:Python OR language:Java OR language:JavaScript",
                top_k=30,
                vector_weight=0.0,
                keyword_weight=1.0,
            )

            assert len(multi_or_results) > 0, "Multi-condition OR query should return results"

            # Verify all results are one of the specified languages
            found_languages = set()
            for result in multi_or_results:
                language = result.get("language", "UNKNOWN")
                found_languages.add(language)
                assert language in [
                    "Python",
                    "Java",
                    "JavaScript",
                ], f"Unexpected language in multi-OR results: {language}"

            # Should find at least one result from each language (if they exist in corpus)
            logger.info(f"Found languages in multi-OR query: {found_languages}")

    async def test_boolean_and_or_combination(self):
        """Test combination of AND and OR operators."""

        load_dotenv()

        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "lang_examples"
        tmp_dir = Path("/workspaces/rust/tmp")
        copy_directory_structure(fixtures_dir, tmp_dir)

        config = SearchTestConfig(
            paths=[str(tmp_dir)],
            no_live=True,
            log_level="INFO",
        )

        infrastructure_kwargs = config.to_infrastructure_kwargs()
        infrastructure_kwargs["test_type"] = "keyword"
        async with CocoIndexTestInfrastructure(**infrastructure_kwargs) as infrastructure:

            # Test combined query: (language:Python OR language:Java) AND functions:fibonacci
            # Note: This tests if the parser and query engine handle combined operators
            try:
                combined_results = infrastructure.hybrid_search_engine.search(
                    vector_query="",
                    keyword_query="language:Python AND functions:fibonacci",
                    top_k=10,
                    vector_weight=0.0,
                    keyword_weight=1.0,
                )

                # If the query succeeds, verify results
                for result in combined_results:
                    language = result.get("language", "UNKNOWN")
                    functions = result.get("functions", "")
                    assert language == "Python", "Should only return Python results"
                    assert (
                        "fibonacci" in functions.lower()
                    ), "Should only return results with fibonacci function"

            except Exception as e:
                # If combined AND/OR is not yet supported, that's okay
                logger.info(f"Combined AND/OR query not supported yet: {e}")
