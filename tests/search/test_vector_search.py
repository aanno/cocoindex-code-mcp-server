#!/usr/bin/env python3

"""
Vector Search Tests

This module contains tests for vector-only search functionality using CocoIndex
infrastructure directly. These tests validate semantic code search capabilities
across different programming languages.
"""

import logging
from pathlib import Path

import pytest
from dotenv import load_dotenv

from ..common import (
    COCOINDEX_AVAILABLE,
    CocoIndexTestInfrastructure,
    copy_directory_structure,
    format_test_failure_report,
    generate_test_timestamp,
    parse_jsonc_file,
    run_cocoindex_vector_search_tests,
)
from ..search_config import SearchTestConfig


@pytest.mark.skipif(not COCOINDEX_AVAILABLE, reason="CocoIndex infrastructure not available")
@pytest.mark.asyncio
class TestVectorSearch:
    """Vector-only search tests using CocoIndex infrastructure."""

    async def test_vector_search_validation(self):
        """Test vector-only search functionality using direct CocoIndex infrastructure."""

        # Load environment variables
        load_dotenv()

        # Generate single timestamp for this entire test run
        run_timestamp = generate_test_timestamp()

        # Copy complete directory structure from lang_examples to /workspaces/rust/tmp/
        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "lang_examples"
        tmp_dir = Path("/workspaces/rust/tmp")

        # Copy complete directory structure to preserve package structure for Java, Haskell, etc.
        copy_directory_structure(fixtures_dir, tmp_dir)

        # Create search test configuration with defaults:
        # --paths /workspaces/rust, --no-live, --default-embedding, --log-level DEBUG
        config = SearchTestConfig(
            # Use default settings which match your requirements:
            # paths=["/workspaces/rust"], no_live=True, default_embedding=True, log_level="DEBUG"
        )

        # Log configuration for debugging
        logger = logging.getLogger(__name__)
        config.log_configuration(logger)

        # Create and initialize infrastructure using configuration with vector test type
        infrastructure_kwargs = config.to_infrastructure_kwargs()
        infrastructure_kwargs["test_type"] = "vector"  # Use separate vector test table
        async with CocoIndexTestInfrastructure(**infrastructure_kwargs) as infrastructure:

            # CocoIndex indexing completes synchronously during infrastructure setup
            # No need to wait - infrastructure is ready for searches

            # Load test cases from fixture file
            fixture_path = Path(__file__).parent.parent / "fixtures" / "vector_search.jsonc"
            test_data = parse_jsonc_file(fixture_path)

            # Run vector search tests using direct infrastructure
            failed_tests = await run_cocoindex_vector_search_tests(
                test_cases=test_data["tests"], infrastructure=infrastructure, run_timestamp=run_timestamp
            )

            # Report results using common helper
            if failed_tests:
                error_msg = format_test_failure_report(failed_tests)
                logging.info(error_msg)
                pytest.fail(error_msg)
            else:
                logging.info("✅ All %s vector search validation tests passed!", len(test_data["tests"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
