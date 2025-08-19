#!/usr/bin/env python3

"""
Direct CocoIndex MCP Tests

This module contains tests that run CocoIndex infrastructure directly
without requiring an integration server. This provides faster test execution
and more control over the infrastructure setup.
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
    run_cocoindex_hybrid_search_tests,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


@pytest.mark.skipif(not COCOINDEX_AVAILABLE, reason="CocoIndex infrastructure not available")
@pytest.mark.asyncio
class TestMCPDirect:
    """Direct CocoIndex MCP tests without integration server."""

    @pytest.mark.xfail(reason="Hybrid search tests not ready for prime time")
    async def test_hybrid_search_validation(self):
        """Test hybrid search functionality using direct CocoIndex infrastructure."""

        # Load environment variables
        load_dotenv()

        # Generate single timestamp for this entire test run
        run_timestamp = generate_test_timestamp()

        # Copy complete directory structure from lang_examples to /workspaces/rust/tmp/
        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "lang_examples"
        tmp_dir = Path("/workspaces/rust/tmp")

        # Copy complete directory structure to preserve package structure for Java, Haskell, etc.
        copy_directory_structure(fixtures_dir, tmp_dir)

        # Set up CocoIndex infrastructure with configurable parameters
        infrastructure_config = {
            "paths": ["/workspaces/rust"],  # Use main workspace directory
            "default_embedding": False,  # Use smart embedding by default
            "default_chunking": False,   # Use custom chunking by default
            "default_language_handler": False,  # Use enhanced language handlers
            "chunk_factor_percent": 100,  # Normal chunk size (can be configured)
            "enable_polling": False,   # --no-live: Disable live updates for tests
            "poll_interval": 30
        }

        # Create and initialize infrastructure  
        async with CocoIndexTestInfrastructure(
            paths=["/workspaces/rust"],
            enable_polling=False,
            default_chunking=False,
            default_language_handler=False
        ) as infrastructure:

            # CocoIndex indexing completes synchronously during infrastructure setup
            # No need to wait - infrastructure is ready for searches

            # Load test cases from fixture file
            fixture_path = Path(__file__).parent.parent / "fixtures" / "hybrid_search.jsonc"
            test_data = parse_jsonc_file(fixture_path)

            # Run hybrid search tests using direct infrastructure
            failed_tests = await run_cocoindex_hybrid_search_tests(
                test_cases=test_data["tests"],
                infrastructure=infrastructure,
                run_timestamp=run_timestamp
            )

            # Report results using common helper
            if failed_tests:
                error_msg = format_test_failure_report(failed_tests)
                logging.info(error_msg)
                pytest.fail(error_msg)
            else:
                logging.info(f"✅ All {len(test_data['tests'])} hybrid search validation tests passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
