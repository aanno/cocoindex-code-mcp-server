#!/usr/bin/env python3

"""
Full Text Search Tests

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
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


@pytest.mark.skipif(not COCOINDEX_AVAILABLE, reason="CocoIndex infrastructure not available")
@pytest.mark.asyncio
class TestFullTextSearch:
    """Full text (vector-only) search tests using CocoIndex infrastructure."""

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
            fixture_path = Path(__file__).parent.parent / "fixtures" / "full_text_search.jsonc"
            test_data = parse_jsonc_file(fixture_path)

            # Run vector search tests using direct infrastructure
            failed_tests = await run_cocoindex_vector_search_tests(
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
                logging.info(f"✅ All {len(test_data['tests'])} vector search validation tests passed!")


async def run_cocoindex_vector_search_tests(
    test_cases: list,
    infrastructure: CocoIndexTestInfrastructure,
    run_timestamp: str
) -> list:
    """
    Run vector-only search tests using CocoIndex infrastructure directly.

    Args:
        test_cases: List of test case definitions
        infrastructure: Initialized CocoIndex infrastructure
        run_timestamp: Timestamp for result saving

    Returns:
        List of failed test cases with error details
    """
    failed_tests = []

    for test_case in test_cases:
        test_name = test_case["name"]
        description = test_case["description"]
        query = test_case["query"]
        expected_results = test_case["expected_results"]

        logging.info(f"Running vector search test: {test_name}")
        logging.info(f"Description: {description}")

        try:
            # Execute vector-only search using infrastructure backend
            search_data = await infrastructure.perform_vector_search(query)

            results = search_data.get("results", [])
            total_results = len(results)

            # Save search results to test-results directory
            save_search_results(test_name, query, search_data, run_timestamp, "search-vector")

            # Check minimum results requirement
            min_results = expected_results.get("min_results", 1)
            if total_results < min_results:
                failed_tests.append({
                    "test": test_name,
                    "error": f"Expected at least {min_results} results, got {total_results}",
                    "query": query
                })
                continue

            # Check expected results using common helper
            if "should_contain" in expected_results:
                for expected_item in expected_results["should_contain"]:
                    found_match = False

                    for result_item in results:
                        from ..common import compare_expected_vs_actual
                        match_found, _ = compare_expected_vs_actual(expected_item, result_item)
                        if match_found:
                            found_match = True
                            break

                    if not found_match:
                        failed_tests.append({
                            "test": test_name,
                            "error": f"No matching result found for expected item: {expected_item}",
                            "query": query,
                            "actual_results": [{
                                "filename": r.get("filename"),
                                "metadata_summary": {
                                    "classes": r.get("classes", []),
                                    "functions": r.get("functions", []),
                                    "imports": r.get("imports", []),
                                    "analysis_method": r.get("metadata_json", {}).get("analysis_method", "unknown")
                                }
                            } for r in results[:3]]  # Show first 3 results for debugging
                        })

        except Exception as e:
            failed_tests.append({
                "test": test_name,
                "error": f"Test execution failed: {str(e)}",
                "query": query
            })

    return failed_tests


def save_search_results(
    test_name: str,
    query: dict,
    search_data: dict,
    run_timestamp: str,
    results_subdir: str = "search-vector"
) -> None:
    """
    Save search results to test-results directory with unique naming.

    Args:
        test_name: Name of the test
        query: The search query that was executed
        search_data: The search results data
        run_timestamp: Timestamp for consistent naming across test run
        results_subdir: Subdirectory name for organizing results
    """
    import datetime
    import json
    import os

    # Use the provided run timestamp for consistent naming across the test run
    filename = f"{test_name}_{run_timestamp}.json"

    # Ensure directory exists
    results_dir = os.path.join("/workspaces/rust/test-results", results_subdir)
    os.makedirs(results_dir, exist_ok=True)

    # Prepare complete result data
    result_data = {
        "test_name": test_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "search_results": search_data
    }

    # Save to file with proper JSON serialization
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"💾 Saved vector search results: {filepath}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
