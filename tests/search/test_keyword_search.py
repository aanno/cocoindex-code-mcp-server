#!/usr/bin/env python3

"""
Keyword Search Tests

This module contains tests for keyword-only search functionality using CocoIndex
infrastructure directly. These tests validate metadata-based filtering and search
capabilities across different programming languages.
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
from ..search_config import SearchTestConfig
from ..db_comparison import compare_test_with_database


@pytest.mark.skipif(not COCOINDEX_AVAILABLE, reason="CocoIndex infrastructure not available")
@pytest.mark.asyncio
class TestKeywordSearch:
    """Keyword-only search tests using CocoIndex infrastructure."""

    async def test_keyword_search_validation(self):
        """Test keyword-only search functionality using direct CocoIndex infrastructure."""

        # Load environment variables
        load_dotenv()

        # Generate single timestamp for this entire test run
        run_timestamp = generate_test_timestamp()

        # Copy complete directory structure from lang_examples to /workspaces/rust/tmp/
        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "lang_examples"
        tmp_dir = Path("/workspaces/rust/tmp")

        # Copy complete directory structure to preserve package structure for Java, Haskell, etc.
        copy_directory_structure(fixtures_dir, tmp_dir)

        # Create search test configuration with only tmp directory for faster processing:
        config = SearchTestConfig(
            paths=[str(tmp_dir)],  # Only process the copied test files, not entire repo
            no_live=True,
            log_level="DEBUG"
        )
        
        # Log configuration for debugging
        logger = logging.getLogger(__name__)
        config.log_configuration(logger)

        # Create and initialize infrastructure using configuration with keyword test type
        infrastructure_kwargs = config.to_infrastructure_kwargs()
        infrastructure_kwargs['test_type'] = 'keyword'  # Use separate keyword test table
        async with CocoIndexTestInfrastructure(**infrastructure_kwargs) as infrastructure:

            # CocoIndex indexing completes synchronously during infrastructure setup
            # No need to wait - infrastructure is ready for searches

            # Load test cases from fixture file
            fixture_path = Path(__file__).parent.parent / "fixtures" / "keyword_search.jsonc"
            test_data = parse_jsonc_file(fixture_path)

            # Run keyword search tests using direct infrastructure
            failed_tests = await run_cocoindex_keyword_search_tests(
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
                logging.info(f"✅ All {len(test_data['tests'])} keyword search validation tests passed!")


async def run_cocoindex_keyword_search_tests(
    test_cases: list,
    infrastructure: CocoIndexTestInfrastructure,
    run_timestamp: str
) -> list:
    """
    Run keyword-only search tests using CocoIndex infrastructure directly.

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
        fail_expected = test_case.get("fail_expected", False)
        fail_reason = test_case.get("fail_reason", "")

        logging.info(f"Running keyword search test: {test_name}")
        logging.info(f"Description: {description}")
        if fail_expected:
            logging.info(f"⚠️  Expected to fail: {fail_reason}")

        try:
            # Execute keyword-only search using infrastructure backend
            search_data = await infrastructure.perform_keyword_search(query)

            results = search_data.get("results", [])
            total_results = len(results)

            # Save search results to test-results directory
            save_search_results(test_name, query, search_data, run_timestamp, "search-keyword")

            # Check minimum results requirement
            min_results = expected_results.get("min_results", 1)
            if total_results < min_results:
                error_msg = f"Expected at least {min_results} results, got {total_results}"
                if fail_expected:
                    logging.info(f"✅ Test failed as expected: {test_name} - {fail_reason}")
                    logging.info(f"   Failure: {error_msg}")
                else:
                    failed_tests.append({
                        "test": test_name,
                        "error": error_msg,
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
                        # Enhanced error reporting with database comparison
                        try:
                            db_comparison = await compare_test_with_database(
                                test_name, query, expected_item, results
                            )
                            db_report = f"\n🔍 Database Comparison Analysis:\n"
                            for discrepancy in db_comparison.discrepancies:
                                db_report += f"  ❌ {discrepancy}\n"
                            
                            if db_comparison.matching_db_records:
                                db_report += f"\n📋 Database has {len(db_comparison.matching_db_records)} matching records\n"
                                # Show sample DB record metadata
                                if db_comparison.matching_db_records:
                                    sample_record = db_comparison.matching_db_records[0]
                                    db_report += f"  Sample DB record: complexity_score={sample_record.get('complexity_score', 'N/A')}, "
                                    db_report += f"has_classes={sample_record.get('has_classes', 'N/A')}, "
                                    db_report += f"functions='{sample_record.get('functions', 'N/A')[:50]}...'\n"
                            
                            error_with_db_analysis = f"No matching result found for expected item: {expected_item}{db_report}"
                        except Exception as db_error:
                            logging.warning(f"Database comparison failed: {db_error}")
                            error_with_db_analysis = f"No matching result found for expected item: {expected_item}"

                        if fail_expected:
                            logging.info(f"✅ Test failed as expected: {test_name} - {fail_reason}")
                            logging.info(f"   Failure: {error_with_db_analysis}")
                        else:
                            failed_tests.append({
                                "test": test_name,
                                "error": error_with_db_analysis,
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
            error_msg = f"Test execution failed: {str(e)}"
            if fail_expected:
                logging.info(f"✅ Test failed as expected: {test_name} - {fail_reason}")
                logging.info(f"   Failure: {error_msg}")
            else:
                failed_tests.append({
                    "test": test_name,
                    "error": error_msg,
                    "query": query
                })

    return failed_tests


def save_search_results(
    test_name: str,
    query: dict,
    search_data: dict,
    run_timestamp: str,
    results_subdir: str = "search-keyword"
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

    print(f"💾 Saved keyword search results: {filepath}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])