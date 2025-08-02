#!/usr/bin/env python3

"""
Common test helpers for CocoIndex MCP server testing.

This module provides shared functionality for test fixture processing,
result comparison, and test result saving.
"""

import json
import logging
import os
import re
import shutil
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def generate_test_timestamp() -> str:
    """Generate a timestamp for test run identification."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds


def copy_directory_structure(source_dir: Path, target_dir: Path) -> None:
    """
    Copy complete directory structure from source to target.
    
    Args:
        source_dir: Source directory path
        target_dir: Target directory path
    """
    # Ensure target directory exists
    target_dir.mkdir(exist_ok=True)
    
    print(f"📁 Copying directory structure from {source_dir} to {target_dir}...")
    
    if source_dir.exists():
        # Remove existing content in target directory first
        if target_dir.exists():
            shutil.rmtree(target_dir)
        
        # Copy the entire directory structure
        shutil.copytree(source_dir, target_dir)
        print(f"  ✅ Copied complete directory structure")
    else:
        print(f"  ❌ Source directory {source_dir} does not exist")


def copy_test_files_legacy(source_dir: Path, target_dir: Path, test_files: List[str]) -> None:
    """
    Legacy method to copy individual test files.
    
    Args:
        source_dir: Source directory path
        target_dir: Target directory path
        test_files: List of filenames to copy
    """
    # Ensure target directory exists
    target_dir.mkdir(exist_ok=True)
    
    print(f"📁 Copying test fixtures from {source_dir} to {target_dir}...")
    for test_file in test_files:
        src = source_dir / test_file
        dst = target_dir / test_file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✅ Copied {test_file}")
        else:
            print(f"  ⚠️  Warning: {test_file} not found in source directory")


def parse_jsonc_file(file_path: Path) -> Dict[str, Any]:
    """
    Parse a JSONC (JSON with comments) file.
    
    Args:
        file_path: Path to the JSONC file
        
    Returns:
        Parsed JSON data as dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the JSON is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"JSONC file not found: {file_path}")
    
    # Read file content
    fixture_content = file_path.read_text()
    
    # Remove comments for JSON parsing
    lines = []
    for line in fixture_content.split('\n'):
        stripped = line.strip()
        if not stripped.startswith('//'):
            lines.append(line)
    
    clean_json = '\n'.join(lines)
    return json.loads(clean_json)


def save_search_results(
    test_name: str,
    query: Dict[str, Any],
    search_data: Dict[str, Any],
    run_timestamp: str,
    results_base_dir: str = "/workspaces/rust/test-results"
) -> None:
    """
    Save search results to test-results directory with unique naming.
    
    Args:
        test_name: Name of the test
        query: The search query that was executed
        search_data: The search results data
        run_timestamp: Timestamp for consistent naming across test run
        results_base_dir: Base directory for test results
    """
    # Use the provided run timestamp for consistent naming across the test run
    filename = f"{test_name}_{run_timestamp}.json"
    
    # Ensure directory exists
    results_dir = os.path.join(results_base_dir, "search-hybrid")
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare complete result data
    result_data = {
        "test_name": test_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "search_results": search_data
    }
    
    # Save to file
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved search results: {filepath}")


def compare_expected_vs_actual(
    expected_item: Dict[str, Any],
    result_item: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Compare expected metadata with actual result metadata.
    
    Args:
        expected_item: Expected result specification
        result_item: Actual search result item
        
    Returns:
        Tuple of (match_found, list_of_errors)
    """
    errors = []
    
    # Check filename pattern if specified
    if "filename_pattern" in expected_item:
        pattern = expected_item["filename_pattern"]
        filename = result_item.get("filename", "")
        if not re.match(pattern, filename):
            return False, [f"Filename '{filename}' does not match pattern '{pattern}'"]
    
    # Check expected metadata
    if "expected_metadata" in expected_item:
        metadata_errors = []
        expected_metadata = expected_item["expected_metadata"]
        
        # Get metadata from both flattened fields and metadata_json
        combined_metadata = dict(result_item)
        if "metadata_json" in result_item and isinstance(result_item["metadata_json"], dict):
            combined_metadata.update(result_item["metadata_json"])
        
        for field, expected_value in expected_metadata.items():
            actual_value = combined_metadata.get(field)
            
            # Handle special comparison operators
            if isinstance(expected_value, str):
                if expected_value.startswith("!"):
                    # Not equal comparison
                    not_expected = expected_value[1:]
                    if str(actual_value) == not_expected:
                        metadata_errors.append(f"{field}: expected not '{not_expected}', got '{actual_value}'")
                elif expected_value.startswith(">"):
                    # Greater than comparison
                    try:
                        threshold = float(expected_value[1:])
                        if not (isinstance(actual_value, (int, float)) and actual_value > threshold):
                            metadata_errors.append(f"{field}: expected > {threshold}, got '{actual_value}'")
                    except ValueError:
                        metadata_errors.append(f"{field}: invalid threshold '{expected_value}'")
                elif expected_value == "!empty":
                    # Not empty check
                    if not actual_value or (isinstance(actual_value, list) and len(actual_value) == 0):
                        metadata_errors.append(f"{field}: expected non-empty, got '{actual_value}'")
                else:
                    # Direct equality
                    if str(actual_value) != expected_value:
                        metadata_errors.append(f"{field}: expected '{expected_value}', got '{actual_value}'")
            elif isinstance(expected_value, bool):
                if actual_value != expected_value:
                    metadata_errors.append(f"{field}: expected {expected_value}, got {actual_value}")
            elif isinstance(expected_value, list):
                if actual_value != expected_value:
                    metadata_errors.append(f"{field}: expected {expected_value}, got {actual_value}")
        
        if metadata_errors:
            errors.extend(metadata_errors)
    
    # Check should_not_be_empty fields
    if "should_not_be_empty" in expected_item:
        # Get metadata from both flattened fields and metadata_json
        combined_metadata = dict(result_item)
        if "metadata_json" in result_item and isinstance(result_item["metadata_json"], dict):
            combined_metadata.update(result_item["metadata_json"])
            
        empty_fields = []
        for field in expected_item["should_not_be_empty"]:
            field_value = combined_metadata.get(field)
            if not field_value or (isinstance(field_value, list) and len(field_value) == 0):
                empty_fields.append(field)
        
        if empty_fields:
            errors.append(f"Fields should not be empty: {empty_fields}")
    
    return len(errors) == 0, errors


def validate_search_results(
    test_cases: List[Dict[str, Any]],
    execute_search_func,
    run_timestamp: str
) -> List[Dict[str, Any]]:
    """
    Validate search results against expected outcomes.
    
    Args:
        test_cases: List of test case definitions
        execute_search_func: Async function to execute search queries
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
        
        logging.info(f"Running hybrid search test: {test_name}")
        logging.info(f"Description: {description}")
        
        try:
            # Execute search (this should be provided by caller)
            search_data = execute_search_func(query)
            
            results = search_data.get("results", [])
            total_results = len(results)
            
            # Save search results to test-results directory
            save_search_results(test_name, query, search_data, run_timestamp)
            
            # Check minimum results requirement
            min_results = expected_results.get("min_results", 1)
            if total_results < min_results:
                failed_tests.append({
                    "test": test_name,
                    "error": f"Expected at least {min_results} results, got {total_results}",
                    "query": query
                })
                continue
            
            # Check expected results
            if "should_contain" in expected_results:
                for expected_item in expected_results["should_contain"]:
                    found_match = False
                    
                    for result_item in results:
                        match_found, errors = compare_expected_vs_actual(expected_item, result_item)
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


def format_test_failure_report(failed_tests: List[Dict[str, Any]]) -> str:
    """
    Format a comprehensive failure report for failed tests.
    
    Args:
        failed_tests: List of failed test dictionaries
        
    Returns:
        Formatted error message string
    """
    if not failed_tests:
        return ""
    
    error_msg = f"Hybrid search validation failed for {len(failed_tests)} test(s):\n"
    
    for failure in failed_tests:
        error_msg += f"\n  Test: {failure['test']}\n"
        error_msg += f"  Query: {failure['query']}\n"
        error_msg += f"  Error: {failure['error']}\n"
        if "actual_results" in failure:
            error_msg += f"  Sample Results: {json.dumps(failure['actual_results'], indent=2)}\n"
    
    return error_msg