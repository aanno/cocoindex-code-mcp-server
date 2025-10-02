#!/usr/bin/env python3

"""Test table name generation for search tests.

This test verifies that the get_test_table_name function correctly generates
table names for different search test types and properly rejects invalid types.
"""

import pytest


def test_table_name_generation():
    """Test that table names are generated correctly for each test type."""
    from tests.search_test_flows import get_test_table_name

    # Test keyword search table name
    keyword_table = get_test_table_name("keyword")
    assert keyword_table == "keywordsearchtest_code_embeddings"

    # Test vector search table name
    vector_table = get_test_table_name("vector")
    assert vector_table == "vectorsearchtest_code_embeddings"

    # Test hybrid search table name
    hybrid_table = get_test_table_name("hybrid")
    assert hybrid_table == "hybridsearchtest_code_embeddings"


def test_invalid_test_type_rejected():
    """Test that invalid test types are properly rejected."""
    from tests.search_test_flows import get_test_table_name

    with pytest.raises(ValueError, match="Unknown test type"):
        get_test_table_name("invalid")


@pytest.mark.parametrize(
    "test_type,expected_table",
    [
        ("keyword", "keywordsearchtest_code_embeddings"),
        ("vector", "vectorsearchtest_code_embeddings"),
        ("hybrid", "hybridsearchtest_code_embeddings"),
    ],
)
def test_table_name_mapping(test_type, expected_table):
    """Test table name mapping using parametrized tests."""
    from tests.search_test_flows import get_test_table_name

    assert get_test_table_name(test_type) == expected_table
