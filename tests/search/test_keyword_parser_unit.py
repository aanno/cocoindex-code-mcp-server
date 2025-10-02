#!/usr/bin/env python3

"""Unit tests for the keyword search parser.

This test verifies that the Lark-based keyword search parser correctly parses
queries and generates proper SQL WHERE clauses for filtering.
"""

import logging

import pytest


logger = logging.getLogger(__name__)


def test_keyword_search_parser_boolean_or():
    """Test the keyword search parser with boolean OR queries."""
    from cocoindex_code_mcp_server.keyword_search_parser_lark import (
        KeywordSearchParser,
        build_sql_where_clause,
    )

    parser = KeywordSearchParser()

    # Test the boolean OR query
    query = "language:rust OR language:c"
    logger.info(f"Testing query: {query}")

    result = parser.parse(query)

    # Check that the result is a SearchGroup with OR operator
    assert hasattr(result, "operator"), "Result should have an operator attribute"
    assert result.operator == "OR", "Operator should be OR"

    # Check that we have conditions
    assert hasattr(result, "conditions"), "Result should have conditions"
    assert len(result.conditions) == 2, "Should have 2 conditions for 'a OR b'"

    # Check individual conditions
    for i, condition in enumerate(result.conditions):
        assert hasattr(condition, "field"), f"Condition {i} should have field"
        assert hasattr(condition, "value"), f"Condition {i} should have value"
        assert condition.field == "language", "Field should be language"
        assert condition.value in [
            "rust",
            "c",
        ], f"Value should be 'rust' or 'c', got {condition.value}"

    # Test SQL generation
    where_clause, params = build_sql_where_clause(result)
    assert where_clause, "WHERE clause should be generated"
    assert params, "Parameters should be generated"
    assert "OR" in where_clause.upper(), "WHERE clause should contain OR"


def test_keyword_search_parser_simple_query():
    """Test the parser with simple single-field queries."""
    from cocoindex_code_mcp_server.keyword_search_parser_lark import (
        KeywordSearchParser,
        build_sql_where_clause,
    )

    parser = KeywordSearchParser()

    # Test simple query
    query = "language:python"
    result = parser.parse(query)

    # Should have a single condition
    assert hasattr(result, "field"), "Simple query should have field"
    assert result.field == "language", "Field should be language"
    assert result.value == "python", "Value should be python"

    # Test SQL generation
    where_clause, params = build_sql_where_clause(result)
    assert where_clause, "WHERE clause should be generated"
    assert params, "Parameters should be generated"


def test_keyword_search_parser_and_query():
    """Test the parser with boolean AND queries."""
    from cocoindex_code_mcp_server.keyword_search_parser_lark import (
        KeywordSearchParser,
        build_sql_where_clause,
    )

    parser = KeywordSearchParser()

    # Test AND query
    query = "language:python AND functions:fibonacci"
    result = parser.parse(query)

    # Check that the result is a SearchGroup with AND operator
    assert hasattr(result, "operator"), "Result should have an operator attribute"
    assert result.operator == "AND", "Operator should be AND"

    # Check that we have conditions
    assert hasattr(result, "conditions"), "Result should have conditions"
    assert len(result.conditions) == 2, "Should have 2 conditions"

    # Test SQL generation
    where_clause, params = build_sql_where_clause(result)
    assert where_clause, "WHERE clause should be generated"
    assert "AND" in where_clause.upper(), "WHERE clause should contain AND"


@pytest.mark.parametrize(
    "query,expected_field,expected_value",
    [
        ("language:rust", "language", "rust"),
        ("functions:fibonacci", "functions", "fibonacci"),
        ("classes:TestClass", "classes", "TestClass"),
        ("filename:test.py", "filename", "test.py"),
    ],
)
def test_keyword_search_parser_field_value_pairs(query, expected_field, expected_value):
    """Test parsing various field:value pairs."""
    from cocoindex_code_mcp_server.keyword_search_parser_lark import KeywordSearchParser

    parser = KeywordSearchParser()
    result = parser.parse(query)

    assert result.field == expected_field, f"Field should be {expected_field}"
    assert result.value == expected_value, f"Value should be {expected_value}"


def test_keyword_search_parser_invalid_query():
    """Test that invalid queries raise appropriate errors."""
    from cocoindex_code_mcp_server.keyword_search_parser_lark import KeywordSearchParser

    parser = KeywordSearchParser()

    # Test invalid query format
    with pytest.raises(Exception):  # Could be SyntaxError or custom parse error
        parser.parse("invalid query without colon")
