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
    logger.info("Testing query: %s", query)

    result = parser.parse(query)

    # Import Operator enum for comparisons
    from cocoindex_code_mcp_server.keyword_search_parser_lark import Operator

    # Check that the result is a SearchGroup with OR operator
    assert hasattr(result, "operator"), "Result should have an operator attribute"
    assert result.operator == Operator.OR, "Operator should be OR"

    # Check that we have conditions
    assert hasattr(result, "conditions"), "Result should have conditions"
    assert len(result.conditions) == 2, "Should have 2 conditions for 'a OR b'"

    # Check individual conditions - OR queries create nested SearchGroups
    for i, group in enumerate(result.conditions):
        assert hasattr(group, "conditions"), f"Group {i} should have conditions"
        assert len(group.conditions) == 1, f"Group {i} should have 1 condition"
        condition = group.conditions[0]
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

    # Should have a single condition in a SearchGroup
    from cocoindex_code_mcp_server.keyword_search_parser_lark import SearchCondition

    assert hasattr(result, "conditions"), "Result should have conditions"
    assert len(result.conditions) == 1, "Should have exactly 1 condition"
    first_cond = result.conditions[0]
    assert isinstance(first_cond, SearchCondition), "First condition should be SearchCondition"
    assert first_cond.field == "language", "Field should be language"
    assert first_cond.value == "python", "Value should be python"

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

    # Import Operator enum for comparisons
    from cocoindex_code_mcp_server.keyword_search_parser_lark import Operator

    # Check that the result is a SearchGroup with AND operator
    assert hasattr(result, "operator"), "Result should have an operator attribute"
    assert result.operator == Operator.AND, "Operator should be AND"

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

    # Parser now returns SearchGroup with conditions list
    from cocoindex_code_mcp_server.keyword_search_parser_lark import SearchCondition

    assert hasattr(result, "conditions"), "Result should have conditions"
    assert len(result.conditions) == 1, "Should have exactly 1 condition"
    first_cond = result.conditions[0]
    assert isinstance(first_cond, SearchCondition), "First condition should be SearchCondition"
    assert first_cond.field == expected_field, f"Field should be {expected_field}"
    assert first_cond.value == expected_value, f"Value should be {expected_value}"


def test_keyword_search_parser_invalid_query():
    """Test that plain text queries are treated as text search."""
    from cocoindex_code_mcp_server.keyword_search_parser_lark import KeywordSearchParser

    parser = KeywordSearchParser()

    # Plain text without colon is now treated as text search
    from cocoindex_code_mcp_server.keyword_search_parser_lark import SearchCondition

    result = parser.parse("invalid query without colon")
    assert hasattr(result, "conditions"), "Result should have conditions"
    assert len(result.conditions) == 1, "Should have exactly 1 condition"
    first_cond = result.conditions[0]
    assert isinstance(first_cond, SearchCondition), "First condition should be SearchCondition"
    # Parser treats plain text as a text field search
    assert first_cond.field == "_text", "Field should be _text for plain text queries"
    assert first_cond.value == "invalid query without colon", "Value should be the query text"
