#!/usr/bin/env python3

"""
Tests for the keyword search parser in hybrid search functionality.
"""

from typing import Union, cast

import pytest

from cocoindex_code_mcp_server.keyword_search_parser_lark import (
    KeywordSearchParser,
    Operator,
    SearchCondition,
    SearchGroup,
    build_sql_where_clause,
)


@pytest.fixture
def parser():
    """Create a KeywordSearchParser instance for testing."""
    return KeywordSearchParser()


# SearchCondition tests
@pytest.mark.unit
@pytest.mark.keyword_parser
class TestSearchCondition:
    """Test SearchCondition class."""

    def test_field_value_condition(self):
        """Test basic field:value condition."""
        condition = SearchCondition(field="language", value="python")
        assert condition.field == "language"
        assert condition.value == "python"
        assert not condition.is_exists_check

    def test_exists_condition(self):
        """Test exists condition."""
        condition = SearchCondition(field="embedding", value="", is_exists_check=True)
        assert condition.field == "embedding"
        assert condition.value == ""
        assert condition.is_exists_check


# SearchGroup tests
@pytest.mark.unit
@pytest.mark.keyword_parser
class TestSearchGroup:
    """Test SearchGroup class."""

    def test_and_group(self):
        """Test AND group creation."""
        condition1 = SearchCondition(field="language", value="python")
        condition2 = SearchCondition(field="filename", value="main_interactive_query.py")
        group = SearchGroup(conditions=[condition1, condition2], operator=Operator.AND)

        assert len(group.conditions) == 2
        assert group.operator == Operator.AND

    def test_or_group(self):
        """Test OR group creation."""
        condition1 = SearchCondition(field="language", value="python")
        condition2 = SearchCondition(field="language", value="rust")
        group = SearchGroup(conditions=[condition1, condition2], operator=Operator.OR)

        assert len(group.conditions) == 2
        assert group.operator == Operator.OR


# KeywordSearchParser tests
@pytest.mark.unit
@pytest.mark.keyword_parser
class TestKeywordSearchParser:
    """Test KeywordSearchParser class."""

    def test_empty_query(self, parser: KeywordSearchParser):
        """Test parsing empty query."""
        result = parser.parse("")
        assert len(result.conditions) == 0

        result = parser.parse("   ")
        assert len(result.conditions) == 0

    def test_simple_field_value(self, parser: KeywordSearchParser):
        """Test parsing simple field:value."""
        result = parser.parse("language:python")

        assert len(result.conditions) == 1
        assert result.operator == Operator.AND

        condition = result.conditions[0]
        assert isinstance(condition, SearchCondition)
        assert condition.field == "language"
        assert condition.value == "python"
        assert not condition.is_exists_check

    def test_quoted_value(self, parser: KeywordSearchParser):
        """Test parsing quoted values."""
        result = parser.parse('filename:"test file.py"')

        assert len(result.conditions) == 1

        condition: Union[SearchCondition, SearchGroup] = result.conditions[0]
        cond = cast(SearchCondition, condition)

        assert cond.field == "filename"
        assert cond.value == "test file.py"

    def test_single_quoted_value(self, parser: KeywordSearchParser):
        """Test parsing single quoted values."""
        result = parser.parse("filename:'test file.py'")

        assert len(result.conditions) == 1

        condition: Union[SearchCondition, SearchGroup] = result.conditions[0]
        cond = cast(SearchCondition, condition)

        assert cond.field == "filename"
        assert cond.value == "test file.py"

    def test_exists_condition(self, parser: KeywordSearchParser):
        """Test parsing exists() condition."""
        result = parser.parse("exists(embedding)")

        assert len(result.conditions) == 1

        condition: Union[SearchCondition, SearchGroup] = result.conditions[0]
        cond = cast(SearchCondition, condition)

        assert cond.field == "embedding"
        assert cond.value == ""
        assert cond.is_exists_check

    def test_exists_case_insensitive(self, parser: KeywordSearchParser):
        """Test exists() is case insensitive."""
        result = parser.parse("EXISTS(embedding)")

        assert len(result.conditions) == 1

        condition: Union[SearchCondition, SearchGroup] = result.conditions[0]
        cond = cast(SearchCondition, condition)

        assert cond.field == "embedding"
        assert cond.is_exists_check

    def test_and_operator(self, parser: KeywordSearchParser):
        """Test AND operator parsing."""
        result = parser.parse("language:python and filename:main_interactive_query.py")

        assert len(result.conditions) == 2
        assert result.operator == Operator.AND

        condition1: Union[SearchCondition, SearchGroup] = result.conditions[0]
        condition2: Union[SearchCondition, SearchGroup] = result.conditions[1]

        cond1 = cast(SearchCondition, condition1)
        cond2 = cast(SearchCondition, condition2)

        assert cond1.field == "language"
        assert cond1.value == "python"
        assert cond2.field == "filename"
        assert cond2.value == "main_interactive_query.py"

    def test_or_operator(self, parser: KeywordSearchParser):
        """Test OR operator parsing."""
        result = parser.parse("language:python or language:rust")

        assert len(result.conditions) == 2
        assert result.operator == Operator.OR

        # Each OR part becomes an AND group with one condition
        group1 = result.conditions[0]
        group2 = result.conditions[1]

        assert isinstance(group1, SearchGroup)
        assert isinstance(group2, SearchGroup)
        assert group1.operator == Operator.AND
        assert group2.operator == Operator.AND

        condition1: Union[SearchCondition, SearchGroup] = group1.conditions[0]
        condition2: Union[SearchCondition, SearchGroup] = group2.conditions[0]

        cond1 = cast(SearchCondition, condition1)
        cond2 = cast(SearchCondition, condition2)

        assert cond1.field == "language"
        assert cond1.value == "python"
        assert cond2.field == "language"
        assert cond2.value == "rust"

    def test_mixed_operators_or_precedence(self, parser: KeywordSearchParser):
        """Test mixed operators with OR having lower precedence."""
        result = parser.parse("language:python and filename:main_interactive_query.py or language:rust")

        # Should be parsed as: (language:python and filename:main_interactive_query.py) or (language:rust)
        assert result.operator == Operator.OR
        assert len(result.conditions) == 2

        # First condition should be an AND group
        first_group: Union[SearchCondition, SearchGroup] = result.conditions[0]
        assert isinstance(first_group, SearchGroup)
        assert first_group.operator == Operator.AND
        assert len(first_group.conditions) == 2

        # Second condition should be an AND group with one condition
        second_group = result.conditions[1]
        assert isinstance(second_group, SearchGroup)
        assert second_group.operator == Operator.AND
        assert len(second_group.conditions) == 1

        second_condition: Union[SearchCondition, SearchGroup] = second_group.conditions[0]
        cond = cast(SearchCondition, second_condition)
        assert cond.field == "language"
        assert cond.value == "rust"

    def test_parentheses_grouping(self, parser: KeywordSearchParser):
        """Test parentheses grouping."""
        result = parser.parse("(language:python or language:rust) and exists(embedding)")

        assert result.operator == Operator.AND
        assert len(result.conditions) == 2

        # First condition should be an OR group
        first_group = result.conditions[0]
        assert isinstance(first_group, SearchGroup)
        assert first_group.operator == Operator.OR
        assert len(first_group.conditions) == 2

        # Second condition should be exists check
        second_condition = result.conditions[1]
        assert isinstance(second_condition, SearchCondition)
        assert second_condition.is_exists_check

    def test_nested_parentheses(self, parser: KeywordSearchParser):
        """Test nested parentheses."""
        result = parser.parse("((language:python or language:rust) and exists(embedding)) or filename:test.py")

        assert result.operator == Operator.OR
        assert len(result.conditions) == 2

        # First condition should be a complex nested group
        first_group = result.conditions[0]
        assert isinstance(first_group, SearchGroup)
        assert first_group.operator == Operator.AND

    def test_general_text_search(self, parser: KeywordSearchParser):
        """Test general text search (no field specified)."""
        result = parser.parse("python function")

        assert len(result.conditions) == 1
        condition: Union[SearchCondition, SearchGroup] = result.conditions[0]
        cond = cast(SearchCondition, condition)
        assert cond.field == "_text"
        assert cond.value == "python function"

    def test_case_insensitive_operators(self, parser: KeywordSearchParser):
        """Test that operators are case insensitive."""
        result1 = parser.parse("language:python AND filename:main_interactive_query.py")
        result2 = parser.parse("language:python Or filename:main_interactive_query.py")

        assert result1.operator == Operator.AND
        assert result2.operator == Operator.OR

    def test_whitespace_handling(self, parser: KeywordSearchParser):
        """Test proper whitespace handling."""
        result = parser.parse("  language:python   and   filename:main_interactive_query.py  ")

        assert len(result.conditions) == 2
        assert result.operator == Operator.AND


# SQL WHERE clause building tests
@pytest.mark.unit
@pytest.mark.keyword_parser
class TestBuildSqlWhereClause:
    """Test SQL WHERE clause building."""

    def test_empty_group(self):
        """Test empty search group."""
        group = SearchGroup(conditions=[])
        where_clause, params = build_sql_where_clause(group)

        assert where_clause == "TRUE"
        assert params == []

    def test_simple_field_condition(self):
        """Test simple field condition."""
        condition = SearchCondition(field="language", value="python")
        group = SearchGroup(conditions=[condition])

        where_clause, params = build_sql_where_clause(group)

        assert where_clause == "LOWER(language) = LOWER(%s)"
        assert params == ["python"]

    def test_exists_condition(self):
        """Test exists condition."""
        condition = SearchCondition(field="embedding", value="", is_exists_check=True)
        group = SearchGroup(conditions=[condition])

        where_clause, params = build_sql_where_clause(group)

        assert where_clause == "embedding IS NOT NULL"
        assert params == []

    def test_text_search_condition(self):
        """Test general text search condition."""
        condition = SearchCondition(field="_text", value="python function")
        group = SearchGroup(conditions=[condition])

        where_clause, params = build_sql_where_clause(group)

        assert where_clause == "code ILIKE %s"
        assert params == ["%python function%"]

    def test_and_conditions(self):
        """Test AND conditions."""
        condition1 = SearchCondition(field="language", value="python")
        condition2 = SearchCondition(field="filename", value="main_interactive_query.py")
        group = SearchGroup(conditions=[condition1, condition2], operator=Operator.AND)

        where_clause, params = build_sql_where_clause(group)

        assert where_clause == "LOWER(language) = LOWER(%s) AND filename ILIKE %s"
        assert params == ["python", "%main_interactive_query.py%"]

    def test_or_conditions(self):
        """Test OR conditions."""
        condition1 = SearchCondition(field="language", value="python")
        condition2 = SearchCondition(field="language", value="rust")
        group = SearchGroup(conditions=[condition1, condition2], operator=Operator.OR)

        where_clause, params = build_sql_where_clause(group)

        assert where_clause == "LOWER(language) = LOWER(%s) OR LOWER(language) = LOWER(%s)"
        assert params == ["python", "rust"]

    def test_nested_groups(self):
        """Test nested groups."""
        # (language:python or language:rust) and exists(embedding)
        inner_condition1 = SearchCondition(field="language", value="python")
        inner_condition2 = SearchCondition(field="language", value="rust")
        inner_group = SearchGroup(conditions=[inner_condition1, inner_condition2], operator=Operator.OR)

        outer_condition = SearchCondition(field="embedding", value="", is_exists_check=True)
        outer_group = SearchGroup(conditions=[inner_group, outer_condition], operator=Operator.AND)

        where_clause, params = build_sql_where_clause(outer_group)

        assert where_clause == "(LOWER(language) = LOWER(%s) OR LOWER(language) = LOWER(%s)) AND embedding IS NOT NULL"
        assert params == ["python", "rust"]

    def test_table_alias(self):
        """Test table alias in WHERE clause."""
        condition = SearchCondition(field="language", value="python")
        group = SearchGroup(conditions=[condition])

        where_clause, params = build_sql_where_clause(group, table_alias="t")

        assert where_clause == "LOWER(t.language) = LOWER(%s)"
        assert params == ["python"]


# Complex query tests
@pytest.mark.integration
@pytest.mark.keyword_parser
class TestComplexQueries:
    """Test complex real-world queries."""

    def test_documentation_example_1(self, parser: KeywordSearchParser):
        """Test: (language:python or language:rust) and exists(embedding)"""
        result = parser.parse("(language:python or language:rust) and exists(embedding)")

        assert result.operator == Operator.AND
        assert len(result.conditions) == 2

        # First condition: OR group
        or_group = result.conditions[0]
        assert isinstance(or_group, SearchGroup)
        assert or_group.operator == Operator.OR
        assert len(or_group.conditions) == 2

        # Second condition: exists check
        exists_condition = result.conditions[1]
        assert isinstance(exists_condition, SearchCondition)
        assert exists_condition.is_exists_check
        assert exists_condition.field == "embedding"

    def test_documentation_example_2(self, parser: KeywordSearchParser):
        """Test: filename:"test file.py" and language:python"""
        result: SearchGroup = parser.parse('filename:"test file.py" and language:python')

        assert result.operator == Operator.AND
        assert len(result.conditions) == 2

        condition1: Union[SearchCondition, SearchGroup] = result.conditions[0]
        condition2: Union[SearchCondition, SearchGroup] = result.conditions[1]

        cond1 = cast(SearchCondition, condition1)
        cond2 = cast(SearchCondition, condition2)
        assert cond1.field == "filename"
        assert cond1.value == "test file.py"
        assert cond2.field == "language"
        assert cond2.value == "python"

    def test_complex_mixed_query(self, parser: KeywordSearchParser):
        """Test complex query with multiple operators and grouping."""
        query = '(filename:main_interactive_query.py or filename:app.py) and (language:python or language:go) and exists(embedding)'
        result = parser.parse(query)

        assert result.operator == Operator.AND
        assert len(result.conditions) == 3

        # All conditions should be present
        filename_group = result.conditions[0]
        language_group = result.conditions[1]
        exists_condition = result.conditions[2]

        assert isinstance(filename_group, SearchGroup)
        assert filename_group.operator == Operator.OR

        assert isinstance(language_group, SearchGroup)
        assert language_group.operator == Operator.OR

        assert isinstance(exists_condition, SearchCondition)
        assert exists_condition.is_exists_check
