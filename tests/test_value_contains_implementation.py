#!/usr/bin/env python3
"""Test cases for value_contains functionality in keyword search parsers."""

import pytest
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'cocoindex-code-mcp-server'))

from keyword_search_parser import KeywordSearchParser as FallbackParser, build_sql_where_clause, SearchCondition, SearchGroup

try:
    from keyword_search_parser_lark import KeywordSearchParser as LarkParser, build_sql_where_clause as lark_build_sql_where_clause
    LARK_AVAILABLE = True
except ImportError:
    LARK_AVAILABLE = False
    lark_build_sql_where_clause = None


class TestValueContainsFallbackParser:
    """Test value_contains functionality in the fallback parser."""
    
    def test_simple_value_contains(self):
        """Test basic value_contains parsing."""
        parser = FallbackParser()
        result = parser.parse('value_contains(code, "function")')
        
        assert len(result.conditions) == 1
        condition = result.conditions[0]
        assert isinstance(condition, SearchCondition)
        assert condition.field == "code"
        assert condition.value == "function"
        assert condition.is_value_contains_check is True
        assert condition.is_exists_check is False
    
    def test_value_contains_with_quoted_value(self):
        """Test value_contains with quoted search string."""
        parser = FallbackParser()
        result = parser.parse('value_contains(filename, "test file")')
        
        condition = result.conditions[0]
        assert condition.field == "filename"
        assert condition.value == "test file"
        assert condition.is_value_contains_check is True
    
    def test_value_contains_with_unquoted_value(self):
        """Test value_contains with unquoted search string."""
        parser = FallbackParser()
        result = parser.parse('value_contains(code, async)')
        
        condition = result.conditions[0]
        assert condition.field == "code"
        assert condition.value == "async"
        assert condition.is_value_contains_check is True
    
    def test_value_contains_with_and_operator(self):
        """Test value_contains combined with AND operator."""
        parser = FallbackParser()
        result = parser.parse('value_contains(code, "function") and language:python')
        
        assert len(result.conditions) == 2
        
        # First condition should be value_contains
        condition1 = result.conditions[0]
        assert condition1.field == "code"
        assert condition1.value == "function"
        assert condition1.is_value_contains_check is True
        
        # Second condition should be regular field:value
        condition2 = result.conditions[1]
        assert condition2.field == "language"
        assert condition2.value == "python"
        assert condition2.is_value_contains_check is False
    
    def test_value_contains_with_parentheses(self):
        """Test value_contains with parentheses grouping."""
        parser = FallbackParser()
        result = parser.parse('(language:python or language:rust) and value_contains(code, "async")')
        
        assert len(result.conditions) == 2
        
        # First should be a group with OR operator
        group = result.conditions[0]
        assert isinstance(group, SearchGroup)
        
        # Second should be value_contains
        condition = result.conditions[1]
        assert condition.field == "code"
        assert condition.value == "async"
        assert condition.is_value_contains_check is True


class TestValueContainsSQLGeneration:
    """Test SQL generation for value_contains conditions."""
    
    def test_value_contains_sql_generation(self):
        """Test that value_contains generates correct SQL."""
        condition = SearchCondition(field="code", value="function", is_value_contains_check=True)
        group = SearchGroup(conditions=[condition])
        
        where_clause, params = build_sql_where_clause(group)
        
        assert where_clause == "code ILIKE %s"
        assert params == ["%function%"]
    
    def test_value_contains_with_table_alias(self):
        """Test value_contains SQL generation with table alias."""
        condition = SearchCondition(field="filename", value="test", is_value_contains_check=True)
        group = SearchGroup(conditions=[condition])
        
        where_clause, params = build_sql_where_clause(group, table_alias="t")
        
        assert where_clause == "t.filename ILIKE %s"
        assert params == ["%test%"]
    
    def test_complex_value_contains_sql(self):
        """Test complex query with value_contains generates correct SQL."""
        parser = FallbackParser()
        result = parser.parse('language:python and value_contains(code, "def")')
        
        where_clause, params = build_sql_where_clause(result)
        
        assert "language = %s" in where_clause
        assert "code ILIKE %s" in where_clause
        assert "AND" in where_clause
        assert params == ["python", "%def%"]


@pytest.mark.skipif(not LARK_AVAILABLE, reason="Lark parser not available")
class TestValueContainsLarkParser:
    """Test value_contains functionality in the Lark parser."""
    
    def test_lark_value_contains_parsing(self):
        """Test that Lark parser handles value_contains correctly."""
        parser = LarkParser()
        result = parser.parse('value_contains(code, "function")')
        
        assert len(result.conditions) == 1
        condition = result.conditions[0]
        assert condition.field == "code"
        assert condition.value == "function"
        assert condition.is_value_contains_check is True
    
    def test_lark_complex_value_contains_query(self):
        """Test complex Lark parser query with value_contains."""
        parser = LarkParser()
        result = parser.parse('(language:python or language:rust) and value_contains(code, "async")')
        
        # Check the structure first
        assert len(result.conditions) == 2
        
        # Check that we have a value_contains condition
        has_value_contains = False
        for condition in result.conditions:
            if hasattr(condition, 'is_value_contains_check') and condition.is_value_contains_check:
                has_value_contains = True
                assert condition.field == "code"
                assert condition.value == "async"
                break
        assert has_value_contains, "Should have a value_contains condition"
        
        where_clause, params = lark_build_sql_where_clause(result)
        
        assert "code ILIKE %s" in where_clause
        assert "%async%" in params
        assert "language = %s" in where_clause  # Should have both languages
        assert params == ['python', 'rust', '%async%']


class TestValueContainsEdgeCases:
    """Test edge cases for value_contains implementation."""
    
    def test_value_contains_empty_string(self):
        """Test value_contains with empty search string."""
        parser = FallbackParser()
        result = parser.parse('value_contains(code, "")')
        
        condition = result.conditions[0]
        assert condition.value == "" or condition.value is None  # Handle both cases
        assert condition.is_value_contains_check is True
        
        where_clause, params = build_sql_where_clause(result)
        # Empty string becomes %% or %None%
        assert params[0] in ["%%", "%None%"]
    
    def test_value_contains_special_characters(self):
        """Test value_contains with special characters."""
        parser = FallbackParser()
        result = parser.parse('value_contains(code, "async/await")')
        
        condition = result.conditions[0]
        assert condition.value == "async/await"
        
        where_clause, params = build_sql_where_clause(result)
        assert params == ["%async/await%"]
    
    def test_value_contains_vs_exists_distinction(self):
        """Test that value_contains and exists are handled differently."""
        parser = FallbackParser()
        
        # Test exists
        exists_result = parser.parse('exists(embedding)')
        exists_condition = exists_result.conditions[0]
        assert exists_condition.is_exists_check is True
        assert exists_condition.is_value_contains_check is False
        
        # Test value_contains
        contains_result = parser.parse('value_contains(code, "test")')
        contains_condition = contains_result.conditions[0]
        assert contains_condition.is_exists_check is False
        assert contains_condition.is_value_contains_check is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])