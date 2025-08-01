#!/usr/bin/env python3

"""
Lark-based keyword search parser for metadata search with enhanced syntax support.
Supports field:value, exists(field), value_contains(field, string), and boolean operators.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Union

try:
    from lark import Lark, Token, Transformer
    from lark.exceptions import LarkError, ParseError
    LARK_AVAILABLE = True
except ImportError:
    LARK_AVAILABLE = False

# Import the fallback parser components
from .keyword_search_parser import KeywordSearchParser as FallbackParser
from .keyword_search_parser import (
    build_sql_where_clause as fallback_build_sql_where_clause,
)
FALLBACK_AVAILABLE = True

# Configure logging
logger = logging.getLogger(__name__)

class Operator(Enum):
    AND = "and"
    OR = "or"


@dataclass
class SearchCondition:
    """Represents a single search condition."""
    field: str
    value: str
    is_exists_check: bool = False
    is_value_contains_check: bool = False


@dataclass
class SearchGroup:
    """Represents a group of search conditions with an operator."""
    conditions: List[Union[SearchCondition, 'SearchGroup']]
    operator: Operator = Operator.AND


class KeywordSearchTransformer(Transformer):
    """Transformer to convert Lark parse tree to SearchGroup objects."""

    def remove_quotes(self, token: str) -> str:
        """Remove surrounding quotes from quoted strings."""
        s = str(token)
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        return s

    def QUOTED_VALUE(self, token):
        """Transform quoted value by removing quotes."""
        return self.remove_quotes(token)

    def field_condition(self, items) -> SearchCondition:
        """Transform field:value condition."""
        field, value = items
        return SearchCondition(field=str(field), value=str(value))

    def exists_condition(self, items) -> SearchCondition:
        """Transform exists(field) condition."""
        field = items[0]
        return SearchCondition(field=str(field), value="", is_exists_check=True)

    def value_contains_condition(self, items) -> SearchCondition:
        """Transform value_contains(field, string) condition."""
        field, value = items
        return SearchCondition(
            field=str(field),
            value=str(value),
            is_value_contains_check=True
        )

    def text_search(self, items) -> SearchCondition:
        """Transform general text search."""
        # Join multiple words with spaces
        value = " ".join(str(item) for item in items)
        return SearchCondition(field="_text", value=value)

    def word(self, items) -> str:
        """Transform word rule."""
        return str(items[0])

    def and_expr(self, items):
        """Transform AND expression."""
        if len(items) == 1:
            # Single item - wrap in a SearchGroup for consistency
            item = items[0]
            if isinstance(item, SearchCondition):
                return SearchGroup(conditions=[item], operator=Operator.AND)
            return item
        return SearchGroup(conditions=list(items), operator=Operator.AND)

    def or_expr(self, items):
        """Transform OR expression."""
        if len(items) == 1:
            # Single item - wrap in a SearchGroup for consistency
            item = items[0]
            if isinstance(item, SearchCondition):
                return SearchGroup(conditions=[item], operator=Operator.OR)
            return item
        return SearchGroup(conditions=list(items), operator=Operator.OR)

    def start(self, items) -> SearchGroup:
        """Transform start rule - always return SearchGroup."""
        item = items[0]
        if isinstance(item, SearchCondition):
            return SearchGroup(conditions=[item], operator=Operator.AND)
        elif isinstance(item, SearchGroup):
            return item
        else:
            # Shouldn't happen, but handle gracefully
            return SearchGroup(conditions=[], operator=Operator.AND)


class KeywordSearchParser:
    """Lark-based parser for keyword search syntax with fallback to regex parser."""

    def __init__(self) -> None:
        """Initialize the parser with Lark grammar or fallback."""
        self.lark_parser = None
        self.transformer = KeywordSearchTransformer()
        self.fallback_parser = None

        if LARK_AVAILABLE:
            try:
                # Load the grammar file
                grammar_path = Path(__file__).parent / "grammars" / "keyword_search.lark"
                if grammar_path.exists():
                    with open(grammar_path, 'r') as f:
                        grammar = f.read()

                    self.lark_parser = Lark(
                        grammar,
                        parser='lalr',
                        transformer=self.transformer,
                        # Make keywords case insensitive
                        lexer_callbacks={
                            'UNQUOTED_VALUE': lambda t: Token('UNQUOTED_VALUE', t.value),
                        }
                    )
                else:
                    logger.warning(f"Grammar file not found at {grammar_path}, falling back to regex parser")
            except Exception as e:
                logger.warning(f"Failed to initialize Lark parser ({e}), falling back to regex parser")

        # Initialize fallback parser if Lark is not available or failed
        if self.lark_parser is None:
            if FALLBACK_AVAILABLE:
                self.fallback_parser = FallbackParser()
                if not LARK_AVAILABLE:
                    logger.warning("Lark not available, using regex-based parser")
            else:
                logger.error("No parser available - neither Lark nor fallback parser could be loaded")

    def parse(self, query: str) -> SearchGroup:
        """
        Parse a keyword search query into a SearchGroup.

        Supported syntax:
        - field:value - match field equals value
        - field:"quoted value" - match field with quoted value
        - exists(field) - check if field exists
        - value_contains(field, "search_string") - check if field value contains string
        - and / or - logical operators
        - (group) - parentheses for grouping

        Examples:
        - language:python and filename:main_interactive_query.py
        - (language:python or language:rust) and exists(embedding)
        - filename:"test file.py" and language:python
        - value_contains(code, "function") and language:python
        """
        if not query or not query.strip():
            return SearchGroup(conditions=[])

        # Try Lark parser first
        if self.lark_parser is not None:
            try:
                # Normalize case for keywords
                normalized_query = self._normalize_keywords(query.strip())
                tree = self.lark_parser.parse(normalized_query)
                # The tree is already transformed to SearchGroup during parsing
                return tree  # type: ignore
            except (LarkError, ParseError) as e:
                logger.warning(f"Lark parser failed ({e}), falling back to regex parser")

        # Fall back to regex parser
        if self.fallback_parser is not None:
            fallback_result = self.fallback_parser.parse(query)
            # Convert fallback SearchGroup to our SearchGroup
            return self._convert_from_fallback(fallback_result)
        else:
            # Last resort: return empty group
            logger.error("No parser available")
            return SearchGroup(conditions=[])

    def _normalize_keywords(self, query: str) -> str:
        """Normalize keywords to lowercase for case-insensitive parsing."""
        import re

        # Replace keywords with lowercase versions, preserving word boundaries
        replacements = {
            r'\bAND\b': 'and',
            r'\bOr\b': 'or',
            r'\bOR\b': 'or',
            r'\bAnd\b': 'and',
            r'\bEXISTS\b': 'exists',
            r'\bExists\b': 'exists',
            r'\bVALUE_CONTAINS\b': 'value_contains',
            r'\bValue_Contains\b': 'value_contains',
            r'\bvalue_Contains\b': 'value_contains',
        }

        result = query
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    def _convert_from_fallback(self, fallback_group: Any) -> SearchGroup:
        """Convert fallback SearchGroup to our SearchGroup."""
        from .keyword_search_parser import SearchGroup as FallbackSearchGroup, SearchCondition as FallbackSearchCondition
        
        if not isinstance(fallback_group, FallbackSearchGroup):
            return SearchGroup(conditions=[])
            
        conditions: List[Union[SearchCondition, SearchGroup]] = []
        for condition in fallback_group.conditions:
            if isinstance(condition, FallbackSearchCondition):
                # Convert SearchCondition
                new_condition = SearchCondition(
                    field=condition.field,
                    value=condition.value,
                    is_exists_check=getattr(condition, 'is_exists_check', False),
                    is_value_contains_check=getattr(condition, 'is_value_contains_check', False)
                )
                conditions.append(new_condition)
            elif isinstance(condition, FallbackSearchGroup):
                # Recursively convert nested groups
                converted_group = self._convert_from_fallback(condition)
                conditions.append(converted_group)
                
        operator = Operator.AND
        if hasattr(fallback_group, 'operator'):
            if fallback_group.operator == "or":
                operator = Operator.OR
                
        return SearchGroup(conditions=conditions, operator=operator)


def build_sql_where_clause(search_group: SearchGroup, table_alias: str = "") -> tuple[str, List[Any]]:
    """
    Build a SQL WHERE clause from a SearchGroup with proper field validation and SQL injection protection.

    Returns:
        tuple: (where_clause, parameters)

    Raises:
        ValueError: If any field names are invalid or don't exist in the schema
    """
    # Import here to avoid circular imports
    from .schema_validator import schema_validator

    if not search_group.conditions:
        return "TRUE", []

    where_parts = []
    params = []
    prefix = f"{table_alias}." if table_alias else ""

    for condition in search_group.conditions:
        if isinstance(condition, SearchCondition):
            # Validate and map field name to prevent SQL injection and unknown column errors
            field_result = schema_validator.validate_field(condition.field)
            if not field_result.is_valid:
                raise ValueError(f"Invalid field in search condition: {field_result.error_message}")

            validated_field = field_result.mapped_field

            if condition.is_exists_check:
                where_parts.append(f"{prefix}{validated_field} IS NOT NULL")
            elif condition.is_value_contains_check:
                # value_contains(field, "search_string") -> field ILIKE %search_string%
                where_parts.append(f"{prefix}{validated_field} ILIKE %s")
                params.append(f"%{condition.value}%")
            elif condition.field == "_text":
                # General text search across code content - map to 'code' field
                where_parts.append(f"{prefix}code ILIKE %s")
                params.append(f"%{condition.value}%")
            else:
                where_parts.append(f"{prefix}{validated_field} = %s")
                params.append(condition.value)
        elif isinstance(condition, SearchGroup):
            sub_where, sub_params = build_sql_where_clause(condition, table_alias)
            where_parts.append(f"({sub_where})")
            params.extend(sub_params)

    operator_str = " OR " if search_group.operator == Operator.OR else " AND "
    where_clause = operator_str.join(where_parts)

    return where_clause, params


# Example usage and testing
if __name__ == "__main__":
    parser = KeywordSearchParser()

    test_queries = [
        "language:python",
        "language:python and filename:main_interactive_query.py",
        "(language:python or language:rust) and exists(embedding)",
        'filename:"test file.py" and language:python',
        "exists(embedding) and (language:rust or language:go)",
        "python function",  # general text search
        'value_contains(code, "function") and language:python',
        'value_contains(filename, "test") or exists(embedding)',
        '(value_contains(code, "async") and language:python) or language:rust',
    ]

    print(f"Using Lark parser: {parser.lark_parser is not None}")
    print(f"LARK_AVAILABLE: {LARK_AVAILABLE}")

    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = parser.parse(query)
            where_clause, params = build_sql_where_clause(result)
            print(f"SQL WHERE: {where_clause}")
            print(f"Params: {params}")
        except Exception as e:
            print(f"Error parsing query: {e}")
