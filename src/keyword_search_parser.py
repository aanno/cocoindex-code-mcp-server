#!/usr/bin/env python3

"""
Keyword search parser for metadata search with 'and', 'or', and 'exists' operators.
"""

import re
from typing import List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum


class Operator(Enum):
    AND = "and"
    OR = "or"


@dataclass
class SearchCondition:
    """Represents a single search condition."""
    field: str
    value: str
    is_exists_check: bool = False


@dataclass
class SearchGroup:
    """Represents a group of search conditions with an operator."""
    conditions: List[Union[SearchCondition, 'SearchGroup']]
    operator: Operator = Operator.AND


class KeywordSearchParser:
    """Parser for keyword search syntax."""
    
    def __init__(self):
        # Pattern for field:value pairs - handle quoted and unquoted values
        self.field_value_pattern = re.compile(r'(\w+):(?:(["\'])([^"\']*?)\2|([^\s]+))')
        # Pattern for exists checks
        self.exists_pattern = re.compile(r'exists\s*\(\s*(\w+)\s*\)', re.IGNORECASE)
        # Pattern for operators
        self.operator_pattern = re.compile(r'\b(and|or)\b', re.IGNORECASE)
        # Pattern for parentheses groups
        self.group_pattern = re.compile(r'\(([^)]+)\)')
    
    def parse(self, query: str) -> SearchGroup:
        """
        Parse a keyword search query into a SearchGroup.
        
        Supported syntax:
        - field:value - match field equals value
        - field:"quoted value" - match field with quoted value
        - exists(field) - check if field exists
        - and / or - logical operators
        - (group) - parentheses for grouping
        
        Examples:
        - language:python and filename:main.py
        - (language:python or language:rust) and exists(embedding)
        - filename:"test file.py" and language:python
        """
        if not query or not query.strip():
            return SearchGroup(conditions=[])
        
        # Parse the query into tokens and build the search tree
        return self._parse_expression(query.strip())
    
    def _parse_expression(self, expr: str) -> SearchGroup:
        """Parse a full expression, handling operators and groups."""
        # Handle parentheses first, but skip exists() function calls
        while '(' in expr:
            match = self.group_pattern.search(expr)
            if not match:
                break
            
            # Check if this is an exists() function call
            start_pos = match.start()
            if start_pos >= 6:  # "exists".length
                before_paren = expr[start_pos-6:start_pos].lower()
                if before_paren == 'exists':
                    # This is an exists() call, not a grouping - find next parentheses
                    next_match = self.group_pattern.search(expr, match.end())
                    if next_match:
                        match = next_match
                    else:
                        break
            
            group_content = match.group(1)
            group_result = self._parse_expression(group_content)
            # Replace the parentheses group with a placeholder
            placeholder = f"__GROUP_{id(group_result)}__"
            expr = expr[:match.start()] + placeholder + expr[match.end():]
            # Store the group for later use
            if not hasattr(self, '_groups'):
                self._groups = {}
            self._groups[placeholder] = group_result
        
        # Split by OR operators (lowest precedence)
        or_parts = self._split_by_operator(expr, 'or')
        if len(or_parts) > 1:
            conditions = []
            for part in or_parts:
                conditions.append(self._parse_and_expression(part.strip()))
            return SearchGroup(conditions=conditions, operator=Operator.OR)
        
        # If no OR, parse as AND expression
        return self._parse_and_expression(expr)
    
    def _parse_and_expression(self, expr: str) -> SearchGroup:
        """Parse an AND expression."""
        and_parts = self._split_by_operator(expr, 'and')
        conditions = []
        
        for part in and_parts:
            part = part.strip()
            if part.startswith('__GROUP_') and part.endswith('__'):
                # Replace group placeholder with actual group
                if hasattr(self, '_groups') and part in self._groups:
                    conditions.append(self._groups[part])
            else:
                # Parse individual condition
                condition = self._parse_condition(part)
                if condition:
                    conditions.append(condition)
        
        return SearchGroup(conditions=conditions, operator=Operator.AND)
    
    def _split_by_operator(self, expr: str, operator: str) -> List[str]:
        """Split expression by operator, respecting quotes and parentheses."""
        parts = []
        current_part = ""
        in_quotes = False
        quote_char = None
        paren_depth = 0
        i = 0
        
        while i < len(expr):
            char = expr[i]
            
            if char in ['"', "'"] and (i == 0 or expr[i-1] != '\\'):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
            
            if not in_quotes:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif paren_depth == 0:
                    # Look for operator with word boundaries
                    remaining = expr[i:]
                    pattern = rf'\b{operator}\b'
                    match = re.match(pattern, remaining, re.IGNORECASE)
                    if match:
                        parts.append(current_part.strip())
                        current_part = ""
                        i += len(match.group(0))
                        # Skip any following whitespace
                        while i < len(expr) and expr[i].isspace():
                            i += 1
                        continue
            
            current_part += char
            i += 1
        
        if current_part.strip():
            parts.append(current_part.strip())
        
        return parts if len(parts) > 1 else [expr]
    
    def _parse_condition(self, condition: str) -> Union[SearchCondition, None]:
        """Parse a single condition."""
        condition = condition.strip()
        
        # Check for exists condition first
        exists_match = self.exists_pattern.search(condition)
        if exists_match:
            field = exists_match.group(1)
            return SearchCondition(field=field, value="", is_exists_check=True)
        
        # Check for field:value condition
        field_value_match = self.field_value_pattern.search(condition)
        if field_value_match:
            field = field_value_match.group(1)
            # Group 3 is quoted value, group 4 is unquoted value
            value = field_value_match.group(3) or field_value_match.group(4)
            return SearchCondition(field=field, value=value)
        
        # If no pattern matches, treat as a general text search
        return SearchCondition(field="_text", value=condition)


def build_sql_where_clause(search_group: SearchGroup, table_alias: str = "") -> tuple[str, List[Any]]:
    """
    Build a SQL WHERE clause from a SearchGroup.
    
    Returns:
        tuple: (where_clause, parameters)
    """
    if not search_group.conditions:
        return "TRUE", []
    
    where_parts = []
    params = []
    prefix = f"{table_alias}." if table_alias else ""
    
    for condition in search_group.conditions:
        if isinstance(condition, SearchCondition):
            if condition.is_exists_check:
                where_parts.append(f"{prefix}{condition.field} IS NOT NULL")
            elif condition.field == "_text":
                # General text search across code content
                where_parts.append(f"{prefix}code ILIKE %s")
                params.append(f"%{condition.value}%")
            else:
                where_parts.append(f"{prefix}{condition.field} = %s")
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
        "language:python and filename:main.py",
        "(language:python or language:rust) and exists(embedding)",
        'filename:"test file.py" and language:python',
        "exists(embedding) and (language:rust or language:go)",
        "python function",  # general text search
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = parser.parse(query)
        where_clause, params = build_sql_where_clause(result)
        print(f"SQL WHERE: {where_clause}")
        print(f"Params: {params}")