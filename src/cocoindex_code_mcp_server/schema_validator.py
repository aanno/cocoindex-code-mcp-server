#!/usr/bin/env python3
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 aanno <aanno@users.noreply.github.com>
#
# This file is part of cocoindex_code_mcp_server from
# https://github.com/aanno/cocoindex-code-mcp-server
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Database schema validation and field mapping for safe SQL query generation.

Prevents SQL injection and unknown column errors.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

# Import single source of truth for field mappings
from .mappers import CONST_FIELD_MAPPINGS

# Valid database columns from single source of truth
VALID_COLUMNS: Set[str] = set(CONST_FIELD_MAPPINGS.keys())

# Field aliases - map user-friendly names to actual column names
FIELD_ALIASES: Dict[str, str] = {
    "path": "filename",
    "file": "filename",
    "filepath": "filename",
    "name": "filename",
    "lang": "language",
    "content": "code",
    "text": "code",
    "source": "code",
    "functions_list": "functions",
    "classes_list": "classes",
    "imports_list": "imports",
    "complexity": "complexity_score",
    "has_hints": "has_type_hints",
    "has_async_functions": "has_async",
    "source_file": "source_name",
    "metadata": "metadata_json",
}

# SQL-safe operators for conditions
SAFE_OPERATORS: Set[str] = {
    "=",
    "!=",
    "<>",
    "<",
    ">",
    "<=",
    ">=",
    "LIKE",
    "ILIKE",
    "IN",
    "NOT IN",
    "IS NULL",
    "IS NOT NULL",
}


@dataclass
class ValidationResult:
    """Result of field validation."""

    is_valid: bool
    mapped_field: Optional[str] = None
    error_message: Optional[str] = None


class SchemaValidator:
    """Validates and maps database fields to prevent SQL injection and schema errors."""

    def __init__(
        self, valid_columns: Optional[Set[str]] = None, field_aliases: Optional[Dict[str, str]] = None
    ) -> None:
        self.valid_columns = valid_columns or VALID_COLUMNS
        self.field_aliases = field_aliases or FIELD_ALIASES

    def validate_field(self, field: str) -> ValidationResult:
        """
        Validate a field name and return the mapped column name.

        Args:
            field: Field name from user query

        Returns:
            ValidationResult with validation status and mapped field
        """
        if not field or not isinstance(field, str):
            return ValidationResult(is_valid=False, error_message="Field name must be a non-empty string")

        # Sanitize field name - only allow alphanumeric and underscore
        if not field.replace("_", "").replace("-", "").isalnum():
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid field name '{field}' - only alphanumeric characters and underscores allowed",
            )

        # Check if field exists directly
        if field in self.valid_columns:
            return ValidationResult(is_valid=True, mapped_field=field)

        # Check field aliases
        if field in self.field_aliases:
            mapped_field = self.field_aliases[field]
            if mapped_field in self.valid_columns:
                return ValidationResult(is_valid=True, mapped_field=mapped_field)

        # Field not found
        valid_fields = sorted(list(self.valid_columns) + list(self.field_aliases.keys()))
        return ValidationResult(
            is_valid=False, error_message=f"Unknown field '{field}'. Valid fields: {', '.join(valid_fields)}"
        )

    def validate_operator(self, operator: str) -> bool:
        """Validate SQL operator to prevent injection."""
        return operator.upper().strip() in SAFE_OPERATORS

    def sanitize_value(self, value: Any) -> Any:
        """Sanitize a value for SQL parameter binding."""
        if isinstance(value, str):
            # Remove any potential SQL injection patterns
            # The actual protection comes from parameter binding, this is additional safety
            return value.replace("'", "''").replace(";", "").replace("--", "")
        return value

    def build_safe_condition(self, field: str, operator: str, value: Any) -> tuple[str, List[Any]]:
        """
        Build a safe SQL condition with proper parameter binding.

        Args:
            field: Field name to query
            operator: SQL operator
            value: Value to compare against

        Returns:
            tuple: (sql_condition, parameters)

        Raises:
            ValueError: If field or operator is invalid
        """
        # Validate field
        field_result = self.validate_field(field)
        if not field_result.is_valid:
            raise ValueError(field_result.error_message)

        mapped_field = field_result.mapped_field

        # Validate operator
        if not self.validate_operator(operator):
            raise ValueError(f"Invalid operator '{operator}'. Allowed: {', '.join(SAFE_OPERATORS)}")

        # Sanitize value
        safe_value = self.sanitize_value(value)

        # Build condition with parameter binding (NOT string concatenation)
        if operator.upper() in ["IS NULL", "IS NOT NULL"]:
            return f"{mapped_field} {operator.upper()}", []
        elif operator.upper() in ["IN", "NOT IN"]:
            if not isinstance(safe_value, (list, tuple)):
                raise ValueError(f"Operator {operator} requires a list/tuple value")
            placeholders = ", ".join(["%s"] * len(safe_value))
            return f"{mapped_field} {operator.upper()} ({placeholders})", list(safe_value)
        else:
            return f"{mapped_field} {operator} %s", [safe_value]

    def get_valid_fields_help(self) -> str:
        """Get help text showing valid fields and aliases."""
        help_text = "Valid database fields:\n"
        help_text += f"  Direct fields: {', '.join(sorted(self.valid_columns))}\n"
        help_text += f"  Aliases: {', '.join(sorted(self.field_aliases.keys()))}\n\n"
        help_text += "Field aliases mapping:\n"
        for alias, real_field in sorted(self.field_aliases.items()):
            help_text += f"  {alias} -> {real_field}\n"
        return help_text


# Global validator instance
schema_validator = SchemaValidator()


# Convenience functions
def validate_field(field: str) -> ValidationResult:
    """Validate a field name using the global validator."""
    return schema_validator.validate_field(field)


def build_safe_condition(field: str, operator: str = "=", value: Any = None) -> tuple[str, List[Any]]:
    """Build a safe SQL condition using the global validator."""
    return schema_validator.build_safe_condition(field, operator, value)


def get_valid_fields_help() -> str:
    """Get help for valid fields."""
    return schema_validator.get_valid_fields_help()


if __name__ == "__main__":
    # Test the validator
    validator = SchemaValidator()

    test_cases = [
        ("filename", "=", "test.py"),
        ("path", "=", "test.py"),  # Should map to filename
        ("invalid_field", "=", "value"),  # Should fail
        ("language", "LIKE", "%python%"),
        ("code", "ILIKE", "%function%"),
        ("complexity_score", ">", 5),
        ("has_async", "IS NOT NULL", None),
    ]

    for field, operator, value in test_cases:
        try:
            condition, params = validator.build_safe_condition(field, operator, value)
            print(f"✅ {field} {operator} {value} -> {condition} with params {params}")
        except ValueError as e:
            print(f"❌ {field} {operator} {value} -> ERROR: {e}")

    print("\n" + validator.get_valid_fields_help())
