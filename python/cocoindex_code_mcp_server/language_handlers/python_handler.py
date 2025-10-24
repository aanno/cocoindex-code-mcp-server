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
Python-specific AST node handler for tree-sitter based analysis.

Handles Python-specific constructs like decorators, async functions, comprehensions, etc.
"""

import os

# Import from parent directory
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# from cocoindex_code_mcp_server.ast_visitor import NodeContext
from tree_sitter import Node

from ..ast_visitor import NodeContext
from . import LOGGER

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@dataclass
class PythonFunction:
    """Represents a Python function with detailed metadata."""

    name: str
    line: int
    column: int
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: Set[str] = field(default_factory=set)
    docstring: Optional[str] = None
    is_async: bool = False
    is_private: bool = False
    is_dunder: bool = False
    complexity: int = 0


@dataclass
class PythonClass:
    """Represents a Python class with detailed metadata."""

    name: str
    line: int
    column: int
    bases: List[str] = field(default_factory=list)
    decorators: Set[str] = field(default_factory=set)
    docstring: Optional[str] = None
    methods: List[str] = field(default_factory=list)
    is_private: bool = False


@dataclass
class PythonImport:
    """Represents a Python import statement."""

    module: str
    names: List[str] = field(default_factory=list)
    alias: Optional[str] = None
    is_from_import: bool = False
    line: int = 0


class PythonNodeHandler:
    """Python-specific tree-sitter node handler."""

    # Python-specific node types mapping
    PYTHON_NODE_TYPES = {
        # Functions and methods
        "function_definition": "function",
        "async_function_definition": "async_function",
        # Classes
        "class_definition": "class",
        # Imports
        "import_statement": "import",
        "import_from_statement": "from_import",
        # Variables and assignments
        "assignment": "assignment",
        "augmented_assignment": "augmented_assignment",
        "expression_statement": "expression",
        # Control flow
        "if_statement": "conditional",
        "for_statement": "loop",
        "while_statement": "loop",
        "try_statement": "exception_handling",
        "with_statement": "context_manager",
        # Expressions
        "call": "function_call",
        "lambda": "lambda",
        "list_comprehension": "comprehension",
        "dictionary_comprehension": "comprehension",
        "set_comprehension": "comprehension",
        "generator_expression": "comprehension",
        # Literals and constants
        "string": "string_literal",
        "integer": "number_literal",
        "float": "number_literal",
        "true": "boolean_literal",
        "false": "boolean_literal",
        "none": "none_literal",
        # Comments and docstrings
        "comment": "comment",
    }

    def __init__(self) -> None:
        self.functions: List[PythonFunction] = []
        self.classes: List[PythonClass] = []
        self.imports: List[PythonImport] = []
        self.variables: List[str] = []
        self.decorators: Set[str] = set()
        self.complexity_score = 0
        self.current_class: Optional[str] = None
        self.scope_stack: List[str] = []

    def can_handle(self, node_type: str) -> bool:
        """Check if this handler can process the given node type."""
        return node_type in self.PYTHON_NODE_TYPES

    def extract_metadata(self, context: NodeContext) -> Dict[str, Any]:
        """Extract metadata from a Python AST node."""
        node_type = context.node.type if hasattr(context.node, "type") else str(type(context.node))

        if not self.can_handle(node_type):
            return {}

        handler_method = f"_handle_{self.PYTHON_NODE_TYPES[node_type]}"

        if hasattr(self, handler_method):
            try:
                result = getattr(self, handler_method)(context) or {}
                LOGGER.debug("Handled %s with result: %s", node_type, result)
                return result
            except Exception as e:
                LOGGER.warning("Error handling %s: %s", node_type, e)
                return {}

        return {}

    def _handle_function(self, context: NodeContext) -> Dict[str, Any]:
        """Handle function definitions."""
        return self._extract_function_info(context, is_async=False)

    def _handle_async_function(self, context: NodeContext) -> Dict[str, Any]:
        """Handle async function definitions."""
        return self._extract_function_info(context, is_async=True)

    def _extract_function_info(self, context: NodeContext, is_async: bool = False) -> Dict[str, Any]:
        """Extract detailed function information."""
        node = context.node
        function_name = self._extract_function_name(node, context.source_text)

        if not function_name:
            return {}

        position = context.get_position()

        func_info = PythonFunction(
            name=function_name,
            line=position.line,
            column=position.column,
            is_async=is_async,
            is_private=function_name.startswith("_"),
            is_dunder=function_name.startswith("__") and function_name.endswith("__"),
        )

        # Extract parameters
        func_info.parameters = self._extract_parameters(node, context.source_text)

        # Extract return type annotation
        func_info.return_type = self._extract_return_type(node, context.source_text)

        # Extract decorators
        func_info.decorators = self._extract_decorators(node, context.source_text)
        self.decorators.update(func_info.decorators)

        # Extract docstring
        func_info.docstring = self._extract_docstring(node, context.source_text)

        # Calculate function-specific complexity
        func_info.complexity = self._calculate_function_complexity(node, context.source_text)
        self.complexity_score += func_info.complexity

        self.functions.append(func_info)

        return {
            "function_found": {
                "name": func_info.name,
                "is_async": func_info.is_async,
                "is_private": func_info.is_private,
                "parameter_count": len(func_info.parameters),
                "has_decorators": len(func_info.decorators) > 0,
                "has_return_type": func_info.return_type is not None,
                "complexity": func_info.complexity,
            }
        }

    def _handle_class(self, context: NodeContext) -> Dict[str, Any]:
        """Handle class definitions."""
        node = context.node
        class_name = self._extract_class_name(node, context.source_text)

        if not class_name:
            return {}

        position = context.get_position()

        class_info = PythonClass(
            name=class_name, line=position.line, column=position.column, is_private=class_name.startswith("_")
        )

        # Extract base classes
        class_info.bases = self._extract_base_classes(node, context.source_text)

        # Extract decorators
        class_info.decorators = self._extract_decorators(node, context.source_text)
        self.decorators.update(class_info.decorators)

        # Extract docstring
        class_info.docstring = self._extract_docstring(node, context.source_text)

        self.classes.append(class_info)
        self.current_class = class_name

        return {
            "class_found": {
                "name": class_info.name,
                "is_private": class_info.is_private,
                "base_count": len(class_info.bases),
                "has_decorators": len(class_info.decorators) > 0,
            }
        }

    def _handle_import(self, context: NodeContext) -> Dict[str, Any]:
        """Handle import statements."""
        return self._extract_import_info(context, is_from_import=False)

    def _handle_from_import(self, context: NodeContext) -> Dict[str, Any]:
        """Handle from...import statements."""
        return self._extract_import_info(context, is_from_import=True)

    def _extract_import_info(self, context: NodeContext, is_from_import: bool) -> Dict[str, Any]:
        """Extract import information."""
        node = context.node
        position = context.get_position()

        import_info = PythonImport(module="", is_from_import=is_from_import, line=position.line)

        # Extract module and names based on import type
        if is_from_import:
            # from module import name1, name2
            module_name, imported_names = self._extract_from_import_parts(node, context.source_text)
            import_info.module = module_name or ""
            import_info.names = imported_names
        else:
            # import module [as alias]
            module_name, alias = self._extract_import_parts(node, context.source_text)
            import_info.module = module_name or ""
            import_info.alias = alias

        self.imports.append(import_info)

        return {
            "import_found": {
                "module": import_info.module,
                "is_from_import": import_info.is_from_import,
                "names": import_info.names if import_info.names else [import_info.module],
            }
        }

    def _handle_assignment(self, context: NodeContext) -> Dict[str, Any]:
        """Handle variable assignments."""
        node = context.node
        var_names = self._extract_assignment_targets(node, context.source_text)

        # Only track module-level variables (not in function/class scope)
        if context.depth <= 1:  # Root level or minimal nesting
            self.variables.extend(var_names)

        return {"assignment_found": {"targets": var_names, "is_module_level": context.depth <= 1}} if var_names else {}

    def _handle_conditional(self, context: NodeContext) -> Dict[str, Any]:
        """Handle if statements."""
        self.complexity_score += 1
        return {"conditional_complexity": 1}

    def _handle_loop(self, context: NodeContext) -> Dict[str, Any]:
        """Handle for/while loops."""
        self.complexity_score += 1
        return {"loop_complexity": 1}

    def _handle_exception_handling(self, context: NodeContext) -> Dict[str, Any]:
        """Handle try/except blocks."""
        self.complexity_score += 1
        return {"exception_handling_complexity": 1}

    def _handle_comprehension(self, context: NodeContext) -> Dict[str, Any]:
        """Handle list/dict/set comprehensions and generator expressions."""
        self.complexity_score += 2
        return {"comprehension_complexity": 2}

    def _handle_lambda(self, context: NodeContext) -> Dict[str, Any]:
        """Handle lambda functions."""
        self.complexity_score += 1
        return {"lambda_complexity": 1}

    # Helper methods for extracting specific information

    def _extract_function_name(self, node: Node, source_text: str) -> Optional[str]:
        """Extract function name from function definition node."""
        if not hasattr(node, "children"):
            return None

        for child in node.children:
            if hasattr(child, "type") and child.type == "identifier":
                return source_text[child.start_byte: child.end_byte]

        return None

    def _extract_class_name(self, node: Node, source_text: str) -> Optional[str]:
        """Extract class name from class definition node."""
        if not hasattr(node, "children"):
            return None

        for child in node.children:
            if hasattr(child, "type") and child.type == "identifier":
                return source_text[child.start_byte: child.end_byte]

        return None

    def _extract_parameters(self, node: Node, source_text: str) -> List[Dict[str, Any]]:
        """Extract function parameters with type annotations and defaults."""
        parameters: List[Dict[str, Any]] = []

        if not hasattr(node, "children"):
            return parameters

        # Find parameters node
        for child in node.children:
            if hasattr(child, "type") and child.type == "parameters":
                parameters = self._parse_parameters_node(child, source_text)
                break

        return parameters

    def _parse_parameters_node(self, params_node: Node, source_text: str) -> List[Dict[str, Any]]:
        """Parse individual parameters from parameters node."""
        parameters: List[Dict[str, Any]] = []

        if not hasattr(params_node, "children"):
            return parameters

        for child in params_node.children:
            if hasattr(child, "type"):
                if child.type == "identifier":
                    # Simple parameter
                    param_name = source_text[child.start_byte: child.end_byte]
                    parameters.append({"name": param_name, "type_annotation": None, "default": None})
                elif child.type == "typed_parameter":
                    # Parameter with type annotation
                    param_info = self._parse_typed_parameter(child, source_text)
                    if param_info:
                        parameters.append(param_info)
                elif child.type == "default_parameter":
                    # Parameter with default value
                    param_info = self._parse_default_parameter(child, source_text)
                    if param_info:
                        parameters.append(param_info)

        return parameters

    def _parse_typed_parameter(self, param_node: Node, source_text: str) -> Optional[Dict[str, Any]]:
        """Parse a typed parameter (name: type)."""
        param_info = {"name": "", "type_annotation": None, "default": None}

        if not hasattr(param_node, "children"):
            return None

        for child in param_node.children:
            if hasattr(child, "type"):
                if child.type == "identifier":
                    param_info["name"] = source_text[child.start_byte: child.end_byte]
                else:
                    # Type annotation
                    param_info["type_annotation"] = source_text[child.start_byte: child.end_byte]

        return param_info if param_info["name"] else None

    def _parse_default_parameter(self, param_node, source_text: str) -> Optional[Dict[str, Any]]:
        """Parse a parameter with default value (name=default)."""
        param_info = {"name": "", "type_annotation": None, "default": None}

        if not hasattr(param_node, "children"):
            return None

        children = list(param_node.children)
        if len(children) >= 2:
            # First child is parameter name
            if hasattr(children[0], "type") and children[0].type == "identifier":
                param_info["name"] = source_text[children[0].start_byte: children[0].end_byte]

            # Last child is default value
            param_info["default"] = source_text[children[-1].start_byte: children[-1].end_byte]

        return param_info if param_info["name"] else None

    def _extract_return_type(self, node: Node, source_text: str) -> Optional[str]:
        """Extract return type annotation from function definition."""
        if not hasattr(node, "children"):
            return None

        for child in node.children:
            if hasattr(child, "type") and child.type == "type":
                return source_text[child.start_byte: child.end_byte]

        return None

    def _extract_decorators(self, node: Node, source_text: str) -> Set[str]:
        """Extract decorators from function or class definition."""
        decorators: Set[str] = set()

        if not hasattr(node, "children"):
            return decorators

        for child in node.children:
            if hasattr(child, "type") and child.type == "decorator":
                decorator_text = source_text[child.start_byte: child.end_byte]
                # Remove @ symbol and extract decorator name
                if decorator_text.startswith("@"):
                    decorator_name = decorator_text[1:].strip()
                    # Handle decorator calls like @decorator()
                    if "(" in decorator_name:
                        decorator_name = decorator_name.split("(")[0]
                    decorators.add(decorator_name)

        return decorators

    def _extract_docstring(self, node: Node, source_text: str) -> Optional[str]:
        """Extract docstring from function or class definition."""
        if not hasattr(node, "children"):
            return None

        # Look for string literal as first statement in body
        for child in node.children:
            if hasattr(child, "type") and child.type in ["block", "suite"]:
                return self._find_docstring_in_block(child, source_text)

        return None

    def _find_docstring_in_block(self, block_node: Node, source_text: str) -> Optional[str]:
        """Find docstring in a code block."""
        if not hasattr(block_node, "children"):
            return None

        for child in block_node.children:
            if hasattr(child, "type"):
                if child.type == "expression_statement":
                    # Check if this expression statement contains a string
                    docstring = self._extract_string_from_expression(child, source_text)
                    if docstring:
                        return docstring
                elif child.type not in ["comment", "newline"]:
                    # First non-comment/newline statement that's not a string means no docstring
                    break

        return None

    def _extract_string_from_expression(self, expr_node: Node, source_text: str) -> Optional[str]:
        """Extract string literal from expression statement."""
        if not hasattr(expr_node, "children"):
            return None

        for child in expr_node.children:
            if hasattr(child, "type") and child.type == "string":
                string_content = source_text[child.start_byte: child.end_byte]
                # Remove quotes and return content
                return self._clean_string_literal(string_content)

        return None

    def _clean_string_literal(self, string_literal: str) -> str:
        """Clean string literal by removing quotes and escapes."""
        # Remove triple quotes or single/double quotes
        if string_literal.startswith('"""') or string_literal.startswith("'''"):
            return string_literal[3:-3].strip()
        elif string_literal.startswith('"') or string_literal.startswith("'"):
            return string_literal[1:-1].strip()

        return string_literal.strip()

    def _extract_base_classes(self, node: Node, source_text: str) -> List[str]:
        """Extract base classes from class definition."""
        bases: List[str] = []

        if not hasattr(node, "children"):
            return bases

        for child in node.children:
            if hasattr(child, "type") and child.type == "argument_list":
                # Extract base class names from argument list
                bases.extend(self._extract_names_from_argument_list(child, source_text))

        return bases

    def _extract_names_from_argument_list(self, arg_list_node: Node, source_text: str) -> List[str]:
        """Extract identifiers from argument list."""
        names: List[str] = []

        if not hasattr(arg_list_node, "children"):
            return names

        for child in arg_list_node.children:
            if hasattr(child, "type"):
                if child.type == "identifier":
                    names.append(source_text[child.start_byte: child.end_byte])
                elif child.type == "attribute":
                    # Handle module.Class syntax
                    names.append(source_text[child.start_byte: child.end_byte])

        return names

    def _extract_import_parts(self, node: Node, source_text: str) -> tuple[Optional[str], Optional[str]]:
        """Extract module name and alias from import statement."""
        module_name = None
        alias = None

        if not hasattr(node, "children"):
            return module_name, alias

        for child in node.children:
            if hasattr(child, "type"):
                if child.type in ["dotted_name", "identifier"]:
                    if not module_name:
                        module_name = source_text[child.start_byte: child.end_byte]
                elif child.type == "aliased_import":
                    # Handle "import module as alias"
                    module_name, alias = self._extract_aliased_import(child, source_text)

        return module_name, alias

    def _extract_from_import_parts(self, node: Node, source_text: str) -> tuple[Optional[str], List[str]]:
        """Extract module and imported names from from...import statement."""
        module_name: Optional[str] = None
        imported_names: List[str] = []

        if not hasattr(node, "children"):
            return module_name, imported_names

        for child in node.children:
            if hasattr(child, "type"):
                if child.type in ["dotted_name", "identifier"] and not module_name:
                    module_name = source_text[child.start_byte: child.end_byte]
                elif child.type == "import_list":
                    imported_names = self._extract_import_list(child, source_text)

        return module_name, imported_names

    def _extract_aliased_import(self, aliased_node: Node, source_text: str) -> tuple[Optional[str], Optional[str]]:
        """Extract module and alias from aliased import."""
        module_name = None
        alias = None

        if not hasattr(aliased_node, "children"):
            return module_name, alias

        children = list(aliased_node.children)
        if len(children) >= 2:
            # First child is module name
            if hasattr(children[0], "type"):
                module_name = source_text[children[0].start_byte: children[0].end_byte]

            # Last child is alias
            if hasattr(children[-1], "type"):
                alias = source_text[children[-1].start_byte: children[-1].end_byte]

        return module_name, alias

    def _extract_import_list(self, import_list_node: Node, source_text: str) -> List[str]:
        """Extract list of imported names."""
        names: List[str] = []

        if not hasattr(import_list_node, "children"):
            return names

        for child in import_list_node.children:
            if hasattr(child, "type"):
                if child.type == "identifier":
                    names.append(source_text[child.start_byte: child.end_byte])
                elif child.type == "aliased_import":
                    # Handle "from module import name as alias"
                    original_name, alias = self._extract_aliased_import(child, source_text)
                    if original_name:
                        names.append(alias or original_name)

        return names

    def _extract_assignment_targets(self, node: Node, source_text: str) -> List[str]:
        """Extract variable names from assignment targets."""
        targets: List[str] = []

        if not hasattr(node, "children"):
            return targets

        for child in node.children:
            if hasattr(child, "type"):
                if child.type == "identifier":
                    targets.append(source_text[child.start_byte: child.end_byte])
                elif child.type in ["pattern_list", "tuple_pattern"]:
                    # Handle multiple assignment like a, b = 1, 2
                    targets.extend(self._extract_pattern_names(child, source_text))

        return targets

    def _extract_pattern_names(self, pattern_node: Node, source_text: str) -> List[str]:
        """Extract names from assignment patterns."""
        names: List[str] = []

        if not hasattr(pattern_node, "children"):
            return names

        for child in pattern_node.children:
            if hasattr(child, "type") and child.type == "identifier":
                names.append(source_text[child.start_byte: child.end_byte])

        return names

    def _calculate_function_complexity(self, node: Node, source_text: str) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity

        # Count control flow statements
        function_text = source_text[node.start_byte: node.end_byte] if hasattr(node, "start_byte") else ""

        # Simple complexity indicators
        complexity += function_text.count("if ")
        complexity += function_text.count("elif ")
        complexity += function_text.count("for ")
        complexity += function_text.count("while ")
        complexity += function_text.count("except ")
        complexity += function_text.count("and ")
        complexity += function_text.count("or ")
        complexity += function_text.count("try:")

        return complexity

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all extracted metadata."""
        return {
            "functions": [func.name for func in self.functions],
            "classes": [cls.name for cls in self.classes],
            "imports": list(set(imp.module for imp in self.imports if imp.module)),
            "variables": list(set(self.variables)),
            "decorators": list(self.decorators),
            "complexity_score": self.complexity_score,
            "has_async": any(func.is_async for func in self.functions),
            "has_classes": len(self.classes) > 0,
            "has_decorators": len(self.decorators) > 0,
            "has_type_hints": any(
                func.return_type or any(p.get("type_annotation") for p in func.parameters) for func in self.functions
            ),
            "private_methods": [func.name for func in self.functions if func.is_private],
            "dunder_methods": [func.name for func in self.functions if func.is_dunder],
            "function_details": [
                {
                    "name": func.name,
                    "line": func.line,
                    "is_async": func.is_async,
                    "parameter_count": len(func.parameters),
                    "complexity": func.complexity,
                    "has_decorators": len(func.decorators) > 0,
                    "has_docstring": func.docstring is not None,
                }
                for func in self.functions
            ],
            "class_details": [
                {
                    "name": cls.name,
                    "line": cls.line,
                    "base_count": len(cls.bases),
                    "has_decorators": len(cls.decorators) > 0,
                    "has_docstring": cls.docstring is not None,
                }
                for cls in self.classes
            ],
            "import_details": [
                {"module": imp.module, "names": imp.names, "is_from_import": imp.is_from_import, "line": imp.line}
                for imp in self.imports
            ],
        }
