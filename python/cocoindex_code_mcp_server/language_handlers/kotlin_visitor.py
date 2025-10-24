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
Kotlin-specific AST visitor for metadata extraction.

Follows the same pattern as haskell_visitor.py by subclassing GenericMetadataVisitor.
"""

import logging
from typing import Any, Dict, List, Optional

# from cocoindex_code_mcp_server.ast_visitor import NodeContext
from tree_sitter import Node

from ..ast_visitor import GenericMetadataVisitor, NodeContext
from ..parser_util import update_defaults

LOGGER = logging.getLogger(__name__)


class KotlinASTVisitor(GenericMetadataVisitor):
    """Specialized visitor for Kotlin language AST analysis."""

    def __init__(self) -> None:
        super().__init__("kotlin")
        self.functions: List[str] = []
        self.classes: List[str] = []
        self.interfaces: List[str] = []
        self.objects: List[str] = []
        self.data_classes: List[str] = []
        self.enums: List[str] = []

    def visit_node(self, context: NodeContext) -> Optional[Dict[str, Any]]:
        """Visit a node and extract Kotlin-specific metadata."""
        node = context.node
        node_type = node.type if hasattr(node, "type") else str(type(node))

        # Track node statistics
        self.node_stats[node_type] = self.node_stats.get(node_type, 0) + 1

        # Update complexity score based on node type (inherited from GenericMetadataVisitor)
        self._update_complexity(node_type)
        # Extract Kotlin-specific constructs
        if node_type == "function_declaration":
            self._extract_function(node)
        elif node_type == "class_declaration":
            self._extract_class(node)
        elif node_type == "interface_declaration":
            self._extract_interface(node)
        elif node_type == "object_declaration":
            self._extract_object(node)
        elif node_type == "enum_class_declaration":
            self._extract_enum(node)

        return None

    def _extract_function(self, node: Node) -> None:
        """Extract function name from function_declaration node."""
        try:
            # Kotlin function structure: function_declaration -> identifier (after 'fun' keyword)
            for child in node.children:
                if child.type == "identifier":
                    text = child.text
                    if text is not None:
                        func_name = text.decode("utf-8")
                        self.functions.append(func_name)
                        LOGGER.debug("Found Kotlin function: %s", func_name)
                        break  # Take the first identifier (function name)
        except Exception as e:
            LOGGER.warning("Error extracting Kotlin function: %s", e)

    def _extract_class(self, node):
        """Extract class name from class_declaration node."""
        try:
            # Look for class name (identifier after 'class' keyword)
            is_data_class = False
            class_name = None

            # Check if it's a data class by looking at modifiers
            for child in node.children:
                if child.type == "modifiers":
                    # Check for data modifier
                    for modifier_child in child.children:
                        if modifier_child.type == "class_modifier":
                            for modifier_grandchild in modifier_child.children:
                                if modifier_grandchild.type == "data":
                                    is_data_class = True
                elif child.type == "identifier":
                    class_name = child.text.decode("utf-8")
                    break

            if class_name:
                if is_data_class:
                    self.data_classes.append(class_name)
                    LOGGER.debug("Found Kotlin data class: %s", class_name)
                else:
                    self.classes.append(class_name)
                    LOGGER.debug("Found Kotlin class: %s", class_name)

        except Exception as e:
            LOGGER.warning("Error extracting Kotlin class: %s", e)

    def _extract_interface(self, node):
        """Extract interface name from interface_declaration node."""
        try:
            # Look for interface name
            for child in node.children:
                if child.type == "identifier":
                    interface_name = child.text.decode("utf-8")
                    self.interfaces.append(interface_name)
                    LOGGER.debug("Found Kotlin interface: %s", interface_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Kotlin interface: %s", e)

    def _extract_object(self, node):
        """Extract object name from object_declaration node."""
        try:
            # Look for object name
            for child in node.children:
                if child.type == "identifier":
                    object_name = child.text.decode("utf-8")
                    self.objects.append(object_name)
                    LOGGER.debug("Found Kotlin object: %s", object_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Kotlin object: %s", e)

    def _extract_enum(self, node):
        """Extract enum name from enum_class_declaration node."""
        try:
            # Look for enum name
            for child in node.children:
                if child.type == "identifier":
                    enum_name = child.text.decode("utf-8")
                    self.enums.append(enum_name)
                    LOGGER.debug("Found Kotlin enum: %s", enum_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Kotlin enum: %s", e)

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary in the expected format."""
        return {
            "functions": self.functions,
            "classes": self.classes,
            "interfaces": self.interfaces,
            "objects": self.objects,
            "data_classes": self.data_classes,
            "enums": self.enums,
            "node_stats": dict(self.node_stats),
            "complexity_score": self.complexity_score,
            "analysis_method": "kotlin_ast_visitor",
        }


def analyze_kotlin_code(code: str, filename: str = "") -> Dict[str, Any]:
    """
    Analyze Kotlin code using the specialized Kotlin AST visitor.
    This function mirrors analyze_haskell_code from haskell_visitor.py
    """
    try:
        from ..ast_visitor import ASTParserFactory, TreeWalker

        # Create parser and parse code
        factory = ASTParserFactory()
        parser = factory.create_parser("kotlin")
        if not parser:
            LOGGER.warning("Kotlin parser not available")
            return {"success": False, "error": "Kotlin parser not available"}

        tree = factory.parse_code(code, "kotlin")
        if not tree:
            LOGGER.warning("Failed to parse Kotlin code")
            return {"success": False, "error": "Failed to parse Kotlin code"}

        # Use specialized Kotlin visitor
        visitor = KotlinASTVisitor()
        walker = TreeWalker(code, tree)
        walker.walk(visitor)

        # Get results from visitor
        result = visitor.get_summary()
        # Use display language name for database storage
        from ..mappers import get_display_language_name

        update_defaults(
            result,
            {
                "success": True,
                "language": get_display_language_name("kotlin"),
                "filename": filename,
                "line_count": code.count("\n") + 1,
                "char_count": len(code),
                "parse_errors": 0,
                "tree_language": str(parser.language) if parser else None,
                # Required metadata fields for promoted column implementation
                # don't set chunking method in analyzer
                # "chunking_method": "ast_tree_sitter",
                # "tree_sitter_chunking_error": False,
                "tree_sitter_analyze_error": False,
                "decorators_used": result.get("annotations", []),  # Kotlin uses annotations like Java
                "has_type_hints": True,  # Kotlin has strong typing
                # Kotlin uses 'suspend' for async
                "has_async": any("suspend" in func.lower() for func in result.get("functions", [])),
                "has_classes": len(result.get("classes", [])) > 0,
            },
        )

        LOGGER.debug(
            "Kotlin analysis completed: %s functions, %s classes found",
            len(result.get("functions", [])),
            len(result.get("classes", [])),
        )
        return result

    except Exception as e:
        LOGGER.error("Kotlin code analysis failed: %s", e)
        return {"success": False, "error": str(e)}
