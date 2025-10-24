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
Java-specific AST visitor for metadata extraction.

Follows the same pattern as other language visitors by subclassing GenericMetadataVisitor.
"""

import logging
from typing import Any, Dict, List, Optional

# from cocoindex_code_mcp_server.ast_visitor import NodeContext
from tree_sitter import Node

from ..ast_visitor import GenericMetadataVisitor, NodeContext
from ..parser_util import update_defaults

LOGGER = logging.getLogger(__name__)


class JavaASTVisitor(GenericMetadataVisitor):
    """Specialized visitor for Java language AST analysis."""

    def __init__(self) -> None:
        super().__init__("java")
        self.functions: List[str] = []
        self.classes: List[str] = []
        self.interfaces: List[str] = []
        self.enums: List[str] = []
        self.packages: List[str] = []
        self.annotations: List[str] = []

    def visit_node(self, context: NodeContext) -> Optional[Dict[str, Any]]:
        """Visit a node and extract Java-specific metadata."""
        node = context.node
        node_type = node.type if hasattr(node, "type") else str(type(node))

        # Track node statistics
        self.node_stats[node_type] = self.node_stats.get(node_type, 0) + 1

        # Update complexity score based on node type (inherited from GenericMetadataVisitor)
        self._update_complexity(node_type)

        # Extract Java-specific constructs
        if node_type == "method_declaration":
            self._extract_method(node)
        elif node_type == "constructor_declaration":
            self._extract_constructor(node)
        elif node_type == "class_declaration":
            self._extract_class(node)
        elif node_type == "interface_declaration":
            self._extract_interface(node)
        elif node_type == "enum_declaration":
            self._extract_enum(node)
        elif node_type == "package_declaration":
            self._extract_package(node)
        elif node_type == "annotation_type_declaration":
            self._extract_annotation(node)

        return None

    def _extract_method(self, node: Node) -> None:
        """Extract method name from method_declaration node."""
        try:
            # Java method structure: method_declaration -> identifier
            for child in node.children:
                if child.type == "identifier":
                    text = child.text
                    if text is not None:
                        method_name = text.decode("utf-8")
                        self.functions.append(method_name)
                    LOGGER.debug("Found Java method: %s", method_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Java method: %s", e)

    def _extract_constructor(self, node: Node) -> None:
        """Extract constructor name from constructor_declaration node."""
        try:
            # Java constructor structure: constructor_declaration -> identifier
            for child in node.children:
                if child.type == "identifier":
                    text = child.text
                    if text is not None:
                        constructor_name = text.decode("utf-8")
                        self.functions.append(constructor_name)  # Treat constructors as functions
                    LOGGER.debug("Found Java constructor: %s", constructor_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Java constructor: %s", e)

    def _extract_class(self, node: Node) -> None:
        """Extract class name from class_declaration node."""
        try:
            # Look for class name (identifier after 'class' keyword)
            for child in node.children:
                if child.type == "identifier":
                    text = child.text
                    if text is not None:
                        class_name = text.decode("utf-8")
                        self.classes.append(class_name)
                    LOGGER.debug("Found Java class: %s", class_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Java class: %s", e)

    def _extract_interface(self, node: Node) -> None:
        """Extract interface name from interface_declaration node."""
        try:
            # Look for interface name
            for child in node.children:
                if child.type == "identifier":
                    text = child.text
                    if text is not None:
                        interface_name = text.decode("utf-8")
                        self.interfaces.append(interface_name)
                    LOGGER.debug("Found Java interface: %s", interface_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Java interface: %s", e)

    def _extract_enum(self, node):
        """Extract enum name from enum_declaration node."""
        try:
            # Look for enum name
            for child in node.children:
                if child.type == "identifier":
                    enum_name = child.text.decode("utf-8")
                    self.enums.append(enum_name)
                    LOGGER.debug("Found Java enum: %s", enum_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Java enum: %s", e)

    def _extract_package(self, node):
        """Extract package name from package_declaration node."""
        try:
            # Look for package name (scoped_identifier or identifier)
            for child in node.children:
                if child.type in ["scoped_identifier", "identifier"]:
                    package_name = child.text.decode("utf-8")
                    self.packages.append(package_name)
                    LOGGER.debug("Found Java package: %s", package_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Java package: %s", e)

    def _extract_annotation(self, node):
        """Extract annotation name from annotation_type_declaration node."""
        try:
            # Look for annotation name
            for child in node.children:
                if child.type == "identifier":
                    annotation_name = child.text.decode("utf-8")
                    self.annotations.append(annotation_name)
                    LOGGER.debug("Found Java annotation: %s", annotation_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Java annotation: %s", e)

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary in the expected format."""
        return {
            "functions": self.functions,
            "classes": self.classes,
            "interfaces": self.interfaces,
            "enums": self.enums,
            "packages": self.packages,
            "annotations": self.annotations,
            "node_stats": dict(self.node_stats),
            "complexity_score": self.complexity_score,
            "analysis_method": "java_ast_visitor",
        }


def analyze_java_code(code: str, filename: str = "") -> Dict[str, Any]:
    """
    Analyze Java code using the specialized Java AST visitor.
    This function mirrors other language analyzers.
    """
    try:
        from ..ast_visitor import ASTParserFactory, TreeWalker

        # Create parser and parse code
        factory = ASTParserFactory()
        parser = factory.create_parser("java")
        if not parser:
            LOGGER.warning("Java parser not available")
            return {"success": False, "error": "Java parser not available"}

        tree = factory.parse_code(code, "java")
        if not tree:
            LOGGER.warning("Failed to parse Java code")
            return {"success": False, "error": "Failed to parse Java code"}

        # Use specialized Java visitor
        visitor = JavaASTVisitor()
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
                "language": get_display_language_name("java"),
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
                "decorators_used": result.get("annotations", []),  # Java uses annotations
                "has_type_hints": True,  # Java has strong typing
                "has_async": False,  # Java doesn't have async/await syntax like JS/Python
                "has_classes": len(result.get("classes", [])) > 0 or len(result.get("interfaces", [])) > 0,
            },
        )

        LOGGER.debug(
            "Java analysis completed: %s functions, %s classes found",
            len(result.get("functions", [])),
            len(result.get("classes", [])),
        )
        return result

    except Exception as e:
        LOGGER.error("Java code analysis failed: %s", e)
        return {"success": False, "error": str(e)}
