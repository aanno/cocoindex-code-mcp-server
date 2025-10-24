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
C++-specific AST visitor for metadata extraction.

Inherits from C visitor since C is largely a subset of C++.
Updated to trigger CocoIndex reprocessing.
"""

import logging
from typing import Any, Dict, List, Optional

# from cocoindex_code_mcp_server.ast_visitor import NodeContext
from tree_sitter import Node

from ..ast_visitor import NodeContext
from .c_visitor import CASTVisitor

LOGGER = logging.getLogger(__name__)


class CppASTVisitor(CASTVisitor):
    """Specialized visitor for C++ language AST analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.language = "CPP"
        # Inherit C functionality: self.functions, self.structs, self.enums, self.typedefs
        # Add C++-specific constructs
        self.classes: List[str] = []
        self.namespaces: List[str] = []
        self.templates: List[str] = []

    def visit_node(self, context: NodeContext) -> Optional[Dict[str, Any]]:
        """Visit a node and extract C++-specific metadata."""
        node = context.node
        node_type = node.type if hasattr(node, "type") else str(type(node))

        # Track node statistics
        self.node_stats[node_type] = self.node_stats.get(node_type, 0) + 1

        # Update complexity score based on node type (inherited from GenericMetadataVisitor)
        self._update_complexity(node_type)
        # Handle C++-specific constructs first
        if node_type == "class_specifier":
            self._extract_class(node)
        elif node_type == "namespace_definition":
            self._extract_namespace(node)
        elif node_type == "template_declaration":
            self._extract_template(node)
        else:
            # Delegate to parent C visitor for common constructs (functions, structs, enums, etc.)
            super().visit_node(context)

        return None

    def _extract_class(self, node: Node) -> None:
        """Extract class name from class_specifier node."""
        try:
            # Look for class name (identifier after 'class' keyword)
            for child in node.children:
                if child.type == "type_identifier":
                    text = child.text
                    if text is not None:
                        class_name = text.decode("utf-8")
                        self.classes.append(class_name)
                        LOGGER.debug("Found C++ class: %s", class_name)
                        break
        except Exception as e:
            LOGGER.warning("Error extracting C++ class: %s", e)

    def _extract_namespace(self, node: Node) -> None:
        """Extract namespace name from namespace_definition node."""
        try:
            # Look for namespace name
            for child in node.children:
                if child.type == "identifier":
                    text = child.text
                    if text is not None:
                        namespace_name = text.decode("utf-8")
                        self.namespaces.append(namespace_name)
                        LOGGER.debug("Found C++ namespace: %s", namespace_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting C++ namespace: %s", e)

    def _extract_template(self, node):
        """Extract template information from template_declaration node."""
        try:
            # Look for the template name (usually in the following declaration)
            # This is a simplified extraction - templates are complex
            for child in node.children:
                if child.type in ["class_specifier", "function_definition"]:
                    # Extract the name from the templated construct
                    if child.type == "class_specifier":
                        for grandchild in child.children:
                            if grandchild.type == "type_identifier":
                                template_name = f"template<{grandchild.text.decode('utf-8')}>"
                                self.templates.append(template_name)
                                LOGGER.debug("Found C++ template: %s", template_name)
                                break
                    break
        except Exception as e:
            LOGGER.warning("Error extracting C++ template: %s", e)

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary in the expected format."""
        # Get base C summary and extend with C++-specific fields
        summary = super().get_summary()
        summary.update(
            {
                "classes": self.classes,
                "namespaces": self.namespaces,
                "templates": self.templates,
                "analysis_method": "cpp_ast_visitor",
            }
        )
        return summary


def analyze_cpp_code(code: str, language: str = "cpp", filename: str = "") -> Dict[str, Any]:
    """
    Analyze C++ code using the specialized C++ AST visitor.
    This function mirrors analyze_haskell_code from haskell_visitor.py

    Args:
        code: C++ source code to analyze
        language: Language identifier ("cpp", "cc", "cxx")
        filename: Optional filename for context
    """
    try:
        from ..ast_visitor import ASTParserFactory, TreeWalker

        # Create parser and parse code
        factory = ASTParserFactory()
        parser = factory.create_parser(language)
        if not parser:
            LOGGER.warning("C++ parser not available for %s", language)
            return {"success": False, "error": f"C++ parser not available for {language}"}

        tree = factory.parse_code(code, language)
        if not tree:
            LOGGER.warning("Failed to parse C++ code")
            return {"success": False, "error": "Failed to parse C++ code"}

        # Use specialized C++ visitor
        visitor = CppASTVisitor()
        walker = TreeWalker(code, tree)
        walker.walk(visitor)

        # Normalize language to Title Case for consistency with database schema
        normalized_language = "C++" if language.lower() in ["cpp", "cc", "cxx", "c++"] else language

        # Get results from visitor
        result = visitor.get_summary()
        result.update(
            {
                "success": True,
                "language": normalized_language,
                "filename": filename,
                "line_count": code.count("\n") + 1,
                "char_count": len(code),
                "parse_errors": 0,
                "tree_language": str(parser.language) if parser else None,
            }
        )

        LOGGER.debug(
            "C++ analysis completed: %s functions, %s classes found",
            len(result.get("functions", [])),
            len(result.get("classes", [])),
        )
        return result

    except Exception as e:
        LOGGER.error("C++ code analysis failed: %s", e)
        return {"success": False, "error": str(e)}
