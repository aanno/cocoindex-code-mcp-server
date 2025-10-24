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
Rust-specific AST visitor for metadata extraction.

Follows the same pattern as haskell_visitor.py by subclassing GenericMetadataVisitor.
"""

import logging
from typing import Any, Dict, List, Optional

# from cocoindex_code_mcp_server.ast_visitor import NodeContext
from tree_sitter import Node

from ..ast_visitor import GenericMetadataVisitor, NodeContext
from ..parser_util import update_defaults

LOGGER = logging.getLogger(__name__)


class RustASTVisitor(GenericMetadataVisitor):
    """Specialized visitor for Rust language AST analysis."""

    def __init__(self) -> None:
        super().__init__("rust")
        self.functions: List[str] = []
        self.structs: List[str] = []
        self.enums: List[str] = []
        self.traits: List[str] = []
        self.impls: List[str] = []
        self.mods: List[str] = []

    def visit_node(self, context: NodeContext) -> Optional[Dict[str, Any]]:
        """Visit a node and extract Rust-specific metadata."""
        node = context.node
        node_type = node.type if hasattr(node, "type") else str(type(node))

        # Track node statistics
        self.node_stats[node_type] = self.node_stats.get(node_type, 0) + 1

        # Update complexity score based on node type (inherited from GenericMetadataVisitor)
        self._update_complexity(node_type)
        # Extract Rust-specific constructs
        if node_type == "function_item":
            self._extract_function(node)
        elif node_type == "struct_item":
            self._extract_struct(node)
        elif node_type == "enum_item":
            self._extract_enum(node)
        elif node_type == "trait_item":
            self._extract_trait(node)
        elif node_type == "impl_item":
            self._extract_impl(node)
        elif node_type == "mod_item":
            self._extract_mod(node)

        return None

    def _extract_function(self, node: Node) -> None:
        """Extract function name from function_item node."""
        try:
            # Rust function structure: function_item -> identifier (after 'fn' keyword)
            for child in node.children:
                if child.type == "identifier":
                    text = child.text
                    if text is not None:
                        func_name = text.decode("utf-8")
                        self.functions.append(func_name)
                        LOGGER.debug("Found Rust function: %s", func_name)
                        break  # Take the first identifier (function name)
        except Exception as e:
            LOGGER.warning("Error extracting Rust function: %s", e)

    def _extract_struct(self, node):
        """Extract struct name from struct_item node."""
        try:
            # Look for struct name (identifier after 'struct' keyword)
            for child in node.children:
                if child.type == "type_identifier":
                    struct_name = child.text.decode("utf-8")
                    self.structs.append(struct_name)
                    LOGGER.debug("Found Rust struct: %s", struct_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Rust struct: %s", e)

    def _extract_enum(self, node):
        """Extract enum name from enum_item node."""
        try:
            # Look for enum name
            for child in node.children:
                if child.type == "type_identifier":
                    enum_name = child.text.decode("utf-8")
                    self.enums.append(enum_name)
                    LOGGER.debug("Found Rust enum: %s", enum_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Rust enum: %s", e)

    def _extract_trait(self, node):
        """Extract trait name from trait_item node."""
        try:
            # Look for trait name
            for child in node.children:
                if child.type == "type_identifier":
                    trait_name = child.text.decode("utf-8")
                    self.traits.append(trait_name)
                    LOGGER.debug("Found Rust trait: %s", trait_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Rust trait: %s", e)

    def _extract_impl(self, node):
        """Extract implementation target from impl_item node."""
        try:
            # Look for the type being implemented
            for child in node.children:
                if child.type == "type_identifier":
                    impl_name = child.text.decode("utf-8")
                    self.impls.append(impl_name)
                    LOGGER.debug("Found Rust impl: %s", impl_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Rust impl: %s", e)

    def _extract_mod(self, node):
        """Extract module name from mod_item node."""
        try:
            # Look for module name
            for child in node.children:
                if child.type == "identifier":
                    mod_name = child.text.decode("utf-8")
                    self.mods.append(mod_name)
                    LOGGER.debug("Found Rust mod: %s", mod_name)
                    break
        except Exception as e:
            LOGGER.warning("Error extracting Rust mod: %s", e)

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary in the expected format."""
        return {
            "functions": self.functions,
            "classes": [],  # Rust doesn't have classes, but has structs/traits
            "structs": self.structs,
            "enums": self.enums,
            "traits": self.traits,
            "impls": self.impls,
            "modules": self.mods,
            "node_stats": dict(self.node_stats),
            "complexity_score": self.complexity_score,
            "analysis_method": "rust_ast_visitor",
        }


def analyze_rust_code(code: str, filename: str = "") -> Dict[str, Any]:
    """
    Analyze Rust code using the specialized Rust AST visitor.
    This function mirrors analyze_haskell_code from haskell_visitor.py
    """
    try:
        from ..ast_visitor import ASTParserFactory, TreeWalker

        # Create parser and parse code
        factory = ASTParserFactory()
        parser = factory.create_parser("rust")
        if not parser:
            LOGGER.warning("Rust parser not available")
            return {"success": False, "error": "Rust parser not available"}

        tree = factory.parse_code(code, "rust")
        if not tree:
            LOGGER.warning("Failed to parse Rust code")
            return {"success": False, "error": "Failed to parse Rust code"}

        # Use specialized Rust visitor
        visitor = RustASTVisitor()
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
                "language": get_display_language_name("rust"),
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
                "decorators_used": [],  # Rust doesn't have decorators
                "has_type_hints": True,  # Rust has strong typing
                "has_async": any("async" in func.lower() for func in result.get("functions", [])),
                "has_classes": len(result.get("structs", [])) > 0 or len(result.get("traits", [])) > 0,
            },
        )

        LOGGER.debug("Rust analysis completed: %s functions found", len(result.get("functions", [])))
        return result

    except Exception as e:
        LOGGER.error("Rust code analysis failed: %s", e)
        return {"success": False, "error": str(e)}
