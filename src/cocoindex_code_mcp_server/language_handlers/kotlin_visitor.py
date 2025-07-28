#!/usr/bin/env python3

"""
Kotlin-specific AST visitor for metadata extraction.
Follows the same pattern as haskell_visitor.py by subclassing GenericMetadataVisitor.
"""

import logging
from typing import Any, Dict, List, Optional

from ..ast_visitor import GenericMetadataVisitor, NodeContext
from cocoindex_code_mcp_server.ast_visitor import NodeContext
from tree_sitter import Node

LOGGER = logging.getLogger(__name__)


class KotlinASTVisitor(GenericMetadataVisitor):
    """Specialized visitor for Kotlin language AST analysis."""

    def __init__(self):
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
        node_type = node.type if hasattr(node, 'type') else str(type(node))

        # Track node statistics
        self.node_stats[node_type] = self.node_stats.get(node_type, 0) + 1

        # Extract Kotlin-specific constructs
        if node_type == 'function_declaration':
            self._extract_function(node)
        elif node_type == 'class_declaration':
            self._extract_class(node)
        elif node_type == 'interface_declaration':
            self._extract_interface(node)
        elif node_type == 'object_declaration':
            self._extract_object(node)
        elif node_type == 'enum_class_declaration':
            self._extract_enum(node)

        return None

    def _extract_function(self, node: Node):
        """Extract function name from function_declaration node."""
        try:
            # Kotlin function structure: function_declaration -> identifier (after 'fun' keyword)
            for child in node.children:
                if child.type == 'identifier':
                    text = child.text
                    if text is not None:
                        func_name = child.text.decode('utf-8')
                    self.functions.append(func_name)
                    LOGGER.debug(f"Found Kotlin function: {func_name}")
                    break  # Take the first identifier (function name)
        except Exception as e:
            LOGGER.warning(f"Error extracting Kotlin function: {e}")

    def _extract_class(self, node):
        """Extract class name from class_declaration node."""
        try:
            # Look for class name (identifier after 'class' keyword)
            is_data_class = False
            class_name = None

            # Check if it's a data class by looking at modifiers
            for child in node.children:
                if child.type == 'modifiers':
                    # Check for data modifier
                    for modifier_child in child.children:
                        if modifier_child.type == 'class_modifier':
                            for modifier_grandchild in modifier_child.children:
                                if modifier_grandchild.type == 'data':
                                    is_data_class = True
                elif child.type == 'identifier':
                    class_name = child.text.decode('utf-8')
                    break

            if class_name:
                if is_data_class:
                    self.data_classes.append(class_name)
                    LOGGER.debug(f"Found Kotlin data class: {class_name}")
                else:
                    self.classes.append(class_name)
                    LOGGER.debug(f"Found Kotlin class: {class_name}")

        except Exception as e:
            LOGGER.warning(f"Error extracting Kotlin class: {e}")

    def _extract_interface(self, node):
        """Extract interface name from interface_declaration node."""
        try:
            # Look for interface name
            for child in node.children:
                if child.type == 'identifier':
                    interface_name = child.text.decode('utf-8')
                    self.interfaces.append(interface_name)
                    LOGGER.debug(f"Found Kotlin interface: {interface_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting Kotlin interface: {e}")

    def _extract_object(self, node):
        """Extract object name from object_declaration node."""
        try:
            # Look for object name
            for child in node.children:
                if child.type == 'identifier':
                    object_name = child.text.decode('utf-8')
                    self.objects.append(object_name)
                    LOGGER.debug(f"Found Kotlin object: {object_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting Kotlin object: {e}")

    def _extract_enum(self, node):
        """Extract enum name from enum_class_declaration node."""
        try:
            # Look for enum name
            for child in node.children:
                if child.type == 'identifier':
                    enum_name = child.text.decode('utf-8')
                    self.enums.append(enum_name)
                    LOGGER.debug(f"Found Kotlin enum: {enum_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting Kotlin enum: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary in the expected format."""
        return {
            'functions': self.functions,
            'classes': self.classes,
            'interfaces': self.interfaces,
            'objects': self.objects,
            'data_classes': self.data_classes,
            'enums': self.enums,
            'node_stats': dict(self.node_stats),
            'complexity_score': self.complexity_score,
            'analysis_method': 'kotlin_ast_visitor'
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
        parser = factory.create_parser('kotlin')
        if not parser:
            LOGGER.warning("Kotlin parser not available")
            return {'success': False, 'error': 'Kotlin parser not available'}

        tree = factory.parse_code(code, 'kotlin')
        if not tree:
            LOGGER.warning("Failed to parse Kotlin code")
            return {'success': False, 'error': 'Failed to parse Kotlin code'}

        # Use specialized Kotlin visitor
        visitor = KotlinASTVisitor()
        walker = TreeWalker(code, tree)
        walker.walk(visitor)

        # Get results from visitor
        result = visitor.get_summary()
        result.update({
            'success': True,
            'language': 'kotlin',
            'filename': filename,
            'line_count': code.count('\n') + 1,
            'char_count': len(code),
            'parse_errors': 0,
            'tree_language': str(parser.language) if parser else None
        })

        LOGGER.debug(
            f"Kotlin analysis completed: {len(result.get('functions', []))} functions, {len(result.get('classes', []))} classes found")
        return result

    except Exception as e:
        LOGGER.error(f"Kotlin code analysis failed: {e}")
        return {'success': False, 'error': str(e)}
