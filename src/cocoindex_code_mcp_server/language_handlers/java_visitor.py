#!/usr/bin/env python3

"""
Java-specific AST visitor for metadata extraction.
Follows the same pattern as other language visitors by subclassing GenericMetadataVisitor.
"""

import logging
from typing import Any, Dict, List, Optional

from ..ast_visitor import GenericMetadataVisitor, NodeContext
from cocoindex_code_mcp_server.ast_visitor import NodeContext
from tree_sitter import Node

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
        node_type = node.type if hasattr(node, 'type') else str(type(node))

        # Track node statistics
        self.node_stats[node_type] = self.node_stats.get(node_type, 0) + 1

        # Extract Java-specific constructs
        if node_type == 'method_declaration':
            self._extract_method(node)
        elif node_type == 'constructor_declaration':
            self._extract_constructor(node)
        elif node_type == 'class_declaration':
            self._extract_class(node)
        elif node_type == 'interface_declaration':
            self._extract_interface(node)
        elif node_type == 'enum_declaration':
            self._extract_enum(node)
        elif node_type == 'package_declaration':
            self._extract_package(node)
        elif node_type == 'annotation_type_declaration':
            self._extract_annotation(node)

        return None

    def _extract_method(self, node: Node):
        """Extract method name from method_declaration node."""
        try:
            # Java method structure: method_declaration -> identifier
            for child in node.children:
                if child.type == 'identifier':
                    text = child.text
                    if text is not None:
                        method_name = text.decode('utf-8')
                    self.functions.append(method_name)
                    LOGGER.debug(f"Found Java method: {method_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting Java method: {e}")

    def _extract_constructor(self, node: Node):
        """Extract constructor name from constructor_declaration node."""
        try:
            # Java constructor structure: constructor_declaration -> identifier
            for child in node.children:
                if child.type == 'identifier':
                    text = child.text
                    if text is not None:
                        constructor_name = text.decode('utf-8')
                    self.functions.append(constructor_name)  # Treat constructors as functions
                    LOGGER.debug(f"Found Java constructor: {constructor_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting Java constructor: {e}")

    def _extract_class(self, node: Node):
        """Extract class name from class_declaration node."""
        try:
            # Look for class name (identifier after 'class' keyword)
            for child in node.children:
                if child.type == 'identifier':
                    text = child.text
                    if text is not None:
                        class_name = text.decode('utf-8')
                    self.classes.append(class_name)
                    LOGGER.debug(f"Found Java class: {class_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting Java class: {e}")

    def _extract_interface(self, node: Node):
        """Extract interface name from interface_declaration node."""
        try:
            # Look for interface name
            for child in node.children:
                if child.type == 'identifier':
                    text = child.text
                    if text is not None:
                        interface_name = text.decode('utf-8')
                    self.interfaces.append(interface_name)
                    LOGGER.debug(f"Found Java interface: {interface_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting Java interface: {e}")

    def _extract_enum(self, node):
        """Extract enum name from enum_declaration node."""
        try:
            # Look for enum name
            for child in node.children:
                if child.type == 'identifier':
                    enum_name = child.text.decode('utf-8')
                    self.enums.append(enum_name)
                    LOGGER.debug(f"Found Java enum: {enum_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting Java enum: {e}")

    def _extract_package(self, node):
        """Extract package name from package_declaration node."""
        try:
            # Look for package name (scoped_identifier or identifier)
            for child in node.children:
                if child.type in ['scoped_identifier', 'identifier']:
                    package_name = child.text.decode('utf-8')
                    self.packages.append(package_name)
                    LOGGER.debug(f"Found Java package: {package_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting Java package: {e}")

    def _extract_annotation(self, node):
        """Extract annotation name from annotation_type_declaration node."""
        try:
            # Look for annotation name
            for child in node.children:
                if child.type == 'identifier':
                    annotation_name = child.text.decode('utf-8')
                    self.annotations.append(annotation_name)
                    LOGGER.debug(f"Found Java annotation: {annotation_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting Java annotation: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary in the expected format."""
        return {
            'functions': self.functions,
            'classes': self.classes,
            'interfaces': self.interfaces,
            'enums': self.enums,
            'packages': self.packages,
            'annotations': self.annotations,
            'node_stats': dict(self.node_stats),
            'complexity_score': self.complexity_score,
            'analysis_method': 'java_ast_visitor'
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
        parser = factory.create_parser('java')
        if not parser:
            LOGGER.warning("Java parser not available")
            return {'success': False, 'error': 'Java parser not available'}

        tree = factory.parse_code(code, 'java')
        if not tree:
            LOGGER.warning("Failed to parse Java code")
            return {'success': False, 'error': 'Failed to parse Java code'}

        # Use specialized Java visitor
        visitor = JavaASTVisitor()
        walker = TreeWalker(code, tree)
        walker.walk(visitor)

        # Get results from visitor
        result = visitor.get_summary()
        result.update({
            'success': True,
            'language': 'java',
            'filename': filename,
            'line_count': code.count('\n') + 1,
            'char_count': len(code),
            'parse_errors': 0,
            'tree_language': str(parser.language) if parser else None
        })

        LOGGER.debug(
            f"Java analysis completed: {len(result.get('functions', []))} functions, {len(result.get('classes', []))} classes found")
        return result

    except Exception as e:
        LOGGER.error(f"Java code analysis failed: {e}")
        return {'success': False, 'error': str(e)}
