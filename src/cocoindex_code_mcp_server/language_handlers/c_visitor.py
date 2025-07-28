#!/usr/bin/env python3

"""
C-specific AST visitor for metadata extraction.
Follows the same pattern as haskell_visitor.py by subclassing GenericMetadataVisitor.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from ..ast_visitor import GenericMetadataVisitor, NodeContext
from cocoindex_code_mcp_server.ast_visitor import NodeContext
from tree_sitter import Node

LOGGER = logging.getLogger(__name__)


class CASTVisitor(GenericMetadataVisitor):
    """Specialized visitor for C language AST analysis."""

    def __init__(self) -> None:
        super().__init__("c")
        self.functions: List[str] = []
        self.structs: List[str] = []
        self.enums: List[str] = []
        self.typedefs: List[str] = []

    def visit_node(self, context: NodeContext) -> Optional[Dict[str, Any]]:
        """Visit a node and extract C-specific metadata."""
        node = context.node
        node_type = node.type if hasattr(node, 'type') else str(type(node))

        # Track node statistics
        self.node_stats[node_type] = self.node_stats.get(node_type, 0) + 1

        # Extract C-specific constructs
        if node_type == 'function_definition':
            self._extract_function(node)
        elif node_type == 'struct_specifier':
            self._extract_struct(node)
        elif node_type == 'enum_specifier':
            self._extract_enum(node)
        elif node_type == 'type_definition':
            self._extract_typedef(node)

        return None

    def _extract_function(self, node: Node):
        """Extract function name from function_definition node."""
        try:
            # C function structure: function_definition -> function_declarator -> identifier
            declarator = self._find_child_by_type(node, 'function_declarator')
            if declarator:
                identifier = self._find_child_by_type(declarator, 'identifier')
                if identifier:
                    text = identifier.text
                    if text is not None:
                        func_name = text.decode('utf-8')
                    self.functions.append(func_name)
                    LOGGER.debug(f"Found C function: {func_name}")
        except Exception as e:
            LOGGER.warning(f"Error extracting C function: {e}")

    def _extract_struct(self, node: Node):
        """Extract struct name from struct_specifier node."""
        try:
            # Look for struct name (identifier after 'struct' keyword)
            for child in node.children:
                if child.type == 'type_identifier':
                    text = child.text
                    if text is not None:
                        struct_name = text.decode('utf-8')
                    self.structs.append(struct_name)
                    LOGGER.debug(f"Found C struct: {struct_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting C struct: {e}")

    def _extract_enum(self, node):
        """Extract enum name from enum_specifier node."""
        try:
            # Look for enum name
            for child in node.children:
                if child.type == 'type_identifier':
                    enum_name = child.text.decode('utf-8')
                    self.enums.append(enum_name)
                    LOGGER.debug(f"Found C enum: {enum_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting C enum: {e}")

    def _extract_typedef(self, node):
        """Extract typedef name from type_definition node."""
        try:
            # Look for the new type name in typedef
            for child in reversed(node.children):  # Name is usually at the end
                if child.type == 'type_identifier':
                    typedef_name = child.text.decode('utf-8')
                    self.typedefs.append(typedef_name)
                    LOGGER.debug(f"Found C typedef: {typedef_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting C typedef: {e}")

    def _find_child_by_type(self, node: Node, target_type: str) -> Union[Node,None]:
        """Find first child node of specified type."""
        for child in node.children:
            if child.type == target_type:
                return child
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary in the expected format."""
        return {
            'functions': self.functions,
            'classes': [],  # C doesn't have classes
            'structs': self.structs,
            'enums': self.enums,
            'typedefs': self.typedefs,
            'node_stats': dict(self.node_stats),
            'complexity_score': self.complexity_score,
            'analysis_method': 'c_ast_visitor'
        }


def analyze_c_code(code: str, filename: str = "") -> Dict[str, Any]:
    """
    Analyze C code using the specialized C AST visitor.
    This function mirrors analyze_haskell_code from haskell_visitor.py
    """
    try:
        from ..ast_visitor import ASTParserFactory, TreeWalker

        # Create parser and parse code
        factory = ASTParserFactory()
        parser = factory.create_parser('c')
        if not parser:
            LOGGER.warning("C parser not available")
            return {'success': False, 'error': 'C parser not available'}

        tree = factory.parse_code(code, 'c')
        if not tree:
            LOGGER.warning("Failed to parse C code")
            return {'success': False, 'error': 'Failed to parse C code'}

        # Use specialized C visitor
        visitor = CASTVisitor()
        walker = TreeWalker(code, tree)
        walker.walk(visitor)

        # Get results from visitor
        result = visitor.get_summary()
        result.update({
            'success': True,
            'language': 'c',
            'filename': filename,
            'line_count': code.count('\n') + 1,
            'char_count': len(code),
            'parse_errors': 0,
            'tree_language': str(parser.language) if parser else None
        })

        LOGGER.debug(f"C analysis completed: {len(result.get('functions', []))} functions found")
        return result

    except Exception as e:
        LOGGER.error(f"C code analysis failed: {e}")
        return {'success': False, 'error': str(e)}
