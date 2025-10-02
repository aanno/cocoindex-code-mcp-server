#!/usr/bin/env python3

"""
C++-specific AST visitor for metadata extraction.
Inherits from C visitor since C is largely a subset of C++.
Updated to trigger CocoIndex reprocessing.
"""

import logging
from typing import Any, Dict, List, Optional

from tree_sitter import Node

from cocoindex_code_mcp_server.ast_visitor import NodeContext

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
        node_type = node.type if hasattr(node, 'type') else str(type(node))

        # Track node statistics
        self.node_stats[node_type] = self.node_stats.get(node_type, 0) + 1

        
        # Update complexity score based on node type (inherited from GenericMetadataVisitor)
        self._update_complexity(node_type)
# Handle C++-specific constructs first
        if node_type == 'class_specifier':
            self._extract_class(node)
        elif node_type == 'namespace_definition':
            self._extract_namespace(node)
        elif node_type == 'template_declaration':
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
                if child.type == 'type_identifier':
                    text = child.text
                    if text is not None:
                        class_name = text.decode('utf-8')
                        self.classes.append(class_name)
                        LOGGER.debug(f"Found C++ class: {class_name}")
                        break
        except Exception as e:
            LOGGER.warning(f"Error extracting C++ class: {e}")

    def _extract_namespace(self, node: Node) -> None:
        """Extract namespace name from namespace_definition node."""
        try:
            # Look for namespace name
            for child in node.children:
                if child.type == 'identifier':
                    text = child.text
                    if text is not None:
                        namespace_name = text.decode('utf-8')
                        self.namespaces.append(namespace_name)
                        LOGGER.debug(f"Found C++ namespace: {namespace_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting C++ namespace: {e}")

    def _extract_template(self, node):
        """Extract template information from template_declaration node."""
        try:
            # Look for the template name (usually in the following declaration)
            # This is a simplified extraction - templates are complex
            for child in node.children:
                if child.type in ['class_specifier', 'function_definition']:
                    # Extract the name from the templated construct
                    if child.type == 'class_specifier':
                        for grandchild in child.children:
                            if grandchild.type == 'type_identifier':
                                template_name = f"template<{grandchild.text.decode('utf-8')}>"
                                self.templates.append(template_name)
                                LOGGER.debug(f"Found C++ template: {template_name}")
                                break
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting C++ template: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary in the expected format."""
        # Get base C summary and extend with C++-specific fields
        summary = super().get_summary()
        summary.update({
            'classes': self.classes,
            'namespaces': self.namespaces,
            'templates': self.templates,
            'analysis_method': 'cpp_ast_visitor'
        })
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
            LOGGER.warning(f"C++ parser not available for {language}")
            return {'success': False, 'error': f'C++ parser not available for {language}'}

        tree = factory.parse_code(code, language)
        if not tree:
            LOGGER.warning("Failed to parse C++ code")
            return {'success': False, 'error': 'Failed to parse C++ code'}

        # Use specialized C++ visitor
        visitor = CppASTVisitor()
        walker = TreeWalker(code, tree)
        walker.walk(visitor)

        # Normalize language to Title Case for consistency with database schema
        normalized_language = "C++" if language.lower() in ["cpp", "cc", "cxx", "c++"] else language

        # Get results from visitor
        result = visitor.get_summary()
        result.update({
            'success': True,
            'language': normalized_language,
            'filename': filename,
            'line_count': code.count('\n') + 1,
            'char_count': len(code),
            'parse_errors': 0,
            'tree_language': str(parser.language) if parser else None
        })

        LOGGER.debug(
            f"C++ analysis completed: {len(result.get('functions', []))} functions, {len(result.get('classes', []))} classes found")
        return result

    except Exception as e:
        LOGGER.error(f"C++ code analysis failed: {e}")
        return {'success': False, 'error': str(e)}
