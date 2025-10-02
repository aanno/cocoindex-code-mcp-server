#!/usr/bin/env python3

"""
TypeScript-specific AST visitor for metadata extraction.
Inherits from JavaScript visitor since TypeScript is JavaScript + types.
"""

import logging
from typing import Any, Dict, List, Optional

from ..ast_visitor import NodeContext
from ..parser_util import update_defaults
from .javascript_visitor import JavaScriptASTVisitor

LOGGER = logging.getLogger(__name__)


class TypeScriptASTVisitor(JavaScriptASTVisitor):
    """Specialized visitor for TypeScript language AST analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.language = "typescript"
        # Inherit JavaScript functionality: self.functions, self.classes, etc.
        # Add TypeScript-specific constructs
        self.interfaces: List[str] = []
        self.types: List[str] = []
        self.enums: List[str] = []
        self.namespaces: List[str] = []
        self.decorators: List[str] = []

    def visit_node(self, context: NodeContext) -> Optional[Dict[str, Any]]:
        """Visit a node and extract TypeScript-specific metadata."""
        node = context.node
        node_type = node.type if hasattr(node, 'type') else str(type(node))

        # Track node statistics
        self.node_stats[node_type] = self.node_stats.get(node_type, 0) + 1

        
        # Update complexity score based on node type (inherited from GenericMetadataVisitor)
        self._update_complexity(node_type)
# Handle TypeScript-specific constructs first
        if node_type == 'interface_declaration':
            self._extract_interface(node)
        elif node_type == 'type_alias_declaration':
            self._extract_type_alias(node)
        elif node_type == 'enum_declaration':
            self._extract_enum(node)
        elif node_type == 'namespace_declaration':
            self._extract_namespace(node)
        elif node_type == 'module_declaration':
            self._extract_module(node)
        elif node_type == 'decorator':
            self._extract_decorator(node)
        else:
            # Delegate to parent JavaScript visitor for common constructs
            super().visit_node(context)

        return None

    def _extract_interface(self, node) -> None:
        """Extract interface name from interface_declaration node."""
        try:
            # Look for interface name (identifier after 'interface' keyword)
            for child in node.children:
                if child.type == 'type_identifier':
                    interface_name = child.text.decode('utf-8')
                    self.interfaces.append(interface_name)
                    LOGGER.debug(f"Found TypeScript interface: {interface_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting TypeScript interface: {e}")

    def _extract_type_alias(self, node):
        """Extract type alias name from type_alias_declaration node."""
        try:
            # Look for type alias name
            for child in node.children:
                if child.type == 'type_identifier':
                    type_name = child.text.decode('utf-8')
                    self.types.append(type_name)
                    LOGGER.debug(f"Found TypeScript type alias: {type_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting TypeScript type alias: {e}")

    def _extract_enum(self, node):
        """Extract enum name from enum_declaration node."""
        try:
            # Look for enum name
            for child in node.children:
                if child.type == 'identifier':
                    enum_name = child.text.decode('utf-8')
                    self.enums.append(enum_name)
                    LOGGER.debug(f"Found TypeScript enum: {enum_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting TypeScript enum: {e}")

    def _extract_namespace(self, node):
        """Extract namespace name from namespace_declaration node."""
        try:
            # Look for namespace name
            for child in node.children:
                if child.type == 'identifier':
                    namespace_name = child.text.decode('utf-8')
                    self.namespaces.append(namespace_name)
                    LOGGER.debug(f"Found TypeScript namespace: {namespace_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting TypeScript namespace: {e}")

    def _extract_module(self, node):
        """Extract module name from module_declaration node."""
        try:
            # TypeScript modules can be similar to namespaces
            for child in node.children:
                if child.type == 'identifier':
                    module_name = child.text.decode('utf-8')
                    self.namespaces.append(module_name)  # Treat modules as namespaces
                    LOGGER.debug(f"Found TypeScript module: {module_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting TypeScript module: {e}")

    def _extract_decorator(self, node):
        """Extract decorator name from decorator node."""
        try:
            # Look for decorator name (typically starts with @)
            for child in node.children:
                if child.type == 'identifier':
                    decorator_name = child.text.decode('utf-8')
                    self.decorators.append(decorator_name)
                    LOGGER.debug(f"Found TypeScript decorator: {decorator_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting TypeScript decorator: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary in the expected format."""
        # Get base JavaScript summary and extend with TypeScript-specific fields
        summary = super().get_summary()
        summary.update({
            'interfaces': self.interfaces,
            'types': self.types,
            'enums': self.enums,
            'namespaces': self.namespaces,
            'decorators': self.decorators,
            'analysis_method': 'typescript_ast_visitor'
        })
        return summary


def analyze_typescript_code(code: str, language: str = "typescript", filename: str = "") -> Dict[str, Any]:
    """
    Analyze TypeScript code using the specialized TypeScript AST visitor.

    Args:
        code: TypeScript source code to analyze
        language: Language identifier ("typescript", "tsx")
        filename: Optional filename for context
    """
    try:
        from ..ast_visitor import ASTParserFactory, TreeWalker

        # Create parser and parse code
        factory = ASTParserFactory()
        parser = factory.create_parser(language)
        if not parser:
            LOGGER.warning(f"TypeScript parser not available for {language}")
            return {'success': False, 'error': f'TypeScript parser not available for {language}'}

        tree = factory.parse_code(code, language)
        if not tree:
            LOGGER.warning("Failed to parse TypeScript code")
            return {'success': False, 'error': 'Failed to parse TypeScript code'}

        # Use specialized TypeScript visitor
        visitor = TypeScriptASTVisitor()
        walker = TreeWalker(code, tree)
        walker.walk(visitor)

        # Normalize language to Title Case for consistency with database schema
        normalized_language = "TypeScript" if language.lower() in ["typescript", "ts", "tsx"] else language

        # Get results from visitor
        result = visitor.get_summary()
        # result.update({
        update_defaults(result, {
            'success': True,
            'language': normalized_language,
            'filename': filename,
            'line_count': code.count('\n') + 1,
            'char_count': len(code),
            'parse_errors': 0,
            'tree_language': str(parser.language) if parser else None,
            # Required metadata fields for promoted column implementation
            # don't set chunking method in analyzer
            # "chunking_method": "ast_tree_sitter",
            # "tree_sitter_chunking_error": False,
            'decorators_used': result.get('decorators', []),  # TypeScript supports decorators
            'has_type_hints': True,  # TypeScript has strong typing
            'has_async': any('async' in func.lower() for func in result.get('functions', [])),
            'has_classes': len(result.get('classes', [])) > 0
        })

        LOGGER.debug(
            f"TypeScript analysis completed: {len(result.get('functions', []))} functions, {len(result.get('classes', []))} classes found")
        return result

    except Exception as e:
        LOGGER.error(f"TypeScript code analysis failed: {e}")
        return {'success': False, 'error': str(e)}
