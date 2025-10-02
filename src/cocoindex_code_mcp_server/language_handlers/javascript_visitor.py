#!/usr/bin/env python3

"""
JavaScript-specific AST visitor for metadata extraction.
Follows the same pattern as other language visitors by subclassing GenericMetadataVisitor.
"""

import logging
from typing import Any, Dict, List, Optional

from tree_sitter import Node

from ..ast_visitor import GenericMetadataVisitor, NodeContext
from ..parser_util import update_defaults

LOGGER = logging.getLogger(__name__)


class JavaScriptASTVisitor(GenericMetadataVisitor):
    """Specialized visitor for JavaScript language AST analysis."""

    def __init__(self) -> None:
        super().__init__("javascript")
        self.functions: List[str] = []
        self.classes: List[str] = []
        self.variables: List[str] = []
        self.imports: List[str] = []
        self.exports: List[str] = []

    def visit_node(self, context: NodeContext) -> Optional[Dict[str, Any]]:
        """Visit a node and extract JavaScript-specific metadata."""
        node = context.node
        node_type = node.type if hasattr(node, 'type') else str(type(node))

        # Track node statistics
        self.node_stats[node_type] = self.node_stats.get(node_type, 0) + 1

        
        # Update complexity score based on node type (inherited from GenericMetadataVisitor)
        self._update_complexity(node_type)
# Extract JavaScript-specific constructs
        if node_type == 'function_declaration':
            self._extract_function(node)
        elif node_type == 'function_expression':
            self._extract_function_expression(node)
        elif node_type == 'arrow_function':
            self._extract_arrow_function(node)
        elif node_type == 'method_definition':
            self._extract_method(node)
        elif node_type == 'class_declaration':
            self._extract_class(node)
        elif node_type == 'variable_declarator':
            self._extract_variable(node)
        elif node_type == 'import_statement':
            self._extract_import(node)
        elif node_type == 'export_statement':
            self._extract_export(node)

        return None

    def _extract_function(self, node) -> None:
        """Extract function name from function_declaration node."""
        try:
            # JavaScript function structure: function_declaration -> identifier
            for child in node.children:
                if child.type == 'identifier':
                    func_name = child.text.decode('utf-8')
                    self.functions.append(func_name)
                    LOGGER.debug(f"Found JavaScript function: {func_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting JavaScript function: {e}")

    def _extract_function_expression(self, node):
        """Extract function name from function_expression node."""
        try:
            # Function expressions may or may not have names
            for child in node.children:
                if child.type == 'identifier':
                    func_name = child.text.decode('utf-8')
                    self.functions.append(func_name)
                    LOGGER.debug(f"Found JavaScript function expression: {func_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting JavaScript function expression: {e}")

    def _extract_arrow_function(self, node: Node) -> None:
        """Extract context for arrow functions (often anonymous)."""
        try:
            # Arrow functions are usually anonymous, but we track them
            # Could look at parent context to see if assigned to variable
            pass  # Most arrow functions are anonymous
        except Exception as e:
            LOGGER.warning(f"Error extracting JavaScript arrow function: {e}")

    def _extract_method(self, node):
        """Extract method name from method_definition node."""
        try:
            # Method structure: method_definition -> property_identifier or identifier
            for child in node.children:
                if child.type in ['property_identifier', 'identifier']:
                    method_name = child.text.decode('utf-8')
                    self.functions.append(method_name)
                    LOGGER.debug(f"Found JavaScript method: {method_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting JavaScript method: {e}")

    def _extract_class(self, node):
        """Extract class name from class_declaration node."""
        try:
            # Look for class name (identifier after 'class' keyword)
            for child in node.children:
                if child.type == 'identifier':
                    class_name = child.text.decode('utf-8')
                    self.classes.append(class_name)
                    LOGGER.debug(f"Found JavaScript class: {class_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting JavaScript class: {e}")

    def _extract_variable(self, node):
        """Extract variable name from variable_declarator node."""
        try:
            # Look for variable name
            for child in node.children:
                if child.type == 'identifier':
                    var_name = child.text.decode('utf-8')
                    # Only track function assignments and significant variables
                    # Look at the value to see if it's a function
                    for sibling in node.children:
                        if sibling.type in ['function_expression', 'arrow_function']:
                            self.functions.append(var_name)
                            LOGGER.debug(f"Found JavaScript function variable: {var_name}")
                            break
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting JavaScript variable: {e}")

    def _extract_import(self, node):
        """Extract import information from import_statement node."""
        try:
            # Track import modules (simplified)
            for child in node.children:
                if child.type == 'string':
                    import_module = child.text.decode('utf-8').strip('"\'')
                    self.imports.append(import_module)
                    LOGGER.debug(f"Found JavaScript import: {import_module}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting JavaScript import: {e}")

    def _extract_export(self, node):
        """Extract export information from export_statement node."""
        try:
            # Track what's being exported (simplified)
            for child in node.children:
                if child.type == 'identifier':
                    export_name = child.text.decode('utf-8')
                    self.exports.append(export_name)
                    LOGGER.debug(f"Found JavaScript export: {export_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Error extracting JavaScript export: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary in the expected format."""
        return {
            'functions': self.functions,
            'classes': self.classes,
            'variables': self.variables,
            'imports': self.imports,
            'exports': self.exports,
            'node_stats': dict(self.node_stats),
            'complexity_score': self.complexity_score,
            'analysis_method': 'javascript_ast_visitor'
        }


def analyze_javascript_code(code: str, language: str = "javascript", filename: str = "") -> Dict[str, Any]:
    """
    Analyze JavaScript code using the specialized JavaScript AST visitor.

    Args:
        code: JavaScript source code to analyze
        language: Language identifier ("javascript", "js")
        filename: Optional filename for context
    """
    try:
        from ..ast_visitor import ASTParserFactory, TreeWalker

        # Create parser and parse code
        factory = ASTParserFactory()
        parser = factory.create_parser('javascript')
        if not parser:
            LOGGER.warning(f"JavaScript parser not available for {language}")
            return {'success': False, 'error': f'JavaScript parser not available for {language}'}

        tree = factory.parse_code(code, 'javascript')
        if not tree:
            LOGGER.warning("Failed to parse JavaScript code")
            return {'success': False, 'error': 'Failed to parse JavaScript code'}

        # Use specialized JavaScript visitor
        visitor = JavaScriptASTVisitor()
        walker = TreeWalker(code, tree)
        walker.walk(visitor)

        # Normalize language to Title Case for consistency with database schema
        normalized_language = "JavaScript" if language.lower() in ["javascript", "js"] else language

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
            'tree_sitter_analyze_error': False,
            'decorators_used': [],  # JavaScript doesn't commonly use decorators
            'has_type_hints': language.lower() in ['typescript', 'ts'],  # TypeScript has type hints
            'has_async': any('async' in func.lower() for func in result.get('functions', [])),
            'has_classes': len(result.get('classes', [])) > 0
        })

        LOGGER.debug(
            f"JavaScript analysis completed: {len(result.get('functions', []))} functions, {len(result.get('classes', []))} classes found")
        return result

    except Exception as e:
        LOGGER.error(f"JavaScript code analysis failed: {e}")
        return {'success': False, 'error': str(e)}
