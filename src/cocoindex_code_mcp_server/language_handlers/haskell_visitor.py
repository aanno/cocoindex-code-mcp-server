#!/usr/bin/env python3

"""
Specialized Haskell AST visitor that works directly with haskell_tree_sitter chunks.
This avoids the complexity of trying to wrap chunks in a generic tree-sitter interface.
"""

import os
import sys
from typing import Any, Dict

from tree_sitter import Node

from cocoindex_code_mcp_server.ast_visitor import Position

from ..ast_visitor import GenericMetadataVisitor, NodeContext
from ..language_handlers.haskell_handler import HaskellNodeHandler
from . import LOGGER

# Import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


try:
    import haskell_tree_sitter
    HASKELL_TREE_SITTER_AVAILABLE = True
except ImportError:
    HASKELL_TREE_SITTER_AVAILABLE = False


class HaskellASTVisitor(GenericMetadataVisitor):
    """Specialized visitor for Haskell code using haskell_tree_sitter chunks."""

    def __init__(self) -> None:
        super().__init__(language='haskell')
        self.haskell_handler = HaskellNodeHandler()
        # Add the handler to the parent's handler list
        self.add_handler(self.haskell_handler)

    def analyze_haskell_code(self, code: str, filename: str = "") -> Dict[str, Any]:
        """Analyze Haskell code directly using haskell_tree_sitter chunks."""
        if not HASKELL_TREE_SITTER_AVAILABLE:
            LOGGER.warning("haskell_tree_sitter not available, falling back to generic analysis")
            return self._fallback_analysis(code, filename)

        try:
            # Use enhanced error-aware chunking
            chunking_result = haskell_tree_sitter.get_haskell_ast_chunks_enhanced(code)

            # Process each chunk using our handler
            for chunk in chunking_result.chunks():
                self._process_chunk(chunk, code)

            # Build metadata result with error information
            error_stats = chunking_result.error_stats()
            # Use display language name for database storage
            from ..mappers import get_display_language_name
            metadata = {
                'language': get_display_language_name('haskell'),
                'filename': filename,
                'line_count': len(code.split('\n')),
                'char_count': len(code),
                'analysis_method': 'haskell_chunk_visitor',
                'chunking_method': chunking_result.chunking_method(),
                'errors': self.errors,
                'node_stats': self.node_stats,
                'complexity_score': self.complexity_score,
                'parse_errors': error_stats.error_count(),
                'error_count': error_stats.error_count(),
                'nodes_with_errors': error_stats.nodes_with_errors(),
                'should_fallback': error_stats.should_fallback(),
                'coverage_complete': chunking_result.coverage_complete(),
                'tree_language': 'haskell_tree_sitter',
                'tree_sitter_analyze_error': 'true' if error_stats.error_count() > 0 else 'false',
                'success': True
            }

            # Add Haskell-specific metadata from handler
            handler_summary = self.haskell_handler.get_summary()
            metadata.update(handler_summary)

            return metadata

        except Exception as e:
            LOGGER.error(f"Haskell chunk analysis failed: {e}")
            return self._fallback_analysis(code, filename)

    def _process_chunk(self, chunk, source_code: str) -> None:
        """Process a single haskell_tree_sitter chunk."""
        # Create a chunk context that works with our handler
        chunk_context = HaskellChunkContext(chunk, source_code)

        # Get the node type
        node_type = chunk.node_type()

        # Track statistics
        self.node_stats[node_type] = self.node_stats.get(node_type, 0) + 1

        # Let the handler process the chunk
        if self.haskell_handler.can_handle(node_type):
            try:
                metadata = self.haskell_handler.extract_metadata(chunk_context)
                if metadata:
                    LOGGER.debug(f"Processed {node_type} chunk: {metadata}")
            except Exception as e:
                error_msg = f"Handler error for {node_type}: {e}"
                self.errors.append(error_msg)
                LOGGER.warning(error_msg)

        # Update complexity
        self._update_complexity(node_type)

    def _fallback_analysis(self, code: str, filename: str) -> Dict[str, Any]:
        """Fallback to basic text analysis when haskell_tree_sitter isn't available."""
        lines = code.split('\n')

        # Use display language name for database storage
        from ..mappers import get_display_language_name
        return {
            'language': get_display_language_name('haskell'),
            'filename': filename,
            'line_count': len(lines),
            'char_count': len(code),
            'analysis_method': 'haskell_fallback',
            'errors': ['haskell_tree_sitter not available'],
            'node_stats': {},
            'complexity_score': 0,
            'parse_errors': 0,
            'tree_language': 'fallback',
            'tree_sitter_analyze_error': 'false',  # No tree-sitter used in fallback

            # Basic Haskell-specific analysis
            'module': None,
            'functions': [],
            'data_types': [],
            'type_classes': [],
            'instances': [],
            'imports': [],
            'has_module_declaration': 'module ' in code,
            'has_exports': False,
            'has_type_signatures': '::' in code,
            'has_data_types': any(keyword in code for keyword in ['data ', 'newtype ', 'type ']),
            'has_type_classes': 'class ' in code,
            'has_instances': 'instance ' in code,
            'qualified_imports': 0,
            'function_details': [],
            'data_type_details': [],
            'import_details': []
        }


class HaskellChunkContext(NodeContext):
    """Specialized NodeContext for haskell_tree_sitter chunks."""

    def __init__(self, chunk: Node, source_code: str) -> None:
        # Create a minimal node context that works with chunks
        super().__init__(
            node=chunk,
            parent=None,
            depth=0,
            scope_stack=[],
            source_text=source_code
        )

    def get_node_text(self) -> str:
        """Get text from the chunk."""
        node = self.node
        if node is None:
            return ""
        # Handle node.text() returning bytes or None
        raw_text = node.text() if hasattr(node, 'text') and callable(node.text) else None
        if raw_text is None:
            return ""
        elif isinstance(raw_text, bytes):
            return raw_text.decode('utf-8', errors='ignore')
        else:
            return str(raw_text)

    def get_position(self) -> Position:
        """Get position from the chunk."""
        from ..ast_visitor import Position
        line = 1
        byte_offset = 0

        # Handle both tree-sitter node and chunk interfaces
        if hasattr(self.node, 'start_point'):
            point = self.node.start_point
            if callable(point):
                point_result = point()
                line = point_result[0] + 1 if hasattr(point_result, '__getitem__') else 1
            elif hasattr(point, '__getitem__'):
                line = point[0] + 1
        elif hasattr(self.node, 'start_line'):
            start_line_attr = getattr(self.node, 'start_line')
            if callable(start_line_attr):
                try:
                    result = start_line_attr()
                    line = int(result) if isinstance(result, (int, float, str)) else 1
                except (ValueError, TypeError):
                    line = 1
            else:
                try:
                    line = int(start_line_attr) if isinstance(start_line_attr, (int, float, str)) else 1
                except (ValueError, TypeError):
                    line = 1

        if hasattr(self.node, 'start_byte'):
            start_byte_attr = getattr(self.node, 'start_byte')
            if callable(start_byte_attr):
                try:
                    result = start_byte_attr()
                    byte_offset = int(result) if isinstance(result, (int, float, str)) else 0
                except (ValueError, TypeError, AttributeError):
                    byte_offset = 0
            else:
                try:
                    byte_offset = int(start_byte_attr) if isinstance(start_byte_attr, (int, float, str)) else 0
                except (ValueError, TypeError):
                    byte_offset = 0

        return Position(
            line=line,
            column=0,  # haskell_tree_sitter doesn't provide column info
            byte_offset=byte_offset
        )


# Factory function for easy use
def analyze_haskell_code(code: str, filename: str = "") -> Dict[str, Any]:
    """Convenience function to analyze Haskell code."""
    visitor = HaskellASTVisitor()
    return visitor.analyze_haskell_code(code, filename)
