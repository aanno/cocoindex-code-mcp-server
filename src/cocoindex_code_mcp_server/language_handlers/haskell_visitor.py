#!/usr/bin/env python3

"""
Specialized Haskell AST visitor that works directly with haskell_tree_sitter chunks.
This avoids the complexity of trying to wrap chunks in a generic tree-sitter interface.
"""

import sys
import os
from typing import Any, Dict

# Import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ..ast_visitor import GenericMetadataVisitor, NodeContext
from ..language_handlers.haskell_handler import HaskellNodeHandler
from . import LOGGER

try:
    import haskell_tree_sitter
    HASKELL_TREE_SITTER_AVAILABLE = True
except ImportError:
    HASKELL_TREE_SITTER_AVAILABLE = False


class HaskellASTVisitor(GenericMetadataVisitor):
    """Specialized visitor for Haskell code using haskell_tree_sitter chunks."""
    
    def __init__(self):
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
            # Get chunks directly from haskell_tree_sitter
            chunks = haskell_tree_sitter.get_haskell_ast_chunks_with_fallback(code)
            
            # Process each chunk using our handler
            for chunk in chunks:
                self._process_chunk(chunk, code)
            
            # Build metadata result
            metadata = {
                'language': 'haskell',
                'filename': filename,
                'line_count': len(code.split('\n')),
                'char_count': len(code),
                'analysis_method': 'haskell_chunk_visitor',
                'errors': self.errors,
                'node_stats': self.node_stats,
                'complexity_score': self.complexity_score,
                'parse_errors': 0,  # haskell_tree_sitter handles parse errors internally
                'tree_language': 'haskell_tree_sitter'
            }
            
            # Add Haskell-specific metadata from handler
            handler_summary = self.haskell_handler.get_summary()
            metadata.update(handler_summary)
            
            return metadata
            
        except Exception as e:
            LOGGER.error(f"Haskell chunk analysis failed: {e}")
            return self._fallback_analysis(code, filename)
    
    def _process_chunk(self, chunk, source_code: str):
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
        
        return {
            'language': 'haskell',
            'filename': filename,
            'line_count': len(lines),
            'char_count': len(code),
            'analysis_method': 'haskell_fallback',
            'errors': ['haskell_tree_sitter not available'],
            'node_stats': {},
            'complexity_score': 0,
            'parse_errors': 0,
            'tree_language': 'fallback',
            
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
    
    def __init__(self, chunk, source_code: str):
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
        return self.node.text()
    
    def get_position(self):
        """Get position from the chunk."""
        from ..ast_visitor import Position
        return Position(
            line=self.node.start_line(),
            column=0,  # haskell_tree_sitter doesn't provide column info
            byte_offset=self.node.start_byte() if hasattr(self.node, 'start_byte') else 0
        )


# Factory function for easy use
def analyze_haskell_code(code: str, filename: str = "") -> Dict[str, Any]:
    """Convenience function to analyze Haskell code."""
    visitor = HaskellASTVisitor()
    return visitor.analyze_haskell_code(code, filename)
