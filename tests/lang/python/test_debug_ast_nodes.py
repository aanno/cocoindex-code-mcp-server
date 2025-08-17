#!/usr/bin/env python3

"""
Debug test to understand what AST nodes are being processed.
"""


import pytest

from cocoindex_code_mcp_server.ast_visitor import (
    ASTParserFactory,
    GenericMetadataVisitor,
    TreeWalker,
)
from cocoindex_code_mcp_server.lang.python.tree_sitter_python_analyzer import (
    TreeSitterPythonAnalyzer,
)
from cocoindex_code_mcp_server.language_handlers import get_handler_for_language


def test_debug_ast_node_processing():
    """Debug what nodes are being processed and which handlers are called."""
    sample_code = '''
class Config:
    """A config class."""

    def __init__(self, name: str):
        self.name = name

    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        """Create from dict."""
        return cls(data["name"])

def utility_function() -> str:
    """A utility function."""
    return "test"
'''

    print("🔍 Debugging AST Node Processing")
    print("=" * 50)

    # Test the full TreeSitterPythonAnalyzer
    print("\n1. Full TreeSitterPythonAnalyzer:")
    analyzer = TreeSitterPythonAnalyzer(prefer_tree_sitter=True)
    metadata = analyzer.analyze_code(sample_code, "debug.py")

    if metadata is None:
        pytest.fail("metadata is None")
    else:
        print(f"Functions found: {metadata.get('functions', [])}")
        print(f"Classes found: {metadata.get('classes', [])}")
        print(f"Analysis method: {metadata.get('analysis_method')}")

        # Test the tree-sitter components directly
        print("\n2. Direct tree-sitter analysis:")
        parser_factory = ASTParserFactory()
        tree = parser_factory.parse_code(sample_code, 'python')

        if tree:
            print(f"Tree parsed successfully: {tree}")

            # Create visitor and handler
            visitor = GenericMetadataVisitor('python')
            python_handler = get_handler_for_language('python')

            if python_handler:
                print(f"Python handler found: {type(python_handler)}")
                visitor.add_handler(python_handler)

                # Walk the tree with debugging
                walker = TreeWalker(sample_code, tree)

                # Monkey patch to debug node types
                original_visit_node = visitor.visit_node

                def debug_visit_node(context):
                    # Debug: Check what attributes the node has
                    node = context.node
                    print(f"  Node attributes: {dir(node)}")

                    # Try different ways to get node type
                    node_type = None
                    if hasattr(node, 'kind'):
                        node_type = node.kind
                        print(f"  Has 'kind' attribute: {node_type}")
                    elif hasattr(node, 'type'):
                        node_type = node.type
                        print(f"  Has 'type' attribute: {node_type}")
                    else:
                        node_type = str(type(node))
                        print(f"  Fallback to str(type): {node_type}")

                    # Only show first few nodes to avoid spam
                    if context.depth < 5:
                        print(f"  Processing node: {node_type} "
                              f"(depth: {context.depth})")

                        # Check if any handler can handle this
                        for handler in visitor.handlers:
                            if handler.can_handle(node_type):
                                handler_name = type(handler).__name__
                                print(f"    ✓ Handler {handler_name} can handle "
                                      f"{node_type}")

                    return original_visit_node(context)

                setattr(visitor, 'visit_node', debug_visit_node)

                metadata = walker.walk(visitor)

                # Get handler summary
                if hasattr(python_handler, 'get_summary'):
                    handler_summary = python_handler.get_summary()
                    print("\n3. Handler Summary:")
                    print(f"  Functions: {handler_summary.get('functions', [])}")
                    print(f"  Classes: {handler_summary.get('classes', [])}")
                    print(f"  Function details: {len(handler_summary.get('function_details', []))}")
                    print(f"  Class details: {len(handler_summary.get('class_details', []))}")
                    metadata.update(handler_summary)

                print("\n4. Final metadata:")
                print(f"  Functions: {metadata.get('functions', [])}")
                print(f"  Classes: {metadata.get('classes', [])}")

                # Add proper assertions for pytest
                assert python_handler is not None, "Python handler should be found"
                assert metadata.get('functions') is not None, "Functions should be detected"
                assert metadata.get('classes') is not None, "Classes should be detected"
                assert 'Config' in metadata.get('classes', []), "Config class should be detected"

                # Verify specific functions are found
                functions = metadata.get('functions', [])
                expected_functions = {'__init__', 'from_dict', 'utility_function'}
                found_functions = set(functions)
                assert expected_functions.issubset(found_functions), (
                    f"Expected functions {expected_functions} not all found in "
                    f"{found_functions}"
                )

            else:
                assert False, "Python handler should be found"
        else:
            assert False, "Tree parsing should succeed"


if __name__ == "__main__":
    test_debug_ast_node_processing()
