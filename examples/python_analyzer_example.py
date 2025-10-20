#!/usr/bin/env python3

"""
Example: Using the Enhanced Python Code Analyzer

This example demonstrates how to use the enhanced Python code analyzer
to extract metadata from Python source code.
"""

import os
import sys

from cocoindex_code_mcp_server.lang.python.python_code_analyzer import (
    analyze_python_code,
)

# Add src to path (examples is one level down from root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Test code
test_code = '''
import os
from typing import List, Optional

@dataclass
class ExampleClass:
    """An example class."""

    def __init__(self, name: str):
        self.name = name

    async def async_method(self) -> str:
        return f"Hello {self.name}"

    def _private_method(self):
        pass

def example_function(x: int = 5) -> int:
    """An example function."""
    return x * 2
'''

print("🧪 Enhanced Python Analyzer Example")
print("=" * 50)

metadata = analyze_python_code(test_code, "example.py")

if metadata is not None:
    print(f"Analysis Method: {metadata.get('analysis_method')}")
    print(f"Functions: {metadata.get('functions', [])}")
    print(f"Classes: {metadata.get('classes', [])}")
    print(f"Has Async: {metadata.get('has_async')}")
    print(f"Has Type Hints: {metadata.get('has_type_hints')}")
    print(f"Complexity Score: {metadata.get('complexity_score')}")
    if metadata.get("function_details"):
        print(f"Function Details: {len(metadata['function_details'])} functions with details")

    print("\n✅ Example completed successfully!")
    print("This demonstrates the enhanced Python analyzer capabilities:")
    print("- Tree-sitter AST analysis (with Python AST fallback)")
    print("- Function and class detection")
    print("- Type hint analysis")
    print("- Async function detection")
    print("- Complexity scoring")
else:
    print("Error: metadata is None")
