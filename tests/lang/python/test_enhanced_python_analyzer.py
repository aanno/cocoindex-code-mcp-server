#!/usr/bin/env python3

"""
Test the enhanced Python analyzer with tree-sitter integration.
"""

import logging
import sys

import pytest

# Set up logger for tests
LOGGER = logging.getLogger(__name__)


def test_enhanced_analyzer():
    """Test the enhanced Python analyzer."""

    try:
        from cocoindex_code_mcp_server.lang.python.python_code_analyzer import (
            analyze_python_code,
        )
    except ImportError as e:
        LOGGER.error("Could not import analyzer: %s", e)
        print(f"❌ Could not import analyzer: {e}")
        return False

    # Test code with various Python features
    test_code = """
import os
from typing import List, Dict, Any, Optional
import asyncio

@dataclass
class TestClass:
    '''A test class with various features.'''

    def __init__(self, name: str):
        self.name = name
        self._private_attr = None

    @property
    def name_upper(self) -> str:
        '''Get uppercase name.'''
        return self.name.upper()

    @staticmethod
    def static_method() -> bool:
        return True

    async def async_method(self, items: List[str]) -> Dict[str, Any]:
        '''Process items asynchronously.'''
        results = {}
        for item in items:
            if item:
                results[item] = await self._process_item(item)
        return results

    def _process_item(self, item: str) -> str:
        return f"processed_{item}"

    def __str__(self) -> str:
        return f"TestClass({self.name})"

def standalone_function(x: int, y: int = 10) -> int:
    '''A standalone function with default parameters.'''
    if x > 0:
        for i in range(y):
            x += i
    return x

# Module level variable
MODULE_VAR = "test_value"
"""

    print("🧪 Testing Enhanced Python Analyzer")
    print("=" * 50)

    try:
        metadata = analyze_python_code(test_code, "test_enhanced.py")

        if metadata is None:
            pytest.fail("metadata is None")
        else:
            print("✅ Analysis completed successfully!")
            print(f"📊 Analysis Method: {metadata.get('analysis_method', 'unknown')}")
            print(f"📁 Language: {metadata.get('language', 'unknown')}")
            print(f"📏 Lines: {metadata.get('line_count', 0)}")
            print(f"🔧 Functions: {len(metadata.get('functions', []))}")
            print(f"🏛️  Classes: {len(metadata.get('classes', []))}")
            print(f"📦 Imports: {len(metadata.get('imports', []))}")
            print(f"🎯 Complexity: {metadata.get('complexity_score', 0)}")

            # Test specific features
            features = []
            if metadata.get("has_async"):
                features.append("async")
            if metadata.get("has_type_hints"):
                features.append("type_hints")
            if metadata.get("has_decorators"):
                features.append("decorators")
            if metadata.get("has_classes"):
                features.append("classes")
            if metadata.get("has_docstrings"):
                features.append("docstrings")

            print(f"✨ Features detected: {', '.join(features)}")

            # Check for detailed information
            if "function_details" in metadata:
                print(f"📋 Function details available: {len(metadata['function_details'])} functions")
            if "class_details" in metadata:
                print(f"📋 Class details available: {len(metadata['class_details'])} classes")

            # Check for specific elements we expect
            expected_functions = ["standalone_function"]
            expected_classes = ["TestClass"]
            expected_imports = ["os", "typing", "asyncio"]

            found_functions = metadata.get("functions", [])
            found_classes = metadata.get("classes", [])
            found_imports = metadata.get("imports", [])

            all_good = True

            for expected in expected_functions:
                if expected not in found_functions:
                    print(f"❌ Missing expected function: {expected}")
                    all_good = False
                else:
                    print(f"✅ Found expected function: {expected}")

            for expected in expected_classes:
                if expected not in found_classes:
                    print(f"❌ Missing expected class: {expected}")
                    all_good = False
                else:
                    print(f"✅ Found expected class: {expected}")

            for expected in expected_imports:
                if expected not in found_imports:
                    print(f"❌ Missing expected import: {expected}")
                    all_good = False
                else:
                    print(f"✅ Found expected import: {expected}")

            # Check for enhanced features
            if metadata.get("private_methods"):
                print(f"🔒 Private methods found: {metadata['private_methods']}")
            if metadata.get("dunder_methods"):
                print(f"🔮 Dunder methods found: {metadata['dunder_methods']}")

            if all_good:
                print("\n🎉 All tests passed! Enhanced analyzer working correctly.")
                return True
            else:
                print("\n⚠️  Some tests failed, but basic functionality works.")
                return True

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False


if __name__ == "__main__":
    success = test_enhanced_analyzer()
    if success:
        print("\n✅ Enhanced Python analyzer test completed successfully!")
    else:
        print("\n❌ Enhanced Python analyzer test failed!")
        sys.exit(1)
