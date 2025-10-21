#!/usr/bin/env python3

"""
Test RAG metadata compliance - ensures Python analyzer generates all recommended metadata fields
from the proposed-rag-metadata.md specification.
"""

import json
import logging
import sys

import pytest

# Set up logger for tests
LOGGER = logging.getLogger(__name__)


def test_rag_metadata_compliance():
    """Test that Python analyzer generates all recommended RAG metadata fields."""

    try:
        from cocoindex_code_mcp_server.lang.python.python_code_analyzer import (
            analyze_python_code,
        )
    except ImportError as e:
        LOGGER.error("Could not import analyzer: %s", e)
        print(f"❌ Could not import analyzer: {e}")
        return False

    # Comprehensive test code with all Python features
    test_code = """
import os
import sys
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import json

# Module level variable
MODULE_CONSTANT = "test_value"
module_var: int = 42

@dataclass
class BaseClass:
    '''Base class with docstring.'''
    name: str
    value: int = 0

    def base_method(self) -> str:
        '''Base method with docstring.'''
        return f"base_{self.name}"

@dataclass
class AdvancedClass(BaseClass):
    '''Advanced class inheriting from BaseClass.'''

    items: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _private_attr: Optional[str] = None

    def __init__(self, name: str, value: int = 10):
        super().__init__(name, value)
        self._private_attr = f"private_{name}"

    @property
    def name_upper(self) -> str:
        '''Property returning uppercase name.'''
        return self.name.upper()

    @staticmethod
    def create_empty() -> 'AdvancedClass':
        '''Static method factory.'''
        return AdvancedClass("empty", 0)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedClass':
        '''Class method factory from dictionary.'''
        return cls(data["name"], data.get("value", 0))

    async def async_process(self, items: List[str]) -> Dict[str, Any]:
        '''Async method processing items.'''
        results = {}
        for item in items:
            if item:
                try:
                    result = await self._async_helper(item)
                    results[item] = result
                except Exception as e:
                    results[item] = f"error: {e}"
        return results

    async def _async_helper(self, item: str) -> str:
        '''Private async helper method.'''
        await asyncio.sleep(0.001)  # Simulate async work
        return f"processed_{item}"

    def complex_method(self, data: Union[str, List[str]], threshold: int = 5) -> Optional[Dict[str, Any]]:
        '''Complex method with multiple control flows.'''
        result = {}

        if isinstance(data, str):
            data = [data]

        for item in data:
            if len(item) > threshold:
                for i, char in enumerate(item):
                    if char.isalpha():
                        result[f"{item}_{i}"] = char.upper()
                    elif char.isdigit():
                        result[f"{item}_{i}"] = int(char) * 2
            else:
                while len(item) < threshold:
                    item += "_pad"
                result[item] = "padded"

        return result if result else None

    def _private_method(self, x: int, y: int = 1) -> int:
        '''Private method with default parameters.'''
        return x * y

    def __str__(self) -> str:
        '''String representation dunder method.'''
        return f"AdvancedClass(name={self.name}, value={self.value})"

    def __len__(self) -> int:
        '''Length dunder method.'''
        return len(self.items)

def standalone_function(x: int, y: int = 10, *args, **kwargs) -> int:
    '''Standalone function with various parameter types.'''
    result = x + y

    # Complex control flow for testing
    if x > 0:
        for i in range(min(y, 10)):
            if i % 2 == 0:
                result += i
            else:
                result -= i
    elif x < 0:
        while result > 0 and y > 0:
            result -= 1
            y -= 1

    # List comprehension
    multipliers = [i * 2 for i in args if isinstance(i, int)]
    result += sum(multipliers)

    return result

async def async_standalone_function(data: List[Dict[str, Any]]) -> List[str]:
    '''Async standalone function.'''
    results = []

    async def process_item(item: Dict[str, Any]) -> str:
        await asyncio.sleep(0.001)
        return json.dumps(item)

    for item in data:
        try:
            result = await process_item(item)
            results.append(result)
        except Exception:
            results.append("error")

    return results

def generator_function(n: int):
    '''Generator function.'''
    for i in range(n):
        yield i * 2

# Lambda function
lambda_func = lambda x, y=5: x + y

# Module level assignments
config_data = {
    "debug": True,
    "timeout": 30,
    "features": ["async", "typing", "dataclasses"]
}

# Complex data structures
nested_structure = [
    {"type": "config", "data": config_data},
    {"type": "class", "data": AdvancedClass("test", 100)},
    {"type": "function", "data": standalone_function}
]
"""

    print("🧪 Testing RAG Metadata Compliance")
    print("=" * 60)

    try:
        metadata = analyze_python_code(test_code, "test_rag_compliance.py")

        if metadata is None:
            pytest.fail("metadata is None")
        else:
            print("✅ Analysis completed successfully!")
            print(f"📊 Analysis Method: {metadata.get('analysis_method', 'unknown')}")

            # Check all required RAG metadata fields
            required_fields = {
                # Basic RAG-pychunk fields
                "file": str,
                "node_type": str,
                "lines_of_code": tuple,
                "hash": str,
                "node_relationships": dict,
                "additional_metadata": dict,
                # Our extended fields
                "language": str,
                "filename": str,
                "line_count": int,
                "char_count": int,
                "start_line": int,
                "end_line": int,
                "start_column": int,
                "end_column": int,
                "content_hash": str,
                # Content analysis
                "functions": list,
                "classes": list,
                "imports": list,
                "variables": list,
                "decorators": list,
                "complexity_score": int,
                # Feature flags
                "has_async": bool,
                "has_classes": bool,
                "has_decorators": bool,
                "has_type_hints": bool,
                "has_docstrings": bool,
                # Detailed information
                "function_details": list,
                "class_details": list,
                "import_details": list,
                "private_methods": list,
                "dunder_methods": list,
                # JSON compatibility
                "metadata_json": str,
            }

            print("\n📋 Checking Required RAG Metadata Fields:")
            print("-" * 40)

            all_fields_present = True
            for field, expected_type in required_fields.items():
                if field in metadata:
                    actual_type = type(metadata[field])
                    if actual_type == expected_type:
                        print(f"✅ {field}: {expected_type.__name__}")
                    else:
                        print(f"⚠️  {field}: expected {expected_type.__name__}, got {actual_type.__name__}")
                        all_fields_present = False
                else:
                    print(f"❌ {field}: MISSING")
                    all_fields_present = False

            # Check node_relationships structure
            print("\n🔗 Checking Node Relationships Structure:")
            print("-" * 40)

            node_rel = metadata.get("node_relationships", {})
            expected_rel_fields = [
                "parent",
                "children",
                "scope",
                "contains",
                "class_hierarchies",
                "import_dependencies",
            ]

            for rel_field in expected_rel_fields:
                if rel_field in node_rel:
                    print(f"✅ node_relationships.{rel_field}")
                else:
                    print(f"❌ node_relationships.{rel_field}: MISSING")
                    all_fields_present = False

            # Check additional_metadata structure
            print("\n📊 Checking Additional Metadata Structure:")
            print("-" * 40)

            add_meta = metadata.get("additional_metadata", {})
            expected_add_fields = [
                "analysis_method",
                "parser_version",
                "extracted_at",
                "total_functions",
                "total_classes",
                "total_imports",
                "code_patterns",
            ]

            for add_field in expected_add_fields:
                if add_field in add_meta:
                    print(f"✅ additional_metadata.{add_field}")
                else:
                    print(f"❌ additional_metadata.{add_field}: MISSING")
                    all_fields_present = False

            # Test content accuracy
            print("\n🎯 Checking Content Accuracy:")
            print("-" * 40)

            expected_content = {
                "functions": ["standalone_function", "async_standalone_function", "generator_function"],
                "classes": ["BaseClass", "AdvancedClass"],
                "imports": ["os", "sys", "typing", "dataclasses", "pathlib", "asyncio", "json"],
                "has_async": True,
                "has_classes": True,
                "has_decorators": True,
                "has_type_hints": True,
                "has_docstrings": True,
            }

            content_accurate = True
            for field, expected in expected_content.items():
                actual = metadata.get(field)
                if field.startswith("has_"):
                    if actual == expected:
                        print(f"✅ {field}: {actual}")
                    else:
                        print(f"❌ {field}: expected {expected}, got {actual}")
                        content_accurate = False
                else:  # Lists
                    actual_list = actual if isinstance(actual, list) else []
                    expected_list = expected if isinstance(expected, list) else []
                    missing = set(expected_list) - set(actual_list)
                    if not missing:
                        print(f"✅ {field}: all {len(expected_list)} expected items found")
                    else:
                        print(f"❌ {field}: missing {missing}")
                        content_accurate = False

            # Check detailed function information
            print("\n🔍 Checking Detailed Function Information:")
            print("-" * 40)

            func_details = metadata.get("function_details", [])
            if func_details:
                sample_func = func_details[0]
                required_func_fields = [
                    "name",
                    "line",
                    "end_line",
                    "column",
                    "end_column",
                    "lines_of_code",
                    "parameters",
                    "return_type",
                    "decorators",
                    "docstring",
                ]

                for func_field in required_func_fields:
                    if func_field in sample_func:
                        print(f"✅ function_details[0].{func_field}")
                    else:
                        print(f"❌ function_details[0].{func_field}: MISSING")
            else:
                print("❌ No function details found")

            # Check JSON serialization
            print("\n📄 Checking JSON Serialization:")
            print("-" * 40)

            try:
                json_str = metadata.get("metadata_json", "")
                if json_str:
                    json.loads(json_str)
                    print(f"✅ metadata_json: valid JSON ({len(json_str)} chars)")
                else:
                    print("❌ metadata_json: empty or missing")
                    all_fields_present = False
            except json.JSONDecodeError as e:
                print(f"❌ metadata_json: invalid JSON - {e}")
                all_fields_present = False

            # Summary
            print("\n📊 Summary:")
            print("=" * 60)
            print(f"📁 File: {metadata.get('filename', 'unknown')}")
            print(f"🏷️  Node Type: {metadata.get('node_type', 'unknown')}")
            print(f"📏 Lines: {metadata.get('line_count', 0)}")
            print(f"🔧 Functions: {len(metadata.get('functions', []))}")
            print(f"🏛️  Classes: {len(metadata.get('classes', []))}")
            print(f"📦 Imports: {len(metadata.get('imports', []))}")
            print(f"🎯 Complexity: {metadata.get('complexity_score', 0)}")
            print(f"🔑 Hash: {metadata.get('hash', 'missing')[:8]}...")

            if all_fields_present and content_accurate:
                print("\n🎉 ALL RAG METADATA COMPLIANCE TESTS PASSED!")
                print("✅ Python analyzer generates complete RAG-compliant metadata")
                return True
            else:
                print("\n⚠️  Some RAG metadata compliance issues found")
                print(f"📋 Fields Present: {'✅' if all_fields_present else '❌'}")
                print(f"🎯 Content Accurate: {'✅' if content_accurate else '❌'}")
                return False

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_specific_metadata_features():
    """Test specific metadata features in detail."""

    try:
        from cocoindex_code_mcp_server.lang.python.python_code_analyzer import (
            analyze_python_code,
        )
    except ImportError as e:
        print(f"❌ Could not import analyzer: {e}")
        return False

    print("\n🔬 Testing Specific Metadata Features:")
    print("=" * 60)

    # Test position information accuracy
    position_test_code = """def first_function():
    pass

class TestClass:
    def method(self):
        return True

def last_function(x: int) -> str:
    return str(x)
"""

    try:
        metadata = analyze_python_code(position_test_code, "position_test.py")

        if metadata is None:
            pytest.fail("metadata is None")
        else:
            print("📍 Position Information Test:")
            print("-" * 30)

            func_details = metadata.get("function_details", [])
            for func in func_details:
                if func["name"] == "first_function":
                    expected_line = 1
                    actual_line = func.get("line", 0)
                    if actual_line == expected_line:
                        print(f"✅ first_function line position: {actual_line}")
                    else:
                        print(f"❌ first_function line position: expected {expected_line}, got {actual_line}")

                # Check lines_of_code tuple format
                lines_of_code = func.get("lines_of_code")
                if isinstance(lines_of_code, tuple) and len(lines_of_code) == 2:
                    print(f"✅ {func['name']} lines_of_code format: {lines_of_code}")
                else:
                    print(f"❌ {func['name']} lines_of_code format invalid: {lines_of_code}")

            # Test hash uniqueness
            print("\n🔑 Hash Uniqueness Test:")
            print("-" * 30)

            hash1 = metadata.get("hash", "")

            # Analyze slightly different code
            modified_code = position_test_code + "\n# comment\n"
            metadata2 = analyze_python_code(modified_code, "position_test.py")

            if metadata2 is None:
                pytest.fail("metadata2 is None")
            else:
                hash2 = metadata2.get("hash", "")

                if hash1 and hash2 and hash1 != hash2:
                    print("✅ Hashes are unique for different content")
                    print(f"   Original: {hash1}")
                    print(f"   Modified: {hash2}")
                else:
                    print("❌ Hash uniqueness test failed")
                    print(f"   Original: {hash1}")
                    print(f"   Modified: {hash2}")

                return True

    except Exception as e:
        print(f"❌ Error in specific features test: {e}")
        return False


if __name__ == "__main__":
    print("🧪 RAG Metadata Compliance Test Suite")
    print("=" * 60)

    success1 = test_rag_metadata_compliance()
    success2 = test_specific_metadata_features()

    if success1 and success2:
        print("\n🎉 ALL RAG METADATA COMPLIANCE TESTS PASSED!")
        print("✅ Python analyzer fully compliant with RAG metadata recommendations")
    else:
        print("\n⚠️  Some RAG metadata compliance tests failed")
        sys.exit(1)
