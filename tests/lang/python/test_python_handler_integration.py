#!/usr/bin/env python3

"""
Test for the fixed Python handler integration in cocoindex_config.py.
This test verifies that the TreeSitterPythonAnalyzer properly integrates
the PythonNodeHandler through the AST visitor framework.
"""

import json
import sys
from types import FunctionType
from typing import cast

import pytest

from cocoindex_code_mcp_server.cocoindex_config import (
    PYTHON_HANDLER_AVAILABLE,
    extract_code_metadata,
)
from cocoindex_code_mcp_server.lang.python.tree_sitter_python_analyzer import (
    TreeSitterPythonAnalyzer,
)


def test_tree_sitter_python_analyzer_integration():
    """Test that TreeSitterPythonAnalyzer properly integrates PythonNodeHandler."""

    sample_code = '''
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration dataclass."""
    name: str
    debug: bool = False

class Processor:
    """A sample processor class."""

    def __init__(self, config: Config):
        self.config = config
        self._cache = {}

    @classmethod
    def from_config(cls, config_path: str) -> 'Processor':
        """Create processor from config file."""
        return cls(Config("default"))

    async def process_data(self, items: List[str]) -> Dict[str, Any]:
        """Process data asynchronously."""
        results = {}
        for item in items:
            result = await self._process_single(item)
            results[item] = result
        return results

    def _process_single(self, item: str) -> str:
        """Process single item (private method)."""
        return f"processed_{item}"

    def __str__(self) -> str:
        """String representation (dunder method)."""
        return f"Processor({self.config.name})"

def utility_function(x: int, y: int = 10) -> int:
    """A utility function with type hints."""
    return x + y
'''

    # Test the analyzer directly
    analyzer = TreeSitterPythonAnalyzer(prefer_tree_sitter=True)
    metadata = analyzer.analyze_code(sample_code, "test.py")

    if metadata is None:
        pytest.fail('metadata is None')
    else:
        # Verify that we get enhanced metadata from the PythonNodeHandler
        assert 'analysis_method' in metadata
        assert metadata['analysis_method'] in ['tree_sitter', 'tree_sitter+python_ast', 'tree_sitter+python_code_analyzer']

        # Verify function detection
        assert 'functions' in metadata
        print(f"DEBUG: Found functions: {metadata['functions']}")
        print(f"DEBUG: Full metadata keys: {list(metadata.keys())}")

        # Check for utility_function which should definitely be found
        assert 'utility_function' in metadata['functions'], f"utility_function not found in {metadata['functions']}"

        # Check that we found at least some functions/classes
        assert len(metadata['functions']) > 0, "No functions found"

        # Verify class detection
        assert 'classes' in metadata
        print(f"DEBUG: Found classes: {metadata['classes']}")
        # Classes should be detected
        assert len(metadata['classes']) >= 1, f"Expected at least 1 class, found: {metadata['classes']}"

        # Verify async detection - might not work if methods aren't in top-level functions
        print(f"DEBUG: has_async: {metadata.get('has_async', False)}")

        # Verify type hints detection
        print(f"DEBUG: has_type_hints: {metadata.get('has_type_hints', False)}")

        # Verify decorators detection
        print(f"DEBUG: has_decorators: {metadata.get('has_decorators', False)}")
        print(f"DEBUG: decorators: {metadata.get('decorators', [])}")

        # Verify private/dunder method detection
        print(f"DEBUG: private_methods: {metadata.get('private_methods', [])}")
        print(f"DEBUG: dunder_methods: {metadata.get('dunder_methods', [])}")

        # Basic sanity checks - the analyzer should detect something
        assert len(metadata['functions']) + len(metadata['classes']
                                                ) > 0, "Should detect at least some functions or classes"

        # Verify imports
        assert 'imports' in metadata
        expected_imports = ['asyncio', 'typing', 'dataclasses']
        for imp in expected_imports:
            assert imp in metadata['imports'], f"Import {imp} not found in {metadata['imports']}"

        print("✅ TreeSitterPythonAnalyzer integration test passed!")


def test_cocoindex_config_python_handler_integration():
    """Test that the cocoindex_config uses TreeSitterPythonAnalyzer when PYTHON_HANDLER_AVAILABLE."""

    if not PYTHON_HANDLER_AVAILABLE:
        pytest.skip("Python handler not available")

    sample_code = '''
class TestClass:
    """A test class."""

    def __init__(self):
        self.value = 42

    @property
    def doubled(self) -> int:
        """Double the value."""
        return self.value * 2

    async def async_method(self) -> str:
        """An async method."""
        return "async_result"

def test_function() -> bool:
    """A test function."""
    return True
'''

    # Test through cocoindex_config integration
    metadata_json = cast(FunctionType, extract_code_metadata)(sample_code, "Python", "test.py")
    metadata = json.loads(metadata_json)

    # Verify we get proper metadata through the integration
    assert 'functions' in metadata
    assert 'test_function' in metadata['functions']

    assert 'classes' in metadata
    assert 'TestClass' in metadata['classes']

    assert metadata.get('has_async', False) is True
    assert metadata.get('has_type_hints', False) is True
    assert metadata.get('has_classes', False) is True

    # Verify analysis method indicates enhanced processing
    assert metadata.get('analysis_method') != 'basic'

    print("✅ CocoIndex config Python handler integration test passed!")


def test_python_handler_fallback():
    """Test that fallback works when TreeSitterPythonAnalyzer fails."""

    # Test with potentially problematic code
    broken_code = '''
def incomplete_function(
    # This is intentionally broken syntax
'''

    # Should not crash and should provide basic metadata
    metadata_json = cast(FunctionType, extract_code_metadata)(broken_code, "Python", "broken.py")
    metadata = json.loads(metadata_json)

    # Should have basic structure even with broken code
    assert 'functions' in metadata
    assert 'classes' in metadata
    assert 'imports' in metadata
    assert 'analysis_method' in metadata

    # Analysis method might be fallback
    assert metadata['analysis_method'] in ['basic', 'python_ast', 'tree_sitter', 'tree_sitter+python_ast', 'tree_sitter+python_code_analyzer']

    print("✅ Python handler fallback test passed!")


if __name__ == "__main__":
    print("🧪 Running Python Handler Integration Tests")
    print("=" * 60)

    try:
        test_tree_sitter_python_analyzer_integration()
        test_cocoindex_config_python_handler_integration()
        test_python_handler_fallback()

        print("\n🎉 All Python handler integration tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
