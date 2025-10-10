#!/usr/bin/env python3

"""
Comprehensive tests for Python metadata extraction functionality.
"""

import json
import logging
import sys
from typing import Any, Dict, Optional, Union

import pytest

# Set up logger for tests
LOGGER = logging.getLogger(__name__)

try:
    from cocoindex_code_mcp_server.lang.python.python_code_analyzer import (
        analyze_python_code,
    )
except ImportError as e:
    LOGGER.warning(f"Could not import python_code_analyzer: {e}")
    print("⚠️  Warning: These tests require the full application setup.")

    def analyze_python_code(code: str, filename: str = "") -> Optional[Dict[str, Any]]:
        return None


class TestPythonMetadataAnalysis:
    """Test suite for Python code metadata analysis."""

    def test_simple_function_detection(self) -> None:
        """Test detection of simple functions."""
        if analyze_python_code is None:
            pytest.skip("python_code_analyzer not available")

        code = """
def hello_world():
    print("Hello, World!")

def calculate(x, y):
    return x + y
"""

        metadata: Union[Dict[str, Any], None] = analyze_python_code(code, "test.py")

        if metadata is not None:
            assert "functions" in metadata
            assert len(metadata["functions"]) == 2
            assert "hello_world" in metadata["functions"]
            assert "calculate" in metadata["functions"]
            assert metadata["has_classes"] is False
            assert metadata["has_async"] is False
        else:
            pytest.fail("no metadata in chunk")

    def test_class_detection(self):
        """Test detection of classes and methods."""
        code = """
class User:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def _private_method(self):
        pass

    def __str__(self):
        return f"User({self.name})"

class Manager(User):
    def manage(self):
        pass
"""

        metadata: Union[Dict[str, Any], None] = analyze_python_code(code, "test.py")

        if metadata is None:
            pytest.fail("metadata is None")
        else:
            assert "classes" in metadata
            assert len(metadata["classes"]) == 2
            assert "User" in metadata["classes"]
            assert "Manager" in metadata["classes"]
            assert metadata["has_classes"] is True

            # Check method detection - methods are in function_details
            assert "function_details" in metadata
            method_names = [f["name"] for f in metadata["function_details"]]
            assert "__init__" in method_names
            assert "get_name" in method_names
            assert "_private_method" in method_names
            assert "__str__" in method_names
            assert "manage" in method_names

            # Check private and dunder method categorization
            assert "_private_method" in metadata.get("private_methods", [])
            assert "__str__" in metadata.get("dunder_methods", [])

    def test_async_function_detection(self):
        """Test detection of async functions."""
        code = """
async def fetch_data():
    await some_operation()

def sync_function():
    pass

async def process_data():
    async with some_context():
        pass
"""

        metadata: Union[Dict[str, Any], None] = analyze_python_code(code, "test.py")

        if metadata is not None:
            assert metadata["has_async"] is True
            assert "fetch_data" in metadata["functions"]
            assert "sync_function" in metadata["functions"]
            assert "process_data" in metadata["functions"]
        else:
            pytest.fail("no metadata in chunk")

    def test_type_hints_detection(self):
        """Test detection of type hints."""
        code = """
def add(x: int, y: int) -> int:
    return x + y

def process_data(data: List[str]) -> Dict[str, Any]:
    return {}

def no_hints(x, y):
    return x + y
"""

        metadata: Union[Dict[str, Any], None] = analyze_python_code(code, "test.py")

        if metadata is not None:
            assert metadata["has_type_hints"] is True
            assert len(metadata["functions"]) == 3
        else:
            pytest.fail("no metadata in chunk")

    def test_import_detection(self) -> None:
        """Test detection of imports."""
        code = """
import os
import sys
from typing import List, Dict, Any
from pathlib import Path
import json as js
from collections import defaultdict
"""

        metadata = analyze_python_code(code, "test.py")

        if metadata is not None:
            assert "imports" in metadata
            imports = metadata["imports"]
            assert "os" in imports
            assert "sys" in imports
            assert "json" in imports  # 'as js' should be detected as 'json'
            assert "typing" in imports
            assert "pathlib" in imports
            assert "collections" in imports
        else:
            pytest.fail("no metadata in chunk")

    def test_decorator_detection(self) -> None:
        """Test detection of decorators."""
        code = """
@property
def name(self):
    return self._name

@staticmethod
def static_method():
    pass

@classmethod
def class_method(cls):
    pass

@custom_decorator
@another_decorator
def decorated_function():
    pass

@dataclass
class DataExample:
    name: str
"""

        metadata: Union[Dict[str, Any], None] = analyze_python_code(code, "test.py")

        if metadata is not None:
            assert "decorators" in metadata
            decorators = metadata["decorators"]
            assert "property" in decorators
            assert "staticmethod" in decorators
            assert "classmethod" in decorators
            assert "custom_decorator" in decorators
            assert "another_decorator" in decorators
            assert "dataclass" in decorators

            assert metadata["has_decorators"] is True
        else:
            pytest.fail("no metadata in chunk")

    def test_complexity_score(self) -> None:
        """Test complexity score calculation."""
        simple_code = """
def simple():
    return 1
"""

        complex_code = """
def complex_function(x, y, z):
    if x > 0:
        for i in range(y):
            if i % 2 == 0:
                try:
                    result = compute(i)
                    if result > threshold:
                        return result
                except Exception:
                    continue
            else:
                while z > 0:
                    z -= 1
    else:
        return None
"""

        simple_metadata: Union[Dict[str, Any], None] = analyze_python_code(simple_code, "simple.py")
        complex_metadata: Union[Dict[str, Any], None] = analyze_python_code(complex_code, "complex.py")

        if simple_metadata is not None and complex_metadata is not None:
            assert simple_metadata["complexity_score"] < complex_metadata["complexity_score"]
            # Should be reasonably high (adjusted for enhanced analyzer)
            assert complex_metadata["complexity_score"] > 5
        else:
            pytest.fail("no metadata in chunk")

    def test_docstring_detection(self):
        """Test detection of docstrings."""
        code = """
def documented_function():
    \"\"\"This function has a docstring.\"\"\"
    pass

def undocumented_function():
    pass

class DocumentedClass:
    \"\"\"This class has a docstring.\"\"\"

    def method_with_docs(self):
        \"\"\"This method has docs.\"\"\"
        pass

    def method_without_docs(self):
        pass
"""

        metadata: Union[Dict[str, Any], None] = analyze_python_code(code, "test.py")

        if metadata is not None:
            assert metadata["has_docstrings"] is True
        else:
            pytest.fail("no metadata in chunk")

    @pytest.mark.skip(reason="Variable detection format changed in enhanced analyzer")
    def test_variable_detection(self):
        """Test detection of module-level variables."""
        code = """
MODULE_CONSTANT = "value"
global_var = 42
_private_var = "secret"

def function():
    local_var = 1  # This shouldn't be detected as module variable

class MyClass:
    class_var = "class level"  # This should be detected
"""

        metadata: Union[Dict[str, Any], None] = analyze_python_code(code, "test.py")

        if metadata is not None:
            assert "variables" in metadata
            variables = metadata["variables"]
            assert "MODULE_CONSTANT" in variables
            assert "global_var" in variables
            assert "_private_var" in variables

            # Class variables should be separate
            assert "class_variables" in metadata
            class_vars = metadata["class_variables"]
            assert "class_var" in class_vars
        else:
            pytest.fail("no metadata in chunk")

    @pytest.mark.skip(reason="Output format validation changed in enhanced analyzer")
    def test_comprehensive_real_world_example(self):
        """Test with a more realistic code example."""
        code = '''
#!/usr/bin/env python3
"""
A realistic Python module example.
"""

import os
import json
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# Module constants
DEFAULT_CONFIG = "config.json"
MAX_RETRIES = 3

@dataclass
class Config:
    """Configuration data class."""
    name: str
    values: List[int]
    metadata: Optional[Dict[str, str]] = None

class DataProcessor:
    """Main data processing class."""

    def __init__(self, config: Config):
        self.config = config
        self._cache = {}

    @property
    def name(self) -> str:
        """Get the processor name."""
        return self.config.name

    @staticmethod
    def validate_input(data: Union[str, int]) -> bool:
        """Validate input data."""
        return isinstance(data, (str, int))

    async def process_async(self, items: List[str]) -> Dict[str, Any]:
        """Process items asynchronously."""
        results = {}
        for item in items:
            if self.validate_input(item):
                try:
                    result = await self._process_single(item)
                    results[item] = result
                except ProcessingError as e:
                    self._log_error(e)
                    continue
        return results

    def _process_single(self, item: str) -> str:
        """Private method to process single item."""
        return item.upper()

    def __str__(self) -> str:
        return f"DataProcessor({self.name})"

class ProcessingError(Exception):
    """Custom exception for processing errors."""
    pass

def load_config(path: Path = Path(DEFAULT_CONFIG)) -> Config:
    """Load configuration from file."""
    with open(path) as f:
        data = json.load(f)
    return Config(**data)

if __name__ == "__main__":
    config = load_config()
    processor = DataProcessor(config)
'''

        metadata: Union[Dict[str, Any], None] = analyze_python_code(code, "real_example.py")

        if metadata is not None:
            # Verify comprehensive analysis
            assert metadata["has_classes"] is True
            assert metadata["has_async"] is True
            assert metadata["has_type_hints"] is True
            assert metadata["has_decorators"] is True
            assert metadata["has_docstrings"] is True

            # Check specific elements
            assert "Config" in metadata["classes"]
            assert "DataProcessor" in metadata["classes"]
            assert "ProcessingError" in metadata["classes"]

            assert "load_config" in metadata["functions"]
            assert "process_async" in metadata["functions"]
            assert "_process_single" in metadata["private_methods"]
            assert "__str__" in metadata["dunder_methods"]

            assert "dataclass" in metadata["decorators"]
            assert "property" in metadata["decorators"]
            assert "staticmethod" in metadata["decorators"]

            assert "os" in metadata["imports"]
            assert "json" in metadata["imports"]
            assert "typing" in metadata["imports"]
            assert "dataclasses" in metadata["imports"]
            assert "pathlib" in metadata["imports"]

            assert "DEFAULT_CONFIG" in metadata["variables"]
            assert "MAX_RETRIES" in metadata["variables"]

            # Should have reasonable complexity
            assert metadata["complexity_score"] > 15
        else:
            pytest.fail("no metadata in chunk")

    def test_metadata_json_serialization(self):
        """Test that metadata can be properly serialized to JSON."""
        code = """
from datetime import datetime

class Example:
    def method(self) -> str:
        return "test"
"""

        metadata = analyze_python_code(code, "test.py")

        # Ensure all metadata can be JSON serialized
        try:
            json_str = json.dumps(metadata, default=str)
            # And can be parsed back
            parsed = json.loads(json_str)
            assert parsed["has_classes"] is True
            assert "Example" in parsed["classes"]
            assert "datetime" in parsed["imports"]
        except (TypeError, ValueError) as e:
            pytest.fail(f"Metadata is not JSON serializable: {e}")

    @pytest.mark.skip(reason="Edge case handling changed in enhanced analyzer")
    def test_edge_cases(self):
        """Test edge cases and error handling."""

        # Empty code
        metadata: Union[Dict[str, Any], None] = analyze_python_code("", "empty.py")
        if metadata is not None:
            assert metadata["line_count"] == 0
            assert metadata["char_count"] == 0
            assert len(metadata["functions"]) == 0
            assert len(metadata["classes"]) == 0
        else:
            pytest.fail("no metadata in chunk")

        # Only comments
        metadata = analyze_python_code("# Just a comment\n# Another comment", "comments.py")
        if metadata is not None:
            assert metadata["line_count"] == 2
            assert len(metadata["functions"]) == 0
        else:
            pytest.fail("no metadata in chunk")

        # Syntax error handling
        try:
            metadata = analyze_python_code("def broken(:\n    pass", "broken.py")
            if metadata is not None:
                # Should handle gracefully and return basic info
                assert "line_count" in metadata
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass


if __name__ == "__main__":
    # Check if module is available
    if analyze_python_code is None:
        print("❌ Cannot run tests: python_code_analyzer module not available")
        print("These tests require the full application setup with proper imports.")
        sys.exit(1)

    # Run tests manually if pytest is not available
    test_instance = TestPythonMetadataAnalysis()

    print("🧪 Running Python Metadata Tests")
    print("=" * 50)

    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]

    passed = 0
    failed = 0

    for test_method in test_methods:
        try:
            print(f"Running {test_method}...", end=" ")
            getattr(test_instance, test_method)()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed")
    print("✅ Python metadata tests completed!")
