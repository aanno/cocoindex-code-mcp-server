#!/usr/bin/env python3

"""
Integration test for the Python analyzer functionality.
Moved from src/python_code_analyzer.py to tests/
"""

import sys
import os
import logging

# Set up logger for tests
LOGGER = logging.getLogger(__name__)

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from lang.python.python_code_analyzer import analyze_python_code


def test_python_analyzer_integration():
    """Test the Python code analyzer with comprehensive sample code."""
    sample_code = '''
import os
from typing import List, Dict, Any
import numpy as np

class DataProcessor:
    """A class for processing data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._cache = {}
    
    @property
    def cache_size(self) -> int:
        """Get the cache size."""
        return len(self._cache)
    
    async def process_data(self, data: List[str], batch_size: int = 10) -> List[Dict[str, Any]]:
        """Process data asynchronously."""
        results = []
        for item in data:
            result = await self._process_item(item)
            results.append(result)
        return results
    
    def _process_item(self, item: str) -> Dict[str, Any]:
        """Process a single item."""
        return {"item": item, "processed": True}

def standalone_function(x: int, y: int = 5) -> int:
    """A standalone function."""
    return x + y

if __name__ == "__main__":
    processor = DataProcessor({"batch_size": 10})
    print("Processing complete")
'''
    
    # Analyze the sample code
    metadata = analyze_python_code(sample_code, "test.py")
    
    # Log the results
    LOGGER.info("📊 Enhanced Python Code Analysis Results:")
    LOGGER.info(f"Analysis Method: {metadata.get('analysis_method', 'unknown')}")
    LOGGER.info(f"Functions: {metadata['functions']}")
    LOGGER.info(f"Classes: {metadata['classes']}")
    LOGGER.info(f"Imports: {metadata['imports']}")
    LOGGER.info(f"Complexity Score: {metadata['complexity_score']}")
    LOGGER.info(f"Has Async: {metadata['has_async']}")
    LOGGER.info(f"Has Type Hints: {metadata['has_type_hints']}")
    LOGGER.info(f"Private Methods: {metadata['private_methods']}")
    LOGGER.info(f"Dunder Methods: {metadata.get('dunder_methods', [])}")
    
    # Show analysis details if available
    if 'function_details' in metadata and metadata['function_details']:
        LOGGER.info(f"Function Details Count: {len(metadata['function_details'])}")
    if 'class_details' in metadata and metadata['class_details']:
        LOGGER.info(f"Class Details Count: {len(metadata['class_details'])}")
    
    # Assertions for testing
    assert metadata['language'].lower() == 'python'  # Handle case variations
    assert 'standalone_function' in metadata['functions']
    assert 'DataProcessor' in metadata['classes']
    assert metadata['has_async'] is True
    assert metadata['has_type_hints'] is True
    assert len(metadata['imports']) >= 3  # os, typing, numpy
    assert metadata['complexity_score'] > 0
    
    # Check for expected metadata structure
    assert 'function_details' in metadata
    assert 'class_details' in metadata
    assert 'import_details' in metadata
    
    print("✅ Python analyzer integration test passed!")


def test_python_analyzer_edge_cases():
    """Test edge cases for the Python analyzer."""
    
    # Test empty code
    empty_metadata = analyze_python_code("", "empty.py")
    assert empty_metadata['line_count'] == 1
    assert len(empty_metadata['functions']) == 0
    assert len(empty_metadata['classes']) == 0
    
    # Test minimal code
    minimal_code = "x = 1"
    minimal_metadata = analyze_python_code(minimal_code, "minimal.py")
    assert minimal_metadata['language'].lower() == 'python'  # Handle case variations
    assert minimal_metadata['line_count'] == 1
    
    # Test syntax error handling
    broken_code = "def broken(:\n    pass"
    broken_metadata = analyze_python_code(broken_code, "broken.py")
    # Should handle gracefully and return basic metadata
    assert 'language' in broken_metadata
    assert broken_metadata['language'].lower() == 'python'  # Handle case variations
    
    print("✅ Python analyzer edge cases test passed!")


def test_python_analyzer_comprehensive():
    """Test comprehensive Python features."""
    
    comprehensive_code = '''
#!/usr/bin/env python3
"""
A comprehensive test module.
"""

import asyncio
from typing import Protocol, TypeVar, Generic
from dataclasses import dataclass
from pathlib import Path

T = TypeVar('T')

@dataclass
class Config:
    """Configuration dataclass."""
    name: str
    debug: bool = False

class Processor(Protocol):
    """Processor protocol."""
    def process(self, data: str) -> str: ...

class GenericProcessor(Generic[T]):
    """Generic processor class."""
    
    def __init__(self, config: Config):
        self.config = config
        self._buffer: List[T] = []
    
    @classmethod
    def from_config(cls, config_path: Path) -> 'GenericProcessor':
        """Create from config file."""
        return cls(Config("default"))
    
    @staticmethod
    def validate_input(data: str) -> bool:
        """Validate input data."""
        return len(data) > 0
    
    async def async_process(self, items: List[T]) -> Dict[str, Any]:
        """Process items asynchronously."""
        results = {}
        async with asyncio.TaskGroup() as tg:
            for item in items:
                task = tg.create_task(self._process_single(item))
                results[str(item)] = await task
        return results
    
    def _process_single(self, item: T) -> str:
        """Process single item."""
        return f"processed_{item}"
    
    def __str__(self) -> str:
        return f"GenericProcessor({self.config.name})"
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._buffer.clear()

def utility_function(x: int, y: int = 10, *args, **kwargs) -> int:
    """Utility function with various parameter types."""
    return sum([x, y] + list(args) + list(kwargs.values()))

# Lambda function
process_lambda = lambda x: x.upper()

# List comprehension
squared_numbers = [x**2 for x in range(10) if x % 2 == 0]

# Module-level constants
VERSION = "1.0.0"
DEBUG = True
'''
    
    metadata = analyze_python_code(comprehensive_code, "comprehensive.py")
    
    # Verify comprehensive analysis
    assert metadata['has_classes'] is True
    assert metadata['has_async'] is True
    assert metadata['has_type_hints'] is True
    assert metadata['has_decorators'] is True
    assert metadata['has_docstrings'] is True
    
    # Check specific elements
    assert 'Config' in metadata['classes']
    assert 'Processor' in metadata['classes']
    assert 'GenericProcessor' in metadata['classes']
    
    assert 'utility_function' in metadata['functions']
    
    expected_imports = ['asyncio', 'typing', 'dataclasses', 'pathlib']
    for imp in expected_imports:
        assert imp in metadata['imports'], f"Missing import: {imp}"
    
    # Check for decorators
    assert 'dataclass' in metadata['decorators']
    assert 'classmethod' in metadata['decorators']
    assert 'staticmethod' in metadata['decorators']
    
    # Should have reasonable complexity (adjusted expectation)
    assert metadata['complexity_score'] > 2
    
    print("✅ Python analyzer comprehensive test passed!")


if __name__ == "__main__":
    print("🧪 Running Python Analyzer Integration Tests")
    print("=" * 50)
    
    try:
        test_python_analyzer_integration()
        test_python_analyzer_edge_cases()
        test_python_analyzer_comprehensive()
        
        print("\n🎉 All Python analyzer integration tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
