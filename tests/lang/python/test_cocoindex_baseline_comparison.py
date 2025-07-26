#!/usr/bin/env python3

"""
CocoIndex baseline comparison test following proper unit testing patterns.

This test compares our custom Python metadata extractor against the true CocoIndex
default baseline by using the configuration flags to control which extractor is used.
Based on the pytest example pattern suggested for testing extractors.
"""

import pytest
import json
import sys
import os
from typing import Dict, Any, List
from cocoindex_config import extract_code_metadata, _global_flow_config

class CocoIndexMetadataExtractor:
    """Wrapper for CocoIndex metadata extraction with configurable behavior."""
    
    def __init__(self, metadata: Dict[str, Any]):
        self.metadata = metadata
        self.provider_name = metadata.get("provider_name", "CocoIndex")
        self.version = metadata.get("version", "1.0")
        self.use_default_handler = metadata.get("use_default_handler", False)
        
    def initialize(self) -> bool:
        """Initialize the extractor with the given configuration."""
        try:
            # Update global config based on our settings
            original_config = _global_flow_config.get('use_default_language_handler', False)
            _global_flow_config['use_default_language_handler'] = self.use_default_handler
            self._original_config = original_config
            return True
        except Exception as e:
            print(f"Failed to initialize extractor: {e}")
            return False
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from the provided code data."""
        try:
            code = data.get("code", "")
            filename = data.get("filename", "test.py")
            language = data.get("language", "Python")
            
            # Extract metadata using CocoIndex
            metadata_json = extract_code_metadata(code, language, filename)
            metadata_dict = json.loads(metadata_json)
            
            # Add extractor information
            result = {
                "provider": self.provider_name,
                "version": self.version,
                "extractor_type": "default" if self.use_default_handler else "custom",
                "analysis_method": metadata_dict.get("analysis_method", "unknown"),
                "metadata": metadata_dict
            }
            
            return result
            
        except Exception as e:
            return {
                "provider": self.provider_name,
                "version": self.version,
                "extractor_type": "default" if self.use_default_handler else "custom",
                "error": str(e),
                "metadata": {}
            }
    
    def cleanup(self):
        """Restore original configuration."""
        if hasattr(self, '_original_config'):
            _global_flow_config['use_default_language_handler'] = self._original_config

@pytest.fixture
def dummy_metadata():
    """Provide dummy metadata for extractor initialization."""
    return {
        "provider_name": "CocoIndex",
        "version": "2.5",
        "source_type": "unit_test"
    }

@pytest.fixture
def sample_python_code():
    """Provide comprehensive Python code sample for testing."""
    return {
        "code": '''
"""Module docstring for testing."""

import os
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class TestProcessor:
    """A test processor class with various method types."""
    
    name: str
    config: Dict[str, Any] = None
    
    def __init__(self, name: str):
        self.name = name
        self._internal_state = {}
    
    @property
    def status(self) -> str:
        """Get the current status."""
        return self._internal_state.get('status', 'idle')
    
    @staticmethod
    def validate_input(data: Dict) -> bool:
        """Validate input data."""
        return isinstance(data, dict) and len(data) > 0
    
    @classmethod
    def create_default(cls) -> 'TestProcessor':
        """Create a default processor instance."""
        return cls("default")
    
    async def process_async(self, items: List[str]) -> List[str]:
        """Process items asynchronously."""
        processed = []
        for item in items:
            await asyncio.sleep(0.01)
            processed.append(self._transform_item(item))
        return processed
    
    def _transform_item(self, item: str) -> str:
        """Private method to transform items."""
        return f"processed_{item}"
    
    def __str__(self) -> str:
        """String representation."""
        return f"TestProcessor(name={self.name})"

def helper_function(x: int, y: int) -> float:
    """A simple helper function with type hints."""
    return (x ** 2 + y ** 2) ** 0.5

async def main():
    """Main async function."""
    processor = TestProcessor.create_default()
    items = ['item1', 'item2', 'item3']
    results = await processor.process_async(items)
    print(f"Results: {results}")

if __name__ == "__main__":
    asyncio.run(main())
''',
        "filename": "test_processor.py",
        "language": "Python"
    }

@pytest.fixture
def simple_python_code():
    """Provide simple Python code for basic testing."""
    return {
        "code": '''
def simple_function(x):
    return x * 2

class SimpleClass:
    def method(self):
        pass
''',
        "filename": "simple.py",
        "language": "Python"
    }

@pytest.fixture
def default_extractor(dummy_metadata):
    """Create and initialize the default CocoIndex extractor."""
    metadata = dummy_metadata.copy()
    metadata["use_default_handler"] = True
    extractor = CocoIndexMetadataExtractor(metadata)
    assert extractor.initialize(), "Default extractor should initialize successfully"
    yield extractor
    extractor.cleanup()

@pytest.fixture
def custom_extractor(dummy_metadata):
    """Create and initialize the custom CocoIndex extractor."""
    metadata = dummy_metadata.copy()
    metadata["use_default_handler"] = False
    extractor = CocoIndexMetadataExtractor(metadata)
    assert extractor.initialize(), "Custom extractor should initialize successfully"
    yield extractor
    extractor.cleanup()

class TestCocoIndexBaselineComparison:
    """Test suite comparing custom and default CocoIndex extractors."""
    
    def test_extractor_initialization(self, default_extractor, custom_extractor):
        """Test that both extractors initialize properly."""
        assert default_extractor.provider_name == "CocoIndex"
        assert custom_extractor.provider_name == "CocoIndex"
        assert default_extractor.use_default_handler is True
        assert custom_extractor.use_default_handler is False
    
    def test_basic_metadata_extraction(self, default_extractor, custom_extractor, simple_python_code):
        """Test basic metadata extraction capabilities."""
        default_result = default_extractor.extract(simple_python_code)
        custom_result = custom_extractor.extract(simple_python_code)
        
        # Both should succeed
        assert "error" not in default_result, f"Default extractor failed: {default_result.get('error')}"
        assert "error" not in custom_result, f"Custom extractor failed: {custom_result.get('error')}"
        
        # Both should have metadata
        assert "metadata" in default_result
        assert "metadata" in custom_result
        
        default_metadata = default_result["metadata"]
        custom_metadata = custom_result["metadata"]
        
        # Both should detect basic elements
        assert "functions" in default_metadata
        assert "functions" in custom_metadata
        assert "classes" in default_metadata
        assert "classes" in custom_metadata
        
        # Both should find the simple function and class
        assert "simple_function" in str(default_metadata["functions"])
        assert "simple_function" in str(custom_metadata["functions"])
        assert "SimpleClass" in str(default_metadata["classes"])
        assert "SimpleClass" in str(custom_metadata["classes"])
    
    def test_comprehensive_extraction_comparison(self, default_extractor, custom_extractor, sample_python_code):
        """Test comprehensive metadata extraction and compare results."""
        default_result = default_extractor.extract(sample_python_code)
        custom_result = custom_extractor.extract(sample_python_code)
        
        # Both should succeed
        assert "error" not in default_result, f"Default extractor failed: {default_result.get('error')}"
        assert "error" not in custom_result, f"Custom extractor failed: {custom_result.get('error')}"
        
        default_metadata = default_result["metadata"]
        custom_metadata = custom_result["metadata"]
        
        # Track analysis methods
        default_method = default_metadata.get("analysis_method", "unknown")
        custom_method = custom_metadata.get("analysis_method", "unknown")
        
        print(f"Default analysis method: {default_method}")
        print(f"Custom analysis method: {custom_method}")
        
        # Both should detect functions
        default_functions = default_metadata.get("functions", [])
        custom_functions = custom_metadata.get("functions", [])
        
        assert len(default_functions) > 0, "Default extractor should find functions"
        assert len(custom_functions) > 0, "Custom extractor should find functions"
        
        # Both should detect classes
        default_classes = default_metadata.get("classes", [])
        custom_classes = custom_metadata.get("classes", [])
        
        assert len(default_classes) > 0, "Default extractor should find classes"
        assert len(custom_classes) > 0, "Custom extractor should find classes"
        
        # Check for expected elements
        expected_functions = ["helper_function", "main", "__init__", "status"]
        expected_classes = ["TestProcessor"]
        
        # At least some expected functions should be found by both
        default_found_functions = sum(1 for func in expected_functions if func in str(default_functions))
        custom_found_functions = sum(1 for func in expected_functions if func in str(custom_functions))
        
        assert default_found_functions >= 2, f"Default should find at least 2 expected functions, found {default_found_functions}"
        assert custom_found_functions >= 2, f"Custom should find at least 2 expected functions, found {custom_found_functions}"
        
        # Both should find the main class
        for cls in expected_classes:
            assert cls in str(default_classes), f"Default extractor should find class: {cls}"
            assert cls in str(custom_classes), f"Custom extractor should find class: {cls}"
    
    def test_async_detection_comparison(self, default_extractor, custom_extractor, sample_python_code):
        """Test async code detection capabilities."""
        default_result = default_extractor.extract(sample_python_code)
        custom_result = custom_extractor.extract(sample_python_code)
        
        default_metadata = default_result["metadata"]
        custom_metadata = custom_result["metadata"]
        
        # Both should detect async code
        default_has_async = default_metadata.get("has_async", False)
        custom_has_async = custom_metadata.get("has_async", False)
        
        assert default_has_async, "Default extractor should detect async code"
        assert custom_has_async, "Custom extractor should detect async code"
    
    def test_decorator_detection_comparison(self, default_extractor, custom_extractor, sample_python_code):
        """Test decorator detection capabilities."""
        default_result = default_extractor.extract(sample_python_code)
        custom_result = custom_extractor.extract(sample_python_code)
        
        default_metadata = default_result["metadata"]
        custom_metadata = custom_result["metadata"]
        
        # Get decorator information (field name may vary)
        default_decorators = (default_metadata.get("decorators_used", []) or 
                            default_metadata.get("decorators", []))
        custom_decorators = (custom_metadata.get("decorators_used", []) or 
                           custom_metadata.get("decorators", []))
        
        # Both should detect some decorators
        # (Be more flexible as the field structure may vary)
        default_has_decorators = (len(default_decorators) > 0 or 
                                default_metadata.get("has_decorators", False))
        custom_has_decorators = (len(custom_decorators) > 0 or 
                               custom_metadata.get("has_decorators", False))
        
        assert default_has_decorators, f"Default extractor should detect decorators. Decorators: {default_decorators}, has_decorators: {default_metadata.get('has_decorators', False)}"
        assert custom_has_decorators, f"Custom extractor should detect decorators. Decorators: {custom_decorators}, has_decorators: {custom_metadata.get('has_decorators', False)}"
        
        # Check for expected decorators
        expected_decorators = ["dataclass", "property", "staticmethod", "classmethod"]
        
        default_decorator_str = str(default_decorators)
        custom_decorator_str = str(custom_decorators)
        
        # At least some decorators should be found
        default_found = sum(1 for dec in expected_decorators if dec in default_decorator_str)
        custom_found = sum(1 for dec in expected_decorators if dec in custom_decorator_str)
        
        assert default_found >= 1, f"Default should find at least 1 decorator, found {default_found}"
        assert custom_found >= 1, f"Custom should find at least 1 decorator, found {custom_found}"
    
    def test_metadata_richness_comparison(self, default_extractor, custom_extractor, sample_python_code):
        """Compare the richness of metadata extraction."""
        default_result = default_extractor.extract(sample_python_code)
        custom_result = custom_extractor.extract(sample_python_code)
        
        default_metadata = default_result["metadata"]
        custom_metadata = custom_result["metadata"]
        
        # Count meaningful fields in each result
        def count_meaningful_fields(metadata: Dict[str, Any]) -> int:
            count = 0
            for key, value in metadata.items():
                if key in ["analysis_method"]:
                    continue  # Skip metadata fields
                if isinstance(value, list) and len(value) > 0:
                    count += 1
                elif isinstance(value, dict) and len(value) > 0:
                    count += 1
                elif isinstance(value, bool) and value is True:
                    count += 1
                elif isinstance(value, (int, float)) and value > 0:
                    count += 1
            return count
        
        default_field_count = count_meaningful_fields(default_metadata)
        custom_field_count = count_meaningful_fields(custom_metadata)
        
        print(f"Default extractor meaningful fields: {default_field_count}")
        print(f"Custom extractor meaningful fields: {custom_field_count}")
        
        # Both should provide meaningful metadata
        assert default_field_count >= 3, f"Default should provide at least 3 meaningful fields, got {default_field_count}"
        assert custom_field_count >= 3, f"Custom should provide at least 3 meaningful fields, got {custom_field_count}"
        
        # Custom extractor should generally provide as much or more information
        # (but allow some flexibility for different analysis approaches)
        if custom_field_count < default_field_count:
            print(f"Warning: Custom extractor provides fewer fields ({custom_field_count}) than default ({default_field_count})")
    
    def test_error_handling_comparison(self, default_extractor, custom_extractor):
        """Test error handling with malformed code."""
        malformed_code = {
            "code": '''
def broken_function(
    missing_closing_paren

class IncompleteClass:
    def method_without_body(self):
            ''',
            "filename": "broken.py",
            "language": "Python"
        }
        
        default_result = default_extractor.extract(malformed_code)
        custom_result = custom_extractor.extract(malformed_code)
        
        # Both should handle errors gracefully and return structured results
        assert isinstance(default_result, dict), "Default extractor should return dict even for malformed code"
        assert isinstance(custom_result, dict), "Custom extractor should return dict even for malformed code"
        
        # Both should have provider information even on error
        assert default_result.get("provider") == "CocoIndex"
        assert custom_result.get("provider") == "CocoIndex"
    
    def test_performance_characteristics(self, default_extractor, custom_extractor, sample_python_code):
        """Compare basic performance characteristics."""
        import time
        
        # Create larger code sample by repeating
        large_code = sample_python_code.copy()
        large_code["code"] = sample_python_code["code"] * 3
        
        # Time default extractor
        start_time = time.time()
        default_result = default_extractor.extract(large_code)
        default_time = time.time() - start_time
        
        # Time custom extractor
        start_time = time.time()
        custom_result = custom_extractor.extract(large_code)
        custom_time = time.time() - start_time
        
        print(f"Default extractor time: {default_time:.3f}s")
        print(f"Custom extractor time: {custom_time:.3f}s")
        
        # Both should complete in reasonable time
        assert default_time < 5.0, f"Default extractor took too long: {default_time}s"
        assert custom_time < 5.0, f"Custom extractor took too long: {custom_time}s"
        
        # Both should produce valid results
        assert "metadata" in default_result
        assert "metadata" in custom_result
    
    def test_edge_case_empty_code(self, default_extractor, custom_extractor):
        """Test handling of edge case: empty code."""
        empty_code = {
            "code": "",
            "filename": "empty.py",
            "language": "Python"
        }
        
        default_result = default_extractor.extract(empty_code)
        custom_result = custom_extractor.extract(empty_code)
        
        # Both should handle empty code gracefully
        assert isinstance(default_result, dict)
        assert isinstance(custom_result, dict)
        assert "metadata" in default_result
        assert "metadata" in custom_result
        
        # Should have empty or minimal metadata
        default_metadata = default_result["metadata"]
        custom_metadata = custom_result["metadata"]
        
        assert default_metadata.get("functions", []) == []
        assert default_metadata.get("classes", []) == []
        assert custom_metadata.get("functions", []) == []
        assert custom_metadata.get("classes", []) == []

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
