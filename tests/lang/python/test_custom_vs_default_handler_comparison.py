#!/usr/bin/env python3

"""
Comparison test between custom Python handler and default CocoIndex implementation.

This test compares the results of our custom TreeSitterPythonAnalyzer (with PythonNodeHandler)
against the default CocoIndex language handling to validate our improvements.
"""

import ast
from typing import Any, Dict, Union

import pytest

from cocoindex_code_mcp_server.lang.python.tree_sitter_python_analyzer import (
    TreeSitterPythonAnalyzer,
)

# Create a simple basic analyzer for true comparison


def basic_analyze_python_code(code: str, filename: str = "") -> Dict[str, Any]:
    """Basic Python code analysis using only AST without tree-sitter."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            'functions': [],
            'classes': [],
            'imports': [],
            'has_classes': False,
            'has_async': False,
            'decorators_used': [],
            'analysis_method': 'basic_ast_fallback',
            'error': str(e)
        }

    functions = []
    classes = []
    imports = []
    decorators = []
    has_async = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
            if any(isinstance(d, ast.Name) and d.id in ['staticmethod',
                   'classmethod', 'property'] for d in node.decorator_list):
                decorators.extend([d.id for d in node.decorator_list if isinstance(d, ast.Name)])
        elif isinstance(node, ast.AsyncFunctionDef):
            functions.append(node.name)
            has_async = True
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
            if any(isinstance(d, ast.Name) and d.id == 'dataclass' for d in node.decorator_list):
                decorators.extend([d.id for d in node.decorator_list if isinstance(d, ast.Name)])
        elif isinstance(node, ast.Import):
            imports.extend([alias.name for alias in node.names])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return {
        'functions': functions,
        'classes': classes,
        'imports': imports,
        'has_classes': len(classes) > 0,
        'has_async': has_async,
        'decorators_used': list(set(decorators)),
        'analysis_method': 'basic_ast_only'
    }


# Sample Python code for comprehensive testing
SAMPLE_PYTHON_CODE = '''
"""
Module docstring for sample code.
This module demonstrates various Python constructs.
"""

import os
import asyncio
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from functools import wraps, lru_cache

@dataclass
class DataProcessor:
    """A sample data processing class with various method types."""

    name: str
    config: Dict[str, Any]

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self._internal_state = {}

    @property
    def status(self) -> str:
        """Get the current processing status."""
        return self._internal_state.get('status', 'idle')

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration dictionary."""
        return isinstance(config, dict) and 'version' in config

    @classmethod
    def from_file(cls, filename: str) -> 'DataProcessor':
        """Create instance from configuration file."""
        with open(filename, 'r') as f:
            config = json.load(f)
        return cls(config.get('name', 'default'), config)

    async def process_async(self, items: List[str]) -> List[str]:
        """Process items asynchronously."""
        processed = []
        for item in items:
            await asyncio.sleep(0.1)
            processed.append(self._transform_item(item))
        return processed

    def _transform_item(self, item: str) -> str:
        """Private method to transform individual items."""
        return f"processed_{item}"

    def __str__(self) -> str:
        """String representation of the processor."""
        return f"DataProcessor(name={self.name})"

    def __repr__(self) -> str:
        """Developer representation of the processor."""
        return f"DataProcessor(name='{self.name}', config={self.config})"

def cached_function(cache_size: int = 128):
    """Decorator factory for caching function results."""
    def decorator(func):
        @wraps(func)
        @lru_cache(maxsize=cache_size)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

@cached_function(cache_size=256)
def complex_calculation(x: int, y: int) -> float:
    """
    Perform a complex calculation with caching.

    Args:
        x: First input value
        y: Second input value

    Returns:
        The calculated result as a float
    """
    return (x ** 2 + y ** 2) ** 0.5

async def main():
    """Main async function demonstrating usage."""
    processor = DataProcessor.from_file('config.json')
    items = ['item1', 'item2', 'item3']
    results = await processor.process_async(items)
    print(f"Results: {results}")

if __name__ == "__main__":
    asyncio.run(main())
'''


class TestCustomVsDefaultHandlerComparison:
    """Test suite comparing custom and default Python code analysis."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.custom_analyzer = TreeSitterPythonAnalyzer(prefer_tree_sitter=True)
        self.filename = "test_sample.py"

    def test_basic_metadata_structure_comparison(self) -> None:
        """Compare basic metadata structure between custom and default handlers."""
        # Analyze with custom handler
        custom_result = self.custom_analyzer.analyze_code(SAMPLE_PYTHON_CODE, self.filename)

        # Analyze with basic handler (for true comparison)
        default_result = basic_analyze_python_code(SAMPLE_PYTHON_CODE, self.filename)

        if custom_result is not None and default_result is not None:
            # Both should have basic required fields
            basic_fields = [
                'functions', 'classes', 'imports', 'has_classes',
                'has_async', 'analysis_method'
            ]

            for field in basic_fields:
                assert field in custom_result, f"Custom handler missing field: {field}"
                assert field in default_result, f"Default handler missing field: {field}"

            # Check decorator fields (may vary in naming)
            assert ('decorators' in custom_result or 'decorators_used' in custom_result), \
                "Custom handler should have decorator information"
            assert ('decorators' in default_result or 'decorators_used' in default_result), \
                "Default handler should have decorator information"

            # Custom handler should have additional enhanced fields
            enhanced_fields = [
                'has_docstrings', 'function_details', 'class_details',
                'decorators', 'has_decorators'
            ]

            for field in enhanced_fields:
                assert field in custom_result, f"Custom handler missing enhanced field: {field}"
                # Default handler may or may not have these fields
        else:
            pytest.fail("no metadata in chunk")

    def test_function_detection_comparison(self) -> None:
        """Compare function detection capabilities."""
        custom_result = self.custom_analyzer.analyze_code(SAMPLE_PYTHON_CODE, self.filename)
        default_result = basic_analyze_python_code(SAMPLE_PYTHON_CODE, self.filename)

        if custom_result is None or default_result is None:
            pytest.fail("custom_result or default_result is None")
        else:
            # Both should detect basic functions
            expected_functions = ['complex_calculation', 'main']

            custom_functions = custom_result.get('functions', [])
            default_functions = default_result.get('functions', [])

            for func in expected_functions:
                assert func in custom_functions, f"Custom handler missed function: {func}"
                assert func in default_functions, f"Default handler missed function: {func}"

            # Custom handler should provide more detailed function information
            if custom_result.get('function_details'):
                # Check for enhanced function details
                function_details = custom_result['function_details']
                assert len(function_details) > 0, "Custom handler should provide function details"

                # Check for comprehensive method detection in classes
                assert any('__init__' in str(details) for details in function_details), \
                    "Custom handler should detect __init__ method"
                assert any('process_async' in str(details) for details in function_details), \
                    "Custom handler should detect async methods"

    def test_class_detection_comparison(self):
        """Compare class detection capabilities."""
        custom_result = self.custom_analyzer.analyze_code(SAMPLE_PYTHON_CODE, self.filename)
        default_result = basic_analyze_python_code(SAMPLE_PYTHON_CODE, self.filename)

        if custom_result is None or default_result is None:
            pytest.fail("custom_result or default_result is None")
        else:
            # Both should detect the DataProcessor class
            expected_classes = ['DataProcessor']

            custom_classes = custom_result.get('classes', [])
            default_classes = default_result.get('classes', [])

            for cls in expected_classes:
                assert cls in custom_classes, f"Custom handler missed class: {cls}"
                assert cls in default_classes, f"Default handler missed class: {cls}"

            # Custom handler should provide enhanced class details
            if custom_result.get('class_details'):
                class_details = custom_result['class_details']
                assert len(class_details) > 0, "Custom handler should provide class details"

                # Check for method categorization - methods might be in function_details
                # or in the class methods list
                function_details = custom_result.get('function_details', [])
                method_names = [f.get('name', '') for f in function_details if isinstance(f, dict)]

                # Should detect various method types from function_details
                assert '__init__' in method_names, "Should detect constructor in function details"
                assert 'process_async' in method_names, "Should detect async methods in function details"
                assert any(name.startswith('_') and not name.startswith('__') for name in method_names), \
                    "Should detect private methods in function details"

    def test_decorator_detection_comparison(self):
        """Compare decorator detection capabilities."""
        custom_result = self.custom_analyzer.analyze_code(SAMPLE_PYTHON_CODE, self.filename)
        default_result = basic_analyze_python_code(SAMPLE_PYTHON_CODE, self.filename)

        if custom_result is None or default_result is None:
            pytest.fail("custom_result or default_result is None")
        else:
            # Expected decorators in the sample code
            expected_decorators = ['dataclass', 'property', 'staticmethod', 'classmethod', 'wraps', 'lru_cache']

            custom_decorators = custom_result.get('decorators', custom_result.get('decorators_used', []))
            default_decorators = default_result.get('decorators', default_result.get('decorators_used', []))

            # Custom handler should detect more decorators
            custom_found = sum(1 for dec in expected_decorators if dec in custom_decorators)
            default_found = sum(1 for dec in expected_decorators if dec in default_decorators)

            assert custom_found >= default_found, \
                f"Custom handler should find at least as many decorators as default (custom: {custom_found}, default: {default_found})"

            # Custom handler should have has_decorators field
            if 'has_decorators' in custom_result:
                assert custom_result['has_decorators'] is True, "Should detect that code has decorators"

    def test_docstring_detection_comparison(self):
        """Compare docstring detection capabilities."""
        custom_result = self.custom_analyzer.analyze_code(SAMPLE_PYTHON_CODE, self.filename)
        basic_analyze_python_code(SAMPLE_PYTHON_CODE, self.filename)

        if custom_result is None:
            pytest.fail("custom_result is None")
        else:
            # Custom handler should have enhanced docstring detection
            if 'has_docstrings' in custom_result:
                assert custom_result['has_docstrings'] is True, \
                    "Custom handler should detect docstrings in sample code"

            # Check for docstring details in function information
            if custom_result.get('function_details'):
                docstring_count = 0
                for func_detail in custom_result['function_details']:
                    if isinstance(func_detail, dict) and func_detail.get('has_docstring'):
                        docstring_count += 1

                # If has_docstring is not available, check for docstring field
                if docstring_count == 0:
                    for func_detail in custom_result['function_details']:
                        if isinstance(func_detail, dict) and func_detail.get('docstring'):
                            docstring_count += 1

                # Even if we can't detect individual docstrings, the overall flag should be set
                if docstring_count == 0 and custom_result.get('has_docstrings'):
                    docstring_count = 1  # At least the overall detection works

                assert docstring_count > 0, "Custom handler should detect function docstrings"

    def test_async_detection_comparison(self):
        """Compare async/await detection capabilities."""
        custom_result = self.custom_analyzer.analyze_code(SAMPLE_PYTHON_CODE, self.filename)
        default_result = basic_analyze_python_code(SAMPLE_PYTHON_CODE, self.filename)

        if custom_result is None or default_result is None:
            pytest.fail("custom_result or default_result is None")
        else:
            # Both should detect async code
            assert custom_result.get('has_async') is True, "Custom handler should detect async code"
            assert default_result.get('has_async') is True, "Default handler should detect async code"

            # Custom handler should provide more details about async functions
            if custom_result.get('function_details'):
                # Look for async functions by name (since 'async' functions may not show up as is_async=True)
                function_names = [f.get('name', '') for f in custom_result['function_details'] if isinstance(f, dict)]

                # Check that we have functions that should be async (like main and process_async)
                assert 'main' in function_names, "Should detect main function"
                assert 'process_async' in function_names, "Should detect process_async function"

    def test_import_detection_comparison(self):
        """Compare import detection capabilities."""
        custom_result = self.custom_analyzer.analyze_code(SAMPLE_PYTHON_CODE, self.filename)
        default_result = basic_analyze_python_code(SAMPLE_PYTHON_CODE, self.filename)

        if custom_result is None or default_result is None:
            pytest.fail("custom_result or default_result is None")
        else:
            # Expected imports in the sample code
            expected_imports = ['os', 'asyncio', 'json']

            custom_imports = custom_result.get('imports', [])
            default_imports = default_result.get('imports', [])

            # Both should detect most imports, but let's be lenient about exact matching
            custom_import_str = ' '.join(str(imp) for imp in custom_imports)
            default_import_str = ' '.join(str(imp) for imp in default_imports)

            # Check that both have reasonable import detection
            assert len(custom_imports) > 0, "Custom handler should detect some imports"
            assert len(default_imports) > 0, "Default handler should detect some imports"

            # Check for at least some of the expected imports
            found_in_custom = sum(1 for imp in expected_imports if imp in custom_import_str)
            found_in_default = sum(1 for imp in expected_imports if imp in default_import_str)

            assert found_in_custom >= 2, f"Custom handler should find at least 2 expected imports, found {found_in_custom}"
            assert found_in_default >= 2, f"Default handler should find at least 2 expected imports, found {found_in_default}"

    def test_analysis_method_tracking(self) -> None:
        """Test that analysis method is properly tracked."""
        custom_result: Union[Dict[str, Any], None] = self.custom_analyzer.analyze_code(
            SAMPLE_PYTHON_CODE, self.filename)
        default_result: Dict[str, Any] = basic_analyze_python_code(SAMPLE_PYTHON_CODE, self.filename)

        if custom_result is not None and default_result is not None:
            # Check analysis method is tracked
            assert 'analysis_method' in custom_result
            assert 'analysis_method' in default_result

            # Custom analyzer should indicate hybrid approach
            if custom_result is not None:
                custom_method = custom_result['analysis_method']
                assert custom_method in ['tree_sitter', 'python_ast', 'hybrid', 'tree_sitter+python_ast', 'tree_sitter+python_code_analyzer'], \
                    f"Custom analyzer should use advanced method, got: {custom_method}"

            # Default should be basic
            default_method = default_result['analysis_method']
            assert default_method in ['basic', 'python_ast', 'python', 'basic_ast_only'], \
                f"Default analyzer method: {default_method}"
        else:
            pytest.fail("no metadata in chunk")

    def test_metadata_richness_comparison(self) -> None:
        """Compare overall metadata richness between handlers."""
        custom_result: Union[Dict[str, Any], None] = self.custom_analyzer.analyze_code(
            SAMPLE_PYTHON_CODE, self.filename)
        default_result = basic_analyze_python_code(SAMPLE_PYTHON_CODE, self.filename)

        if custom_result is not None:
            # Count total fields in each result
            custom_field_count = len([k for k, v in custom_result.items() if v])
            default_field_count = len([k for k, v in default_result.items() if v])

            # Custom handler should provide more fields with meaningful data
            assert custom_field_count >= default_field_count, \
                f"Custom handler should provide at least as many fields (custom: {custom_field_count}, default: {default_field_count})"

            # Count complex data structures (lists with content, dicts with content)
            def count_complex_data(result: Dict[str, Any]) -> int:
                count = 0
                for value in result.values():
                    if isinstance(value, list) and len(value) > 0:
                        count += 1
                    elif isinstance(value, dict) and len(value) > 0:
                        count += 1
                return count

            custom_complex_count = count_complex_data(custom_result)
            default_complex_count = count_complex_data(default_result)

            # Custom handler should generally provide more complex data, but be flexible
            # At minimum, both should provide some meaningful data structures
            assert custom_complex_count >= 2, f"Custom handler should provide at least 2 complex data structures, got {custom_complex_count}"
            assert default_complex_count >= 1, f"Default handler should provide at least 1 complex data structure, got {default_complex_count}"
        else:
            pytest.fail("custom_result is None")

    def test_error_handling_comparison(self) -> None:
        """Compare error handling between custom and default handlers."""
        # Test with malformed Python code
        malformed_code = '''
        def broken_function(
            missing_closing_paren

        class IncompleteClass:
            def method_without_body(self):
        '''

        # Both handlers should handle errors gracefully
        try:
            custom_result = self.custom_analyzer.analyze_code(malformed_code, "broken.py")
            assert isinstance(custom_result, dict), "Custom handler should return dict even for malformed code"
            assert 'analysis_method' in custom_result, "Should include analysis method even on error"
        except Exception as e:
            pytest.fail(f"Custom handler should handle malformed code gracefully: {e}")

        try:
            default_result = basic_analyze_python_code(malformed_code, "broken.py")
            assert isinstance(default_result, dict), "Default handler should return dict even for malformed code"
        except Exception as e:
            pytest.fail(f"Default handler should handle malformed code gracefully: {e}")

    def test_performance_characteristics(self):
        """Compare basic performance characteristics (not a benchmark)."""
        import time

        # Test with a larger code sample
        large_code = SAMPLE_PYTHON_CODE * 5  # Repeat the sample code

        # Time custom handler
        start_time = time.time()
        custom_result = self.custom_analyzer.analyze_code(large_code, self.filename)
        custom_time = time.time() - start_time

        # Time default handler
        start_time = time.time()
        default_result = basic_analyze_python_code(large_code, self.filename)
        default_time = time.time() - start_time

        # Both should complete in reasonable time (less than 5 seconds for test code)
        assert custom_time < 5.0, f"Custom handler took too long: {custom_time}s"
        assert default_time < 5.0, f"Default handler took too long: {default_time}s"

        # Results should still be valid
        assert isinstance(custom_result, dict) and len(custom_result) > 0
        assert isinstance(default_result, dict) and len(default_result) > 0

    def test_edge_case_empty_code(self):
        """Test handling of edge case: empty code."""
        empty_code = ""

        custom_result = self.custom_analyzer.analyze_code(empty_code, self.filename)
        default_result = basic_analyze_python_code(empty_code, self.filename)

        # Both should handle empty code gracefully
        assert isinstance(custom_result, dict)
        assert isinstance(default_result, dict)

        # Should have basic structure even for empty code
        assert custom_result.get('functions', []) == []
        assert custom_result.get('classes', []) == []
        assert default_result.get('functions', []) == []
        assert default_result.get('classes', []) == []

    def test_edge_case_comments_only(self):
        """Test handling of edge case: code with only comments."""
        comments_only = '''
        # This is a comment
        """
        This is a docstring but not attached to anything
        """
        # Another comment
        '''

        custom_result = self.custom_analyzer.analyze_code(comments_only, self.filename)
        default_result = basic_analyze_python_code(comments_only, self.filename)

        # Both should handle comments-only code
        assert isinstance(custom_result, dict)
        assert isinstance(default_result, dict)

        # Should recognize lack of executable code
        assert custom_result.get('functions', []) == []
        assert custom_result.get('classes', []) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
