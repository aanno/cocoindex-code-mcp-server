#!/usr/bin/env python3
"""Test C++ visitor inheritance from C visitor."""

import pytest
from cocoindex_code_mcp_server.language_handlers.cpp_visitor import analyze_cpp_code


class TestCppInheritance:
    """Test C++ visitor inheritance functionality."""

    def test_cpp_with_c_features(self):
        """Test that C++ visitor can handle both C and C++ features."""
        # C++ code with both C and C++ features
        cpp_code = '''#include <iostream>

// C-style function
int add(int a, int b) {
    return a + b;
}

// C++ class
class Calculator {
public:
    int multiply(int a, int b) {
        return a * b;
    }
};

// C-style struct
struct Point {
    int x, y;
};

// C++ namespace
namespace Math {
    int subtract(int a, int b) {
        return a - b;
    }
}

int main() {
    Calculator calc;
    return 0;
}'''
        
        result = analyze_cpp_code(cpp_code, 'cpp', 'test.cpp')
        
        # Basic success check
        assert result.get('success', False), f"Analysis failed: {result}"
        
        # Check that we found functions
        functions_found = result.get('functions', [])
        expected_functions = ['add', 'multiply', 'subtract', 'main']
        
        assert len(functions_found) > 0, "No functions found"
        
        # Check for at least some expected functions
        found_set = set(functions_found)
        expected_set = set(expected_functions)
        overlap = found_set & expected_set
        
        assert len(overlap) >= 2, f"Expected at least 2 functions from {expected_functions}, got {functions_found}"
        
        # Check for classes if supported
        classes_found = result.get('classes', [])
        if classes_found:
            assert 'Calculator' in classes_found, f"Expected Calculator class, got {classes_found}"
        
        # Check analysis method is reported
        assert 'analysis_method' in result, "Analysis method not reported"

    def test_cpp_class_detection(self):
        """Test that C++ classes are properly detected."""
        cpp_code = '''class MyClass {
public:
    void method1();
    int method2(int x);
};'''
        
        result = analyze_cpp_code(cpp_code, 'cpp', 'class_test.cpp')
        
        assert result.get('success', False), "Analysis should succeed"
        
        # Check for class detection if supported
        classes = result.get('classes', [])
        functions = result.get('functions', [])
        
        # Either classes should be detected or methods should be detected as functions
        assert len(classes) > 0 or len(functions) > 0, "Should detect either classes or methods"

    def test_cpp_namespace_detection(self):
        """Test that C++ namespaces are handled."""
        cpp_code = '''namespace TestNamespace {
    int func1() { return 1; }
    class TestClass {
    public:
        void method() {}
    };
}'''
        
        result = analyze_cpp_code(cpp_code, 'cpp', 'namespace_test.cpp')
        
        assert result.get('success', False), "Analysis should succeed"
        
        # Should find at least the function
        functions = result.get('functions', [])
        assert len(functions) > 0, "Should find functions in namespace"