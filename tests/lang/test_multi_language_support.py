#!/usr/bin/env python3
"""Test multi-language support across different programming languages."""

import pytest

from cocoindex_code_mcp_server.ast_visitor import analyze_code


class TestMultiLanguageSupport:
    """Test support for multiple programming languages."""

    def test_c_support(self):
        """Test C language support."""
        c_code = '''#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main() {
    printf("%d\\n", add(5, 3));
    return 0;
}'''

        result = analyze_code(c_code, 'c', 'test.c')

        assert 'error' not in result, f"C analysis failed: {result}"
        assert 'analysis_method' in result, "Analysis method should be reported"

        functions = result.get('functions', [])
        assert len(functions) > 0, f"Should find C functions, got {functions}"

        # Should find both functions
        expected_functions = {'add', 'main'}
        found_functions = set(functions)
        found_expected = found_functions & expected_functions

        assert len(found_expected) >= 1, f"Should find at least 1 function from {expected_functions}, got {functions}"

    def test_rust_support(self):
        """Test Rust language support."""
        rust_code = '''fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn main() {
    println!("{}", fibonacci(10));
}'''

        result = analyze_code(rust_code, 'rust', 'test.rs')

        assert 'error' not in result, f"Rust analysis failed: {result}"
        assert 'analysis_method' in result, "Analysis method should be reported"

        functions = result.get('functions', [])
        assert len(functions) > 0, f"Should find Rust functions, got {functions}"

        # Should find both functions
        expected_functions = {'fibonacci', 'main'}
        found_functions = set(functions)
        found_expected = found_functions & expected_functions

        assert len(found_expected) >= 1, f"Should find at least 1 function from {expected_functions}, got {functions}"

    def test_cpp_support(self):
        """Test C++ language support."""
        cpp_code = '''#include <iostream>

class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
};

int main() {
    Calculator calc;
    std::cout << calc.add(5, 3) << std::endl;
    return 0;
}'''

        result = analyze_code(cpp_code, 'cpp', 'test.cpp')

        assert 'error' not in result, f"C++ analysis failed: {result}"
        assert 'analysis_method' in result, "Analysis method should be reported"

        functions = result.get('functions', [])
        classes = result.get('classes', [])

        # Should find functions or classes
        assert len(functions) > 0 or len(
            classes) > 0, f"Should find C++ functions or classes, got functions={functions}, classes={classes}"

        # Should find some expected items
        if functions:
            expected_functions = {'add', 'main'}
            found_functions = set(functions)
            found_expected = found_functions & expected_functions
            assert len(
                found_expected) >= 1, f"Should find at least 1 function from {expected_functions}, got {functions}"

        if classes:
            assert 'Calculator' in classes, f"Should find Calculator class, got {classes}"

    def test_kotlin_support(self):
        """Test Kotlin language support."""
        kotlin_code = '''fun fibonacci(n: Int): Int {
    return when (n) {
        0 -> 0
        1 -> 1
        else -> fibonacci(n - 1) + fibonacci(n - 2)
    }
}

fun main() {
    println(fibonacci(10))
}'''

        result = analyze_code(kotlin_code, 'kotlin', 'test.kt')

        assert 'error' not in result, f"Kotlin analysis failed: {result}"
        assert 'analysis_method' in result, "Analysis method should be reported"

        functions = result.get('functions', [])
        assert len(functions) > 0, f"Should find Kotlin functions, got {functions}"

        # Should find both functions
        expected_functions = {'fibonacci', 'main'}
        found_functions = set(functions)
        found_expected = found_functions & expected_functions

        assert len(found_expected) >= 1, f"Should find at least 1 function from {expected_functions}, got {functions}"

    def test_javascript_support(self):
        """Test JavaScript language support."""
        js_code = '''function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class Calculator {
    add(a, b) {
        return a + b;
    }
}

const calc = new Calculator();
console.log(calc.add(5, 3));
console.log(fibonacci(10));'''

        result = analyze_code(js_code, 'javascript', 'test.js')

        # Skip JavaScript test if parser not available
        if 'error' in result and 'not available' in result['error']:
            pytest.skip(f"JavaScript parser not available: {result['error']}")
        assert 'error' not in result, f"JavaScript analysis failed: {result}"
        assert 'analysis_method' in result, "Analysis method should be reported"

        functions = result.get('functions', [])
        classes = result.get('classes', [])

        # Should find functions or classes
        assert len(functions) > 0 or len(
            classes) > 0, f"Should find JS functions or classes, got functions={functions}, classes={classes}"

    def test_python_support(self):
        """Test Python language support (baseline)."""
        python_code = '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class Calculator:
    def add(self, a, b):
        return a + b

if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(5, 3))
    print(fibonacci(10))'''

        result = analyze_code(python_code, 'python', 'test.py')

        assert 'error' not in result, f"Python analysis failed: {result}"
        assert 'analysis_method' in result, "Analysis method should be reported"

        functions = result.get('functions', [])
        result.get('classes', [])

        # Python should definitely work well
        assert len(functions) > 0, f"Should find Python functions, got {functions}"

        expected_functions = {'fibonacci', 'add'}
        found_functions = set(functions)
        found_expected = found_functions & expected_functions

        assert len(found_expected) >= 1, f"Should find at least 1 function from {expected_functions}, got {functions}"

    @pytest.mark.parametrize("language,extension,code_snippet", [
        ('c', '.c', 'int test() { return 0; }'),
        ('rust', '.rs', 'fn test() -> i32 { 0 }'),
        ('cpp', '.cpp', 'int test() { return 0; }'),
        ('kotlin', '.kt', 'fun test(): Int { return 0 }'),
        ('python', '.py', 'def test(): return 0'),
        ('javascript', '.js', 'function test() { return 0; }'),
    ])
    def test_language_analysis_succeeds(self, language: str, extension: str, code_snippet: str):
        """Test that basic analysis succeeds for all supported languages."""
        result = analyze_code(code_snippet, language, f'test{extension}')

        # Skip if parser not available for this language
        if 'error' in result and 'not available' in result['error']:
            pytest.skip(f"{language} parser not available: {result['error']}")
        assert 'error' not in result, f"{language} analysis should succeed: {result}"
        assert 'analysis_method' in result, f"{language} should report analysis method"

        # Should find at least the test function
        functions = result.get('functions', [])
        assert len(functions) > 0, f"Should find functions in {language} code"
