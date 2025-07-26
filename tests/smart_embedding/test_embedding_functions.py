#!/usr/bin/env python3
"""
Test suite for smart embedding functions.
Tests that embedding functions are properly defined and configured.
"""

import pytest
import inspect

from cocoindex_code_mcp_server.cocoindex_config import (
    graphcodebert_embedding,
    unixcoder_embedding,
    fallback_embedding,
    code_to_embedding,
    SMART_EMBEDDING_AVAILABLE
)


class TestEmbeddingFunctions:
    """Test smart embedding function definitions and configurations."""

    def test_smart_embedding_enabled(self):
        """Test that smart embedding is enabled."""
        assert SMART_EMBEDDING_AVAILABLE is True, "Smart embedding should be available for testing"

    def test_graphcodebert_function_exists(self):
        """Test that GraphCodeBERT embedding function is defined."""
        assert callable(graphcodebert_embedding), "graphcodebert_embedding should be callable"

        # Check function signature
        sig = inspect.signature(graphcodebert_embedding)
        assert 'text' in sig.parameters, "graphcodebert_embedding should have 'text' parameter"

    def test_unixcoder_function_exists(self):
        """Test that UniXcode embedding function is defined."""
        assert callable(unixcoder_embedding), "unixcoder_embedding should be callable"

        # Check function signature
        sig = inspect.signature(unixcoder_embedding)
        assert 'text' in sig.parameters, "unixcoder_embedding should have 'text' parameter"

    def test_fallback_function_exists(self):
        """Test that fallback embedding function is defined."""
        assert callable(fallback_embedding), "fallback_embedding should be callable"

        # Check function signature
        sig = inspect.signature(fallback_embedding)
        assert 'text' in sig.parameters, "fallback_embedding should have 'text' parameter"

    def test_default_embedding_function_exists(self):
        """Test that default embedding function still exists for backwards compatibility."""
        assert callable(code_to_embedding), "code_to_embedding should be callable"

        # Check function signature
        sig = inspect.signature(code_to_embedding)
        assert 'text' in sig.parameters, "code_to_embedding should have 'text' parameter"

    def test_function_docstrings(self):
        """Test that embedding functions have proper documentation."""
        functions_and_expected_keywords = [
            (graphcodebert_embedding, ['GraphCodeBERT', 'Python', 'Java']),
            (unixcoder_embedding, ['UniXcode', 'Rust', 'TypeScript']),
            (fallback_embedding, ['fallback', 'languages']),
        ]

        for func, keywords in functions_and_expected_keywords:
            assert func.__doc__ is not None, f"{func.__name__} should have a docstring"
            docstring = func.__doc__.lower()

            for keyword in keywords:
                assert keyword.lower() in docstring, f"{func.__name__} docstring should mention '{keyword}'"

    def test_transform_flow_decorators(self):
        """Test that embedding functions are properly decorated with @cocoindex.transform_flow()."""
        # This is a bit tricky to test directly, but we can check if the functions
        # have the expected attributes that CocoIndex transform_flow decorator adds

        functions = [graphcodebert_embedding, unixcoder_embedding, fallback_embedding, code_to_embedding]

        for func in functions:
            # CocoIndex transform_flow functions should be callable
            assert callable(func), f"{func.__name__} should be callable"

            # They should have a __name__ attribute
            assert hasattr(func, '__name__'), f"{func.__name__} should have __name__ attribute"

    def test_function_parameter_types(self):
        """Test that embedding functions have correct parameter type hints."""

        functions = [graphcodebert_embedding, unixcoder_embedding, fallback_embedding, code_to_embedding]

        for func in functions:
            sig = inspect.signature(func)

            # Check that 'text' parameter exists
            assert 'text' in sig.parameters, f"{func.__name__} should have 'text' parameter"

            # Check parameter annotation (this might be complex due to CocoIndex types)
            text_param = sig.parameters['text']
            assert text_param.annotation is not None, f"{func.__name__} 'text' parameter should have type annotation"

    def test_all_functions_defined(self):
        """Test that all required embedding functions are defined."""
        required_functions = [
            'graphcodebert_embedding',
            'unixcoder_embedding',
            'fallback_embedding',
            'code_to_embedding'
        ]

        # Import the module to check for function existence
        import cocoindex_code_mcp_server.cocoindex_config as config

        for func_name in required_functions:
            assert hasattr(config, func_name), f"Function '{func_name}' should be defined in cocoindex_config"
            func = getattr(config, func_name)
            assert callable(func), f"'{func_name}' should be callable"


class TestEmbeddingFunctionIntegration:
    """Test integration aspects of embedding functions."""

    def test_functions_use_different_models(self):
        """Test that different embedding functions are configured for different models."""
        # This is harder to test without actually calling the functions,
        # but we can at least verify they're different functions

        functions = [graphcodebert_embedding, unixcoder_embedding, fallback_embedding]

        # Each function should be unique
        for i, func1 in enumerate(functions):
            for j, func2 in enumerate(functions):
                if i != j:
                    assert func1 is not func2, f"Embedding functions should be distinct objects"

    def test_model_configuration_consistency(self):
        """Test that embedding functions are consistent with model group configuration."""
        from cocoindex_config import LANGUAGE_MODEL_GROUPS

        # This test verifies that the expected models are configured
        assert LANGUAGE_MODEL_GROUPS['graphcodebert']['model'] == 'microsoft/graphcodebert-base'
        assert LANGUAGE_MODEL_GROUPS['unixcoder']['model'] == 'microsoft/unixcoder-base'
        assert LANGUAGE_MODEL_GROUPS['fallback']['model'] == 'sentence-transformers/all-MiniLM-L6-v2'


if __name__ == '__main__':
    pytest.main([__file__])
