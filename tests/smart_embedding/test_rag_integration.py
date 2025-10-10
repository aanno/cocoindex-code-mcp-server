#!/usr/bin/env python3
"""
Integration tests for smart embedding with RAG system.
Tests that different languages are properly processed and embedded with correct models.
"""

import os
from types import FunctionType
from typing import cast

import pytest

from cocoindex_code_mcp_server.cocoindex_config import (
    LANGUAGE_MODEL_GROUPS,
    SMART_EMBEDDING_AVAILABLE,
    extract_language,
    get_embedding_model_group,
)


class TestRAGIntegration:
    """Test smart embedding integration with RAG system."""

    def test_fixture_files_exist(self, fixture_files):
        """Test that all fixture files exist and are readable."""
        for language, filepath in fixture_files.items():
            assert os.path.exists(filepath), f"Fixture file for {language} should exist: {filepath}"
            assert os.path.isfile(filepath), f"Fixture path for {language} should be a file: {filepath}"

            # Test file is readable
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 0, f"Fixture file for {language} should not be empty"

    @pytest.mark.parametrize("language,expected_group,expected_model", [
        ('python', 'graphcodebert', 'microsoft/graphcodebert-base'),
        ('rust', 'unixcoder', 'microsoft/unixcoder-base'),
        ('javascript', 'graphcodebert', 'microsoft/graphcodebert-base'),
        ('typescript', 'unixcoder', 'microsoft/unixcoder-base'),
        ('haskell', 'fallback', 'sentence-transformers/all-mpnet-base-v2'),
    ])
    def test_language_to_model_pipeline(self, language, expected_group, expected_model, fixture_files):
        """Test complete pipeline from language detection to model selection."""
        if language not in fixture_files:
            pytest.skip(f"No fixture file for {language}")

        filepath = fixture_files[language]
        filename = os.path.basename(filepath)

        # Test language extraction from filename
        detected_language = cast(FunctionType, extract_language)(filename)
        assert detected_language.lower() == language.lower(), f"Language extraction failed for {filename}"

        # Test model group selection
        model_group = cast(FunctionType, get_embedding_model_group)(detected_language)
        assert model_group == expected_group, f"Model group selection failed for {language}"

        # Test model mapping
        actual_model = LANGUAGE_MODEL_GROUPS[model_group]['model']
        assert actual_model == expected_model, f"Model mapping failed for {language}"

    def test_file_content_analysis(self, fixture_files, test_code_samples):
        """Test that fixture files contain expected code patterns."""
        language_patterns = {
            'python': ['def ', 'class ', ':'],
            'rust': ['fn ', 'struct ', 'impl '],
            'javascript': ['function ', 'class ', '=>'],
            'typescript': ['interface ', 'class ', ':'],
            'haskell': ['data ', '::'],
        }

        for language, patterns in language_patterns.items():
            if language in fixture_files:
                filepath = fixture_files[language]
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                for pattern in patterns:
                    assert pattern in content, f"Pattern '{pattern}' not found in {language} fixture"

    def test_smart_embedding_configuration(self):
        """Test that smart embedding is properly configured for testing."""
        assert SMART_EMBEDDING_AVAILABLE, "Smart embedding should be available for integration tests"

        # Test all model groups are configured
        required_groups = ['graphcodebert', 'unixcoder', 'fallback']
        for group in required_groups:
            assert group in LANGUAGE_MODEL_GROUPS, f"Model group '{group}' should be configured"

            group_info = LANGUAGE_MODEL_GROUPS[group]
            assert 'model' in group_info, f"Group '{group}' should have model configured"
            assert 'languages' in group_info, f"Group '{group}' should have languages configured"

    def test_language_coverage_in_fixtures(self, fixture_files):
        """Test that fixtures cover all major language groups."""
        # Should have at least one file for each model group
        graphcodebert_languages = set()
        unixcoder_languages = set()
        fallback_languages = set()

        for language, filepath in fixture_files.items():
            model_group = cast(FunctionType, get_embedding_model_group)(language)
            if model_group == 'graphcodebert':
                graphcodebert_languages.add(language)
            elif model_group == 'unixcoder':
                unixcoder_languages.add(language)
            else:
                fallback_languages.add(language)

        assert len(graphcodebert_languages) > 0, "Should have GraphCodeBERT language fixtures"
        assert len(unixcoder_languages) > 0, "Should have UniXcode language fixtures"
        assert len(fallback_languages) > 0, "Should have fallback language fixtures"

        # Log coverage for debugging
        print(f"GraphCodeBERT fixtures: {graphcodebert_languages}")
        print(f"UniXcode fixtures: {unixcoder_languages}")
        print(f"Fallback fixtures: {fallback_languages}")


class TestSmartEmbeddingFlow:
    """Test smart embedding flow behavior."""

    def test_model_selection_consistency(self, fixture_files):
        """Test that model selection is consistent across multiple calls."""
        for language, filepath in fixture_files.items():
            filename = os.path.basename(filepath)

            # Multiple calls should return same result
            results = []
            for _ in range(3):
                detected_lang = cast(FunctionType, extract_language)(filename)
                model_group = cast(FunctionType, get_embedding_model_group)(detected_lang)
                results.append((detected_lang, model_group))

            # All results should be identical
            assert len(set(results)) == 1, f"Model selection inconsistent for {language}: {results}"

    def test_case_insensitive_handling(self, fixture_files):
        """Test that language detection handles case variations."""
        for language, filepath in fixture_files.items():
            filename = os.path.basename(filepath)

            # Test original case
            original_lang = cast(FunctionType, extract_language)(filename)
            original_group = cast(FunctionType, get_embedding_model_group)(original_lang)

            # Test different cases
            upper_group = cast(FunctionType, get_embedding_model_group)(original_lang.upper())
            lower_group = cast(FunctionType, get_embedding_model_group)(original_lang.lower())

            assert original_group == upper_group == lower_group, f"Case sensitivity issue for {language}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
