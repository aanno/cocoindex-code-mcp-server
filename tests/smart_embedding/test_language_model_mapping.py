#!/usr/bin/env python3
"""
Test suite for smart embedding language-to-model mapping.
Tests that different programming languages are correctly mapped to appropriate embedding models.
"""

from types import FunctionType
from typing import cast

import pytest

from cocoindex_code_mcp_server.cocoindex_config import (
    LANGUAGE_MODEL_GROUPS,
    SMART_EMBEDDING_AVAILABLE,
    get_embedding_model_group,
)


class TestLanguageModelMapping:
    """Test language to embedding model mapping functionality."""

    def test_smart_embedding_available(self):
        """Test that smart embedding is properly enabled."""
        assert SMART_EMBEDDING_AVAILABLE is True, "Smart embedding should be available"

    def test_language_model_groups_structure(self):
        """Test that language model groups are properly configured."""
        assert 'graphcodebert' in LANGUAGE_MODEL_GROUPS
        assert 'unixcoder' in LANGUAGE_MODEL_GROUPS
        assert 'fallback' in LANGUAGE_MODEL_GROUPS

        # Check each group has required keys
        for group_name, group_info in LANGUAGE_MODEL_GROUPS.items():
            assert 'model' in group_info, f"Group {group_name} missing model key"
            assert 'languages' in group_info, f"Group {group_name} missing languages key"
            assert isinstance(group_info['languages'], set), f"Group {group_name} languages should be a set"

    def test_graphcodebert_languages(self):
        """Test GraphCodeBERT language mapping."""
        graphcodebert_languages = LANGUAGE_MODEL_GROUPS['graphcodebert']['languages']
        expected_languages = {'python', 'java', 'javascript', 'php', 'ruby', 'go', 'c', 'c++'}

        assert graphcodebert_languages == expected_languages
        assert LANGUAGE_MODEL_GROUPS['graphcodebert']['model'] == 'microsoft/graphcodebert-base'

    def test_unixcoder_languages(self):
        """Test UniXcode language mapping."""
        unixcoder_languages = LANGUAGE_MODEL_GROUPS['unixcoder']['languages']
        expected_languages = {'rust', 'typescript', 'tsx', 'c#', 'kotlin', 'scala', 'swift', 'dart'}

        assert unixcoder_languages == expected_languages
        assert LANGUAGE_MODEL_GROUPS['unixcoder']['model'] == 'microsoft/unixcoder-base'

    def test_fallback_configuration(self):
        """Test fallback model configuration."""
        fallback_info = LANGUAGE_MODEL_GROUPS['fallback']
        assert fallback_info['model'] == 'sentence-transformers/all-mpnet-base-v2'
        assert fallback_info['languages'] == set()  # Empty set for fallback

    @pytest.mark.parametrize("language,expected_group", [
        # GraphCodeBERT languages
        ('python', 'graphcodebert'),
        ('java', 'graphcodebert'),
        ('javascript', 'graphcodebert'),
        ('php', 'graphcodebert'),
        ('ruby', 'graphcodebert'),
        ('go', 'graphcodebert'),
        ('c', 'graphcodebert'),
        ('c++', 'graphcodebert'),

        # UniXcode languages
        ('rust', 'unixcoder'),
        ('typescript', 'unixcoder'),
        ('tsx', 'unixcoder'),
        ('c#', 'unixcoder'),
        ('kotlin', 'unixcoder'),
        ('scala', 'unixcoder'),
        ('swift', 'unixcoder'),
        ('dart', 'unixcoder'),

        # Fallback languages
        ('haskell', 'fallback'),
        ('ocaml', 'fallback'),
        ('fortran', 'fallback'),
        ('unknown_language', 'fallback'),
    ])
    def test_language_to_group_mapping(self, language, expected_group):
        """Test that languages are correctly mapped to their embedding model groups."""
        actual_group = cast(FunctionType, get_embedding_model_group)(language)
        assert actual_group == expected_group, f"Language '{language}' should map to '{expected_group}', got '{actual_group}'"

    def test_case_insensitive_mapping(self):
        """Test that language mapping is case insensitive."""
        test_cases = [
            ('Python', 'graphcodebert'),
            ('RUST', 'unixcoder'),
            ('JavaScript', 'graphcodebert'),
            ('TypeScript', 'unixcoder'),
            ('HASKELL', 'fallback'),
        ]

        for language, expected_group in test_cases:
            actual_group = cast(FunctionType, get_embedding_model_group)(language)
            assert actual_group == expected_group, f"Case insensitive test failed for '{language}'"

    def test_model_group_to_model_mapping(self):
        """Test that model groups map to correct embedding models."""
        model_mappings = {
            'graphcodebert': 'microsoft/graphcodebert-base',
            'unixcoder': 'microsoft/unixcoder-base',
            'fallback': 'sentence-transformers/all-mpnet-base-v2'
        }

        for group, expected_model in model_mappings.items():
            actual_model = LANGUAGE_MODEL_GROUPS[group]['model']
            assert actual_model == expected_model, f"Group '{group}' should use model '{expected_model}'"

    def test_no_language_overlap(self):
        """Test that no language appears in multiple groups (except fallback)."""
        graphcodebert_langs = LANGUAGE_MODEL_GROUPS['graphcodebert']['languages']
        unixcoder_langs = LANGUAGE_MODEL_GROUPS['unixcoder']['languages']

        # Check no overlap between specialized groups
        overlap = set(graphcodebert_langs) & set(unixcoder_langs)
        assert len(overlap) == 0, f"Languages should not overlap between groups: {overlap}"

    def test_comprehensive_language_coverage(self):
        """Test that common programming languages are covered."""
        # Languages that should be explicitly supported
        important_languages = [
            'python', 'java', 'javascript', 'typescript', 'rust', 'go',
            'c', 'c++', 'c#', 'php', 'ruby', 'kotlin', 'scala', 'swift'
        ]

        for lang in important_languages:
            group = cast(FunctionType, get_embedding_model_group)(lang)
            assert group in ['graphcodebert',
                             'unixcoder'], f"Important language '{lang}' should use specialized model, got '{group}'"


if __name__ == '__main__':
    pytest.main([__file__])
