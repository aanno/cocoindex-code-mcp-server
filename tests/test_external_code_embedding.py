#!/usr/bin/env python3

"""
Tests for external language-aware code embedding functionality.
Tests the smart_code_embedding module without modifying CocoIndex.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

from smart_code_embedding import (
    LanguageModelSelector,
    create_smart_code_embedding,
    create_python_embedding,
    create_rust_embedding,
    get_supported_languages,
    get_supported_extensions
)


class TestLanguageModelSelector:
    """Test suite for LanguageModelSelector class."""
    
    def test_initialization(self):
        """Test that selector initializes correctly."""
        selector = LanguageModelSelector()
        assert selector.fallback_model == "sentence-transformers/all-MiniLM-L6-v2"
        
        custom_selector = LanguageModelSelector(fallback_model="custom-model")
        assert custom_selector.fallback_model == "custom-model"
    
    def test_language_normalization(self):
        """Test language name normalization."""
        selector = LanguageModelSelector()
        
        test_cases = [
            ("Python", "python"),
            ("JavaScript", "javascript"),
            ("js", "javascript"),
            ("ts", "typescript"),
            ("rs", "rust"),
            ("C++", "cpp"),
            ("c#", "csharp"),
            ("CS", "csharp"),
            ("kt", "kotlin"),
            (None, None),
            ("", None),
            ("  JAVA  ", "java"),
        ]
        
        for input_lang, expected in test_cases:
            result = selector.normalize_language(input_lang)
            assert result == expected, f"Failed for {input_lang}: expected {expected}, got {result}"
    
    def test_extension_detection(self):
        """Test language detection from file extensions."""
        selector = LanguageModelSelector()
        
        test_cases = [
            (".py", "python"),
            (".pyi", "python"),
            (".rs", "rust"),
            (".js", "javascript"),
            (".ts", "typescript"),
            (".java", "java"),
            (".kt", "kotlin"),
            (".cpp", "cpp"),
            (".go", "go"),
            (".hs", "haskell"),
            (".unknown", None),
            (None, None),
            ("", None),
        ]
        
        for extension, expected in test_cases:
            result = selector.detect_language_from_extension(extension)
            assert result == expected, f"Extension {extension}: expected {expected}, got {result}"
    
    def test_extension_normalization(self):
        """Test that extensions are normalized correctly."""
        selector = LanguageModelSelector()
        
        # Test with and without leading dot
        test_cases = [
            ("py", "python"),
            (".py", "python"),
            ("JS", "javascript"),
            (".JS", "javascript"),
            ("Rs", "rust"),
            (".Rs", "rust"),
        ]
        
        for extension, expected in test_cases:
            result = selector.detect_language_from_extension(extension)
            assert result == expected, f"Extension {extension}: expected {expected}, got {result}"
    
    def test_model_selection_graphcodebert(self):
        """Test GraphCodeBERT selection for supported languages."""
        selector = LanguageModelSelector()
        
        graphcodebert_languages = ["python", "java", "javascript", "php", "ruby", "go", "c", "cpp"]
        
        for lang in graphcodebert_languages:
            selected_model = selector.select_model(language=lang)
            assert selected_model == "microsoft/graphcodebert-base", \
                f"Expected GraphCodeBERT for {lang}, got {selected_model}"
    
    def test_model_selection_unixcode(self):
        """Test UniXcode selection for supported languages."""
        selector = LanguageModelSelector()
        
        unixcode_languages = ["rust", "typescript", "csharp", "kotlin", "scala", "swift", "dart"]
        
        for lang in unixcode_languages:
            selected_model = selector.select_model(language=lang)
            assert selected_model == "microsoft/unixcoder-base", \
                f"Expected UniXcode for {lang}, got {selected_model}"
    
    def test_model_selection_fallback(self):
        """Test fallback model selection for unsupported languages."""
        selector = LanguageModelSelector(fallback_model="custom-fallback")
        
        unsupported_languages = ["haskell", "ocaml", "erlang", "cobol", None]
        
        for lang in unsupported_languages:
            selected_model = selector.select_model(language=lang)
            assert selected_model == "custom-fallback", \
                f"Expected fallback for {lang}, got {selected_model}"
    
    def test_force_model_override(self):
        """Test that force_model overrides automatic selection."""
        selector = LanguageModelSelector()
        
        test_languages = ["python", "rust", "haskell", None]
        
        for lang in test_languages:
            selected_model = selector.select_model(language=lang, force_model="override-model")
            assert selected_model == "override-model", \
                f"Expected override for {lang}, got {selected_model}"
    
    def test_extension_based_selection(self):
        """Test model selection based on file extension."""
        selector = LanguageModelSelector()
        
        test_cases = [
            (".py", "microsoft/graphcodebert-base"),
            (".rs", "microsoft/unixcoder-base"),
            (".js", "microsoft/graphcodebert-base"),
            (".ts", "microsoft/unixcoder-base"),
            (".hs", "sentence-transformers/all-MiniLM-L6-v2"),  # fallback
        ]
        
        for extension, expected_model in test_cases:
            selected_model = selector.select_model(file_extension=extension)
            assert selected_model == expected_model, \
                f"Extension {extension}: expected {expected_model}, got {selected_model}"
    
    def test_model_args_handling(self):
        """Test model arguments handling."""
        selector = LanguageModelSelector()
        
        # Test Microsoft model gets trust_remote_code
        microsoft_args = selector.get_model_args("microsoft/graphcodebert-base")
        assert microsoft_args["trust_remote_code"] == True
        
        # Test custom args are preserved
        custom_args = {"device": "cpu", "custom_param": "value"}
        result_args = selector.get_model_args("microsoft/unixcoder-base", custom_args)
        assert result_args["device"] == "cpu"
        assert result_args["custom_param"] == "value"
        assert result_args["trust_remote_code"] == True
        
        # Test non-Microsoft model doesn't get trust_remote_code automatically
        general_args = selector.get_model_args("sentence-transformers/all-MiniLM-L6-v2")
        assert "trust_remote_code" not in general_args


class TestExternalAPIFunctions:
    """Test suite for external API functions."""
    
    @patch('smart_code_embedding.cocoindex')
    def test_create_smart_code_embedding(self, mock_cocoindex):
        """Test smart code embedding creation."""
        # Mock the SentenceTransformerEmbed function
        mock_embed_func = Mock()
        mock_cocoindex.functions.SentenceTransformerEmbed.return_value = mock_embed_func
        
        # Test automatic detection
        result = create_smart_code_embedding(file_extension=".py")
        
        # Verify it was called with GraphCodeBERT for Python
        mock_cocoindex.functions.SentenceTransformerEmbed.assert_called_with(
            model="microsoft/graphcodebert-base",
            args={"trust_remote_code": True}
        )
        assert result == mock_embed_func
    
    @patch('smart_code_embedding.cocoindex')
    def test_create_smart_code_embedding_manual_language(self, mock_cocoindex):
        """Test smart code embedding with manual language."""
        mock_embed_func = Mock()
        mock_cocoindex.functions.SentenceTransformerEmbed.return_value = mock_embed_func
        
        result = create_smart_code_embedding(language="rust")
        
        # Verify it was called with UniXcode for Rust
        mock_cocoindex.functions.SentenceTransformerEmbed.assert_called_with(
            model="microsoft/unixcoder-base",
            args={"trust_remote_code": True}
        )
        assert result == mock_embed_func
    
    @patch('smart_code_embedding.cocoindex')
    def test_create_smart_code_embedding_force_model(self, mock_cocoindex):
        """Test smart code embedding with forced model."""
        mock_embed_func = Mock()
        mock_cocoindex.functions.SentenceTransformerEmbed.return_value = mock_embed_func
        
        result = create_smart_code_embedding(
            language="python",
            force_model="custom-model"
        )
        
        # Verify it was called with custom model
        mock_cocoindex.functions.SentenceTransformerEmbed.assert_called_with(
            model="custom-model",
            args={}  # No trust_remote_code for non-Microsoft model
        )
        assert result == mock_embed_func
    
    @patch('smart_code_embedding.cocoindex')
    def test_create_smart_code_embedding_fallback(self, mock_cocoindex):
        """Test smart code embedding fallback for unsupported language."""
        mock_embed_func = Mock()
        mock_cocoindex.functions.SentenceTransformerEmbed.return_value = mock_embed_func
        
        result = create_smart_code_embedding(language="haskell")
        
        # Verify it was called with fallback model
        mock_cocoindex.functions.SentenceTransformerEmbed.assert_called_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            args={}
        )
        assert result == mock_embed_func
    
    @patch('smart_code_embedding.cocoindex')
    def test_language_specific_functions(self, mock_cocoindex):
        """Test language-specific convenience functions."""
        mock_embed_func = Mock()
        mock_cocoindex.functions.SentenceTransformerEmbed.return_value = mock_embed_func
        
        # Test Python function
        create_python_embedding()
        mock_cocoindex.functions.SentenceTransformerEmbed.assert_called_with(
            model="microsoft/graphcodebert-base",
            args={"trust_remote_code": True}
        )
        
        # Test Rust function
        create_rust_embedding()
        mock_cocoindex.functions.SentenceTransformerEmbed.assert_called_with(
            model="microsoft/unixcoder-base",
            args={"trust_remote_code": True}
        )
    
    def test_get_supported_languages(self):
        """Test getting supported languages mapping."""
        languages = get_supported_languages()
        
        assert isinstance(languages, dict)
        assert "python" in languages
        assert "rust" in languages
        assert "javascript" in languages
        assert languages["python"] == "microsoft/graphcodebert-base"
        assert languages["rust"] == "microsoft/unixcoder-base"
    
    def test_get_supported_extensions(self):
        """Test getting supported extensions mapping."""
        extensions = get_supported_extensions()
        
        assert isinstance(extensions, dict)
        assert ".py" in extensions
        assert ".rs" in extensions
        assert ".js" in extensions
        assert extensions[".py"] == "python"
        assert extensions[".rs"] == "rust"


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_complete_file_processing_flow(self):
        """Test complete flow from file extension to model selection."""
        selector = LanguageModelSelector()
        
        test_scenarios = [
            # (file_extension, expected_language, expected_model)
            (".py", "python", "microsoft/graphcodebert-base"),
            (".rs", "rust", "microsoft/unixcoder-base"),
            (".js", "javascript", "microsoft/graphcodebert-base"),
            (".ts", "typescript", "microsoft/unixcoder-base"),
            (".java", "java", "microsoft/graphcodebert-base"),
            (".kt", "kotlin", "microsoft/unixcoder-base"),
            (".hs", "haskell", "sentence-transformers/all-MiniLM-L6-v2"),
        ]
        
        for file_ext, expected_lang, expected_model in test_scenarios:
            # Step 1: Extension detection
            detected_language = selector.detect_language_from_extension(file_ext)
            assert detected_language == expected_lang, \
                f"Extension {file_ext}: expected language {expected_lang}, got {detected_language}"
            
            # Step 2: Model selection
            selected_model = selector.select_model(language=detected_language)
            assert selected_model == expected_model, \
                f"Language {detected_language}: expected model {expected_model}, got {selected_model}"
    
    @patch('smart_code_embedding.cocoindex')
    def test_file_context_integration(self, mock_cocoindex):
        """Test integration with file context from CocoIndex flows."""
        from smart_code_embedding import create_smart_embedding_from_file_context
        
        mock_embed_func = Mock()
        mock_cocoindex.functions.SentenceTransformerEmbed.return_value = mock_embed_func
        
        # Mock file record from CocoIndex
        file_record = {
            "extension": ".py",
            "file_path": "/path/to/file.py",
            "content": "def hello(): pass"
        }
        
        result = create_smart_embedding_from_file_context(file_record)
        
        # Should detect Python and use GraphCodeBERT
        mock_cocoindex.functions.SentenceTransformerEmbed.assert_called_with(
            model="microsoft/graphcodebert-base",
            args={"trust_remote_code": True}
        )
        assert result == mock_embed_func


def test_model_configuration_consistency():
    """Test that model configurations are consistent."""
    selector = LanguageModelSelector()
    
    # Ensure no overlap between GraphCodeBERT and UniXcode languages
    graphcodebert_langs = selector.GRAPHCODEBERT_LANGUAGES
    unixcode_langs = selector.UNIXCODE_LANGUAGES
    
    overlap = graphcodebert_langs.intersection(unixcode_langs)
    assert len(overlap) == 0, f"Found overlap between language sets: {overlap}"
    
    # Test model mappings exist
    assert "graphcodebert" in selector.MODEL_MAP
    assert "unixcode" in selector.MODEL_MAP
    assert selector.MODEL_MAP["graphcodebert"].startswith("microsoft/")
    assert selector.MODEL_MAP["unixcode"].startswith("microsoft/")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
