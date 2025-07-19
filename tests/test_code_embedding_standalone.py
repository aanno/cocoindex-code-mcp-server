#!/usr/bin/env python3

"""
Standalone tests for language-aware code embedding functionality.
Tests the logic without requiring full CocoIndex imports.
"""

import pytest
from unittest.mock import Mock
import sys
import os

# Test the core logic without importing the full module
class MockCodeEmbeddingExecutor:
    """Mock version of CodeEmbeddingExecutor for testing logic."""
    
    # Language support mappings based on research and model documentation
    GRAPHCODEBERT_LANGUAGES = {
        "python", "java", "javascript", "php", "ruby", "go", "c", "cpp", "c++", "js"
    }
    
    UNIXCODE_LANGUAGES = {
        "rust", "typescript", "csharp", "c#", "cs", "kotlin", "scala", "swift",
        "dart", "ts", "rs", "kt", "scala"
    }
    
    # Model name mappings
    MODEL_MAP = {
        "graphcodebert": "microsoft/graphcodebert-base",
        "unixcode": "microsoft/unixcoder-base",
    }
    
    def __init__(self, force_model=None, fallback_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.force_model = force_model
        self.fallback_model = fallback_model
    
    def _normalize_language(self, language: str | None) -> str | None:
        """Normalize language names to standard format."""
        if not language:
            return None
        
        # Convert common variations to standard names
        lang_map = {
            "js": "javascript",
            "ts": "typescript", 
            "rs": "rust",
            "kt": "kotlin",
            "cs": "csharp",
            "c#": "csharp",
            "c++": "cpp",
            "cc": "cpp",
            "cxx": "cpp",
            "py": "python",
        }
        
        normalized = language.lower().strip()
        return lang_map.get(normalized, normalized)
    
    def _select_model(self, language: str | None) -> str:
        """Select the best embedding model for the given language."""
        if self.force_model:
            return self.force_model
        
        normalized_lang = self._normalize_language(language)
        
        if normalized_lang in self.GRAPHCODEBERT_LANGUAGES:
            return self.MODEL_MAP["graphcodebert"]
        elif normalized_lang in self.UNIXCODE_LANGUAGES:
            return self.MODEL_MAP["unixcode"]
        else:
            # Use fallback for unsupported languages
            return self.fallback_model


class MockSmartCodeEmbeddingExecutor:
    """Mock version of SmartCodeEmbeddingExecutor for testing logic."""
    
    # File extension to language mapping
    EXTENSION_TO_LANGUAGE = {
        ".py": "python",
        ".pyi": "python", 
        ".rs": "rust",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript", 
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".scala": "scala",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".cxx": "cpp",
        ".cc": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".go": "go",
        ".php": "php",
        ".rb": "ruby",
        ".swift": "swift",
        ".dart": "dart",
        ".hs": "haskell",
        ".lhs": "haskell",
    }
    
    def __init__(self, file_extension=None, language_hint=None):
        self.file_extension = file_extension
        self.language_hint = language_hint
    
    def _detect_language(self) -> str | None:
        """Detect programming language from context."""
        # Use explicit language hint if provided
        if self.language_hint:
            return self.language_hint
        
        # Detect from file extension
        if self.file_extension:
            ext = self.file_extension.lower()
            if not ext.startswith('.'):
                ext = '.' + ext
            return self.EXTENSION_TO_LANGUAGE.get(ext)
        
        return None


class TestCodeEmbeddingLogic:
    """Test suite for CodeEmbedding logic."""
    
    def test_language_normalization(self):
        """Test that language names are properly normalized."""
        executor = MockCodeEmbeddingExecutor()
        
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
            result = executor._normalize_language(input_lang)
            assert result == expected, f"Failed for {input_lang}: expected {expected}, got {result}"
    
    def test_graphcodebert_language_selection(self):
        """Test that GraphCodeBERT is selected for supported languages."""
        executor = MockCodeEmbeddingExecutor()
        
        graphcodebert_languages = ["python", "java", "javascript", "php", "ruby", "go", "c", "cpp"]
        
        for lang in graphcodebert_languages:
            selected_model = executor._select_model(lang)
            assert selected_model == "microsoft/graphcodebert-base", \
                f"Expected GraphCodeBERT for {lang}, got {selected_model}"
    
    def test_unixcode_language_selection(self):
        """Test that UniXcode is selected for supported languages."""
        executor = MockCodeEmbeddingExecutor()
        
        unixcode_languages = ["rust", "typescript", "csharp", "kotlin", "scala", "swift", "dart"]
        
        for lang in unixcode_languages:
            selected_model = executor._select_model(lang)
            assert selected_model == "microsoft/unixcoder-base", \
                f"Expected UniXcode for {lang}, got {selected_model}"
    
    def test_fallback_language_selection(self):
        """Test that fallback model is selected for unsupported languages."""
        executor = MockCodeEmbeddingExecutor(fallback_model="custom-fallback-model")
        
        unsupported_languages = ["haskell", "ocaml", "erlang", "elixir", "cobol", None]
        
        for lang in unsupported_languages:
            selected_model = executor._select_model(lang)
            assert selected_model == "custom-fallback-model", \
                f"Expected fallback model for {lang}, got {selected_model}"
    
    def test_force_model_override(self):
        """Test that force_model overrides automatic selection."""
        executor = MockCodeEmbeddingExecutor(force_model="custom-override-model")
        
        # Should use override regardless of language
        test_languages = ["python", "rust", "haskell", None]
        
        for lang in test_languages:
            selected_model = executor._select_model(lang)
            assert selected_model == "custom-override-model", \
                f"Expected override model for {lang}, got {selected_model}"
    
    def test_language_variations(self):
        """Test that language variations are handled correctly."""
        executor = MockCodeEmbeddingExecutor()
        
        # Test common variations that should map to the same model
        test_cases = [
            (["js", "javascript"], "microsoft/graphcodebert-base"),
            (["ts", "typescript"], "microsoft/unixcoder-base"),
            (["rs", "rust"], "microsoft/unixcoder-base"),
            (["c++", "cpp"], "microsoft/graphcodebert-base"),
            (["c#", "csharp", "cs"], "microsoft/unixcoder-base"),
        ]
        
        for variations, expected_model in test_cases:
            models = [executor._select_model(var) for var in variations]
            assert all(model == expected_model for model in models), \
                f"Inconsistent model selection for variations {variations}: {models}"


class TestSmartCodeEmbeddingLogic:
    """Test suite for SmartCodeEmbedding logic."""
    
    def test_extension_to_language_mapping(self):
        """Test that file extensions are correctly mapped to languages."""
        test_cases = [
            (".py", "python"),
            (".pyi", "python"),
            (".rs", "rust"),
            (".js", "javascript"),
            (".mjs", "javascript"),
            (".ts", "typescript"),
            (".tsx", "typescript"),
            (".java", "java"),
            (".kt", "kotlin"),
            (".scala", "scala"),
            (".cs", "csharp"),
            (".cpp", "cpp"),
            (".go", "go"),
            (".php", "php"),
            (".rb", "ruby"),
            (".hs", "haskell"),
            (".unknown", None),
        ]
        
        for extension, expected_language in test_cases:
            executor = MockSmartCodeEmbeddingExecutor(file_extension=extension)
            detected = executor._detect_language()
            assert detected == expected_language, \
                f"Extension {extension}: expected {expected_language}, got {detected}"
    
    def test_language_hint_override(self):
        """Test that language_hint overrides extension detection."""
        executor = MockSmartCodeEmbeddingExecutor(
            file_extension=".py",  # Would normally detect as Python
            language_hint="rust"   # But override with Rust
        )
        
        detected = executor._detect_language()
        assert detected == "rust"
    
    def test_extension_normalization(self):
        """Test that extensions are normalized correctly."""
        test_cases = [
            ("py", "python"),
            (".py", "python"),
            ("JS", "javascript"),
            (".JS", "javascript"),
            ("Rs", "rust"),
            (".Rs", "rust"),
        ]
        
        for extension, expected_language in test_cases:
            executor = MockSmartCodeEmbeddingExecutor(file_extension=extension)
            detected = executor._detect_language()
            assert detected == expected_language, \
                f"Extension {extension}: expected {expected_language}, got {detected}"


class TestLanguageSpecificConfigurations:
    """Test specific configurations for different programming languages."""
    
    def test_supported_language_coverage(self):
        """Test that all major programming languages are covered."""
        executor = MockCodeEmbeddingExecutor()
        
        # Languages that should be supported by GraphCodeBERT or UniXcode
        supported_languages = [
            "python", "java", "javascript", "typescript", "rust", "go", 
            "cpp", "csharp", "kotlin", "scala", "php", "ruby"
        ]
        
        for lang in supported_languages:
            model = executor._select_model(lang)
            assert "microsoft/" in model, f"Language {lang} should use a specialized model, got {model}"
    
    def test_comprehensive_language_model_mapping(self):
        """Test comprehensive mapping of languages to models."""
        executor = MockCodeEmbeddingExecutor()
        
        expected_mappings = {
            # GraphCodeBERT languages
            "python": "microsoft/graphcodebert-base",
            "java": "microsoft/graphcodebert-base", 
            "javascript": "microsoft/graphcodebert-base",
            "js": "microsoft/graphcodebert-base",
            "php": "microsoft/graphcodebert-base",
            "ruby": "microsoft/graphcodebert-base",
            "go": "microsoft/graphcodebert-base",
            "c": "microsoft/graphcodebert-base",
            "cpp": "microsoft/graphcodebert-base",
            "c++": "microsoft/graphcodebert-base",
            
            # UniXcode languages
            "rust": "microsoft/unixcoder-base",
            "rs": "microsoft/unixcoder-base",
            "typescript": "microsoft/unixcoder-base",
            "ts": "microsoft/unixcoder-base",
            "csharp": "microsoft/unixcoder-base",
            "c#": "microsoft/unixcoder-base",
            "cs": "microsoft/unixcoder-base",
            "kotlin": "microsoft/unixcoder-base",
            "kt": "microsoft/unixcoder-base",
            "scala": "microsoft/unixcoder-base",
            "swift": "microsoft/unixcoder-base",
            "dart": "microsoft/unixcoder-base",
            
            # Fallback languages
            "haskell": "sentence-transformers/all-MiniLM-L6-v2",
            "ocaml": "sentence-transformers/all-MiniLM-L6-v2",
            "unknown": "sentence-transformers/all-MiniLM-L6-v2",
        }
        
        for language, expected_model in expected_mappings.items():
            actual_model = executor._select_model(language)
            assert actual_model == expected_model, \
                f"Language {language}: expected {expected_model}, got {actual_model}"


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_complete_file_processing_flow(self):
        """Test complete flow from file extension to model selection."""
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
            smart_executor = MockSmartCodeEmbeddingExecutor(file_extension=file_ext)
            detected_language = smart_executor._detect_language()
            assert detected_language == expected_lang, \
                f"Extension {file_ext}: expected language {expected_lang}, got {detected_language}"
            
            # Step 2: Model selection
            code_executor = MockCodeEmbeddingExecutor()
            selected_model = code_executor._select_model(detected_language)
            assert selected_model == expected_model, \
                f"Language {detected_language}: expected model {expected_model}, got {selected_model}"
    
    def test_override_scenarios(self):
        """Test various override scenarios."""
        # Test force model override
        executor = MockCodeEmbeddingExecutor(force_model="custom-model")
        assert executor._select_model("python") == "custom-model"
        assert executor._select_model("rust") == "custom-model"
        assert executor._select_model("haskell") == "custom-model"
        
        # Test custom fallback
        executor = MockCodeEmbeddingExecutor(fallback_model="custom-fallback")
        assert executor._select_model("haskell") == "custom-fallback"
        assert executor._select_model("unsupported") == "custom-fallback"
        
        # Test language hint override
        smart_executor = MockSmartCodeEmbeddingExecutor(
            file_extension=".py",
            language_hint="rust"
        )
        assert smart_executor._detect_language() == "rust"


def test_model_configuration_consistency():
    """Test that model configurations are consistent."""
    # Ensure no overlap between GraphCodeBERT and UniXcode languages
    graphcodebert_langs = MockCodeEmbeddingExecutor.GRAPHCODEBERT_LANGUAGES
    unixcode_langs = MockCodeEmbeddingExecutor.UNIXCODE_LANGUAGES
    
    overlap = graphcodebert_langs.intersection(unixcode_langs)
    assert len(overlap) == 0, f"Found overlap between language sets: {overlap}"
    
    # Test model mappings exist
    model_map = MockCodeEmbeddingExecutor.MODEL_MAP
    assert "graphcodebert" in model_map
    assert "unixcode" in model_map
    assert model_map["graphcodebert"].startswith("microsoft/")
    assert model_map["unixcode"].startswith("microsoft/")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])