#!/usr/bin/env python3
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 aanno <aanno@users.noreply.github.com>
#
# This file is part of cocoindex_code_mcp_server from
# https://github.com/aanno/cocoindex-code-mcp-server
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
External Language-Aware Code Embedding for CocoIndex

This module provides intelligent code embedding functionality that automatically
selects the best embedding model based on programming language, without
modifying the CocoIndex source code.

Usage as external wrapper around CocoIndex's SentenceTransformerEmbed.
"""

from typing import Any, Dict

import cocoindex
from cocoindex_code_mcp_server import LOGGER


class LanguageModelSelector:
    """
    Smart model selector for code embeddings based on programming language.

    Uses GraphCodeBERT for languages it supports well, UniXcode for others,
    and falls back to general-purpose models for unsupported languages.
    """

    # Language support mappings based on research and model documentation
    GRAPHCODEBERT_LANGUAGES = {"python", "java", "javascript", "php", "ruby", "go", "c", "cpp", "c++", "js"}

    UNIXCODE_LANGUAGES = {
        "rust",
        "typescript",
        "csharp",
        "c#",
        "cs",
        "kotlin",
        "scala",
        "swift",
        "dart",
        "ts",
        "rs",
        "kt",
    }

    # Model name mappings
    MODEL_MAP = {
        "graphcodebert": "microsoft/graphcodebert-base",
        "unixcode": "microsoft/unixcoder-base",
    }

    # File extension to language mapping - uses centralized mapping from mappers.py
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

    def __init__(self, fallback_model: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        """
        Initialize the language model selector.

        Args:
            fallback_model: Model to use for unsupported languages
        """
        self.fallback_model = fallback_model

    def normalize_language(self, language: str | None) -> str | None:
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

    def detect_language_from_extension(self, file_extension: str | None) -> str | None:
        """Detect programming language from file extension."""
        if not file_extension:
            return None

        # Create a dummy filename to use the centralized mapping
        dummy_filename = f"dummy{file_extension.lower()}"
        if not file_extension.startswith("."):
            dummy_filename = f"dummy.{file_extension.lower()}"

        from .mappers import get_internal_language_name, get_language_from_extension

        display_language = get_language_from_extension(dummy_filename)
        if display_language.lower() == "unknown":
            return None

        # Convert to internal processing name for embedding model selection
        return get_internal_language_name(display_language)

    def select_model(
        self, language: str | None = None, file_extension: str | None = None, force_model: str | None = None
    ) -> str:
        """
        Select the best embedding model for the given language or file extension.

        Args:
            language: Programming language (e.g., "python", "rust")
            file_extension: File extension (e.g., ".py", ".rs")
            force_model: Override automatic selection with specific model

        Returns:
            Model name to use for embedding
        """
        if force_model:
            return force_model

        # Try to detect language from extension if not provided
        if not language and file_extension:
            language = self.detect_language_from_extension(file_extension)

        normalized_lang = self.normalize_language(language)

        if normalized_lang in self.GRAPHCODEBERT_LANGUAGES:
            return self.MODEL_MAP["graphcodebert"]
        elif normalized_lang in self.UNIXCODE_LANGUAGES:
            return self.MODEL_MAP["unixcode"]
        else:
            # Use fallback for unsupported languages
            return self.fallback_model

    def get_model_args(self, model_name: str, custom_args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Get appropriate model arguments for the selected model.

        Args:
            model_name: Name of the model
            custom_args: Custom arguments to merge

        Returns:
            Dictionary of model arguments
        """
        args = custom_args or {}

        # Add trust_remote_code=True for Microsoft models
        if "microsoft/" in model_name and "trust_remote_code" not in args:
            args = {**args, "trust_remote_code": True}

        return args


def create_smart_code_embedding(
    language: str | None = None,
    file_extension: str | None = None,
    force_model: str | None = None,
    fallback_model: str = "sentence-transformers/all-mpnet-base-v2",
    model_args: Dict[str, Any] | None = None,
) -> cocoindex.functions.SentenceTransformerEmbed:
    """
    Create a CocoIndex SentenceTransformerEmbed function with intelligent model selection.

    This is the main external interface that wraps CocoIndex's existing functionality
    with language-aware model selection.

    Args:
        language: Programming language (e.g., "python", "rust", "java")
        file_extension: File extension for auto-detection (e.g., ".py", ".rs")
        force_model: Override automatic selection with specific model
        fallback_model: Model to use for unsupported languages
        model_args: Additional arguments to pass to the model constructor

    Returns:
        CocoIndex SentenceTransformerEmbed function configured with optimal model

    Example:
        # Automatic detection from file extension
        embedding_func = create_smart_code_embedding(file_extension=".py")

        # Manual language specification
        embedding_func = create_smart_code_embedding(language="rust")

        # Force specific model
        embedding_func = create_smart_code_embedding(
            language="python",
            force_model="microsoft/graphcodebert-base"
        )

        # Use in CocoIndex flow
        chunk["embedding"] = chunk["text"].transform(embedding_func)
    """
    selector = LanguageModelSelector(fallback_model=fallback_model)

    # Select the optimal model
    selected_model = selector.select_model(language=language, file_extension=file_extension, force_model=force_model)

    # Prepare model arguments
    final_args = selector.get_model_args(selected_model, model_args)

    # Log the selection for debugging
    LOGGER.info("Selected embedding model: %s for language: %s", selected_model, language or "auto")

    # Return CocoIndex SentenceTransformerEmbed with selected model
    return cocoindex.functions.SentenceTransformerEmbed(model=selected_model, args=final_args)


def create_smart_embedding_from_file_context(
    file_record: Dict[str, Any],
    extension_field: str = "extension",
    fallback_model: str = "sentence-transformers/all-mpnet-base-v2",
    model_args: Dict[str, Any] | None = None,
) -> cocoindex.functions.SentenceTransformerEmbed:
    """
    Create smart code embedding function from file context in a CocoIndex flow.

    This helper function extracts the file extension from a file record and
    creates the appropriate embedding function.

    Args:
        file_record: File record from CocoIndex flow
        extension_field: Field name containing file extension
        fallback_model: Model for unsupported languages
        model_args: Additional model arguments

    Returns:
        CocoIndex SentenceTransformerEmbed function

    Example:
        # In a CocoIndex flow
        with flow_builder.read_files(data_scope.input_directory) as file:
            embedding_func = create_smart_embedding_from_file_context(file)
            chunk["embedding"] = chunk["text"].transform(embedding_func)
    """
    file_extension = file_record.get(extension_field)

    return create_smart_code_embedding(
        file_extension=file_extension, fallback_model=fallback_model, model_args=model_args
    )


# Convenience functions for common languages
def create_python_embedding(model_args: Dict[str, Any] | None = None) -> cocoindex.functions.SentenceTransformerEmbed:
    """Create embedding function optimized for Python code."""
    return create_smart_code_embedding(language="python", model_args=model_args)


def create_rust_embedding(model_args: Dict[str, Any] | None = None) -> cocoindex.functions.SentenceTransformerEmbed:
    """Create embedding function optimized for Rust code."""
    return create_smart_code_embedding(language="rust", model_args=model_args)


def create_javascript_embedding(
    model_args: Dict[str, Any] | None = None,
) -> cocoindex.functions.SentenceTransformerEmbed:
    """Create embedding function optimized for JavaScript code."""
    return create_smart_code_embedding(language="javascript", model_args=model_args)


def create_typescript_embedding(
    model_args: Dict[str, Any] | None = None,
) -> cocoindex.functions.SentenceTransformerEmbed:
    """Create embedding function optimized for TypeScript code."""
    return create_smart_code_embedding(language="typescript", model_args=model_args)


def get_supported_languages() -> Dict[str, str]:
    """
    Get mapping of supported languages to their optimal models.

    Returns:
        Dictionary mapping language names to model names
    """
    selector = LanguageModelSelector()
    languages = {}

    # Add GraphCodeBERT languages
    for lang in selector.GRAPHCODEBERT_LANGUAGES:
        languages[lang] = selector.MODEL_MAP["graphcodebert"]

    # Add UniXcode languages
    for lang in selector.UNIXCODE_LANGUAGES:
        languages[lang] = selector.MODEL_MAP["unixcode"]

    return languages


def get_supported_extensions() -> Dict[str, str]:
    """
    Get mapping of supported file extensions to their detected languages.

    Returns:
        Dictionary mapping file extensions to language names
    """
    selector = LanguageModelSelector()
    return selector.EXTENSION_TO_LANGUAGE.copy()


# Example usage and testing
if __name__ == "__main__":
    print("=== Smart Code Embedding for CocoIndex ===")
    print()

    print("Supported Languages:")
    for lang, model in get_supported_languages().items():
        print(f"  {lang:12} → {model}")

    print()
    print("Supported Extensions:")
    for ext, lang in get_supported_extensions().items():
        print(f"  {ext:6} → {lang}")

    print()
    print("Example Usage:")
    print(
        """
    # In your CocoIndex flow:
    from cocoindex_code_mcp_server.smart_code_embedding import create_smart_code_embedding

    # Automatic model selection
    embedding_func = create_smart_code_embedding(file_extension=".py")
    chunk["embedding"] = chunk["text"].transform(embedding_func)

    # Manual language selection
    embedding_func = create_smart_code_embedding(language="rust")
    chunk["embedding"] = chunk["text"].transform(embedding_func)
    """
    )
