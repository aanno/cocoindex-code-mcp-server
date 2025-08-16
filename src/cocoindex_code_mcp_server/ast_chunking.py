#!/usr/bin/env python3

"""
AST-based code chunking integration for CocoIndex using ASTChunk library.
This module provides a CocoIndex operation that leverages ASTChunk for
structure-aware code chunking.
"""

from cocoindex.op import FunctionSpec
import logging
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, List, Optional, Union

import cocoindex
from cocoindex_code_mcp_server import LOGGER


@dataclass
class Chunk:
    """Represents a code chunk with text and location metadata."""
    content: str
    metadata: cocoindex.Json  # Use cocoindex.Json for union-safe metadata
    location: str = ""
    start: int = 0
    end: int = 0
    
    def __getitem__(self, key: Union[str, int]) -> Any:
        """Allow dictionary-style access."""
        if isinstance(key, str):
            if hasattr(self, key):
                return getattr(self, key)
            elif key in self.metadata:
                return self.metadata[key]
            else:
                raise KeyError(f"Key '{key}' not found in chunk")
        else:
            # For integer access, treat this chunk as if it's the only item in a list
            if key == 0:
                return self
            else:
                raise IndexError(f"Chunk index {key} out of range (only index 0 is valid)")
    
    def __setitem__(self, key: str, value) -> None:
        """Allow dictionary-style assignment."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.metadata[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in chunk (for 'key in chunk' syntax)."""
        return hasattr(self, key) or key in self.metadata
    
    def get(self, key: str, default=""):
        """Dictionary-style get method."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def __getattr__(self, name: str):
        """Allow accessing metadata fields as attributes."""
        if name in self.metadata:
            return self.metadata[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def keys(self):
        """Return available keys (attribute names + metadata keys)."""
        chunk_attrs = ["content", "metadata", "location", "start", "end"]
        metadata_keys = list(self.metadata.keys()) if isinstance(self.metadata, dict) else []
        return chunk_attrs + metadata_keys
    
    def to_dict(self) -> dict:
        """Convert chunk to dictionary for CocoIndex compatibility."""
        result = {
            "content": self.content,
            "location": self.location,
            "start": self.start,
            "end": self.end
        }
        # Merge metadata
        result.update(self.metadata)
        return result



try:
    from astchunk import ASTChunkBuilder  # type: ignore
except ImportError as e:
    logging.warning(f"ASTChunk not available: {e}")
    ASTChunkBuilder = None  # type: ignore

# Import CocoIndex conditionally to avoid circular imports
cocoindex_module: Optional[ModuleType] = None
try:
    import cocoindex
    cocoindex_module = cocoindex
except ImportError:
    LOGGER.warning("CocoIndex not available")


def detect_language_from_filename(filename: str) -> str:
    """
    Detect programming language from filename using centralized mapping.
    """
    from .mappers import get_language_from_extension
    return get_language_from_extension(filename)


class CocoIndexASTChunker:
    """
    CocoIndex-integrated AST chunker that combines ASTChunk with our existing
    Haskell tree-sitter functionality.
    """

    # Language mapping from CocoIndex to ASTChunk (only for actually supported languages)
    LANGUAGE_MAP = {
        "Python": "python",
        "Java": "java",
        "C#": "csharp",
        "TypeScript": "typescript",
        "JavaScript": "typescript",  # ASTChunk uses TypeScript parser for JS
        "TSX": "typescript",
        # Note: C++, C, Kotlin, Rust are not supported by ASTChunk library
        # These languages continue to use their language-specific AST visitors
    }

    def __init__(self, max_chunk_size: int = 1800,
                 chunk_overlap: int = 0,
                 chunk_expansion: bool = False,
                 metadata_template: str = "default"):
        """
        Initialize the AST chunker with configuration.

        Args:
            max_chunk_size: Maximum non-whitespace characters per chunk
            chunk_overlap: Number of AST nodes to overlap between chunks
            chunk_expansion: Whether to add metadata headers to chunks
            metadata_template: Format for chunk metadata (default, repoeval, swebench)
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_expansion = chunk_expansion
        self.metadata_template = metadata_template

        # Cache for ASTChunkBuilder instances by language
        self._builders: Dict[str, ASTChunkBuilder] = {}

    def _get_builder(self, language: str) -> Optional[ASTChunkBuilder]:
        """Get or create an ASTChunkBuilder for the given language."""
        if ASTChunkBuilder is None:
            LOGGER.warning("ASTChunkBuilder not available")
            return None

        if language not in self._builders:
            try:
                configs = {
                    "max_chunk_size": self.max_chunk_size,
                    "language": language,
                    "metadata_template": self.metadata_template,
                    "chunk_expansion": self.chunk_expansion
                }
                self._builders[language] = ASTChunkBuilder(**configs)
            except Exception as e:
                LOGGER.error(f"Failed to create ASTChunkBuilder for {language}: {e}")
                return None

        return self._builders[language]

    def is_supported_language(self, language: str) -> bool:
        """Check if the language is supported by ASTChunk."""
        return language in self.LANGUAGE_MAP

    def chunk_code(self, code: str, language: str, file_path: str = "") -> List[Chunk]:
        """
        Chunk code using AST-based chunking.

        Args:
            code: Source code to chunk
            language: Programming language (CocoIndex format)
            file_path: Optional file path for metadata

        Returns:
            List of chunk dictionaries with content and metadata
        """
        # Map CocoIndex language to ASTChunk language
        astchunk_language = self.LANGUAGE_MAP.get(language)
        if not astchunk_language:
            LOGGER.info(f"🔍 Language {language} not supported by ASTChunk - using fallback")
            return self._fallback_chunking(code, language, file_path)
        else:
            LOGGER.info(f"🔍 Language {language} IS supported by ASTChunk - proceeding with astchunk_language={astchunk_language}")

        # Get ASTChunkBuilder for this language
        builder = self._get_builder(astchunk_language)
        if not builder:
            LOGGER.warning(f"Failed to get builder for {astchunk_language}")
            return self._fallback_chunking(code, language, file_path)

        try:
            # Create chunks using ASTChunk
            configs = {
                "max_chunk_size": self.max_chunk_size,
                "language": astchunk_language,
                "metadata_template": self.metadata_template,
                "chunk_expansion": self.chunk_expansion
            }

            chunks = builder.chunkify(code, **configs)

            # Convert ASTChunk format to CocoIndex format
            result_chunks: List[Chunk] = []
            for i, chunk in enumerate(chunks):
                # Extract content and metadata
                content = chunk.get('content', chunk.get('context', ''))
                metadata = chunk.get('metadata', {})

                # Enhance metadata with our information
                enhanced_metadata = {
                    "chunk_id": i,
                    "chunking_method": "astchunk_library",  # Use specific name for ASTChunk library usage
                    "language": language,
                    "file_path": file_path,
                    "chunk_size": len(content.strip()),
                    "line_count": len(content.split('\n')),
                    "tree_sitter_chunking_error": False,  # ASTChunk succeeded
                    "tree_sitter_analyze_error": False,   # ASTChunk succeeded
                    **metadata
                }

                chunk_obj = Chunk(
                    content=content,
                    metadata=enhanced_metadata
                )
                # Convert to dictionary to make metadata available as top-level fields for CocoIndex
                result_chunks.append(chunk_obj.to_dict())

            LOGGER.info(f"AST chunking created {len(result_chunks)} chunks for {language}")
            return result_chunks

        except Exception as e:
            LOGGER.error(f"AST chunking failed for {language}: {e}")
            return self._fallback_chunking(code, language, file_path)

    def _fallback_chunking(self, code: str, language: str, file_path: str) -> List[Chunk]:
        """
        Fallback to our existing Haskell chunking or simple text chunking.

        Args:
            code: Source code to chunk
            language: Programming language
            file_path: Optional file path for metadata

        Returns:
            List of chunk dictionaries
        """
        # Use our FIXED Haskell chunking for Haskell code with rust_haskell_* methods
        if language == "Haskell":
            try:
                # Import and call Haskell chunker
                import importlib
                haskell_module = importlib.import_module('.lang.haskell.haskell_ast_chunker', 'cocoindex_code_mcp_server')
                extract_func = getattr(haskell_module, 'extract_haskell_ast_chunks')
                chunks = extract_func(code)

                result_chunks: List[Chunk] = []
                for i, chunk_dict in enumerate(chunks):
                    # Extract data from the chunk dictionary returned by Haskell chunker
                    content = chunk_dict.get("text", "")
                    original_metadata = chunk_dict.get("metadata", {})
                    
                    # Build metadata preserving the original Rust chunking method
                    metadata = {
                        "chunk_id": i,
                        "language": language,
                        "file_path": file_path,
                        "chunk_size": len(content),
                        "line_count": chunk_dict.get("end", 0) - chunk_dict.get("start", 0) + 1,
                        "start_line": chunk_dict.get("start", 0),
                        "end_line": chunk_dict.get("end", 0),
                        "node_type": chunk_dict.get("node_type", "unknown"),
                        # Preserve the Rust chunking method names like 'ast_recursive', 'regex_fallback'
                        **original_metadata
                    }

                    chunk_obj = Chunk(
                        content=content,
                        metadata=metadata
                    )
                    # Keep as Chunk object
                    result_chunks.append(chunk_obj)

                LOGGER.info(f"✅ NEW: Haskell AST chunking created {len(result_chunks)} chunks with proper Rust method names")
                return result_chunks

            except Exception as e:
                LOGGER.error(f"Haskell AST chunking failed: {e}")

        # Simple text chunking as last resort
        return self._simple_text_chunking(code, language, file_path, "ast_fallback_unavailable")

    def _simple_text_chunking(self, code: str, language: str, file_path: str, chunking_method: str = "simple_text_chunking") -> List[Chunk]:
        """
        Simple text-based chunking as a fallback.

        Args:
            code: Source code to chunk
            language: Programming language
            file_path: Optional file path for metadata

        Returns:
            List of chunk dictionaries
        """
        lines = code.split('\n')
        chunks: list[Chunk] = []
        chunk_size = self.max_chunk_size // 10  # Rough estimate for lines

        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            content = '\n'.join(chunk_lines)

            if content.strip():
                metadata = {
                    "chunk_id": len(chunks),
                    "chunking_method": chunking_method,
                    "language": language,
                    "file_path": file_path,
                    "chunk_size": len(content),
                    "line_count": len(chunk_lines),
                    "start_line": i + 1,
                    "end_line": i + len(chunk_lines),
                    "tree_sitter_chunking_error": chunking_method == "ast_fallback_unavailable",  # True if fallback
                    "tree_sitter_analyze_error": False,   # Simple text chunking doesn't use analysis
                }

                chunk_obj = Chunk(
                    content=content,
                    metadata=metadata
                )
                # Keep as Chunk object
                chunks.append(chunk_obj)

        LOGGER.info(f"Simple text chunking created {len(chunks)} chunks")
        return chunks


def create_ast_chunking_operation() -> FunctionSpec:
    """
    Create a CocoIndex AST chunking operation using @cocoindex.op.function() decorator.
    Returns a function registered with CocoIndex that works with .transform() method.
    """
    if cocoindex is None:
        raise ImportError("CocoIndex not available")

    @cocoindex.op.function()
    def ast_chunk_content(content: str, language: str = "auto", max_chunk_size: int = 1800,
                          chunk_overlap: int = 0, chunk_expansion: bool = False,
                          metadata_template: str = "default") -> List[Chunk]:
        LOGGER.info(f"🚀 ast_chunk_content called with language={language}, content_length={len(content)}")
        """
        AST-based code chunking function for CocoIndex.

        Args:
            content: Text content to chunk
            language: Programming language
            max_chunk_size: Maximum non-whitespace characters per chunk
            chunk_overlap: Number of AST nodes to overlap between chunks
            chunk_expansion: Whether to add metadata headers to chunks
            metadata_template: Format for chunk metadata

        Returns:
            List of chunk dictionaries with text and location metadata
        """
        # Auto-detect language if needed
        if language == "auto":
            detected_language = "Python"  # Default fallback for now
        else:
            detected_language = language

        # Create chunker with helper class for complex logic
        chunker = CocoIndexASTChunker(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_expansion=chunk_expansion,
            metadata_template=metadata_template
        )

        # Generate chunks from the content
        raw_chunks = chunker.chunk_code(content, detected_language, "")
        LOGGER.info(f"🔍 AST chunking produced {len(raw_chunks)} raw chunks")

        # Convert dictionaries to Chunk dataclass instances
        chunks = []
        for i, chunk in enumerate(raw_chunks):
            chunk_content = chunk.get("content", "")
            LOGGER.info(f"  Raw chunk {i+1}: content_len={len(chunk_content)}")
            if len(chunk_content) == 0:
                LOGGER.error(f"❌ Raw chunk {i+1} has NO CONTENT! Keys: {list(chunk.keys())}")
                LOGGER.error(f"   Chunk data: {chunk}")
            metadata = chunk.get("metadata", {})
            # Construct location from file path and line numbers, ensuring uniqueness
            file_path = metadata.get("file_path", "")
            start_line = metadata.get("start_line", 0)
            chunk_id = metadata.get("chunk_id", i)

            # Create unique location using chunk index to avoid duplicates
            if file_path:
                location = f"{file_path}:{start_line}#{chunk_id}"
            else:
                location = f"line:{start_line}#{chunk_id}"

            chunk_obj = Chunk(
                content=chunk.get("content", ""),
                metadata=metadata,
                location=location,
                start=start_line,
                end=metadata.get("end_line", 0)
            )
            # Keep as Chunk object but ensure metadata is accessible
            LOGGER.info(f"🔍 ASTChunk created chunk for {language} with chunking_method: {chunk_obj.metadata.get('chunking_method', 'NOT_SET')}")
            chunks.append(chunk_obj)

        return chunks

    return ast_chunk_content


# Create the operation conditionally
ASTChunkOperation = None
if cocoindex is not None:
    try:
        ASTChunkOperation = create_ast_chunking_operation()
        LOGGER.info("ASTChunkOperation created successfully")
    except Exception as e:
        LOGGER.warning(f"Failed to create ASTChunkOperation: {e}")


def create_hybrid_chunking_operation() -> FunctionSpec:
    """
    Create a hybrid chunking operation that uses AST-based chunking for supported
    languages and falls back to regex-based chunking for others.
    """
    if cocoindex is None:
        raise ImportError("CocoIndex not available")

    @cocoindex.op.function()
    def HybridChunkProcessor(source_field: str,
                             language: str,
                             filename: str = "",
                             chunk_size: int = 1200,
                             min_chunk_size: int = 300,
                             chunk_overlap: int = 200):
        """
        Hybrid chunking that uses AST-based chunking for supported languages
        and falls back to regex-based chunking for others.
        """
        def process_record(record):
            code = record.get(source_field, "")
            lang = record.get(language, "")
            file_path = record.get(filename, "")

            # Check if language is supported by ASTChunk
            chunker = CocoIndexASTChunker(
                max_chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_expansion=False,
                metadata_template="default"
            )

            if chunker.is_supported_language(lang):
                # Use AST-based chunking
                chunks: List[Chunk] = chunker.chunk_code(code, lang, file_path)
                LOGGER.debug(f"AST chunking created {len(chunks)} chunks for {lang}")

                # Convert to CocoIndex format
                result = []
                for chunk in chunks:
                    metadata = chunk["metadata"]
                    chunk_record = {
                        "text": chunk["content"],
                        "location": f"{file_path}:{metadata.get('start_line', 0) if metadata is not None else None}",
                        "start": metadata.get("start_line", 0) if metadata is not None else None,
                        "end": metadata.get("end_line", 0) if metadata is not None else None,
                        "metadata": metadata
                    }
                    result.append(chunk_record)

                return result
            else:
                # Fall back to regex-based chunking
                try:
                    # Import locally to avoid circular dependency
                    # Use basic CocoIndex chunking if available
                    if cocoindex_module is not None:
                        # Import CUSTOM_LANGUAGES for proper SplitRecursively usage
                        try:
                            from cocoindex_code_mcp_server.cocoindex_config import CUSTOM_LANGUAGES
                            splitter = cocoindex_module.functions.SplitRecursively(custom_languages=CUSTOM_LANGUAGES)
                        except ImportError:
                            splitter = cocoindex_module.functions.SplitRecursively()
                        
                        LOGGER.debug(f"Using CocoIndex SplitRecursively for {lang} code")
                        
                        # SplitRecursively expects to be called directly on content, not as a transform
                        # Call it with proper arguments
                        result = splitter(
                            code,
                            language=lang,
                            chunk_size=chunk_size,
                            min_chunk_size=min_chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                    else:
                        raise ImportError("CocoIndex not available")

                    return result
                except ImportError:
                    # If we can't import CUSTOM_LANGUAGES, fall back to simple text chunking
                    LOGGER.warning(f"Could not import CUSTOM_LANGUAGES, using simple text chunking for {lang}")
                    temp_chunker = CocoIndexASTChunker(max_chunk_size=chunk_size)
                    chunks = temp_chunker._simple_text_chunking(code, lang, file_path)

                    # Convert to CocoIndex format
                    result = []
                    for chunk in chunks:
                        metadata = chunk["metadata"]
                        chunk_record = {
                            "text": chunk["content"],
                            "location": f"{file_path}:{metadata.get('start_line', 0) if metadata is not None else None}",
                            "start": metadata.get("start_line", 0) if metadata is not None else None,
                            "end": metadata.get("end_line", 0) if metadata is not None else None,
                            "metadata": metadata
                        }
                        result.append(chunk_record)

                    return result

        return process_record

    return HybridChunkProcessor


if __name__ == "__main__":
    print("AST chunking module - use tests/test_ast_chunking_integration.py for testing")
