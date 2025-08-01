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
    from astchunk import ASTChunkBuilder
except ImportError as e:
    logging.warning(f"ASTChunk not available: {e}")
    ASTChunkBuilder = None

# Import CocoIndex conditionally to avoid circular imports
cocoindex_module: Optional[ModuleType] = None
try:
    import cocoindex
    cocoindex_module = cocoindex
except ImportError:
    LOGGER.warning("CocoIndex not available")


def detect_language_from_filename(filename: str) -> str:
    """
    Detect programming language from filename.
    Simplified version to avoid circular imports.
    """
    # Tree-sitter language mapping
    LANGUAGE_MAP = {
        ".c": "C",
        ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".h": "C++", ".hpp": "C++",
        ".cs": "C#",
        ".go": "Go",
        ".java": "Java",
        ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
        ".py": "Python", # ".pyi": "Python",
        ".rb": "Ruby",
        ".rs": "Rust",
        ".scala": "Scala",
        ".swift": "Swift",
        ".tsx": "TSX",
        ".ts": "TypeScript",
        ".hs": "Haskell", ".lhs": "Haskell",
        ".kt": "Kotlin", ".kts": "Kotlin",
    }

    basename = os.path.basename(filename)

    # Handle special files
    if basename.lower() in ["makefile", "dockerfile", "jenkinsfile"]:
        return basename.lower()

    # Get extension
    ext = os.path.splitext(filename)[1].lower()

    # Map to language
    return LANGUAGE_MAP.get(ext, "Unknown")


class CocoIndexASTChunker:
    """
    CocoIndex-integrated AST chunker that combines ASTChunk with our existing
    Haskell tree-sitter functionality.
    """

    # Language mapping from CocoIndex to ASTChunk
    LANGUAGE_MAP = {
        "Python": "python",
        "Java": "java",
        "C#": "csharp",
        "TypeScript": "typescript",
        "JavaScript": "typescript",  # ASTChunk uses TypeScript parser for JS
        "TSX": "typescript",
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
            LOGGER.warning(f"Language {language} not supported by ASTChunk")
            return self._fallback_chunking(code, language, file_path)

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
                    "chunk_method": "ast_chunking",
                    "language": language,
                    "file_path": file_path,
                    "chunk_size": len(content.strip()),
                    "line_count": len(content.split('\n')),
                    **metadata
                }

                result_chunks.append(Chunk(
                    content=content,
                    metadata=enhanced_metadata
                ))

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
        # Use our existing Haskell chunking for Haskell code
        if language == "Haskell":
            try:
                # Import and call Haskell chunker
                import importlib
                haskell_module = importlib.import_module('.lang.haskell.haskell_ast_chunker', 'cocoindex_code_mcp_server')
                extract_func = getattr(haskell_module, 'extract_haskell_ast_chunks')
                chunks = extract_func(code)

                result_chunks: List[Chunk] = []
                for i, chunk in enumerate(chunks):
                    metadata = {
                        "chunk_id": i,
                        "chunk_method": "haskell_ast_chunking",
                        "language": language,
                        "file_path": file_path,
                        "chunk_size": len(chunk.text()),
                        "line_count": chunk.end_line() - chunk.start_line() + 1,
                        "start_line": chunk.start_line(),
                        "end_line": chunk.end_line(),
                        "node_type": chunk.node_type(),
                        **chunk.metadata()
                    }

                    result_chunks.append(Chunk(
                        content=chunk.text(),
                        metadata=metadata
                    ))

                LOGGER.info(f"Haskell AST chunking created {len(result_chunks)} chunks")
                return result_chunks

            except Exception as e:
                LOGGER.error(f"Haskell AST chunking failed: {e}")

        # Simple text chunking as last resort
        return self._simple_text_chunking(code, language, file_path)

    def _simple_text_chunking(self, code: str, language: str, file_path: str) -> List[Chunk]:
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
                    "chunk_method": "simple_text_chunking",
                    "language": language,
                    "file_path": file_path,
                    "chunk_size": len(content),
                    "line_count": len(chunk_lines),
                    "start_line": i + 1,
                    "end_line": i + len(chunk_lines)
                }

                chunks.append(Chunk(
                    content=content,
                    metadata=metadata
                ))

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

            chunks.append(Chunk(
                content=chunk.get("content", ""),
                metadata=metadata,
                location=location,
                start=start_line,
                end=metadata.get("end_line", 0)
            ))

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
