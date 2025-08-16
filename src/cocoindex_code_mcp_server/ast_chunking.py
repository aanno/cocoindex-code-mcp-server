#!/usr/bin/env python3

"""
AST-based code chunking integration for CocoIndex using ASTChunk library.
This module provides a CocoIndex operation that leverages ASTChunk for
structure-aware code chunking.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

import cocoindex
from cocoindex import op
from cocoindex_code_mcp_server import LOGGER


@dataclass
class ASTChunkRow:
    """CocoIndex table row for AST chunk data."""
    content: str
    location: str
    start: int
    end: int
    chunking_method: str


class ChunkRow(TypedDict):
    """Represents a code chunk with text and location metadata (internal use)."""
    content: str
    location: str
    start: int
    end: int
    chunking_method: str


try:
    from astchunk import ASTChunkBuilder  # type: ignore
    ASTCHUNK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ASTChunk not available: {e}")
    ASTChunkBuilder = None  # type: ignore
    ASTCHUNK_AVAILABLE = False


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


class ASTChunkSpec(op.FunctionSpec):
    """AST-based code chunking function spec for CocoIndex."""
    max_chunk_size: int = 1800
    chunk_overlap: int = 0
    chunk_expansion: bool = False
    metadata_template: str = "default"


@op.executor_class()
class ASTChunkExecutor:
    """Executor for AST-based code chunking."""
    
    spec: ASTChunkSpec
    _builders: Dict[str, Any] = {}
    
    def analyze(self, content: Any, language: Any = "auto") -> type:
        """Analyze method required by CocoIndex to determine return type."""
        return list[ASTChunkRow]
    
    def _convert_chunks_to_ast_chunk_rows(self, chunks: List[ChunkRow]) -> list[ASTChunkRow]:
        """Convert internal ChunkRow objects to ASTChunkRow dataclass instances."""
        return [
            ASTChunkRow(
                content=chunk["content"],
                location=chunk["location"], 
                start=chunk["start"],
                end=chunk["end"],
                chunking_method=chunk["chunking_method"]
            )
            for chunk in chunks
        ]
    
    def _get_builder(self, language: str) -> Optional[Any]:
        """Get or create an ASTChunkBuilder for the given language."""
        if not ASTCHUNK_AVAILABLE:
            LOGGER.warning("ASTChunkBuilder not available")
            return None

        if language not in self._builders:
            try:
                configs = {
                    "max_chunk_size": self.spec.max_chunk_size,
                    "language": language,
                    "metadata_template": self.spec.metadata_template,
                    "chunk_expansion": self.spec.chunk_expansion
                }
                self._builders[language] = ASTChunkBuilder(**configs)
            except Exception as e:
                LOGGER.error(f"Failed to create ASTChunkBuilder for {language}: {e}")
                return None

        return self._builders[language]

    def _is_supported_language(self, language: str) -> bool:
        """Check if the language is supported by ASTChunk."""
        return language in LANGUAGE_MAP

    def _fallback_chunking(self, code: str, language: str) -> List[ChunkRow]:
        """Fallback to Haskell chunking or simple text chunking."""
        # Use our FIXED Haskell chunking for Haskell code
        if language == "Haskell":
            try:
                # Import and call Haskell chunker
                import importlib
                haskell_module = importlib.import_module('.lang.haskell.haskell_ast_chunker', 'cocoindex_code_mcp_server')
                extract_func = getattr(haskell_module, 'extract_haskell_ast_chunks')
                chunks = extract_func(code)

                result_chunks: List[ChunkRow] = []
                for i, chunk_dict in enumerate(chunks):
                    # Extract data from the chunk dictionary returned by Haskell chunker
                    content = chunk_dict.get("text", "")
                    original_metadata = chunk_dict.get("metadata", {})
                    start_line = chunk_dict.get("start", 0)
                    end_line = chunk_dict.get("end", 0)
                    
                    # Build location
                    location = f"line:{start_line}#{i}"
                    
                    # Get chunking method from original metadata (preserves Rust method names)
                    chunking_method = original_metadata.get("chunking_method", "rust_haskell_ast")

                    chunk_row = ChunkRow({
                        "content": content,
                        "location": location,
                        "start": start_line,
                        "end": end_line,
                        "chunking_method": chunking_method
                    })
                    result_chunks.append(chunk_row)

                LOGGER.info(f"✅ Haskell AST chunking created {len(result_chunks)} chunks with proper Rust method names")
                return result_chunks

            except Exception as e:
                LOGGER.error(f"Haskell AST chunking failed: {e}")

        # Simple text chunking as last resort
        return self._simple_text_chunking(code, language, "ast_fallback_unavailable")

    def _simple_text_chunking(self, code: str, language: str, chunking_method: str = "simple_text_chunking") -> List[ChunkRow]:
        """Simple text-based chunking as a fallback."""
        lines = code.split('\n')
        chunks: List[ChunkRow] = []
        chunk_size = self.spec.max_chunk_size // 10  # Rough estimate for lines

        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            content = '\n'.join(chunk_lines)

            if content.strip():
                location = f"line:{i + 1}#{len(chunks)}"
                
                chunk_row = ChunkRow({
                    "content": content,
                    "location": location,
                    "start": i + 1,
                    "end": i + len(chunk_lines),
                    "chunking_method": chunking_method
                })
                chunks.append(chunk_row)

        LOGGER.info(f"Simple text chunking created {len(chunks)} chunks")
        return chunks
    

    def __call__(self, content: str, language: str = "auto") -> list[ASTChunkRow]:
        """Main chunking function - returns typed chunk structures for CocoIndex."""
        LOGGER.info(f"🚀 ASTChunk called with language={language}, content_length={len(content)}")
        
        # Auto-detect language if needed
        if language == "auto":
            detected_language = "Python"  # Default fallback for now
        else:
            detected_language = language

        # Map CocoIndex language to ASTChunk language
        astchunk_language = LANGUAGE_MAP.get(detected_language)
        if not astchunk_language:
            LOGGER.info(f"🔍 Language {detected_language} not supported by ASTChunk - using fallback")
            chunks = self._fallback_chunking(content, detected_language)
            return self._convert_chunks_to_ast_chunk_rows(chunks)
        else:
            LOGGER.info(f"🔍 Language {detected_language} IS supported by ASTChunk - proceeding with astchunk_language={astchunk_language}")

        # Get ASTChunkBuilder for this language
        builder = self._get_builder(astchunk_language)
        if not builder:
            LOGGER.warning(f"Failed to get builder for {astchunk_language}")
            chunks = self._fallback_chunking(content, detected_language)
            return self._convert_chunks_to_ast_chunk_rows(chunks)

        try:
            # Create chunks using ASTChunk
            configs = {
                "max_chunk_size": self.spec.max_chunk_size,
                "language": astchunk_language,
                "metadata_template": self.spec.metadata_template,
                "chunk_expansion": self.spec.chunk_expansion
            }

            astchunk_results = builder.chunkify(content, **configs)

            # Convert ASTChunk format to our ChunkRow format
            result_chunks: List[ChunkRow] = []
            for i, chunk in enumerate(astchunk_results):
                # Extract content and metadata
                chunk_content = chunk.get('content', chunk.get('context', ''))
                metadata = chunk.get('metadata', {})

                # Build location
                start_line = metadata.get('start_line', 0)
                location = f"line:{start_line}#{i}"

                chunk_row = ChunkRow({
                    "content": chunk_content,
                    "location": location,
                    "start": start_line,
                    "end": metadata.get('end_line', 0),
                    "chunking_method": "astchunk_library"  # Use specific name for ASTChunk library usage
                })
                result_chunks.append(chunk_row)

            LOGGER.info(f"AST chunking created {len(result_chunks)} chunks for {detected_language}")
            return self._convert_chunks_to_ast_chunk_rows(result_chunks)

        except Exception as e:
            LOGGER.error(f"AST chunking failed for {detected_language}: {e}")
            chunks = self._fallback_chunking(content, detected_language)
            return self._convert_chunks_to_ast_chunk_rows(chunks)


# Create the operation  
ASTChunkOperation = ASTChunkSpec()

if __name__ == "__main__":
    print("AST chunking module - use tests/test_ast_chunking_integration.py for testing")
