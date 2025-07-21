#!/usr/bin/env python3

"""
AST-based code chunking integration for CocoIndex using ASTChunk library.
This module provides a CocoIndex operation that leverages ASTChunk for 
structure-aware code chunking.
"""

import sys
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from __init__ import LOGGER

try:
    from astchunk import ASTChunkBuilder
except ImportError as e:
    logging.warning(f"ASTChunk not available: {e}")
    ASTChunkBuilder = None

# Import CocoIndex conditionally to avoid circular imports
cocoindex = None
try:
    import cocoindex
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
        ".py": "Python", ".pyi": "Python",
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
    
    def chunk_code(self, code: str, language: str, file_path: str = "") -> List[Dict[str, Any]]:
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
            result_chunks = []
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
                
                result_chunks.append({
                    "content": content,
                    "metadata": enhanced_metadata
                })
            
            LOGGER.info(f"AST chunking created {len(result_chunks)} chunks for {language}")
            return result_chunks
            
        except Exception as e:
            LOGGER.error(f"AST chunking failed for {language}: {e}")
            return self._fallback_chunking(code, language, file_path)
    
    def _fallback_chunking(self, code: str, language: str, file_path: str) -> List[Dict[str, Any]]:
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
                from lang.haskell.haskell_support import extract_haskell_ast_chunks
                chunks = extract_haskell_ast_chunks(code)
                
                result_chunks = []
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
                    
                    result_chunks.append({
                        "content": chunk.text(),
                        "metadata": metadata
                    })
                
                LOGGER.info(f"Haskell AST chunking created {len(result_chunks)} chunks")
                return result_chunks
                
            except Exception as e:
                LOGGER.error(f"Haskell AST chunking failed: {e}")
        
        # Simple text chunking as last resort
        return self._simple_text_chunking(code, language, file_path)
    
    def _simple_text_chunking(self, code: str, language: str, file_path: str) -> List[Dict[str, Any]]:
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
        chunks = []
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
                
                chunks.append({
                    "content": content,
                    "metadata": metadata
                })
        
        LOGGER.info(f"Simple text chunking created {len(chunks)} chunks")
        return chunks


def create_ast_chunking_operation():
    """
    Create a CocoIndex operation for AST-based chunking.
    """
    if cocoindex is None:
        raise ImportError("CocoIndex not available")
    
    @cocoindex.op.function()
    def ASTChunk(source_field: str, 
                 language: str = "auto",
                 max_chunk_size: int = 1800,
                 chunk_overlap: int = 0,
                 chunk_expansion: bool = False,
                 metadata_template: str = "default"):
        """
        Structure-aware code chunking using AST analysis.
        
        Args:
            source_field: Field containing the source code
            language: Programming language (auto-detect if "auto")
            max_chunk_size: Maximum non-whitespace characters per chunk
            chunk_overlap: Number of AST nodes to overlap between chunks
            chunk_expansion: Whether to add metadata headers to chunks
            metadata_template: Format for chunk metadata
            
        Returns:
            List of chunks with content and metadata
        """
        def process_record(record):
            code = record.get(source_field, "")
            file_path = record.get("file_path", "")
            
            # Auto-detect language if needed
            if language == "auto":
                if file_path:
                    detected_language = detect_language_from_filename(file_path)
                else:
                    detected_language = "Python"  # Default fallback
            else:
                detected_language = language
            
            # Create chunker
            chunker = CocoIndexASTChunker(
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_expansion=chunk_expansion,
                metadata_template=metadata_template
            )
            
            # Generate chunks
            chunks = chunker.chunk_code(code, detected_language, file_path)
            
            # Return chunks as separate records
            result = []
            for chunk in chunks:
                new_record = record.copy()
                new_record.update(chunk)
                result.append(new_record)
            
            return result
        
        return process_record
    
    return ASTChunk


# Create the operation conditionally
ASTChunkOperation = None
if cocoindex is not None:
    try:
        ASTChunkOperation = create_ast_chunking_operation()
    except Exception as e:
        LOGGER.warning(f"Failed to create ASTChunkOperation: {e}")


def create_hybrid_chunking_operation():
    """
    Create a hybrid chunking operation that uses AST-based chunking for supported
    languages and falls back to regex-based chunking for others.
    """
    if cocoindex is None:
        raise ImportError("CocoIndex not available")
    
    @cocoindex.op.function()
    def HybridChunk(source_field: str, 
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
                chunks = chunker.chunk_code(code, lang, file_path)
                LOGGER.debug(f"AST chunking created {len(chunks)} chunks for {lang}")
                
                # Convert to CocoIndex format
                result = []
                for chunk in chunks:
                    chunk_record = {
                        "text": chunk["content"],
                        "location": f"{file_path}:{chunk['metadata'].get('start_line', 0)}",
                        "start": chunk["metadata"].get("start_line", 0),
                        "end": chunk["metadata"].get("end_line", 0),
                        "metadata": chunk["metadata"]
                    }
                    result.append(chunk_record)
                
                return result
            else:
                # Fall back to regex-based chunking
                try:
                    # Import locally to avoid circular dependency
                    # Use basic CocoIndex chunking if available
                    if cocoindex is not None:
                        split_func = cocoindex.functions.SplitRecursively()
                        LOGGER.debug(f"Using CocoIndex SplitRecursively for {lang} code")
                    else:
                        raise ImportError("CocoIndex not available")
                    
                    # Create a temporary record for the split function
                    temp_record = {source_field: code}
                    result = split_func(
                        temp_record,
                        language=lang,
                        chunk_size=chunk_size,
                        min_chunk_size=min_chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    return result
                except ImportError:
                    # If we can't import CUSTOM_LANGUAGES, fall back to simple text chunking
                    LOGGER.warning(f"Could not import CUSTOM_LANGUAGES, using simple text chunking for {lang}")
                    temp_chunker = CocoIndexASTChunker(max_chunk_size=chunk_size)
                    chunks = temp_chunker._simple_text_chunking(code, lang, file_path)
                    
                    # Convert to CocoIndex format
                    result = []
                    for chunk in chunks:
                        chunk_record = {
                            "text": chunk["content"],
                            "location": f"{file_path}:{chunk['metadata'].get('start_line', 0)}",
                            "start": chunk["metadata"].get("start_line", 0),
                            "end": chunk["metadata"].get("end_line", 0),
                            "metadata": chunk["metadata"]
                        }
                        result.append(chunk_record)
                    
                    return result
        
        return process_record
    
    return HybridChunk


if __name__ == "__main__":
    print("AST chunking module - use tests/test_ast_chunking_integration.py for testing")
