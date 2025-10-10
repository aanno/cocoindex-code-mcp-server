#!/usr/bin/env python3

"""
Enhanced Haskell-specific functionality for AST-based code chunking.
Incorporates techniques from ASTChunk for improved chunking quality.
Modernized to align with current ast_chunking.py patterns (January 2025).
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from warnings import deprecated

import haskell_tree_sitter

import cocoindex
from cocoindex import op

from . import LOGGER


@dataclass
class HaskellChunkRow:
    """CocoIndex table row for Haskell AST chunk data."""
    content: str
    location: str
    start: int
    end: int
    chunking_method: str
    
    def __getitem__(self, key: Union[str, int]) -> Any:
        """Allow dictionary-style access."""
        if isinstance(key, str):
            if hasattr(self, key):
                return getattr(self, key)
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
            raise KeyError(f"Cannot set unknown attribute '{key}' on HaskellChunkRow")
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in chunk (for 'key in chunk' syntax)."""
        return hasattr(self, key)
    
    def get(self, key: str, default=""):
        """Dictionary-style get method."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def keys(self):
        """Return available keys (attribute names)."""
        return ["content", "location", "start", "end", "chunking_method"]
    
    def to_dict(self) -> dict:
        """Convert chunk to dictionary for CocoIndex compatibility."""
        return {
            "content": self.content,
            "location": self.location,
            "start": self.start,
            "end": self.end,
            "chunking_method": self.chunking_method
        }


class ChunkRow:
    """Legacy chunk format for backward compatibility (internal use)."""
    def __init__(self, data: Dict[str, Any]):
        self.data = data
    
    def get(self, key: str, default=None):
        return self.data.get(key, default)
    
    def __getitem__(self, key: str):
        return self.data[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self.data


class HaskellChunkConfig:
    """Configuration for Haskell chunking with ASTChunk-inspired features."""

    def __init__(self,
                 max_chunk_size: int = 1800,
                 chunk_overlap: int = 0,
                 chunk_expansion: bool = False,
                 metadata_template: str = "default",
                 preserve_imports: bool = True,
                 preserve_exports: bool = True) -> None:
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_expansion = chunk_expansion
        self.metadata_template = metadata_template
        self.preserve_imports = preserve_imports
        self.preserve_exports = preserve_exports


def get_enhanced_haskell_separators() -> List[str]:
    """
    Get enhanced separators for Haskell that combine regex patterns with AST knowledge.
    This provides better chunking boundaries than pure regex.
    Inspired by ASTChunk's language-specific separator approach.
    """
    base_separators = haskell_tree_sitter.get_haskell_separators()

    # Add additional AST-aware separators with priority ordering
    enhanced_separators = base_separators + [
        # High priority: Module and import boundaries (should rarely be split)
        r"\nmodule\s+[A-Z][a-zA-Z0-9_.']*",
        r"\nimport\s+(qualified\s+)?[A-Z][a-zA-Z0-9_.']*",

        # Medium priority: Type and data definitions
        r"\ndata\s+[A-Z][a-zA-Z0-9_']*",
        r"\nnewtype\s+[A-Z][a-zA-Z0-9_']*",
        r"\ntype\s+[A-Z][a-zA-Z0-9_']*",
        r"\nclass\s+[A-Z][a-zA-Z0-9_']*",
        r"\ninstance\s+[A-Z][a-zA-Z0-9_']*",

        # Medium priority: Function definitions with type signatures
        r"\n[a-zA-Z][a-zA-Z0-9_']*\s*::",  # Type signatures
        r"\n[a-zA-Z][a-zA-Z0-9_']*.*\s*=",  # Function definitions

        # Lower priority: Block structures
        r"\nwhere\s*$",
        r"\nlet\s+",
        r"\nin\s+",
        r"\ndo\s*$",

        # Language pragmas (usually at file top, high priority)
        r"\n\{-#\s*[A-Z]+",

        # Comment blocks (can be good separation points)
        r"\n--\s*[=-]{3,}",  # Comment separators like "-- ==="
        r"\n\{-\s*[=-]{3,}",  # Block comment separators
    ]

    return enhanced_separators


@deprecated("Use HaskellChunkExecutor with @op.executor_class() pattern instead")
class EnhancedHaskellChunker:
    """
    Enhanced Haskell chunker inspired by ASTChunk techniques.
    Provides configurable chunking with rich metadata and multiple fallback strategies.

    Note: This class is deprecated but still used internally by HaskellChunkExecutor
    as a fallback implementation when Rust-based chunking fails.
    """

    def __init__(self, config: Optional[HaskellChunkConfig] = None) -> None:
        self.config = config or HaskellChunkConfig()
        self._cache: dict[str, Any] = {}  # Cache for expensive operations

    def chunk_code(self, content: str, file_path: str = "") -> List[Dict[str, Any]]:
        """
        Main chunking method with multiple strategies and rich metadata.
        """
        try:
            # Try AST-based chunking first
            chunks = self._ast_based_chunking(content, file_path)

            # Apply size optimization if needed
            chunks = self._optimize_chunk_sizes(chunks)

            # Add overlapping if configured
            if self.config.chunk_overlap > 0:
                chunks = self._add_chunk_overlap(chunks, content)

            # Apply chunk expansion if configured
            if self.config.chunk_expansion:
                chunks = self._expand_chunks_with_context(chunks, file_path)

            # Enhance metadata
            chunks = self._enhance_metadata(chunks, content, file_path)

            LOGGER.info(f"Successfully created {len(chunks)} Haskell chunks using AST method")
            return chunks

        except Exception as e:
            LOGGER.warning(f"AST chunking failed for Haskell code: {e}")
            return self._fallback_chunking(content, file_path)

    def _ast_based_chunking(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """AST-based chunking using tree-sitter with configurable parameters."""
        try:
            # Create chunking parameters from config
            params = haskell_tree_sitter.ChunkingParams(
                chunk_size=self.config.max_chunk_size,
                min_chunk_size=min(self.config.max_chunk_size // 4, 400),  # Conservative min size
                chunk_overlap=self.config.chunk_overlap,
                max_chunk_size=self.config.max_chunk_size
            )

            # Use the new parameterized AST chunking with recursive splitting
            chunking_result = haskell_tree_sitter.get_haskell_ast_chunks_with_params(content, params)
            ast_chunks = chunking_result.chunks()

        except Exception as e:
            # Fallback to the original method if parameterized version fails
            LOGGER.warning(f"Parameterized chunking failed, using fallback: {e}")
            ast_chunks = haskell_tree_sitter.get_haskell_ast_chunks_with_fallback(content)

        result = []
        for i, chunk in enumerate(ast_chunks):
            # Get original metadata from Rust including proper chunking method
            original_metadata = chunk.metadata()

            chunk_dict = {
                "content": chunk.text(),
                "start_line": chunk.start_line(),
                "end_line": chunk.end_line(),
                "start_byte": chunk.start_byte(),
                "end_byte": chunk.end_byte(),
                "node_type": chunk.node_type(),
                "chunk_id": i,
                "method": original_metadata.get("chunking_method", "haskell_ast_with_context"),
                "original_metadata": original_metadata,
            }
            result.append(chunk_dict)

        return result

    def _optimize_chunk_sizes(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize chunk sizes to stay within configured limits.
        Split large chunks and merge small ones where appropriate.
        """
        optimized = []

        for chunk in chunks:
            content = chunk["content"]
            content_size = len(content.replace(" ", "").replace("\n", "").replace("\t", ""))

            if content_size > self.config.max_chunk_size:
                # Split large chunks
                sub_chunks = self._split_large_chunk(chunk)
                optimized.extend(sub_chunks)
            else:
                optimized.append(chunk)

        return optimized

    def _split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a large chunk into smaller ones using enhanced separators."""
        content = chunk["content"]
        lines = content.split('\n')
        separators = get_enhanced_haskell_separators()

        # Find good split points
        split_points = [0]
        current_size = 0

        for i, line in enumerate(lines):
            line_size = len(line.replace(" ", "").replace("\t", ""))

            if current_size + line_size > self.config.max_chunk_size:
                # Look for a good separator near this point
                best_split = self._find_best_split_point(lines, i, separators)
                if best_split > split_points[-1]:
                    split_points.append(best_split)
                    current_size = 0
                else:
                    # Force split if no good point found
                    split_points.append(i)
                    current_size = line_size
            else:
                current_size += line_size

        if split_points[-1] < len(lines):
            split_points.append(len(lines))

        # Create sub-chunks
        sub_chunks = []
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]

            sub_lines = lines[start_idx:end_idx]
            sub_content = '\n'.join(sub_lines)

            if sub_content.strip():
                sub_chunk = chunk.copy()
                sub_chunk.update({
                    "content": sub_content,
                    "start_line": chunk["start_line"] + start_idx,
                    "end_line": chunk["start_line"] + end_idx - 1,
                    "chunk_id": f"{chunk['chunk_id']}.{i}",
                    "is_split": True,
                    "split_reason": "size_optimization"
                })
                sub_chunks.append(sub_chunk)

        return sub_chunks

    def _find_best_split_point(self, lines: List[str], target_idx: int, separators: List[str]) -> int:
        """Find the best split point near target_idx using separator patterns."""
        # Search in a window around target_idx
        search_window = min(10, len(lines) // 4)
        start_search = max(0, target_idx - search_window)
        end_search = min(len(lines), target_idx + search_window)

        best_score = -1
        best_idx = target_idx

        for i in range(start_search, end_search):
            line = lines[i]
            score = 0

            # Higher score for lines matching separators
            for j, separator in enumerate(separators):
                # Remove leading \n but handle special cases like \\n\\n+
                pattern = separator
                if pattern.startswith('\\n'):
                    pattern = pattern[2:]  # Remove \n
                    # Handle double newlines and other special cases
                    if pattern.startswith('\\n'):
                        if pattern == '\\n+':
                            pattern = '^$'  # Match empty lines
                        else:
                            pattern = pattern[2:] + '$'  # Make it end-of-line match for empty lines

                try:
                    if re.match(pattern, line):
                        # Earlier separators have higher priority
                        score = len(separators) - j
                        break
                except re.error:
                    # Skip invalid regex patterns
                    continue

            # Prefer splits closer to target
            distance_penalty = abs(i - target_idx) * 0.1
            final_score = score - distance_penalty

            if final_score > best_score:
                best_score = int(final_score)
                best_idx = i

        return best_idx

    def _add_chunk_overlap(self, chunks: List[Dict[str, Any]], content: str) -> List[Dict[str, Any]]:
        """Add overlapping context between chunks."""
        if len(chunks) <= 1:
            return chunks

        lines = content.split('\n')
        overlap_lines = self.config.chunk_overlap

        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced_chunk = chunk.copy()

            # Add lines from previous chunk
            if i > 0:
                prev_end = chunks[i - 1]["end_line"]
                overlap_start = max(0, prev_end - overlap_lines)
                prev_lines = lines[overlap_start:prev_end]

                if prev_lines:
                    overlap_content = '\n'.join(prev_lines)
                    enhanced_chunk["content"] = overlap_content + '\n' + chunk["content"]
                    enhanced_chunk["has_prev_overlap"] = True

            # Add lines from next chunk
            if i < len(chunks) - 1:
                next_start = chunks[i + 1]["start_line"]
                overlap_end = min(len(lines), next_start + overlap_lines)
                next_lines = lines[next_start:overlap_end]

                if next_lines:
                    overlap_content = '\n'.join(next_lines)
                    enhanced_chunk["content"] = chunk["content"] + '\n' + overlap_content
                    enhanced_chunk["has_next_overlap"] = True

            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def _expand_chunks_with_context(self, chunks: List[Dict[str, Any]], file_path: str) -> List[Dict[str, Any]]:
        """Add contextual headers to chunks (similar to ASTChunk expansion)."""
        expanded_chunks = []

        for chunk in chunks:
            content = chunk["content"]

            # Create context header
            header_parts = []
            if file_path:
                header_parts.append(f"File: {file_path}")

            header_parts.append(f"Lines: {chunk['start_line']}-{chunk['end_line']}")
            header_parts.append(f"Node type: {chunk.get('node_type', 'unknown')}")

            if chunk.get("method"):
                header_parts.append(f"Method: {chunk['method']}")

            header = "-- " + " | ".join(header_parts) + "\n"

            expanded_chunk = chunk.copy()
            expanded_chunk["content"] = header + content
            expanded_chunk["has_expansion_header"] = True

            expanded_chunks.append(expanded_chunk)

        return expanded_chunks

    def _enhance_metadata(self, chunks: List[Dict[str, Any]], content: str, file_path: str) -> List[Dict[str, Any]]:
        """Add rich metadata inspired by ASTChunk templates."""
        enhanced_chunks = []

        for chunk in chunks:
            metadata = {
                "chunk_id": chunk.get("chunk_id", 0),
                "chunk_method": chunk.get("method", "haskell_ast"),
                "language": "Haskell",
                "file_path": file_path,

                # Size metrics
                "chunk_size": len(chunk["content"]),
                "non_whitespace_size": len(chunk["content"].replace(" ", "").replace("\n", "").replace("\t", "")),
                "line_count": len(chunk["content"].split('\n')),

                # Location info
                "start_line": chunk.get("start_line", 0),
                "end_line": chunk.get("end_line", 0),
                "start_byte": chunk.get("start_byte", 0),
                "end_byte": chunk.get("end_byte", 0),

                # AST info
                "node_type": chunk.get("node_type", "unknown"),
                "is_split": chunk.get("is_split", False),

                # Chunking method tracking - preserve Rust chunking method names
                "chunking_method": chunk.get("original_metadata", {}).get("chunking_method", chunk.get("method", "haskell_ast_enhanced")),

                # Tree-sitter error tracking
                "tree_sitter_chunking_error": chunk.get("original_metadata", {}).get("tree_sitter_chunking_error", "false"),
                "has_tree_sitter_error": chunk.get("original_metadata", {}).get("has_error", "false") == "true",

                # Context info
                "has_imports": "import " in chunk["content"],
                "has_exports": "module " in chunk["content"] and "(" in chunk["content"],
                "has_type_signatures": "::" in chunk["content"],
                "has_data_types": any(keyword in chunk["content"] for keyword in ["data ", "newtype ", "type "]),
                "has_instances": "instance " in chunk["content"],
                "has_classes": "class " in chunk["content"],
            }

            # Template-specific metadata
            if self.config.metadata_template == "repoeval":
                metadata.update({
                    "functions": self._extract_function_names(chunk["content"]),
                    "types": self._extract_type_names(chunk["content"]),
                })
            elif self.config.metadata_template == "swebench":
                metadata.update({
                    "complexity_score": self._calculate_complexity(chunk["content"]),
                    "dependencies": self._extract_dependencies(chunk["content"]),
                })

            enhanced_chunk = {
                "content": chunk["content"],
                "metadata": metadata
            }
            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def _extract_function_names(self, content: str) -> List[str]:
        """Extract function names from Haskell code."""
        function_pattern = r'^([a-zA-Z][a-zA-Z0-9_\']*)\s*::'
        matches = re.findall(function_pattern, content, re.MULTILINE)
        return list(set(matches))

    def _extract_type_names(self, content: str) -> List[str]:
        """Extract type names from Haskell code."""
        type_patterns = [
            r'^data\s+([A-Z][a-zA-Z0-9_\']*)',
            r'^newtype\s+([A-Z][a-zA-Z0-9_\']*)',
            r'^type\s+([A-Z][a-zA-Z0-9_\']*)',
            r'^class\s+([A-Z][a-zA-Z0-9_\']*)',
        ]

        types = []
        for pattern in type_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            types.extend(matches)

        return list(set(types))

    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract import dependencies from Haskell code."""
        import_pattern = r'^import\s+(?:qualified\s+)?([A-Z][a-zA-Z0-9_\.]*)'
        matches = re.findall(import_pattern, content, re.MULTILINE)
        return list(set(matches))

    def _calculate_complexity(self, content: str) -> int:
        """Calculate a simple complexity score based on Haskell constructs."""
        complexity = 0

        # Count various constructs that add complexity
        complexity += content.count("case ")
        complexity += content.count("if ")
        complexity += content.count("where")
        complexity += content.count("let ")
        complexity += content.count("do")
        complexity += content.count(">>")  # Monadic operations
        complexity += content.count(">>=")
        complexity += len(re.findall(r'\$', content))  # Function application
        complexity += len(re.findall(r'::', content))  # Type signatures

        return complexity

    def _fallback_chunking(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Enhanced regex-based fallback chunking."""
        LOGGER.info("Using enhanced regex fallback for Haskell chunking")
        return create_enhanced_regex_fallback_chunks(content, file_path, self.config)


class CompatibleChunk:
    """Chunk wrapper that provides the interface expected by AST chunking code."""

    def __init__(self, content: str, metadata: dict, start_line: int = 1,
                 end_line: int = 1, node_type: str = "haskell_chunk") -> None:
        self._content = content
        self._metadata = metadata
        self._start_line = start_line
        self._end_line = end_line
        self._node_type = node_type

    def text(self) -> str:
        return self._content

    def start_line(self) -> int:
        return self._start_line

    def end_line(self) -> int:
        return self._end_line

    def node_type(self) -> str:
        return self._node_type

    def metadata(self) -> dict:
        return self._metadata


@deprecated("Use HaskellChunkExecutor with @op.executor_class() pattern instead")
def extract_haskell_ast_chunks(content: str) -> List[Dict[str, Any]]:
    """
    Enhanced AST-based Haskell chunking using the fixed Rust implementation.

    This function maintains backward compatibility for existing usage points and is
    still used internally by utility code (ast_chunking.py, cocoindex_config.py).

    Args:
        content: Haskell source code

    Returns:
        List of chunk dictionaries with proper rust_haskell_* chunking method names
    """
    try:
        # Call the fixed Rust function directly
        rust_chunks = haskell_tree_sitter.get_haskell_ast_chunks(content)

        # Convert from Rust HaskellChunk objects to legacy format for backward compatibility
        legacy_chunks = []
        for chunk in rust_chunks:
            metadata = chunk.metadata()
            legacy_chunk = {
                "text": chunk.text(),
                "start": chunk.start_line(),
                "end": chunk.end_line(),
                "location": f"{chunk.start_line()}:{chunk.end_line()}",
                "start_byte": chunk.start_byte(),
                "end_byte": chunk.end_byte(),
                "node_type": chunk.node_type(),
                "metadata": metadata,  # This now contains rust_haskell_* method names!
            }
            legacy_chunks.append(legacy_chunk)

        # Check if chunks have function metadata before claiming success
        chunks_with_functions = 0
        for legacy_chunk in legacy_chunks:
            chunk_metadata = legacy_chunk.get('metadata', {})
            if isinstance(chunk_metadata, dict) and 'function_name' in chunk_metadata:
                chunks_with_functions += 1

        if chunks_with_functions > 0:
            LOGGER.info(f"✅ Rust Haskell chunking produced {len(legacy_chunks)} chunks, {chunks_with_functions} with function metadata")
        else:
            LOGGER.warning(f"⚠️ Rust Haskell chunking produced {len(legacy_chunks)} chunks but NO function metadata - likely using regex fallback")
            if legacy_chunks:
                first_chunk = legacy_chunks[0]
                if 'metadata' in first_chunk and isinstance(first_chunk['metadata'], dict):
                    LOGGER.debug(f"Sample chunk metadata keys: {list(first_chunk['metadata'].keys())}")
                else:
                    LOGGER.debug("Sample chunk metadata keys: none")
        return legacy_chunks

    except Exception as e:
        LOGGER.error(f"❌ Rust Haskell chunking failed: {e}")
        LOGGER.info("⚠️ Falling back to enhanced Python chunking")

        # Fallback to the existing Python implementation if Rust fails
        chunk_config = HaskellChunkConfig()
        chunker = EnhancedHaskellChunker(chunk_config)
        chunks = chunker.chunk_code(content)

        # Convert to legacy CocoIndex format for backward compatibility
        legacy_chunks = []
        for chunk_dict in chunks:
            # Type cast to ensure mypy knows this is a dict
            chunk_data: dict = chunk_dict
            legacy_chunk = {
                "text": chunk_data["content"],
                "start": chunk_data["metadata"]["start_line"],
                "end": chunk_data["metadata"]["end_line"],
                "location": f"{chunk_data['metadata']['start_line']}:{chunk_data['metadata']['end_line']}",
                "start_byte": chunk_data["metadata"].get("start_byte", 0),
                "end_byte": chunk_data["metadata"].get("end_byte", len(chunk_data["content"].encode('utf-8'))),
                "node_type": chunk_data["metadata"].get("node_type", "haskell_chunk"),
                "metadata": {
                    "category": chunk_data["metadata"].get("node_type", "haskell_ast"),
                    "method": chunk_data["metadata"]["chunk_method"],
                    "chunking_method": "python_haskell_fallback",  # Mark as Python fallback
                    **chunk_data["metadata"]
                },
            }
            legacy_chunks.append(legacy_chunk)

        return legacy_chunks


def create_enhanced_regex_fallback_chunks(content: str, file_path: str,
                                          config: HaskellChunkConfig) -> List[Dict[str, Any]]:
    """
    Enhanced fallback chunking using regex patterns when AST parsing fails.
    Incorporates ASTChunk-inspired improvements for better chunk quality.
    """
    separators = get_enhanced_haskell_separators()
    lines = content.split('\n')
    chunks: List[Dict[str, Any]] = []

    current_start = 0
    current_size = 0

    for i, line in enumerate(lines):
        line_size = len(line.replace(" ", "").replace("\t", ""))
        is_separator = False
        separator_priority = 0

        # Check for separator patterns with priority
        for priority, separator in enumerate(separators):
            # Remove leading \n but handle special cases like \\n\\n+
            pattern = separator
            if pattern.startswith('\\n'):
                pattern = pattern[2:]  # Remove \n
                # Handle double newlines and other special cases
                if pattern.startswith('\\n'):
                    if pattern == '\\n+':
                        pattern = '^$'  # Match empty lines
                    else:
                        pattern = pattern[2:] + '$'  # Make it end-of-line match for empty lines

            try:
                if re.match(pattern, line):
                    is_separator = True
                    separator_priority = len(separators) - priority
                    break
            except re.error:
                # Skip invalid regex patterns
                continue

        # Force split if chunk gets too large
        force_split = current_size + line_size > config.max_chunk_size

        if (is_separator or force_split) and current_start < i:
            chunk_lines = lines[current_start:i]
            chunk_text = '\n'.join(chunk_lines)

            if chunk_text.strip():
                metadata = {
                    "chunk_id": len(chunks),
                    "chunk_method": "rust_haskell_regex_fallback_python",
                    "chunking_method": "rust_haskell_regex_fallback_python",
                    "language": "Haskell",
                    "file_path": file_path,
                    "chunk_size": len(chunk_text),
                    "non_whitespace_size": len(chunk_text.replace(" ", "").replace("\n", "").replace("\t", "")),
                    "line_count": len(chunk_lines),
                    "start_line": current_start + 1,
                    "end_line": i,
                    "separator_priority": separator_priority,
                    "was_force_split": force_split and not is_separator,

                    # Tree-sitter error tracking (false for regex fallback)
                    "tree_sitter_chunking_error": "false",
                    "has_tree_sitter_error": False,

                    # Haskell-specific content analysis
                    "has_imports": "import " in chunk_text,
                    "has_exports": "module " in chunk_text and "(" in chunk_text,
                    "has_type_signatures": "::" in chunk_text,
                    "has_data_types": any(keyword in chunk_text for keyword in ["data ", "newtype ", "type "]),
                    "has_instances": "instance " in chunk_text,
                    "has_classes": "class " in chunk_text,
                }

                chunk_dict = {
                    "content": chunk_text,
                    "metadata": metadata
                }
                chunks.append(chunk_dict)

            current_start = i
            current_size = line_size
        else:
            current_size += line_size

    # Handle the last chunk
    if current_start < len(lines):
        chunk_lines = lines[current_start:]
        chunk_text = '\n'.join(chunk_lines)

        if chunk_text.strip():
            metadata = {
                "chunk_id": len(chunks),
                "chunk_method": "rust_haskell_regex_fallback_python",
                "chunking_method": "rust_haskell_regex_fallback_python",
                "language": "Haskell",
                "file_path": file_path,
                "chunk_size": len(chunk_text),
                "non_whitespace_size": len(chunk_text.replace(" ", "").replace("\n", "").replace("\t", "")),
                "line_count": len(chunk_lines),
                "start_line": current_start + 1,
                "end_line": len(lines),
                "separator_priority": 0,
                "was_force_split": False,
                "is_final_chunk": True,

                # Tree-sitter error tracking (false for regex fallback)
                "tree_sitter_chunking_error": "false",
                "has_tree_sitter_error": False,

                # Haskell-specific content analysis
                "has_imports": "import " in chunk_text,
                "has_exports": "module " in chunk_text and "(" in chunk_text,
                "has_type_signatures": "::" in chunk_text,
                "has_data_types": any(keyword in chunk_text for keyword in ["data ", "newtype ", "type "]),
                "has_instances": "instance " in chunk_text,
                "has_classes": "class " in chunk_text,
            }

            chunk_dict = {
                "content": chunk_text,
                "metadata": metadata
            }
            chunks.append(chunk_dict)

    LOGGER.info(f"Enhanced regex fallback created {len(chunks)} Haskell chunks")
    return chunks


def create_regex_fallback_chunks_python(content: str) -> List[Dict[str, Any]]:
    """
    Legacy fallback function for backward compatibility.
    Redirects to enhanced version with default config.
    """
    config = HaskellChunkConfig()
    enhanced_chunks = create_enhanced_regex_fallback_chunks(content, "", config)

    # Convert to legacy format for compatibility
    legacy_chunks = []
    for chunk in enhanced_chunks:
        legacy_chunk = {
            "text": chunk["content"],
            "start": chunk["metadata"]["start_line"],
            "end": chunk["metadata"]["end_line"],
            "location": f"{chunk['metadata']['start_line']}:{chunk['metadata']['end_line']}",
            "start_byte": 0,
            "end_byte": len(chunk["content"].encode('utf-8')),
            "node_type": "regex_chunk",
            "metadata": {
                "category": "regex_fallback",
                "method": "regex",
                **chunk["metadata"]
            },
        }
        legacy_chunks.append(legacy_chunk)

    return legacy_chunks


def get_haskell_language_spec(config: Optional[HaskellChunkConfig] = None) -> cocoindex.functions.CustomLanguageSpec:
    """
    Get the enhanced Haskell language specification for CocoIndex.

    Args:
        config: Optional configuration for Haskell chunking

    Returns:
        Enhanced CustomLanguageSpec with ASTChunk-inspired features
    """
    if config is None:
        config = HaskellChunkConfig()

    return cocoindex.functions.CustomLanguageSpec(
        language_name="Haskell",
        aliases=[".hs", ".lhs"],
        separators_regex=get_enhanced_haskell_separators()
    )


class HaskellChunkSpec(op.FunctionSpec):
    """Haskell chunking function spec for CocoIndex."""
    max_chunk_size: int = 1800
    chunk_overlap: int = 0
    chunk_expansion: bool = False
    metadata_template: str = "default"
    preserve_imports: bool = True
    preserve_exports: bool = True


@op.executor_class()
class HaskellChunkExecutor:
    """Executor for Haskell AST-based code chunking."""

    spec: HaskellChunkSpec

    def analyze(self, content: Any, language: Any = "Haskell") -> type:
        """Analyze method required by CocoIndex to determine return type."""
        return list[HaskellChunkRow]

    def _convert_chunks_to_haskell_chunk_rows(self, chunks: List[Dict[str, Any]]) -> list[HaskellChunkRow]:
        """Convert internal chunk dictionaries to HaskellChunkRow dataclass instances with unique locations."""
        seen_locations = set()
        result = []

        for i, chunk_dict in enumerate(chunks):
            # Handle both enhanced chunker format and direct Rust format
            if "content" in chunk_dict:
                # Direct format from enhanced chunker
                content = chunk_dict["content"]
                metadata = chunk_dict.get("metadata", {})
                start_line = metadata.get("start_line", 0)
                end_line = metadata.get("end_line", 0)
                chunking_method = metadata.get("chunking_method", "haskell_ast_enhanced")
            else:
                # Rust format from extract_haskell_ast_chunks
                content = chunk_dict.get("text", "")
                metadata = chunk_dict.get("metadata", {})
                start_line = chunk_dict.get("start", 0)
                end_line = chunk_dict.get("end", 0)
                chunking_method = metadata.get("chunking_method", "rust_haskell_ast")

            # Ensure unique location
            base_location = f"line:{start_line}#{i}"
            unique_location = base_location
            suffix = 0
            while unique_location in seen_locations:
                suffix += 1
                unique_location = f"{base_location}_dup{suffix}"
            seen_locations.add(unique_location)

            result.append(HaskellChunkRow(
                content=content,
                location=unique_location,
                start=start_line,
                end=end_line,
                chunking_method=chunking_method
            ))

        return result

    def __call__(self, content: str, language: str = "Haskell") -> list[HaskellChunkRow]:
        """Main Haskell chunking function - returns typed chunk structures for CocoIndex."""
        LOGGER.info(f"🚀 HaskellChunkExecutor called with language={language}, content_length={len(content)}")

        try:
            # Try Rust-based Haskell chunking first
            chunks = extract_haskell_ast_chunks(content)
            # More accurate logging about chunk content
            LOGGER.info(f"✅ Rust Haskell chunking produced {len(chunks)} chunks (via HaskellChunkExecutor)")
            return self._convert_chunks_to_haskell_chunk_rows(chunks)
            
        except Exception as e:
            LOGGER.warning(f"Rust Haskell chunking failed: {e}, falling back to enhanced Python chunking")
            
            # Fallback to enhanced Python chunker
            config = HaskellChunkConfig(
                max_chunk_size=self.spec.max_chunk_size,
                chunk_overlap=self.spec.chunk_overlap,
                chunk_expansion=self.spec.chunk_expansion,
                metadata_template=self.spec.metadata_template,
                preserve_imports=self.spec.preserve_imports,
                preserve_exports=self.spec.preserve_exports
            )
            
            chunker = EnhancedHaskellChunker(config)
            enhanced_chunks = chunker.chunk_code(content, "")
            
            LOGGER.info(f"✅ Enhanced Python Haskell chunking produced {len(enhanced_chunks)} chunks")
            return self._convert_chunks_to_haskell_chunk_rows(enhanced_chunks)


# Create the operation
HaskellChunkOperation = HaskellChunkSpec()


if __name__ == "__main__":
    print("Enhanced Haskell support module - use tests/test_haskell_ast_chunker_integration.py for testing")
