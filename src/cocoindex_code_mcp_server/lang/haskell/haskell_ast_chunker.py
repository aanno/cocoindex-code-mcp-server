#!/usr/bin/env python3

"""
Enhanced Haskell-specific functionality for AST-based code chunking.
Incorporates techniques from ASTChunk for improved chunking quality.
"""

import re
from typing import Any, Dict, List, Optional

import haskell_tree_sitter

import cocoindex

from . import LOGGER


class HaskellChunkConfig:
    """Configuration for Haskell chunking with ASTChunk-inspired features."""

    def __init__(self,
                 max_chunk_size: int = 1800,
                 chunk_overlap: int = 0,
                 chunk_expansion: bool = False,
                 metadata_template: str = "default",
                 preserve_imports: bool = True,
                 preserve_exports: bool = True):
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


class EnhancedHaskellChunker:
    """
    Enhanced Haskell chunker inspired by ASTChunk techniques.
    Provides configurable chunking with rich metadata and multiple fallback strategies.
    """

    def __init__(self, config: Optional[HaskellChunkConfig] = None):
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
        """AST-based chunking using tree-sitter."""
        # Use the tree-sitter AST chunking
        ast_chunks = haskell_tree_sitter.get_haskell_ast_chunks_with_fallback(content)

        result = []
        for i, chunk in enumerate(ast_chunks):
            chunk_dict = {
                "content": chunk.text(),
                "start_line": chunk.start_line(),
                "end_line": chunk.end_line(),
                "start_byte": chunk.start_byte(),
                "end_byte": chunk.end_byte(),
                "node_type": chunk.node_type(),
                "chunk_id": i,
                "method": "haskell_ast",
                "original_metadata": chunk.metadata(),
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
    
    def __init__(self, content: str, metadata: dict, start_line: int = 1, end_line: int = 1, node_type: str = "haskell_chunk"):
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


@cocoindex.op.function()
def extract_haskell_ast_chunks(content: str):
    """
    Enhanced AST-based Haskell chunking with default configuration.

    Args:
        content: Haskell source code

    Returns:
        List of chunk dictionaries with enhanced metadata
    """
    # Use default configuration
    chunk_config = HaskellChunkConfig()

    chunker = EnhancedHaskellChunker(chunk_config)
    chunks = chunker.chunk_code(content)

    # Convert to legacy CocoIndex format for backward compatibility
    legacy_chunks = []
    for chunk in chunks:
        legacy_chunk = {
            "text": chunk["content"],
            "start": chunk["metadata"]["start_line"],
            "end": chunk["metadata"]["end_line"],
            "location": f"{chunk['metadata']['start_line']}:{chunk['metadata']['end_line']}",
            "start_byte": chunk["metadata"].get("start_byte", 0),
            "end_byte": chunk["metadata"].get("end_byte", len(chunk["content"].encode('utf-8'))),
            "node_type": chunk["metadata"].get("node_type", "haskell_chunk"),
            "metadata": {
                "category": chunk["metadata"].get("node_type", "haskell_ast"),
                "method": chunk["metadata"]["chunk_method"],
                **chunk["metadata"]
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
                    "chunk_method": "enhanced_regex_fallback",
                    "language": "Haskell",
                    "file_path": file_path,
                    "chunk_size": len(chunk_text),
                    "non_whitespace_size": len(chunk_text.replace(" ", "").replace("\n", "").replace("\t", "")),
                    "line_count": len(chunk_lines),
                    "start_line": current_start + 1,
                    "end_line": i,
                    "separator_priority": separator_priority,
                    "was_force_split": force_split and not is_separator,

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
                "chunk_method": "enhanced_regex_fallback",
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


def create_enhanced_haskell_chunking_operation():
    """
    Create a CocoIndex operation for enhanced Haskell chunking.
    Provides a high-level interface similar to ASTChunk operations.
    """
    @cocoindex.op.function()
    def EnhancedHaskellChunk(source_field: str,
                             max_chunk_size: int = 1800,
                             chunk_overlap: int = 0,
                             chunk_expansion: bool = False,
                             metadata_template: str = "default",
                             preserve_imports: bool = True,
                             preserve_exports: bool = True):
        """
        Enhanced Haskell chunking with ASTChunk-inspired features.

        Args:
            source_field: Field containing the Haskell source code
            max_chunk_size: Maximum non-whitespace characters per chunk
            chunk_overlap: Number of lines to overlap between chunks
            chunk_expansion: Whether to add contextual headers to chunks
            metadata_template: Format for chunk metadata (default, repoeval, swebench)
            preserve_imports: Try to keep imports together
            preserve_exports: Try to keep module exports together

        Returns:
            List of chunks with enhanced metadata
        """
        def process_record(record):
            code = record.get(source_field, "")
            file_path = record.get("file_path", "")

            # Create configuration
            config = HaskellChunkConfig(
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_expansion=chunk_expansion,
                metadata_template=metadata_template,
                preserve_imports=preserve_imports,
                preserve_exports=preserve_exports
            )

            # Create chunker and process
            chunker = EnhancedHaskellChunker(config)
            chunks = chunker.chunk_code(code, file_path)

            # Return chunks as separate records
            result = []
            for chunk in chunks:
                new_record = record.copy()
                new_record.update(chunk)
                result.append(new_record)

            return result

        return process_record

    return EnhancedHaskellChunk


if __name__ == "__main__":
    print("Enhanced Haskell support module - use tests/test_haskell_ast_chunker_integration.py for testing")
