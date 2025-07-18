#!/usr/bin/env python3

"""
Haskell-specific functionality for AST-based code chunking.
"""

from typing import List, Dict, Any
import haskell_tree_sitter
import cocoindex


def get_enhanced_haskell_separators() -> List[str]:
    """
    Get enhanced separators for Haskell that combine regex patterns with AST knowledge.
    This provides better chunking boundaries than pure regex.
    """
    base_separators = haskell_tree_sitter.get_haskell_separators()
    
    # Add additional AST-aware separators
    enhanced_separators = base_separators + [
        # More precise function definitions
        r"\n[a-zA-Z][a-zA-Z0-9_']*\s*::",  # Type signatures
        r"\n[a-zA-Z][a-zA-Z0-9_']*.*\s*=",  # Function definitions  
        # Class and instance boundaries
        r"\nclass\s+[A-Z][a-zA-Z0-9_']*",
        r"\ninstance\s+[A-Z][a-zA-Z0-9_']*",
        # Data type definitions
        r"\ndata\s+[A-Z][a-zA-Z0-9_']*",
        r"\nnewtype\s+[A-Z][a-zA-Z0-9_']*",
        r"\ntype\s+[A-Z][a-zA-Z0-9_']*",
        # Module boundaries
        r"\nmodule\s+[A-Z][a-zA-Z0-9_.']*",
        r"\nimport\s+(qualified\s+)?[A-Z][a-zA-Z0-9_.']*",
        # Block structures
        r"\nwhere\s*$",
        r"\nlet\s+",
        r"\nin\s+",
        # Pragmas and language extensions
        r"\n{-#\s*[A-Z]+",
    ]
    
    return enhanced_separators


@cocoindex.op.function()
def extract_haskell_ast_chunks(content: str) -> List[Dict[str, Any]]:
    """
    Extract AST-based chunks from Haskell code using tree-sitter.
    Returns a list of chunk dictionaries with enhanced metadata.
    """
    try:
        # Use the new AST-based chunking function with fallback
        chunks = haskell_tree_sitter.get_haskell_ast_chunks_with_fallback(content)
        
        # Convert HaskellChunk objects to dictionaries for CocoIndex
        result = []
        for chunk in chunks:
            chunk_dict = {
                "text": chunk.text(),
                "start": {"line": chunk.start_line(), "column": 0},
                "end": {"line": chunk.end_line(), "column": 0},
                "location": f"{chunk.start_line()}:{chunk.end_line()}",
                "start_byte": chunk.start_byte(),
                "end_byte": chunk.end_byte(),
                "node_type": chunk.node_type(),
                "metadata": chunk.metadata(),
            }
            result.append(chunk_dict)
        
        return result
    
    except Exception as e:
        # Fallback to regex-based chunking if AST parsing fails
        print(f"AST chunking failed for Haskell code: {e}")
        return create_regex_fallback_chunks_python(content)


def create_regex_fallback_chunks_python(content: str) -> List[Dict[str, Any]]:
    """
    Fallback chunking using regex patterns when AST parsing fails.
    """
    separators = haskell_tree_sitter.get_haskell_separators()
    lines = content.split('\n')
    chunks = []
    
    current_start = 0
    
    for i, line in enumerate(lines):
        is_separator = False
        
        for separator in separators:
            pattern = separator.lstrip('\\n')
            if line.startswith(pattern.replace('\\s+', ' ').replace('\\w+', '')):
                is_separator = True
                break
        
        if is_separator and current_start < i:
            chunk_lines = lines[current_start:i]
            chunk_text = '\n'.join(chunk_lines)
            
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "start": {"line": current_start, "column": 0},
                    "end": {"line": i, "column": 0},
                    "location": f"{current_start}:{i}",
                    "start_byte": 0,
                    "end_byte": len(chunk_text.encode('utf-8')),
                    "node_type": "regex_chunk",
                    "metadata": {
                        "category": "regex_fallback",
                        "method": "regex",
                    },
                })
            current_start = i
    
    # Handle the last chunk
    if current_start < len(lines):
        chunk_lines = lines[current_start:]
        chunk_text = '\n'.join(chunk_lines)
        
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "start": {"line": current_start, "column": 0},
                "end": {"line": len(lines), "column": 0},
                "location": f"{current_start}:{len(lines)}",
                "start_byte": 0,
                "end_byte": len(chunk_text.encode('utf-8')),
                "node_type": "regex_chunk",
                "metadata": {
                    "category": "regex_fallback",
                    "method": "regex",
                },
            })
    
    return chunks


def get_haskell_language_spec() -> cocoindex.functions.CustomLanguageSpec:
    """Get the Haskell language specification for CocoIndex."""
    return cocoindex.functions.CustomLanguageSpec(
        language_name="Haskell",
        aliases=[".hs", ".lhs"],
        separators_regex=get_enhanced_haskell_separators()
    )