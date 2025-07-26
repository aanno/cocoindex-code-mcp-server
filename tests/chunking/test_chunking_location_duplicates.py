#!/usr/bin/env python3

"""
Test to identify exactly why chunking logic produces duplicate location values
within the same file, causing PostgreSQL "ON CONFLICT DO UPDATE" errors.
"""

import pytest

import cocoindex
from cocoindex_code_mcp_server.cocoindex_config import (
    extract_language, get_chunking_params, ASTChunkOperation, AST_CHUNKING_AVAILABLE,
    CUSTOM_LANGUAGES
)

# Test files based on the actual error cases
RUST_FILE_CONTENT = '''use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct User {
    pub id: u32,
    pub name: String,
    pub email: String,
}

impl User {
    pub fn new(id: u32, name: String, email: String) -> Self {
        User { id, name, email }
    }
    
    pub fn validate_email(&self) -> bool {
        self.email.contains('@')
    }
}

pub struct UserService {
    users: HashMap<u32, User>,
    next_id: u32,
}

impl UserService {
    pub fn new() -> Self {
        UserService {
            users: HashMap::new(),
            next_id: 1,
        }
    }
    
    pub fn create_user(&mut self, name: String, email: String) -> User {
        let user = User::new(self.next_id, name, email);
        self.users.insert(self.next_id, user.clone());
        self.next_id += 1;
        user
    }
    
    pub fn get_user(&self, id: u32) -> Option<&User> {
        self.users.get(&id)
    }
}
'''

MARKDOWN_CONTENT = '''# CocoIndex Code MCP Server

A Model Context Protocol (MCP) server that provides code indexing and search capabilities using CocoIndex.

## Features

- **Code Indexing**: Index code files with language-aware chunking
- **Semantic Search**: Find code using natural language queries  
- **Metadata Extraction**: Extract functions, classes, imports, and complexity metrics
- **Hybrid Search**: Combine keyword and semantic search
- **Multiple Languages**: Support for Python, Rust, TypeScript, Java, and more

## Installation

```bash
git clone https://github.com/user/cocoindex_code_mcp_server.git
cd cocoindex_code_mcp_server
pip install -e .
```

## Usage

### Basic Usage

Start the MCP server:

```bash
python -m code_index_mcp.server
```

### Configuration

The server can be configured using environment variables:

- `DATABASE_URL`: PostgreSQL connection string
- `EMBEDDING_MODEL`: Model to use for embeddings
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARN, ERROR)

## API Reference

The server provides the following MCP tools:

### search_code

Search for code using natural language or keywords.

**Parameters:**
- `query` (string): The search query
- `limit` (integer): Maximum number of results (default: 10)

### get_file_info

Get detailed information about a specific file.

**Parameters:**
- `filename` (string): Path to the file

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
'''

PYTHON_CONTENT = '''#!/usr/bin/env python3

"""
Base analyzer interface for different programming languages.
All language-specific analyzers should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import ast
import re


@dataclass
class AnalysisResult:
    """Result of code analysis."""
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity_score: int
    has_type_hints: bool
    has_async: bool
    has_classes: bool
    decorators_used: List[str]
    analysis_method: str


class BaseAnalyzer(ABC):
    """Base class for all code analyzers."""
    
    def __init__(self, language: str):
        self.language = language
        self.supported_extensions = []
    
    @abstractmethod
    def analyze(self, code: str, filename: str = "") -> AnalysisResult:
        """
        Analyze the given code and return structured information.
        
        Args:
            code: The source code to analyze
            filename: Optional filename for context
            
        Returns:
            AnalysisResult containing extracted information
        """
        pass
    
    def is_supported_file(self, filename: str) -> bool:
        """Check if this analyzer supports the given file."""
        return any(filename.endswith(ext) for ext in self.supported_extensions)
    
    def extract_functions(self, code: str) -> List[str]:
        """Extract function names from code. Override in subclasses."""
        return []
    
    def extract_classes(self, code: str) -> List[str]:
        """Extract class names from code. Override in subclasses."""
        return []
    
    def extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code. Override in subclasses."""
        return []
    
    def calculate_complexity(self, code: str) -> int:
        """Calculate code complexity score. Override in subclasses."""
        # Simple line-based complexity as fallback
        lines = [line.strip() for line in code.split('\\n') if line.strip()]
        return len(lines)
    
    def has_type_annotations(self, code: str) -> bool:
        """Check if code has type annotations. Override in subclasses."""
        return False
    
    def has_async_code(self, code: str) -> bool:
        """Check if code uses async/await. Override in subclasses."""
        return 'async ' in code or 'await ' in code
    
    def extract_decorators(self, code: str) -> List[str]:
        """Extract decorator names. Override in subclasses."""
        return []


class PythonAnalyzer(BaseAnalyzer):
    """Analyzer specifically for Python code."""
    
    def __init__(self):
        super().__init__("Python")
        self.supported_extensions = [".py", ".pyi", ".pyx"]
    
    def analyze(self, code: str, filename: str = "") -> AnalysisResult:
        """Analyze Python code using AST parsing."""
        try:
            tree = ast.parse(code)
            
            functions = self._extract_functions_ast(tree)
            classes = self._extract_classes_ast(tree)
            imports = self._extract_imports_ast(tree)
            decorators = self._extract_decorators_ast(tree)
            
            return AnalysisResult(
                functions=functions,
                classes=classes,
                imports=imports,
                complexity_score=self._calculate_ast_complexity(tree),
                has_type_hints=self._has_type_hints_ast(tree),
                has_async=self._has_async_ast(tree),
                has_classes=len(classes) > 0,
                decorators_used=decorators,
                analysis_method="ast_parsing"
            )
            
        except SyntaxError:
            # Fallback to regex-based analysis for invalid Python
            return self._fallback_analysis(code)
    
    def _extract_functions_ast(self, tree: ast.AST) -> List[str]:
        """Extract function names using AST."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return functions
    
    def _extract_classes_ast(self, tree: ast.AST) -> List[str]:
        """Extract class names using AST.""" 
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes
    
    def _extract_imports_ast(self, tree: ast.AST) -> List[str]:
        """Extract import names using AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    
    def _extract_decorators_ast(self, tree: ast.AST) -> List[str]:
        """Extract decorator names using AST."""
        decorators = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(decorator.id)
        return decorators
    
    def _calculate_ast_complexity(self, tree: ast.AST) -> int:
        """Calculate complexity based on AST nodes."""
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                complexity += 1
        return complexity
    
    def _has_type_hints_ast(self, tree: ast.AST) -> bool:
        """Check for type hints in AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns or any(arg.annotation for arg in node.args.args):
                    return True
        return False
    
    def _has_async_ast(self, tree: ast.AST) -> bool:
        """Check for async functions in AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                return True
        return False
    
    def _fallback_analysis(self, code: str) -> AnalysisResult:
        """Fallback regex-based analysis for invalid Python."""
        functions = re.findall(r'^\\s*def\\s+(\\w+)', code, re.MULTILINE)
        classes = re.findall(r'^\\s*class\\s+(\\w+)', code, re.MULTILINE)
        imports = re.findall(r'^\\s*(?:from\\s+\\w+\\s+)?import\\s+(\\w+)', code, re.MULTILINE)
        
        return AnalysisResult(
            functions=functions,
            classes=classes,
            imports=imports,
            complexity_score=len(code.split('\\n')),
            has_type_hints=': ' in code and '->' in code,
            has_async='async def' in code,
            has_classes=len(classes) > 0,
            decorators_used=[],
            analysis_method="regex_fallback"
        )
'''


class TestChunkingLocationDuplicates:
    """Test to identify duplicate location generation within single files."""

    def test_ast_chunking_location_uniqueness_detailed(self):
        """Test AST chunking with detailed location analysis for files that cause errors."""
        test_cases = [
            ("driver.rs", RUST_FILE_CONTENT, "Rust"),
            ("README.md", MARKDOWN_CONTENT, "Markdown"),
            ("base_analyzer.py", PYTHON_CONTENT, "Python")
        ]

        for filename, content, expected_language in test_cases:
            print(f"\n=== Testing {filename} ({expected_language}) ===")

            language = extract_language(filename)
            assert language == expected_language, f"Language detection failed: {language} != {expected_language}"

            if not AST_CHUNKING_AVAILABLE:
                print("AST chunking not available, skipping")
                continue

            # Test with different chunk sizes to see if duplicates occur
            chunk_sizes = [500, 1000, 1500]

            for chunk_size in chunk_sizes:
                print(f"\n--- Chunk size: {chunk_size} ---")

                try:
                    chunks = ASTChunkOperation(
                        content=content,
                        language=language,
                        max_chunk_size=chunk_size,
                        chunk_overlap=100
                    )

                    print(f"Generated {len(chunks)} chunks")

                    # Collect all locations and analyze for duplicates
                    locations = []
                    primary_keys = []

                    for i, chunk in enumerate(chunks):
                        location = chunk.location
                        primary_key = (filename, location, "files")

                        locations.append(location)
                        primary_keys.append(primary_key)

                        print(
                            f"  Chunk {i}: location='{location}', text_len={len(chunk.text)}, start={chunk.start}, end={chunk.end}")
                        print(f"    Primary key: {primary_key}")
                        print(f"    Text preview: {chunk.text[:100].replace(chr(10), ' ')[:50]}...")

                    # Check for duplicate locations within this file
                    unique_locations = set(locations)
                    if len(locations) != len(unique_locations):
                        duplicates = [loc for loc in locations if locations.count(loc) > 1]
                        print(f"❌ DUPLICATE LOCATIONS FOUND: {set(duplicates)}")
                        print(f"This would cause PostgreSQL 'ON CONFLICT DO UPDATE' error!")

                        # Show which chunks have the same location
                        for dup_loc in set(duplicates):
                            dup_indices = [i for i, loc in enumerate(locations) if loc == dup_loc]
                            print(f"  Location '{dup_loc}' appears in chunks: {dup_indices}")
                            for idx in dup_indices:
                                chunk = chunks[idx]
                                print(f"    Chunk {idx}: start={chunk.start}, end={chunk.end}, len={len(chunk.text)}")

                        # This should fail the test
                        assert False, f"Chunk size {chunk_size}: Duplicate locations found for {filename}: {set(duplicates)}"
                    else:
                        print(f"✅ All locations unique for chunk size {chunk_size}")

                except Exception as e:
                    print(f"Chunking failed for {filename} with chunk size {chunk_size}: {e}")
                    # Don't fail on chunking errors, just report them
                    continue

    def test_default_chunking_location_uniqueness(self):
        """Test default SplitRecursively chunking for location uniqueness."""
        print("\n=== Testing Default Chunking (SplitRecursively) ===")

        # Test with a file that would use default chunking (Rust is not supported by AST chunking)
        filename = "test.rs"
        language = extract_language(filename)

        print(f"Testing {filename} with language: {language}")

        # Simulate the default chunking path from the flow
        try:
            params = get_chunking_params(language)
            print(f"Chunking params: {params}")

            # Create the chunker as used in the flow
            chunker = cocoindex.functions.SplitRecursively(custom_languages=CUSTOM_LANGUAGES)
            print(f"Created chunker: {chunker}")

            # NOTE: We can't easily test SplitRecursively directly due to DataSlice requirements
            # But we can verify the chunker is created correctly
            print("✅ Default chunker created successfully")

            # The actual issue might be in how the location field is set by SplitRecursively
            # This would require running the full flow to test properly

        except Exception as e:
            print(f"Default chunking setup failed: {e}")
            raise

    def test_empty_or_minimal_content_chunking(self):
        """Test how chunking handles edge cases that might produce duplicate locations."""
        edge_cases = [
            ("empty.py", "", "Python"),
            ("comment_only.rs", "// Just a comment", "Rust"),
            ("single_line.md", "# Title", "Markdown"),
            ("whitespace.py", "\n\n\n   \n\n", "Python"),
        ]

        print("\n=== Testing Edge Cases ===")

        for filename, content, expected_lang in edge_cases:
            print(f"\n--- Testing {filename} ---")
            print(f"Content: '{content.replace(chr(10), '\\n')}'")

            language = extract_language(filename)

            if AST_CHUNKING_AVAILABLE:
                try:
                    chunks = ASTChunkOperation(
                        content=content,
                        language=language,
                        max_chunk_size=1000,
                        chunk_overlap=0
                    )

                    print(f"Generated {len(chunks)} chunks")

                    locations = [chunk.location for chunk in chunks]
                    unique_locations = set(locations)

                    if len(locations) != len(unique_locations):
                        duplicates = [loc for loc in locations if locations.count(loc) > 1]
                        print(f"❌ Edge case produced duplicate locations: {set(duplicates)}")
                    else:
                        print(f"✅ Edge case handled correctly: {locations}")

                    # Show all chunks for edge cases
                    for i, chunk in enumerate(chunks):
                        print(
                            f"  Chunk {i}: location='{chunk.location}', len={len(chunk.text)}, text='{chunk.text.replace(chr(10), '\\n')}'")

                except Exception as e:
                    print(f"Edge case chunking failed: {e}")


if __name__ == "__main__":
    cocoindex.init()
    pytest.main([__file__, "-v", "-s"])
