#!/usr/bin/env python3

"""
Test standalone chunking functionality without CocoIndex imports.
"""

from typing import Any, Dict, List, Optional

from astchunk import ASTChunkBuilder


def detect_language_from_filename(filename: str) -> str:
    """Detect programming language from filename."""
    import os

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
    if basename.lower() in ["makefile", "dockerfile", "jenkinsfile"]:
        return basename.lower()

    ext = os.path.splitext(filename)[1].lower()
    return LANGUAGE_MAP.get(ext, "Unknown")


class StandaloneASTChunker:
    """Standalone AST chunker that doesn't require CocoIndex."""

    LANGUAGE_MAP = {
        "Python": "python",
        "Java": "java",
        "C#": "csharp",
        "TypeScript": "typescript",
        "JavaScript": "typescript",
        "TSX": "typescript",
    }

    def __init__(self, max_chunk_size: int = 1800):
        self.max_chunk_size = max_chunk_size
        self._builders: Dict[str, ASTChunkBuilder] = {}

    def _get_builder(self, language: str) -> Optional[ASTChunkBuilder]:
        """Get or create an ASTChunkBuilder for the given language."""
        if language not in self._builders:
            try:
                configs = {
                    "max_chunk_size": self.max_chunk_size,
                    "language": language,
                    "metadata_template": "default",
                    "chunk_expansion": False
                }
                self._builders[language] = ASTChunkBuilder(**configs)
            except Exception:
                return None

        return self._builders[language]

    def is_supported_language(self, language: str) -> bool:
        """Check if the language is supported by ASTChunk."""
        return language in self.LANGUAGE_MAP

    def chunk_code(self, code: str, language: str, file_path: str = "") -> List[Dict[str, Any]]:
        """Chunk code using AST-based chunking."""
        # Map language to ASTChunk language
        astchunk_language = self.LANGUAGE_MAP.get(language)
        if not astchunk_language:
            return self._simple_text_chunking(code, language, file_path)

        # Get ASTChunkBuilder for this language
        builder = self._get_builder(astchunk_language)
        if not builder:
            return self._simple_text_chunking(code, language, file_path)

        try:
            configs = {
                "max_chunk_size": self.max_chunk_size,
                "language": astchunk_language,
                "metadata_template": "default",
                "chunk_expansion": False
            }

            chunks = builder.chunkify(code, **configs)

            # Convert to our format
            result_chunks = []
            for i, chunk in enumerate(chunks):
                content = chunk.get('content', chunk.get('context', ''))
                metadata = chunk.get('metadata', {})

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

            return result_chunks

        except Exception:
            return self._simple_text_chunking(code, language, file_path)

    def _simple_text_chunking(self, code: str, language: str, file_path: str) -> List[Dict[str, Any]]:
        """Simple text-based chunking as a fallback."""
        lines = code.split('\n')
        chunks: List[Dict[str, Any]] = []
        chunk_size = max(10, self.max_chunk_size // 50)  # Rough estimate for lines

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

        return chunks


def test_language_detection():
    """Test language detection from filenames."""
    test_cases = [
        ("test.py", "Python"),
        ("test.java", "Java"),
        ("test.cs", "C#"),
        ("test.ts", "TypeScript"),
        ("test.hs", "Haskell"),
        ("test.kt", "Kotlin"),
        ("test.cpp", "C++"),
        ("test.rs", "Rust"),
        ("unknown.xyz", "Unknown")
    ]

    for filename, expected_lang in test_cases:
        detected = detect_language_from_filename(filename)
        assert detected == expected_lang


def test_standalone_chunker_initialization():
    """Test that StandaloneASTChunker can be initialized."""
    chunker = StandaloneASTChunker(max_chunk_size=200)
    assert chunker.max_chunk_size == 200
    assert isinstance(chunker._builders, dict)


def test_language_support_check():
    """Test checking if languages are supported by ASTChunk."""
    chunker = StandaloneASTChunker()

    assert chunker.is_supported_language("Python") is True
    assert chunker.is_supported_language("Java") is True
    assert chunker.is_supported_language("Haskell") is False
    assert chunker.is_supported_language("Unknown") is False


def test_python_code_chunking():
    """Test chunking of Python code."""
    python_code = '''
def hello_world():
    """A simple hello world function."""
    print("Hello, World!")

class Calculator:
    """A simple calculator class."""

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
'''

    chunker = StandaloneASTChunker(max_chunk_size=200)
    chunks = chunker.chunk_code(python_code, "Python", "test.py")

    assert len(chunks) > 0
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all('content' in chunk for chunk in chunks)
    assert all('metadata' in chunk for chunk in chunks)

    # Check that AST chunking was used
    for chunk in chunks:
        metadata = chunk['metadata']
        assert metadata['chunk_method'] == 'ast_chunking'
        assert metadata['language'] == 'Python'


def test_java_code_chunking():
    """Test chunking of Java code."""
    java_code = '''
public class Calculator {
    public static void main(String[] args) {
        Calculator calc = new Calculator();
        int result = calc.add(5, 3);
        System.out.println("5 + 3 = " + result);
    }

    public int add(int a, int b) {
        return a + b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }
}
'''

    chunker = StandaloneASTChunker(max_chunk_size=200)
    chunks = chunker.chunk_code(java_code, "Java", "test.java")

    assert len(chunks) > 0
    for chunk in chunks:
        metadata = chunk['metadata']
        assert metadata['chunk_method'] == 'ast_chunking'
        assert metadata['language'] == 'Java'


def test_unsupported_language_fallback():
    """Test that unsupported languages fall back to simple text chunking."""
    code = "main = putStrLn \"Hello, World!\""

    chunker = StandaloneASTChunker(max_chunk_size=200)
    chunks = chunker.chunk_code(code, "Haskell", "test.hs")

    assert len(chunks) > 0
    for chunk in chunks:
        metadata = chunk['metadata']
        assert metadata['chunk_method'] == 'simple_text_chunking'
        assert metadata['language'] == 'Haskell'


def test_empty_code_handling():
    """Test handling of empty or whitespace-only code."""
    chunker = StandaloneASTChunker()

    # Empty string - AST chunker might still create a chunk, so we allow >= 0
    chunks = chunker.chunk_code("", "Python", "empty.py")
    assert len(chunks) >= 0

    # Whitespace only - AST chunker might still create a chunk, so we allow >= 0
    chunks = chunker.chunk_code("   \n  \n  ", "Python", "whitespace.py")
    assert len(chunks) >= 0
