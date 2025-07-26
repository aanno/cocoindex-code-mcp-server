#!/usr/bin/env python3

"""
Simple test of ASTChunk functionality.
"""

from astchunk import ASTChunkBuilder


def test_astchunk_import():
    """Test that ASTChunk can be imported successfully."""
    assert ASTChunkBuilder is not None


def test_astchunk_builder_creation():
    """Test that ASTChunkBuilder can be created with configuration."""
    configs = {
        "max_chunk_size": 300,
        "language": "python",
        "metadata_template": "default",
        "chunk_expansion": False
    }

    builder = ASTChunkBuilder(**configs)
    assert builder is not None


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

if __name__ == "__main__":
    calc = Calculator()
    result = calc.add(5, 3)
    print(f"5 + 3 = {result}")
    hello_world()
    '''

    configs = {
        "max_chunk_size": 300,
        "language": "python",
        "metadata_template": "default",
        "chunk_expansion": False
    }

    builder = ASTChunkBuilder(**configs)
    chunks = builder.chunkify(python_code, **configs)

    assert len(chunks) > 0
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all('content' in chunk for chunk in chunks)
    assert all('metadata' in chunk for chunk in chunks)

    # Verify metadata contains expected fields
    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        assert isinstance(metadata, dict)
