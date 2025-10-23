# Language-Aware Code Embeddings

CocoIndex now supports intelligent, language-aware code embeddings that automatically select the most suitable embedding model based on the programming language being processed.

## Overview

The new `CodeEmbedding` and `SmartCodeEmbedding` functions provide:

- **Automatic model selection** based on programming language
- **GraphCodeBERT** for languages it supports (Python, Java, JavaScript, PHP, Ruby, Go, C, C++)
- **UniXcode** for other supported languages (Rust, TypeScript, C#, Kotlin, Scala, Swift, Dart)
- **Fallback to general-purpose models** for unsupported languages
- **Seamless integration** with existing CocoIndex workflows

## Quick Start

### Basic Usage with SmartCodeEmbedding

```python
import cocoindex

@cocoindex.flow_def
def code_embedding_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    with flow_builder.read_files(data_scope.input_directory) as file:
        # Chunk the code using Tree-sitter
        file["chunks"] = file["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language=file["extension"],
            chunk_size=1000,
            chunk_overlap=200,
        )

        with file["chunks"].row() as chunk:
            # Smart embedding automatically detects language and selects best model
            chunk["embedding"] = chunk["text"].transform(
                cocoindex.functions.SmartCodeEmbedding(
                    file_extension=file["extension"]
                )
            )

        # Store in vector database
        file["chunks"].save(cocoindex.targets.QdrantTarget(
            collection_name="code_embeddings"
        ))
```

### Manual Language Specification

```python
# Explicitly specify the language
chunk["embedding"] = chunk["text"].transform(
    cocoindex.functions.CodeEmbedding(language="python")
)

# Force a specific model
chunk["embedding"] = chunk["text"].transform(
    cocoindex.functions.CodeEmbedding(
        language="rust",
        force_model="microsoft/unixcoder-base"
    )
)

# Custom fallback for unsupported languages
chunk["embedding"] = chunk["text"].transform(
    cocoindex.functions.CodeEmbedding(
        language="haskell",  # Not supported by GraphCodeBERT/UniXcode
        fallback_model="sentence-transformers/all-mpnet-base-v2"
    )
)
```

## Language Support

### GraphCodeBERT Languages

GraphCodeBERT is automatically selected for:

- Python (`.py`, `.pyi`)
- Java (`.java`)
- JavaScript (`.js`, `.mjs`, `.cjs`)
- PHP (`.php`)
- Ruby (`.rb`)
- Go (`.go`)
- C (`.c`, `.h`)
- C++ (`.cpp`, `.cxx`, `.cc`, `.hpp`)

### UniXcode Languages

UniXcode is automatically selected for:

- Rust (`.rs`)
- TypeScript (`.ts`, `.tsx`)
- C# (`.cs`)
- Kotlin (`.kt`, `.kts`)
- Scala (`.scala`)
- Swift (`.swift`)
- Dart (`.dart`)

### Fallback Languages

For unsupported languages (e.g., Haskell, OCaml, Erlang), the system falls back to general-purpose sentence transformers.

## API Reference

### CodeEmbedding

```python
@dataclasses.dataclass
class CodeEmbedding(op.FunctionSpec):
    language: str | None = None              # Programming language
    force_model: str | None = None           # Override model selection
    fallback_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_args: dict[str, Any] | None = None # Additional model arguments
```

### SmartCodeEmbedding

```python
@dataclasses.dataclass
class SmartCodeEmbedding(op.FunctionSpec):
    file_extension: str | None = None        # File extension for language detection
    language_hint: str | None = None         # Override auto-detection
    fallback_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_args: dict[str, Any] | None = None # Additional model arguments
```

## Advanced Usage

### Multi-Model Embeddings

Create multiple embeddings per code chunk for enhanced retrieval:

```python
with file["chunks"].row() as chunk:
    # General purpose embedding
    chunk["embedding_general"] = chunk["text"].transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # Code-specific embedding
    chunk["embedding_code"] = chunk["text"].transform(
        cocoindex.functions.SmartCodeEmbedding(
            file_extension=file["extension"]
        )
    )
```

### Custom Model Arguments

```python
# Configure model with custom parameters
chunk["embedding"] = chunk["text"].transform(
    cocoindex.functions.CodeEmbedding(
        language="python",
        model_args={
            "device": "cuda",
            "trust_remote_code": True,
            "batch_size": 32
        }
    )
)
```

### Language-Conditional Embedding

```python
def get_embedding_function(extension: str):
    """Select embedding function based on file extension."""
    if extension in [".py", ".js", ".java"]:
        # Use GraphCodeBERT for well-supported languages
        return cocoindex.functions.CodeEmbedding(
            language=extension[1:],  # Remove dot
            force_model="microsoft/graphcodebert-base"
        )
    elif extension in [".rs", ".ts"]:
        # Use UniXcode for these languages
        return cocoindex.functions.CodeEmbedding(
            language=extension[1:],
            force_model="microsoft/unixcoder-base"
        )
    else:
        # Use general-purpose for others
        return cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

# In your flow
embedding_func = get_embedding_function(file["extension"])
chunk["embedding"] = chunk["text"].transform(embedding_func)
```

## Performance Considerations

### Model Loading and Caching

- Models are automatically cached after first load
- GPU acceleration is enabled by default when available
- Use `model_args={"device": "cpu"}` to force CPU execution

### Memory Usage

- GraphCodeBERT: ~500MB VRAM
- UniXcode: ~1.2GB VRAM
- Consider using smaller models for memory-constrained environments:

```python
chunk["embedding"] = chunk["text"].transform(
    cocoindex.functions.CodeEmbedding(
        language="python",
        fallback_model="sentence-transformers/all-MiniLM-L6-v2"  # Smaller model
    )
)
```

### Batch Processing

For large codebases, consider processing files in language-specific batches:

```python
# Group files by language for efficient processing
python_files = [f for f in files if f.endswith('.py')]
rust_files = [f for f in files if f.endswith('.rs')]

# Process each group with optimized settings
for file_group, language in [(python_files, "python"), (rust_files, "rust")]:
    # Process with language-specific optimizations
    pass
```

## Migration from SentenceTransformerEmbed

Existing code using `SentenceTransformerEmbed` can be easily upgraded:

### Before

```python
chunk["embedding"] = chunk["text"].transform(
    cocoindex.functions.SentenceTransformerEmbed(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
)
```

### After

```python
chunk["embedding"] = chunk["text"].transform(
    cocoindex.functions.SmartCodeEmbedding(
        file_extension=file["extension"]
    )
)
```

## Installation

The new embedding functions are included in the `embeddings` optional dependency:

```bash
pip install 'cocoindex[embeddings]'
```

This includes:

- `sentence-transformers>=3.3.1`
- `transformers>=4.21.0`
- `torch>=1.12.0`

## Troubleshooting

### Model Download Issues

Models are downloaded automatically on first use. If you encounter download issues:

```python
# Pre-download models
from sentence_transformers import SentenceTransformer

# Download GraphCodeBERT
SentenceTransformer("microsoft/graphcodebert-base", trust_remote_code=True)

# Download UniXcode
SentenceTransformer("microsoft/unixcoder-base", trust_remote_code=True)
```

### Memory Issues

If you encounter CUDA out of memory errors:

```python
chunk["embedding"] = chunk["text"].transform(
    cocoindex.functions.CodeEmbedding(
        language="python",
        model_args={"device": "cpu"}  # Force CPU execution
    )
)
```

### Unsupported Languages

For languages not supported by GraphCodeBERT or UniXcode:

```python
chunk["embedding"] = chunk["text"].transform(
    cocoindex.functions.CodeEmbedding(
        language="haskell",
        fallback_model="sentence-transformers/all-mpnet-base-v2"  # Better fallback
    )
)
```

## Examples

See the `/examples` directory for complete working examples:

- `examples/code_embedding_basic.py` - Basic language-aware embedding
- `examples/code_embedding_advanced.py` - Multi-model embedding pipeline
- `examples/code_search_with_embeddings.py` - Code search using language-specific embeddings
