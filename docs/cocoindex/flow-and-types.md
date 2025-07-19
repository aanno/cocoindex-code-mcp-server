# CocoIndex Flow and Types: Gotchas and Best Practices

This document covers important gotchas, type system quirks, and best practices when working with CocoIndex flows and operations.

## Type System Gotchas

### 1. Unsupported Type Annotations

**❌ Problem**: CocoIndex function decorators don't support certain Python type annotations.

```python
from typing import List, Dict, Any

# THIS WILL FAIL
@cocoindex.op.function()
def my_function(text: str) -> List[Dict[str, Any]]:
    return [{"key": "value"}]
```

**Error Message**:

```text
ValueError: Unsupported as a specific type annotation for CocoIndex data type: typing.Any
```

**✅ Solution**: Remove problematic return type annotations:

```python
@cocoindex.op.function()
def my_function(text: str):  # No return type annotation
    return [{"key": "value"}]
```

### 2. inspect._empty Type Issues

**❌ Problem**: When no return type is specified, Python's `inspect` module returns `inspect._empty`, which CocoIndex doesn't recognize.

**Error Message**:

```text
ValueError: Unsupported as a specific type annotation for CocoIndex data type: <class 'inspect._empty'>
```

**✅ Solution**: Either provide a supported return type or remove the annotation entirely.

### 3. Supported Type Annotations

**✅ These work well**:

```python
@cocoindex.op.function()
def extract_extension(filename: str) -> str:
    return os.path.splitext(filename)[1]

@cocoindex.transform_flow()
def code_to_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[NDArray[np.float32]]:
    return text.transform(...)
```

**✅ CocoIndex-specific types**:

- `cocoindex.DataSlice[T]`
- `cocoindex.Json` (instead of `Dict[str, Any]`)
- Basic Python types: `str`, `int`, `float`, `bool`
- NumPy types: `NDArray[np.float32]`

## Decorator Gotchas

### 1. Incorrect Decorator Names

**❌ Old/Incorrect**:

```python
@cocoindex.operation  # This doesn't exist
def my_function():
    pass
```

**✅ Correct**:

```python
@cocoindex.op.function()  # Note the parentheses
def my_function():
    pass
```

### 2. Transform Flow vs Function

**For data transformations within flows**:

```python
@cocoindex.transform_flow()
def code_to_embedding(text: cocoindex.DataSlice[str]) -> cocoindex.DataSlice[NDArray[np.float32]]:
    return text.transform(cocoindex.functions.SentenceTransformerEmbed(...))
```

**For utility functions**:

```python
@cocoindex.op.function()
def extract_language(filename: str) -> str:
    return os.path.splitext(filename)[1]
```

## Flow Definition Gotchas

### 1. Setup Required

**❌ Problem**: Flows need to be set up before use.

**Error Message**:

```text
Exception: Setup for flow `CodeEmbedding` is not up-to-date. Please run `cocoindex setup` to update the setup.
```

**✅ Solution**: Run setup command:

```bash
cocoindex setup src/your_config.py
```

### 2. CocoIndex Initialization

**❌ Problem**: CocoIndex library not initialized.

**Error Message**:

```text
CocoIndex library is not initialized or already stopped
```

**✅ Solution**: Always initialize before using flows:

```python
import cocoindex
from dotenv import load_dotenv

load_dotenv()
cocoindex.init()

# Now you can use flows
stats = my_flow.update()
```

### 3. Environment Variables

**Required environment variables**:

```bash
COCOINDEX_DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

Load them properly:

```python
from dotenv import load_dotenv
load_dotenv()  # Load .env file
```

## Data Flow Gotchas

### 1. Custom Operations in Flows

**❌ Problem**: Custom operations might not work directly in flows.

```python
# This might fail
file["chunks"] = file["content"].transform(
    create_hybrid_chunking_operation(),  # Returns a function
    language=file["language"]
)
```

**✅ Solution**: Use built-in operations or properly structured custom operations:

```python
file["chunks"] = file["content"].transform(
    cocoindex.functions.SplitRecursively(),  # Built-in operation
    language=file["language"],
    chunk_size=1000
)
```

### 2. Field Access in Records

**❌ Wrong way to access fields**:

```python
def process_record(record):
    lang = record.get(language, "")  # Variable name, not string
```

**✅ Correct way**:

```python
def process_record(record):
    lang = record.get("language", "")  # String key
    file_path = record.get("filename", "")
```

### 3. Return Value Structure

**CocoIndex expects specific return structures**:

```python
@cocoindex.op.function()
def my_chunking_operation():
    def process_record(record):
        # Must return records with specific fields
        return [{
            "text": chunk_content,
            "location": f"{file_path}:{line_number}",
            "start": {"line": start_line, "column": 0},
            "end": {"line": end_line, "column": 0}
        }]
    return process_record
```

## Database Integration

### 1. Connection Pool Setup

**For search operations, you need a connection pool**:

```python
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
import os

pool = ConnectionPool(os.getenv("COCOINDEX_DATABASE_URL"))

# Register vector extension for each connection
with pool.connection() as conn:
    register_vector(conn)
```

### 2. Table Name Discovery

**Get the actual table name CocoIndex creates**:

```python
table_name = cocoindex.utils.get_target_default_name(
    your_flow, "export_target_name"
)
```

Example:

```python
table_name = cocoindex.utils.get_target_default_name(
    code_embedding_flow, "code_embeddings"
)
# Returns: "codeembedding__code_embeddings"
```

## Best Practices

### 1. Type Annotations

```python
# ✅ Good: Simple types or no annotation
@cocoindex.op.function()
def extract_extension(filename: str) -> str:
    return os.path.splitext(filename)[1]

@cocoindex.op.function()
def complex_operation(text: str):  # No return annotation
    return complex_result

# ✅ Good: CocoIndex-specific types
@cocoindex.transform_flow()
def embedding_transform(
    text: cocoindex.DataSlice[str]
) -> cocoindex.DataSlice[NDArray[np.float32]]:
    return text.transform(...)
```

### 2. Error Handling

```python
try:
    stats = flow.update()
    print(f"✅ Flow updated: {stats}")
except Exception as e:
    print(f"❌ Flow update failed: {e}")
    # Check initialization, setup, environment variables
```

### 3. Development Workflow

1. **Setup environment**:

   ```bash
   # Load environment
   source .env
   
   # Build Rust extensions (if modified)
   cd cocoindex && maturin develop
   ```

2. **Initialize and setup**:

   ```python
   import cocoindex
   from dotenv import load_dotenv
   
   load_dotenv()
   cocoindex.init()
   ```

3. **Run setup** (after flow changes):

   ```bash
   cocoindex setup src/your_config.py
   ```

4. **Test your flow**:

   ```python
   stats = your_flow.update()
   ```

### 4. Custom Language Support

```python
# ✅ Correct CustomLanguageSpec usage
custom_lang = cocoindex.functions.CustomLanguageSpec(
    language_name="Haskell",
    aliases=[".hs", ".lhs"],
    separators_regex=[
        r"\\nmodule\\s+[A-Z][a-zA-Z0-9_.']*",
        r"\\ndata\\s+[A-Z][a-zA-Z0-9_']*",
        # ... more patterns
    ]
    # Don't include unsupported parameters like chunk_config
)
```

## Common Error Messages and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Unsupported as a specific type annotation: typing.Any` | Using `typing.Any` in return types | Remove return type annotation |
| `module 'cocoindex' has no attribute 'operation'` | Using old decorator syntax | Use `@cocoindex.op.function()` |
| `Setup for flow is not up-to-date` | Flow not set up | Run `cocoindex setup src/config.py` |
| `CocoIndex library is not initialized` | Missing initialization | Call `cocoindex.init()` |
| `unexpected keyword argument 'chunk_config'` | Unsupported parameter | Remove unsupported parameters |

## Summary

- **Avoid `typing.Any`** in function signatures
- **Use `@cocoindex.op.function()`** not `@cocoindex.operation`
- **Always call `cocoindex.init()`** before using flows
- **Run setup** after flow definition changes
- **Use built-in operations** when possible
- **Check environment variables** are loaded
- **Use proper table name discovery** for database operations

Following these practices will help you avoid common pitfalls when working with CocoIndex's powerful but sometimes finicky type system and flow architecture.
