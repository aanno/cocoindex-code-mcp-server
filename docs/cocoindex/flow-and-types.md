# CocoIndex Flow and Types: Gotchas and Best Practices

This document covers important gotchas, type system quirks, and best practices when working with CocoIndex flows and operations.

## Type System Gotchas

### 1. Vector Types Are Required (Not Optional!)

**✅ CORRECT**: CocoIndex requires proper Vector type annotations for embeddings.

```python
from cocoindex.typing import Vector
from typing import Literal
import numpy as np

@cocoindex.op.function()
def embed_text(text: str) -> Vector[np.float32, Literal[384]]:
    """Must use Vector type with dimension for pgvector compatibility."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode(text)
    return embedding.astype(np.float32)  # Return numpy array, not .tolist()
```

**❌ WRONG**: Using generic types or lists causes pgvector issues.

```python
# These cause "operator class vector_cosine_ops does not accept data type jsonb"
def embed_text(text: str) -> NDArray[np.float32]:  # Too generic
    return embedding.tolist()  # Python list gets stored as JSONB

def embed_text(text: str) -> list[float]:  # Also becomes JSONB
    return embedding.tolist()
```

### 2. Return Type Annotations ARE Required for Complex Types

**❌ OUTDATED ADVICE**: The old advice to "remove return type annotations" is wrong for modern CocoIndex.

**✅ CURRENT PRACTICE**: CocoIndex requires specific type annotations for:

```python
# Vector types (essential for embeddings)
@cocoindex.op.function()
def create_embedding(text: str) -> Vector[np.float32, Literal[384]]:
    return embedding.astype(np.float32)

# Transform flows (still required)
@cocoindex.transform_flow()
def code_to_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[NDArray[np.float32]]:
    return text.transform(...)

# Simple types (optional but recommended)
@cocoindex.op.function()
def extract_extension(filename: str) -> str:
    return os.path.splitext(filename)[1]
```

### 3. Supported Type Annotations (Updated)

**✅ Current supported types**:

```python
from cocoindex.typing import Vector
from typing import Literal
import numpy as np

# CocoIndex Vector types (REQUIRED for embeddings)
Vector[np.float32]                           # Dynamic dimension
Vector[np.float32, Literal[384]]            # Fixed dimension (preferred for pgvector)
Vector[cocoindex.Float32, Literal[128]]     # Using CocoIndex float types

# CocoIndex flow types
cocoindex.DataSlice[str]
cocoindex.DataSlice[NDArray[np.float32]]

# Basic Python types
str, int, float, bool, bytes

# Date/time types
datetime.datetime, datetime.date, uuid.UUID

# NumPy types (but Vector is preferred for embeddings)
NDArray[np.float32], NDArray[np.int64]
```

**❌ Still unsupported**:
- `typing.Any`
- Generic `List`, `Dict` without type parameters
- Complex unions with incompatible types

## Database Integration (Updated)

### 1. Vector Storage and pgvector

**✅ CRITICAL**: For PostgreSQL with pgvector, use fixed-dimension Vector types:

```python
@cocoindex.op.function()
def embed_code(code: str) -> Vector[np.float32, Literal[384]]:
    """Fixed dimension required for pgvector indexes."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode(code)
    return embedding.astype(np.float32)  # NOT .tolist()
```

This creates proper PostgreSQL schema:
```sql
-- ✅ Correct schema
embedding vector(384)  -- Supports vector indexes

-- ❌ Wrong schema (from old approach)
embedding jsonb        -- Cannot use vector indexes
```

### 2. Vector Index Configuration

```python
code_embeddings.export(
    "code_embeddings",
    cocoindex.targets.Postgres(),
    primary_key_fields=["filename", "location"],
    vector_indexes=[
        cocoindex.VectorIndexDef(
            field_name="embedding",
            metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
        )
    ],
)
```

## Custom Metadata Fields (IMPORTANT)

### 1. Collecting Custom Metadata

**✅ CocoIndex supports rich custom metadata collection**:

```python
code_embeddings.collect(
    filename=file["filename"],
    language=file["language"],
    location=chunk["location"],
    code=chunk["text"],
    embedding=chunk["embedding"],
    # Custom metadata fields
    metadata_json=chunk["metadata"],
    functions=str(metadata_dict.get("functions", [])),
    classes=str(metadata_dict.get("classes", [])),
    imports=str(metadata_dict.get("imports", [])),
    complexity_score=metadata_dict.get("complexity_score", 0),
    has_type_hints=metadata_dict.get("has_type_hints", False),
    has_async=metadata_dict.get("has_async", False),
    has_classes=metadata_dict.get("has_classes", False)
)
```

These fields should appear in both database schema and evaluation outputs.

## Flow Definition Gotchas (Still Relevant)

### 1. Setup Required
Still true - always run setup after schema changes:

```bash
cocoindex setup src/your_config.py
```

### 2. Model Loading Best Practices

**✅ CURRENT APPROACH**: Load models at module level to avoid repeated loading:

```python
from sentence_transformers import SentenceTransformer

# Load once at import time
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@cocoindex.op.function()
def embed_text(text: str) -> Vector[np.float32, Literal[384]]:
    """Use global model instance."""
    embedding = model.encode(text)
    return embedding.astype(np.float32)
```

**❌ AVOID**: Loading models inside functions (causes repeated loading):

```python
@cocoindex.op.function()
def embed_text(text: str) -> Vector[np.float32, Literal[384]]:
    # BAD: Loads model every time
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode(text)
    return embedding.astype(np.float32)
```

## Common Error Messages and Solutions (Updated)

| Error | Cause | Solution |
|-------|-------|----------|
| `operator class "vector_cosine_ops" does not accept data type jsonb` | Using Python lists instead of Vector types | Use `Vector[np.float32, Literal[dim]]` and return numpy arrays |
| `NameError: name 'lang' is not defined` | Variable name mismatch in function | Check function parameter names match usage |
| `Unsupported as a specific type annotation: typing.Any` | Using `typing.Any` in return types | Remove or use specific types |
| `Setup for flow is not up-to-date` | Flow not set up | Run `cocoindex setup src/config.py` |
| `CocoIndex library is not initialized` | Missing initialization | Call `cocoindex.init()` |

## Best Practices (Updated)

### 1. Vector Types for Embeddings

```python
# ✅ ALWAYS use Vector types for embeddings
from cocoindex.typing import Vector
from typing import Literal

@cocoindex.op.function()
def embed_text(text: str) -> Vector[np.float32, Literal[384]]:
    embedding = model.encode(text)
    return embedding.astype(np.float32)
```

### 2. Custom Metadata Collection

```python
# ✅ Collect rich metadata for better search
code_embeddings.collect(
    # Standard fields
    filename=file["filename"],
    code=chunk["text"], 
    embedding=chunk["embedding"],
    # Custom metadata that should appear in exports
    functions=extract_functions(chunk["text"]),
    classes=extract_classes(chunk["text"]),
    complexity_score=calculate_complexity(chunk["text"]),
    has_type_hints=check_type_hints(chunk["text"])
)
```

### 3. Error Handling with Better Debugging

```python
try:
    stats = flow.update()
    print(f"✅ Flow updated: {stats}")
except Exception as e:
    print(f"❌ Flow update failed: {e}")
    # Check:
    # 1. Vector type annotations correct?
    # 2. Model loading working?
    # 3. Custom metadata fields properly defined?
    # 4. Database schema up to date?
```

## Summary

- **USE Vector types** with fixed dimensions for embeddings: `Vector[np.float32, Literal[384]]`
- **Return numpy arrays** (`.astype(np.float32)`), NOT Python lists (`.tolist()`)
- **Type annotations ARE required** for Vector and complex types
- **Custom metadata fields** should appear in evaluation outputs and database
- **Load models once** at module level, not inside functions
- **Always run setup** after changing collection schema or vector dimensions

Following these updated practices will ensure proper pgvector integration and rich metadata collection in your CocoIndex flows.