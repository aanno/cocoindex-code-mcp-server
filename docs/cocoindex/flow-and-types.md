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

## Chunking and Primary Key Management (CRITICAL)

### 1. Unique Chunk Locations Required

**⚠️ CRITICAL GOTCHA**: CocoIndex chunking functions don't guarantee unique location identifiers within the same file, causing PostgreSQL conflicts.

```python
# ❌ PROBLEM: SplitRecursively may produce duplicate locations
file["chunks"] = file["content"].transform(
    cocoindex.functions.SplitRecursively(),
    language=file["language"],
    chunk_size=1000
)
# Multiple chunks may have same location → PostgreSQL error:
# "ON CONFLICT DO UPDATE command cannot affect row a second time"
```

**✅ SOLUTION**: Always post-process chunks to ensure unique locations:

```python
@cocoindex.op.function()
def ensure_unique_chunk_locations(chunks) -> list:
    """Post-process chunks to ensure location fields are unique within the file."""
    if not chunks:
        return chunks
    
    chunk_list = list(chunks) if hasattr(chunks, '__iter__') else [chunks]
    seen_locations = set()
    unique_chunks = []
    
    for i, chunk in enumerate(chunk_list):
        if hasattr(chunk, 'location'):
            # AST chunking Chunk dataclass
            base_loc = chunk.location
        elif isinstance(chunk, dict):
            # Default chunking dictionary format
            base_loc = chunk.get("location", f"chunk_{i}")
        else:
            base_loc = f"chunk_{i}"
        
        # Make location unique
        unique_loc = base_loc
        suffix = 0
        while unique_loc in seen_locations:
            suffix += 1
            unique_loc = f"{base_loc}#{suffix}"
        
        seen_locations.add(unique_loc)
        
        # Update chunk with unique location
        if hasattr(chunk, 'location'):
            from dataclasses import replace
            unique_chunk = replace(chunk, location=unique_loc)
        elif isinstance(chunk, dict):
            unique_chunk = chunk.copy()
            unique_chunk["location"] = unique_loc
        else:
            unique_chunk = chunk
            
        unique_chunks.append(unique_chunk)
    
    return unique_chunks

# ✅ Apply to ALL chunking methods
# Default chunking
raw_chunks = file["content"].transform(
    cocoindex.functions.SplitRecursively(custom_languages=CUSTOM_LANGUAGES),
    language=file["language"],
    chunk_size=file["chunking_params"]["chunk_size"],
    min_chunk_size=file["chunking_params"]["min_chunk_size"],
    chunk_overlap=file["chunking_params"]["chunk_overlap"],
)
file["chunks"] = raw_chunks.transform(ensure_unique_chunk_locations)

# AST chunking
raw_chunks = file["content"].transform(
    ASTChunkOperation,
    language=file["language"],
    max_chunk_size=file["chunking_params"]["chunk_size"],
)
file["chunks"] = raw_chunks.transform(ensure_unique_chunk_locations)
```

### 2. Primary Key Design Considerations

**✅ BEST PRACTICE**: Use comprehensive primary keys that prevent conflicts:

```python
code_embeddings.export(
    "code_embeddings",
    cocoindex.targets.Postgres(),
    # Include source_name to handle multiple sources with same files
    primary_key_fields=["filename", "location", "source_name"],
    vector_indexes=[
        cocoindex.VectorIndexDef(
            field_name="embedding",
            metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
        )
    ],
)
```

**Why this matters**:
- `filename` alone isn't unique (same file in multiple sources)
- `location` alone isn't unique (SplitRecursively may produce duplicates)
- `source_name` prevents conflicts when same file appears in multiple paths

### 3. Chunking Method Selection

**CocoIndex chunking method hierarchy**:

```python
# 1. AST chunking (best for supported languages: Python, TypeScript, Java)
if language in ["Python", "TypeScript", "JavaScript", "Java"]:
    chunks = content.transform(ASTChunkOperation, language=language)
    
# 2. Default chunking (for unsupported languages: Rust, Go, C++)
else:
    chunks = content.transform(
        cocoindex.functions.SplitRecursively(custom_languages=CUSTOM_LANGUAGES),
        language=language
    )

# 3. ALWAYS ensure unique locations regardless of method
chunks = chunks.transform(ensure_unique_chunk_locations)
```

## Common Error Messages and Solutions (Updated)

| Error | Cause | Solution |
|-------|-------|----------|
| `operator class "vector_cosine_ops" does not accept data type jsonb` | Using Python lists instead of Vector types | Use `Vector[np.float32, Literal[dim]]` and return numpy arrays |
| `ON CONFLICT DO UPDATE command cannot affect row a second time` | Duplicate chunk locations within same file | Post-process chunks with `ensure_unique_chunk_locations()` |
| `data did not match any variant of untagged enum ValueType` | Union types in dataclass fields (e.g., `Dict[str, Union[...]]`) | Use `cocoindex.Json` for flexible metadata fields |
| `Type mismatch for metadata_json: passed in Json, declared <class 'str'>` | Functions expecting `str` but receiving `cocoindex.Json` | Update function signatures to accept `cocoindex.Json` |
| `Untyped dict is not accepted as a specific type annotation` | Using generic `dict` or `list` return types | Use specific types like `list[SomeClass]` or `cocoindex.Json` |
| `regex parse error: repetition quantifier expects a valid decimal` | Unescaped `{` in regex patterns | Escape curly braces: `r"\{-#"` instead of `r"{-#"` |
| `NameError: name 'lang' is not defined` | Variable name mismatch in function | Check function parameter names match usage |
| `Unsupported as a specific type annotation: typing.Any` | Using `typing.Any` in return types | Remove or use specific types |
| `Setup for flow is not up-to-date` | Flow not set up | Run `cocoindex setup src/config.py` |
| `CocoIndex library is not initialized` | Missing initialization | Call `cocoindex.init()` |
| `'SplitRecursively' object is not callable` | Missing custom_languages parameter | Use `SplitRecursively(custom_languages=CUSTOM_LANGUAGES)` |

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

## Metadata Strategy: Development vs Production

### Development Phase Strategy
Use `cocoindex.Json` for flexible metadata experimentation without frequent schema migrations:

```python
@dataclass
class Chunk:
    """Development-friendly chunk with flexible metadata."""
    content: str
    metadata: cocoindex.Json  # Flexible bag for experimental fields
    location: str = ""
    start: int = 0
    end: int = 0
```

**Benefits:**
- **Fast iteration** - no `cocoindex setup` needed for metadata changes
- **Experimental fields** - test complexity scores, type hints, etc. without schema impact
- **Rapid prototyping** - validate metadata usefulness before production commitment

### Production Migration Strategy
Extract proven metadata fields into dedicated PostgreSQL columns:

```python
# After validating metadata fields in development, promote to production schema
code_embeddings.collect(
    filename=file["filename"],
    content=chunk["content"],
    location=chunk["location"],
    # Promoted from metadata to dedicated columns for performance:
    functions=chunk["functions"],           # str column with indexing
    classes=chunk["classes"],              # str column with indexing  
    complexity_score=chunk["complexity"],   # int column for range queries
    has_type_hints=chunk["has_type_hints"], # bool column for filtering
    # Keep metadata for remaining experimental fields
    metadata_json=chunk["metadata"]         # json column for edge cases
)
```

**Production advantages:**
- **Query performance** - dedicated columns enable proper indexing
- **pgvector integration** - direct column access for vector operations
- **Type safety** - PostgreSQL enforces column types
- **Backward compatibility** - metadata_json remains for experimental fields

## Summary

- **USE Vector types** with fixed dimensions for embeddings: `Vector[np.float32, Literal[384]]`
- **Return numpy arrays** (`.astype(np.float32)`), NOT Python lists (`.tolist()`)
- **Type annotations ARE required** for Vector and complex types
- **Use cocoindex.Json for development metadata** to avoid frequent schema changes
- **Promote successful metadata to dedicated columns** for production performance
- **ALWAYS ensure unique chunk locations** with post-processing to prevent PostgreSQL conflicts
- **Use comprehensive primary keys** including `source_name` to handle multiple sources
- **Apply `SplitRecursively(custom_languages=CUSTOM_LANGUAGES)`** with proper parameters
- **Custom metadata fields** should appear in evaluation outputs and database
- **Load models once** at module level, not inside functions
- **Always run setup** after changing collection schema or vector dimensions

Following these updated practices will ensure proper pgvector integration, prevent database conflicts, and enable rich metadata collection in your CocoIndex flows.

## Real-World Lessons Learned

### PostgreSQL Conflict Resolution
The most common production issue is duplicate primary keys causing `ON CONFLICT DO UPDATE command cannot affect row a second time` errors. This happens because:

1. **SplitRecursively doesn't guarantee unique locations** for chunks within the same file
2. **Multiple processing runs** of the same file generate identical keys  
3. **Path overlaps** cause the same file to be processed multiple times

The solution is **mandatory post-processing** of all chunks to ensure location uniqueness, regardless of chunking method used.

### Chunking Method Selection Strategy
- **AST chunking**: Use for languages with good AST support (Python, TypeScript, Java, C#)
- **Default chunking**: Fallback for other languages (Rust, Go, Markdown, etc.)
- **Always post-process**: Both methods require unique location enforcement

### Database Schema Evolution
When adding new metadata fields to collection, remember:
1. Run `cocoindex setup` to update schema
2. Test with `cocoindex evaluate` to verify field population
3. Check that evaluation outputs show your custom fields
4. Ensure primary key covers all uniqueness requirements

### ValueType and Type System Debugging

The most complex CocoIndex issues often involve the type system and serialization. Here are key debugging lessons:

#### 1. ValueType Deserialization Errors
**Root cause:** Union types in dataclass fields cause serialization failures.

```python
# ❌ PROBLEMATIC: Union types in nested structures
@dataclass
class Chunk:
    metadata: Dict[str, Union[str, int, float, bool]]  # Breaks ValueType enum

# ✅ SOLUTION: Use cocoindex.Json for flexible metadata
@dataclass  
class Chunk:
    metadata: cocoindex.Json  # Handles any JSON-serializable data
```

#### 2. Function Parameter Type Mismatches
**Root cause:** Functions expecting one type but receiving another due to schema changes.

```python
# ❌ PROBLEMATIC: Function expects string but receives Json
@cocoindex.op.function()
def extract_field(metadata_json: str) -> str:
    return json.loads(metadata_json)["field"]

# ✅ SOLUTION: Update to accept cocoindex.Json
@cocoindex.op.function()
def extract_field(metadata_json: cocoindex.Json) -> str:
    metadata_dict = metadata_json if isinstance(metadata_json, dict) else json.loads(str(metadata_json))
    return str(metadata_dict.get("field", ""))
```

#### 3. Regex Pattern Issues in Language Configurations
**Root cause:** Unescaped special characters in regex patterns.

```python
# ❌ PROBLEMATIC: Unescaped curly braces
separators = [
    r"\n{-#\s*[A-Z]+",  # Breaks: { is quantifier syntax
]

# ✅ SOLUTION: Escape special regex characters  
separators = [
    r"\n\{-#\s*[A-Z]+",  # Works: \{ matches literal brace
]
```

#### 4. Development Workflow for Type Issues
When encountering type system errors:

1. **Isolate the problem** - Create minimal reproduction script
2. **Check union types** - Replace `Union[...]` in dataclasses with `cocoindex.Json`
3. **Verify function signatures** - Ensure parameter types match data being passed
4. **Test incrementally** - Fix one type issue at a time
5. **Use development metadata strategy** - Keep experimental fields in `cocoindex.Json` until proven

#### 5. Type System Best Practices Summary
- **Avoid unions in dataclass fields** - Use `cocoindex.Json` for flexible metadata
- **Keep type annotations** - They are required, not optional in modern CocoIndex
- **Handle both dict and string inputs** - Functions may receive either depending on context
- **Escape regex special characters** - Language configuration regexes need proper escaping
- **Test with minimal examples** - Isolate type issues before fixing in main codebase