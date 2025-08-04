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

## DataSlice to String Conversion (CRITICAL)

### 1. DataSlice Objects in Database Collection

**⚠️ CRITICAL GOTCHA**: DataSlice objects are not automatically converted to strings when passed to collect() calls, resulting in empty database content.

```python
# ❌ PROBLEM: DataSlice objects stored as empty strings
code_embeddings.collect(
    filename=file["filename"],
    code=chunk["content"],  # chunk["content"] is a DataSlice object
    embedding=chunk["embedding"]
)
# Result: Database shows code="" for all chunks
```

**✅ SOLUTION**: Always convert DataSlice objects to strings using transform:

```python
@cocoindex.op.function()
def convert_dataslice_to_string(content) -> str:
    """Convert CocoIndex DataSlice content to string."""
    try:
        result = str(content) if content else ""
        LOGGER.info(f"🔍 DataSlice conversion: input type={type(content)}, output_len={len(result)}")
        if len(result) == 0:
            LOGGER.error(f"❌ DataSlice conversion produced empty string! Input: {repr(content)}")
        return result
    except Exception as e:
        LOGGER.error(f"Failed to convert content to string: {e}")
        return ""

# ✅ CORRECT: Convert DataSlice to string before collection
code_embeddings.collect(
    filename=file["filename"],
    code=chunk["content"].transform(convert_dataslice_to_string),  # Transform DataSlice to string
    embedding=chunk["embedding"]
)
```

**Why this matters:**
- DataSlice objects represent lazy evaluation pipelines, not immediate values
- Database storage requires concrete string values
- Without conversion, database stores empty strings instead of actual code content
- This breaks hybrid search functionality completely

## Chunking Dictionary Key Compatibility (CRITICAL)

### 1. AST vs Default Chunking Key Differences

**⚠️ CRITICAL GOTCHA**: Different chunking methods use different dictionary keys for content, causing content loss in post-processing functions.

```python
# AST chunking creates chunks with "content" key:
{
    "content": "def function():\n    pass",  # AST chunks use "content"
    "metadata": {...},
    "location": "file.py:10"
}

# Default/recursive chunking creates chunks with "text" key:
{
    "text": "def function():\n    pass",      # Default chunking uses "text"
    "location": "file.py:10",
    "start": 10,
    "end": 15
}
```

**❌ PROBLEM**: Post-processing functions that assume one key format lose content:

```python
@cocoindex.op.function()
def ensure_unique_chunk_locations(chunks) -> list:
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text", "")  # ❌ Loses AST chunk content (uses "content" key)
```

**✅ SOLUTION**: Handle both key formats in post-processing functions:

```python
@cocoindex.op.function()
def ensure_unique_chunk_locations(chunks) -> list:
    for chunk in chunks:
        if isinstance(chunk, dict):
            # Try "content" first (AST chunks), fallback to "text" (default chunks)
            text = chunk.get("content", chunk.get("text", ""))  # ✅ Handles both formats
```

**Key insight:** Any function that processes chunks from multiple chunking methods must handle both "content" and "text" keys to avoid content loss.

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
| `Type mismatch for metadata_json: passed in Str, declared typing.Annotated[typing.Any, TypeKind(kind='Json')] (Json)` | Transform functions return different types (str vs Json) that cause type conflicts | Make metadata functions return consistent types (all strings via `json.dumps()`) |
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

#### 4. Metadata Transform Function Type Consistency
**Root cause:** Inconsistent return types between metadata creation functions cause type mismatches in transform chains.

```python
# ❌ PROBLEMATIC: Inconsistent return types
@cocoindex.op.function()
def create_default_metadata(content: str) -> cocoindex.Json:
    return {"functions": [], "classes": []}  # Returns dict

@cocoindex.op.function() 
def extract_code_metadata(text: str, language: str) -> str:
    return json.dumps({"functions": [], "classes": []})  # Returns string

# When used in transforms:
chunk["metadata"] = chunk["content"].transform(create_default_metadata)  # -> dict
chunk["metadata"] = chunk["content"].transform(extract_code_metadata)   # -> str

# Later functions expecting consistent types fail:
@cocoindex.op.function()
def extract_functions(metadata_json: cocoindex.Json) -> str:  # Expects Json but gets Str
    return str(metadata_json.get("functions", []))
```

**✅ SOLUTION:** Make all metadata functions return consistent types (JSON strings):

```python
@cocoindex.op.function()
def create_default_metadata(content: str) -> str:
    default_metadata = {"functions": [], "classes": []}
    return json.dumps(default_metadata)  # Consistent string output

@cocoindex.op.function()
def extract_code_metadata(text: str, language: str) -> str:
    return json.dumps({"functions": [], "classes": []})  # Already string

# Update extract functions to accept strings:
@cocoindex.op.function()
def extract_functions(metadata_json: str) -> str:
    metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
    return str(metadata_dict.get("functions", []))
```

**Key insight:** CocoIndex transforms serialize return values, so functions receiving transformed data should expect serialized types (strings for JSON), not the original types (dicts for `cocoindex.Json`).

#### 5. Development Workflow for Type Issues
When encountering type system errors:

1. **Isolate the problem** - Create minimal reproduction script
2. **Check union types** - Replace `Union[...]` in dataclasses with `cocoindex.Json`
3. **Verify function signatures** - Ensure parameter types match data being passed
4. **Test incrementally** - Fix one type issue at a time
5. **Use development metadata strategy** - Keep experimental fields in `cocoindex.Json` until proven

#### 6. Type System Best Practices Summary
- **Avoid unions in dataclass fields** - Use `cocoindex.Json` for flexible metadata
- **Keep type annotations** - They are required, not optional in modern CocoIndex
- **Ensure consistent metadata function types** - All metadata creation functions should return the same type (preferably JSON strings)
- **Handle both dict and string inputs** - Functions may receive either depending on context
- **Escape regex special characters** - Language configuration regexes need proper escaping
- **Test with minimal examples** - Isolate type issues before fixing in main codebase

## Multi-Language Analysis Integration (January 2025)

### 1. Backend vs Frontend Analysis Synchronization

**⚠️ CRITICAL GOTCHA**: Multi-language analyzers may work in CocoIndex flows but fail in MCP server backends due to analysis function location mismatches.

```python
# ❌ PROBLEM: Backend only uses Python analyzer
class PostgresBackend:
    def _format_result(self, row, score_type):
        if language.lower() == "python":
            from ..lang.python.python_code_analyzer import analyze_python_code
            metadata = analyze_python_code(code, filename)
        else:
            # All non-Python languages get empty metadata
            metadata = {"analysis_method": "none", "functions": [], "classes": []}
```

**✅ SOLUTION**: Ensure backends use the same multi-language analysis as CocoIndex flows:

```python
class PostgresBackend:
    def _format_result(self, row, score_type):
        try:
            # Use the same multi-language analyzer as CocoIndex flows
            from ..cocoindex_config import extract_code_metadata
            metadata_json_str = extract_code_metadata(code, language, filename)
            analysis_metadata = json.loads(metadata_json_str)
            
            if analysis_metadata is not None:
                pg_row.update({
                    "functions": analysis_metadata.get("functions", []),
                    "classes": analysis_metadata.get("classes", []),
                    "imports": analysis_metadata.get("imports", []),
                    "analysis_method": analysis_metadata.get("analysis_method", "unknown")
                })
        except Exception as e:
            # Fallback to basic metadata with error logging
            LOGGER.error(f"Multi-language analysis failed: {e}")
            pg_row.update({"analysis_method": "error", "functions": [], "classes": []})
```

### 2. Language Case Sensitivity and Normalization

**⚠️ CRITICAL GOTCHA**: Language strings from different sources use inconsistent casing, causing analyzer selection failures.

```python
# Sources of language case mismatches:
"Python" vs "python" vs "PYTHON"    # File extensions vs user input vs database
"C++" vs "cpp" vs "CPP"             # Database storage vs query parameters  
"JavaScript" vs "javascript" vs "js" # Full names vs abbreviations
```

**✅ SOLUTION**: Implement comprehensive case-insensitive language matching:

```python
def select_language_analyzer(language: str) -> callable:
    """Select appropriate analyzer with case-insensitive matching."""
    lang_lower = language.lower() if language else ""
    
    # Handle all common variations
    if lang_lower in ["python", "py"]:
        from .language_handlers.python_visitor import analyze_python_code
        return analyze_python_code
    elif lang_lower == "rust":
        from .language_handlers.rust_visitor import analyze_rust_code
        return analyze_rust_code
    elif lang_lower == "java":
        from .language_handlers.java_visitor import analyze_java_code
        return analyze_java_code
    elif lang_lower in ["javascript", "js"]:
        from .language_handlers.javascript_visitor import analyze_javascript_code
        return analyze_javascript_code
    elif lang_lower in ["typescript", "ts"]:
        from .language_handlers.typescript_visitor import analyze_typescript_code
        return analyze_typescript_code
    elif lang_lower in ["cpp", "c++", "cxx"]:
        from .language_handlers.cpp_visitor import analyze_cpp_code
        return analyze_cpp_code
    elif lang_lower == "c":
        from .language_handlers.c_visitor import analyze_c_code
        return analyze_c_code
    elif lang_lower in ["kotlin", "kt"]:
        from .language_handlers.kotlin_visitor import analyze_kotlin_code
        return analyze_kotlin_code
    elif lang_lower in ["haskell", "hs"]:
        from .language_handlers.haskell_visitor import analyze_haskell_code
        return analyze_haskell_code
    else:
        return None  # Use fallback basic analysis
```

### 3. SQL Query Language Matching

**⚠️ CRITICAL GOTCHA**: Database queries with exact language matching fail when user input case differs from stored case.

```python
# ❌ PROBLEM: Exact case matching fails
WHERE language = 'cpp'      # Fails if database has 'C++'
WHERE language = 'CPP'      # Fails if user searches for 'cpp'
```

**✅ SOLUTION**: Use case-insensitive SQL comparisons for language fields:

```python
def build_sql_where_clause(search_group, table_alias=""):
    """Build SQL WHERE clause with case-insensitive language matching."""
    for condition in search_group.conditions:
        if condition.field == "language":
            # Use case-insensitive comparison for language field
            where_parts.append(f"LOWER({prefix}language) = LOWER(%s)")
            params.append(condition.value)
        else:
            # Use exact matching for other fields
            where_parts.append(f"{prefix}{condition.field} = %s")
            params.append(condition.value)
```

### 4. Language-Specific Dependencies and Fallbacks

**⚠️ CRITICAL GOTCHA**: Missing tree-sitter language parsers cause silent analysis failures without clear error messages.

```python
# ❌ PROBLEM: Silent fallback when dependencies missing
try:
    import tree_sitter_javascript
    language_obj = tree_sitter.Language(tree_sitter_javascript.language())
except ImportError:
    # Silent fallback - no clear indication of missing dependency
    return None
```

**✅ SOLUTION**: Explicit dependency management with clear error reporting:

```python
def get_language_parser(language: str):
    """Get tree-sitter parser with explicit dependency handling."""
    try:
        if language == 'javascript':
            import tree_sitter_javascript
            return tree_sitter.Language(tree_sitter_javascript.language())
        elif language == 'rust':
            import tree_sitter_rust
            return tree_sitter.Language(tree_sitter_rust.language())
        # ... other languages
    except ImportError as e:
        LOGGER.error(f"Missing tree-sitter dependency for {language}: {e}")
        LOGGER.info(f"Install with: pip install tree-sitter-{language}")
        return None
    except Exception as e:
        LOGGER.error(f"Failed to load {language} parser: {e}")
        return None
```

**Dependency Management Best Practices:**
```python
# In pyproject.toml, ensure all required tree-sitter languages are listed:
dependencies = [
    "tree-sitter>=0.20.0",
    "tree-sitter-python>=0.20.0",
    "tree-sitter-rust>=0.20.0", 
    "tree-sitter-java>=0.20.0",
    "tree-sitter-javascript>=0.23.1",  # Note version requirements
    "tree-sitter-typescript>=0.20.0",
    # Add as needed for new language support
]
```

### 5. Analysis Result Validation and Fallbacks

**⚠️ CRITICAL GOTCHA**: Language analyzers may return incomplete results without proper success indicators.

```python
# ❌ PROBLEM: Missing success field causes fallback to basic analysis
def analyze_language_code(code, language, filename):
    """Analyzer without success indicator."""
    return {
        "functions": ["func1", "func2"],
        "classes": ["Class1"],
        "analysis_method": "language_ast_visitor"
        # Missing: "success": True
    }

# Later validation fails:
if not result.get("success", False):
    # Falls back to basic analysis, losing rich metadata
    return create_basic_metadata()
```

**✅ SOLUTION**: Ensure all language analyzers include success indicators:

```python
def analyze_language_code(code, language, filename):
    """Language analyzer with proper success indication."""
    try:
        # Perform analysis...
        metadata = {
            'language': language,
            'filename': filename,
            'functions': extracted_functions,
            'classes': extracted_classes,
            'analysis_method': f'{language}_ast_visitor',
            'success': True,  # ✅ Critical success indicator
            'parse_errors': 0,
            'complexity_score': calculated_complexity
        }
        return metadata
    except Exception as e:
        LOGGER.error(f"{language} analysis failed: {e}")
        return {
            'success': False,  # ✅ Clear failure indication
            'error': str(e),
            'analysis_method': 'basic_fallback'
        }
```

### 6. Integration Testing for Multi-Language Support

**✅ BEST PRACTICE**: Create comprehensive test suites that validate multi-language analysis across the entire stack:

```python
# Test matrix covering all layers:
@pytest.mark.parametrize("language,file_extension,expected_functions", [
    ("python", ".py", ["test_function"]),
    ("rust", ".rs", ["fibonacci", "main"]),
    ("java", ".java", ["calculateSum", "Person"]),
    ("javascript", ".js", ["processData"]),
    ("typescript", ".ts", ["validateInput"]),
    ("cpp", ".cpp", ["fibonacci", "Person"]),
    ("c", ".c", ["test_function"]),
    ("kotlin", ".kt", ["dataProcessor"]),
    ("haskell", ".hs", ["quicksort"])
])
def test_end_to_end_language_analysis(language, file_extension, expected_functions):
    """Test complete pipeline from file reading to database storage."""
    # 1. Test CocoIndex flow analysis
    cocoindex_result = run_cocoindex_analysis(test_code, language)
    assert cocoindex_result["analysis_method"] != "none"
    
    # 2. Test MCP server backend analysis  
    mcp_result = query_mcp_server(f"language:{language}")
    assert len(mcp_result["results"]) > 0
    
    # 3. Test database storage consistency
    db_result = query_database(f"language = '{language}'")
    assert db_result["analysis_method"] != "none"
    
    # 4. Test function extraction across all layers
    for expected_func in expected_functions:
        assert expected_func in cocoindex_result["functions"]
        assert expected_func in mcp_result["results"][0]["functions"]
        assert expected_func in db_result["functions"]
```

### 7. Multi-Language Analysis Migration Checklist

When adding new language support or fixing existing languages:

- [ ] **Add tree-sitter dependency** to pyproject.toml
- [ ] **Create language-specific analyzer** in language_handlers/
- [ ] **Update cocoindex_config.py** with case-insensitive language matching
- [ ] **Update postgres_backend.py** to use extract_code_metadata
- [ ] **Add SQL case-insensitive matching** for language queries
- [ ] **Include success indicators** in analyzer return values
- [ ] **Add integration tests** covering CocoIndex flow + MCP server + database
- [ ] **Test with real files** to verify metadata extraction
- [ ] **Document language-specific quirks** and dependencies

### 8. Language Analysis Performance Considerations

**✅ OPTIMIZATION**: Cache language analyzers and parsers to avoid repeated loading:

```python
# Global cache for expensive parser initialization
_LANGUAGE_PARSERS = {}
_LANGUAGE_ANALYZERS = {}

def get_cached_analyzer(language: str):
    """Get cached language analyzer to avoid repeated initialization."""
    lang_key = language.lower()
    
    if lang_key not in _LANGUAGE_ANALYZERS:
        analyzer = select_language_analyzer(language)
        if analyzer:
            _LANGUAGE_ANALYZERS[lang_key] = analyzer
            LOGGER.info(f"Cached analyzer for {language}")
    
    return _LANGUAGE_ANALYZERS.get(lang_key)
```

This caching is particularly important for:
- **Tree-sitter parser initialization** (expensive grammar loading)
- **Model loading** for embedding-based analysis
- **Regex compilation** for pattern-based metadata extraction

## CocoIndex Type Inference and Fallback Models (January 2025)

### 1. Understanding Fallback Model Warnings

**⚠️ INFORMATIONAL GOTCHA**: CocoIndex may log "Using fallback model for DataSlice" warnings that appear concerning but are actually informational messages about the type inference system.

```python
# Example warning message that appears during flow compilation:
cocoindex_code_mcp_server.cocoindex_config: INFO     Using fallback model for DataSlice(Str; [_root] [files AS files_1] .language)
```

**What this means:**
- CocoIndex's static type analysis cannot determine the exact string type of certain fields during flow compilation
- The engine falls back to a generic model for type inference rather than using specialized type handlers
- This is a limitation of CocoIndex's static analysis, not an error in your code
- The actual functionality (language detection, processing) works correctly regardless

### 2. Common Scenarios Triggering Fallback Models

**Language Detection Fields:**
```python
# This pattern often triggers fallback model warnings:
file["language"] = file["filename"].transform(extract_language)

# Even with proper function definition:
@cocoindex.op.function()
def extract_language(filename: str) -> str:
    """Extract language from filename - works correctly."""
    ext = os.path.splitext(filename)[1].lower()
    return LANGUAGE_MAP.get(ext, "unknown")
```

**Dynamic Field Creation:**
- Fields created through transform operations on DataSlices
- String fields with values determined at runtime
- Fields that depend on file content analysis

### 3. Fallback Model Impact and Mitigation

**✅ FUNCTIONAL IMPACT**: Minimal to none
- Your flows execute correctly
- Data is processed and stored properly  
- Type checking still works at runtime

**✅ PERFORMANCE IMPACT**: Generally negligible
- Fallback models are typically just less optimized code paths
- No significant performance degradation observed

**⚠️ WHEN TO INVESTIGATE**: Only if you notice:
- Actual processing failures
- Incorrect data in outputs
- Significant performance issues

### 4. Improving Language Detection to Reduce Fallbacks

**Enhanced language detection with better fallback handling:**

```python
@cocoindex.op.function()
def extract_language(filename: str) -> str:
    """Extract language with explicit fallback handling."""
    basename = os.path.basename(filename)
    
    # Handle special files without extensions
    if basename.lower() in ["makefile", "dockerfile", "jenkinsfile"]:
        return basename.lower()
    
    # Handle special patterns
    special_patterns = {
        "cmakelists": "cmake",
        "build.gradle": "gradle", 
        "pom.xml": "maven",
        "docker-compose": "dockerfile",
        "go.": "go"
    }
    
    for pattern, lang in special_patterns.items():
        if pattern in basename.lower():
            return lang
    
    # Get extension and map to language
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in TREE_SITTER_LANGUAGE_MAP:
        return TREE_SITTER_LANGUAGE_MAP[ext]
    elif ext:
        # Return clean extension name for unknown extensions
        return ext[1:] if ext.startswith('.') else ext
    else:
        # Explicit fallback for files without extensions
        return "unknown"
```

**Benefits of explicit fallback handling:**
- Consistent return values
- Better debugging information
- Reduced ambiguity in type inference

### 5. Transform() Function Requirements for Type Safety

**⚠️ CRITICAL GOTCHA**: CocoIndex's `transform()` method requires functions decorated with `@cocoindex.op.function()`, not lambda functions.

```python
# ❌ WRONG: Lambda functions cause "transform() can only be called on a CocoIndex function" errors
chunk["analysis_method"] = chunk["metadata"].transform(
    lambda x: json.loads(x).get("analysis_method", "unknown") if x else "unknown"
)

# ✅ CORRECT: Use proper CocoIndex function decorators
@cocoindex.op.function()
def extract_analysis_method_field(metadata_json: str) -> str:
    """Extract analysis_method field from metadata JSON."""
    try:
        if not metadata_json:
            return "unknown"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        return str(metadata_dict.get("analysis_method", "unknown"))
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for analysis_method: {e}")
        return "unknown"

# Use the proper function:
chunk["analysis_method"] = chunk["metadata"].transform(extract_analysis_method_field)
```

**Why this matters:**
- CocoIndex needs to serialize and track function dependencies
- Lambda functions cannot be properly serialized by the flow system
- Decorated functions provide proper type information for the engine

### 6. Creating Extractor Functions for Metadata Fields

**Pattern for creating metadata field extractors:**

```python
# Template for metadata field extractors:
@cocoindex.op.function()
def extract_[field_name]_field(metadata_json: str) -> [return_type]:
    """Extract [field_name] field from metadata JSON."""
    try:
        if not metadata_json:
            return [default_value]
        # Parse JSON string to dict
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        value = metadata_dict.get("[field_name]", [default_value])
        return [type_conversion](value) if value is not None else [default_value]
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for [field_name]: {e}")
        return [default_value]

# Examples:
@cocoindex.op.function()
def extract_functions_field(metadata_json: str) -> str:
    """Extract functions field from metadata JSON."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        functions = metadata_dict.get("functions", [])
        return str(functions) if isinstance(functions, list) else str([functions]) if functions else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for functions: {e}")
        return "[]"

@cocoindex.op.function()
def extract_complexity_score_field(metadata_json: str) -> int:
    """Extract complexity_score field from metadata JSON."""
    try:
        if not metadata_json:
            return 0
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        score = metadata_dict.get("complexity_score", 0)
        return int(score) if isinstance(score, (int, float, str)) and str(score).isdigit() else 0
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for complexity_score: {e}")
        return 0
```

### 7. Debugging Type Inference Issues

**When encountering fallback model warnings:**

1. **Verify functionality first** - Check if your flow actually works despite warnings
2. **Check function decorators** - Ensure all transform functions use `@cocoindex.op.function()`
3. **Validate return types** - Make sure function return types are consistent
4. **Test with minimal examples** - Isolate the specific field causing issues
5. **Consider if action is needed** - Most fallback warnings can be safely ignored

**Debugging workflow:**
```python
# Test individual functions outside of flows:
def test_language_extraction():
    test_files = ['test.py', 'example.rs', 'README.md', 'unknown.xyz']
    for filename in test_files:
        result = extract_language(filename)
        print(f"{filename} -> {result} (type: {type(result)})")

# Verify the function works before worrying about type inference warnings
```

### 8. Best Practices for Type-Safe CocoIndex Functions

**✅ Type Safety Checklist:**
- Always use `@cocoindex.op.function()` decorators for transform functions
- Provide explicit type annotations for parameters and return values
- Handle edge cases with explicit default values
- Use try-catch blocks for JSON parsing operations
- Return consistent types from similar functions
- Test functions independently before integration

**✅ Acceptable Fallback Scenarios:**
- Language detection from dynamic file analysis
- Fields derived from runtime content analysis
- Transform chains with complex data dependencies

**⚠️ Investigate Further When:**
- Functions fail to execute (not just warnings)
- Data appears incorrectly in database
- Performance significantly degrades
- Type mismatches cause flow failures

### Summary: Fallback Model Warnings

- **Fallback model warnings are usually informational**, not errors
- **Functionality typically works correctly** despite warnings
- **Focus on actual failures** rather than warning messages
- **Use proper `@cocoindex.op.function()` decorators** for all transform functions
- **Improve language detection logic** with explicit fallback handling
- **Create dedicated extractor functions** for metadata fields instead of lambda functions
- **Test individual components** to verify they work before worrying about type inference