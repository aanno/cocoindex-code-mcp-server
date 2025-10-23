# CocoIndex Flow/Pipeline Debugging Guide

This document provides a comprehensive guide for debugging CocoIndex flow and pipeline issues, based on real-world debugging of the hybrid search functionality.

## Overview

CocoIndex flows can be complex with multiple transformation stages. When debugging, it's important to understand the flow architecture and have systematic approaches to isolate issues.

## Flow Architecture

The typical CocoIndex flow for code analysis follows this pipeline:

1. **Source Reading** - LocalFile source reads files from disk
2. **Language Detection** - `extract_language()` determines programming language
3. **Chunking** - Either AST chunking or recursive text chunking
4. **Unique Location Processing** - `ensure_unique_chunk_locations()` prevents duplicates
5. **Embedding Generation** - Language-aware or default embedding models
6. **Metadata Extraction** - `extract_code_metadata()` analyzes code for functions, classes, etc.
7. **Database Export** - Results stored in PostgreSQL with pgvector

## Common Issues and Debugging Techniques

### 1. Empty Code Chunks

**Symptoms:**

- Database shows `code: ""` for all chunks
- Metadata shows `analysis_method: "unknown"`
- Metadata shows `chunking_method: "unknown"`
- Hybrid search fails because there's no content to search

**Root Causes:**

- Bug in `ensure_unique_chunk_locations()` discarding chunk content
- **CRITICAL**: Wrong dictionary key in `ensure_unique_chunk_locations()` (using "text" instead of "content")
- DataSlice objects not being converted to strings properly
- Incorrect chunking operation configuration
- File reading issues

**Debugging Steps:**

1. **Use CocoIndex Evaluation Mode:**

   ```bash
   cocoindex evaluate cocoindex_code_mcp_server.cocoindex_config
   ```

   This creates `eval_CodeEmbedding_YYYYMMDD_HHMMSS/` directory with raw flow outputs

2. **Examine Raw Chunks:**

   ```bash
   # Find your test file
   find eval_CodeEmbedding_* -name "*your_test_file*"

   # Check first few lines for code content
   head -20 eval_CodeEmbedding_*/files@path%2Fto%2Ffile.py.yaml
   ```

3. **Test Individual Components:**
   Create test scripts to isolate each stage:

   ```python
   # Test AST chunking directly
   from cocoindex_code_mcp_server.ast_chunking import ASTChunkOperation
   chunks = ASTChunkOperation(content=test_code, language="Python")
   ```

4. **Check Configuration:**
   Verify flow configuration flags:

   ```python
   from cocoindex_code_mcp_server.cocoindex_config import _global_flow_config
   print(f"AST chunking: {not _global_flow_config.get('use_default_chunking', False)}")
   print(f"Language handler: {not _global_flow_config.get('use_default_language_handler', False)}")
   ```

### 2. Metadata Extraction Issues

**Symptoms:**

- All metadata shows `analysis_method: "unknown"`
- Missing function/class information in database
- Python analyzer not being used

**Debugging Steps:**

1. **Test Language Handler Directly:**

   ```python
   from cocoindex_code_mcp_server.lang.python.python_code_analyzer import analyze_python_code
   result = analyze_python_code(test_code, "test.py")
   print(f"Analysis method: {result.get('analysis_method')}")
   ```

2. **Check Configuration Flags:**

   ```python
   # Look for these debug logs in CocoIndex output
   # "🔍 DEBUGGING extract_code_metadata for filename"
   # "use_default_handler: False" (should be False for proper analysis)
   ```

3. **Verify Metadata Preservation:**
   Check that `ensure_unique_chunk_locations()` preserves metadata:

   ```python
   # Before fix: metadata={} (lost)
   # After fix: metadata=chunk.metadata (preserved)
   ```

### 3. Database Content Issues

**Symptoms:**

- Query results don't match expectations
- Missing or incorrect data in database

**Debugging Steps:**

1. **Direct Database Query:**

   ```sql
   SELECT filename, location,
          LEFT(code, 100) as code_preview,
          LEFT(metadata_json, 200) as metadata_preview
   FROM code_embeddings
   WHERE filename LIKE '%your_test_file%';
   ```

2. **Check Data Types:**

   ```sql
   SELECT filename,
          CASE WHEN code = '' THEN 'EMPTY' ELSE 'HAS_CONTENT' END as code_status,
          CASE WHEN metadata_json = '{}' THEN 'NO_METADATA' ELSE 'HAS_METADATA' END as metadata_status
   FROM code_embeddings
   LIMIT 10;
   ```

### 4. Chunking Method and Error Tracking

**New Features (2025):**
CocoIndex now tracks comprehensive metadata about chunking methods and tree-sitter error handling.

**Key Metadata Properties:**

- `chunking_method`: Tracks the actual chunking method used (`ast_recursive`, `ast_recursive_with_errors`, `regex_fallback`, etc.)
- `tree_sitter_chunking_error`: Boolean indicating if tree-sitter error nodes were encountered during chunking
- `tree_sitter_analyze_error`: Boolean indicating if tree-sitter errors occurred during analysis
- `analysis_method`: Existing field now complemented by chunking-specific tracking

**Debugging chunking issues:**

```sql
-- Check chunking method distribution
SELECT
    JSON_EXTRACT(metadata_json, '$.chunking_method') as method,
    COUNT(*) as count
FROM code_embeddings
GROUP BY method;

-- Find chunks with tree-sitter errors
SELECT filename, location,
    JSON_EXTRACT(metadata_json, '$.tree_sitter_chunking_error') as chunk_errors,
    JSON_EXTRACT(metadata_json, '$.tree_sitter_analyze_error') as analyze_errors
FROM code_embeddings
WHERE JSON_EXTRACT(metadata_json, '$.tree_sitter_chunking_error') = 'true'
   OR JSON_EXTRACT(metadata_json, '$.tree_sitter_analyze_error') = 'true';
```

**Testing chunking method tracking:**

```python
from cocoindex_code_mcp_server.lang.haskell.haskell_ast_chunker import EnhancedHaskellChunker, HaskellChunkConfig

config = HaskellChunkConfig(max_chunk_size=500)
chunker = EnhancedHaskellChunker(config)
chunks = chunker.chunk_code(code, 'test.hs')

for chunk in chunks:
    metadata = chunk.get('metadata', {})
    print(f"chunking_method: {metadata.get('chunking_method', 'unknown')}")
    print(f"tree_sitter_chunking_error: {metadata.get('tree_sitter_chunking_error', 'unknown')}")
```

### 5. Flow Configuration Issues

**Symptoms:**

- Wrong chunking method being used
- Unexpected file filtering
- Performance issues

**Debugging Steps:**

1. **Verify Path Configuration:**

   ```python
   update_flow_config(
       paths=['specific/test/file.py'],  # Test with single file first
       use_default_chunking=False,
       use_default_language_handler=False
   )
   ```

2. **Check Flow Logs:**
   Look for these log messages:
   + "Using AST chunking extension" vs "Using default recursive splitting"
   + "Using custom language handler extension" vs "Using default language handler"
   + File count: "Adding source: path as 'files'"

## Debugging Tools and Techniques

### 1. CocoIndex Evaluation Mode

The most powerful debugging tool. Creates raw output files without database changes:

```bash
cocoindex evaluate cocoindex_code_mcp_server.cocoindex_config
```

### 2. Component Isolation Testing

Test individual components in isolation:

```python
# Test chunking
from cocoindex_code_mcp_server.ast_chunking import CocoIndexASTChunker
chunker = CocoIndexASTChunker(max_chunk_size=500)
chunks = chunker.chunk_code(code, "Python", "test.py")

# Test metadata extraction
from cocoindex_code_mcp_server.cocoindex_config import extract_code_metadata
metadata = extract_code_metadata(code, "Python", "test.py")
```

### 3. Configuration Debugging

```python
# Check current configuration
from cocoindex_code_mcp_server.cocoindex_config import _global_flow_config
print("Current config:", _global_flow_config)

# Test with minimal config
update_flow_config(
    paths=['single_test_file.py'],
    use_default_chunking=False,
    use_default_language_handler=False
)
```

### 4. Database State Inspection

```sql
-- Check for empty chunks
SELECT COUNT(*) as empty_chunks FROM code_embeddings WHERE code = '';

-- Check analysis methods
SELECT
    JSON_EXTRACT(metadata_json, '$.analysis_method') as analysis_method,
    JSON_EXTRACT(metadata_json, '$.chunking_method') as chunking_method,
    COUNT(*) as count
FROM code_embeddings
GROUP BY analysis_method, chunking_method;

-- Sample content
SELECT filename, LEFT(code, 200) as preview
FROM code_embeddings
WHERE code != ''
LIMIT 5;
```

## Common Fixes

### 1. Metadata Preservation Fix

In `ensure_unique_chunk_locations()`, ensure metadata is preserved:

```python
# WRONG (loses metadata):
unique_chunk = Chunk(
    content=text,
    metadata={},  # ❌ Empty metadata
    location=unique_loc,
    start=start,
    end=end
)

# CORRECT (preserves metadata):
unique_chunk = Chunk(
    content=text,
    metadata=metadata,  # ✅ Preserved metadata
    location=unique_loc,
    start=start,
    end=end
)
```

### 2. Dictionary Key Compatibility Fix

**CRITICAL FIX**: AST chunks use "content" key while other chunkers use "text" key:

```python
# WRONG (loses AST chunk content):
elif isinstance(chunk, dict):
    text = chunk.get("text", "")  # ❌ AST chunks use "content" key

# CORRECT (preserves all chunk content):
elif isinstance(chunk, dict):
    text = chunk.get("content", chunk.get("text", ""))  # ✅ Try "content" first, fallback to "text"
```

### 3. DataSlice to String Conversion

Ensure DataSlice objects are converted to strings before database storage:

```python
# In collect() call, use transform to convert DataSlice to string:
code_embeddings.collect(
    filename=file["filename"],
    language=file["language"],
    location=chunk["location"],
    code=chunk["content"].transform(convert_dataslice_to_string),  # ✅ Convert DataSlice to string
    embedding=chunk["embedding"],
    # ... other fields
)
```

### 4. Chunk Class Dictionary Compatibility

Add missing methods to Chunk class for dictionary-style access:

```python
def __contains__(self, key: str) -> bool:
    """Check if key exists in chunk (for 'key in chunk' syntax)."""
    return hasattr(self, key) or key in self.metadata

def __getitem__(self, key: Union[str, int]) -> Any:
    """Allow dictionary-style access."""
    if isinstance(key, str):
        if hasattr(self, key):
            return getattr(self, key)
        elif key in self.metadata:
            return self.metadata[key]
        else:
            raise KeyError(f"Key '{key}' not found in chunk")
    # Handle integer access for compatibility
    elif key == 0:
        return self
    else:
        raise IndexError(f"Chunk index {key} out of range")
```

### 3. Configuration Validation

Always validate configuration before running:

```python
def validate_flow_config():
    """Validate flow configuration for debugging."""
    config = _global_flow_config
    print(f"Paths: {config.get('paths')}")
    print(f"AST chunking enabled: {not config.get('use_default_chunking', False)}")
    print(f"Custom language handler: {not config.get('use_default_language_handler', False)}")
    print(f"Smart embedding: {config.get('use_smart_embedding', False)}")
```

## Best Practices

1. **Start Small:** Test with a single file before running on entire codebase
2. **Use Evaluation Mode:** Always use `cocoindex evaluate` for debugging
3. **Check Each Stage:** Test chunking, metadata extraction, and database export separately
4. **Preserve State:** Use version control to track configuration changes
5. **Log Everything:** Enable debug logging to trace data flow
6. **Validate Assumptions:** Don't assume components work - test them individually

## Troubleshooting Checklist

- [ ] Is the file being read correctly?
- [ ] Is the language detected properly?
- [ ] Are chunks being generated with content?
- [ ] Is metadata being extracted?
- [ ] Is metadata being preserved through transformations?
- [ ] Are embeddings being generated?
- [ ] Is data reaching the database correctly?
- [ ] Are queries working as expected?

## Real-World Case Studies

### Case Study 1: Empty Code Chunks (Resolved)

**Problem:** Hybrid search returning no results despite database containing files.

**Investigation Process:**

1. Checked database - found `analysis_method: "unknown"` for all entries
2. Tested Python analyzer directly - worked correctly
3. Used CocoIndex evaluation - found all `code: ""` (empty chunks)
4. Tested AST chunking directly - worked correctly, chunks had content
5. **BREAKTHROUGH**: Traced pipeline step-by-step and found `ensure_unique_chunk_locations()` was using wrong dictionary key
6. **ROOT CAUSE**: Function looked for `chunk.get("text", "")` but AST chunks use `"content"` key
7. **FIX**: Changed to `chunk.get("content", chunk.get("text", ""))` to handle both formats
8. **ADDITIONAL FIX**: Added DataSlice to string conversion in collect() call
9. **VERIFICATION**: Database now shows chunks with substantial content (1684, 1562, 1503+ characters)
10. **SUCCESS**: Vector search tests pass, hybrid search functional

**Key Lessons:**

- **Systematic debugging** is essential - test each pipeline stage individually
- **The issue wasn't in obvious places** (Python analyzer, AST chunking) but in a utility function
- **Different chunking methods use different dictionary keys** - AST uses "content", others use "text"
- **DataSlice objects require explicit conversion** to strings before database storage
- **Pipeline debugging** requires understanding data flow transformations, not just individual components

### Case Study 2: Multi-Language Analysis Failures (January 2025)

**Problem:** MCP server test suite failing - 10 out of 15 multi-language hybrid search tests failing with missing metadata.

**Investigation Process:**

1. **Initial Discovery**: Tests showed `analysis_method: "none"` for all non-Python languages
2. **Database Analysis**: Found files being indexed but with empty metadata arrays:

   ```json
   {
     "functions": [],
     "classes": [],
     "imports": [],
     "analysis_method": "none"
   }
   ```

3. **Source Code Investigation**: Found multiple root causes:
   + **Root Cause 1**: `postgres_backend.py` was hardcoded to only use Python analyzer
   + **Root Cause 2**: Multi-language analyzers existed but weren't being called
   + **Root Cause 3**: Case sensitivity issues in language matching

**Systematic Fix Process:**

**Fix 1 - PostgreSQL Backend Multi-Language Support:**

```python
# BEFORE (postgres_backend.py): Only Python analysis
if language.lower() == "python":
    metadata = analyze_python_code(code, filename)
else:
    pg_row.update({
        "analysis_method": "none",  # ❌ Hardcoded for all non-Python
        "functions": [],
        "classes": []
    })

# AFTER: Full multi-language support
from ..cocoindex_config import extract_code_metadata
metadata_json_str = extract_code_metadata(code, language, filename)
analysis_metadata = json.loads(metadata_json_str)
```

**Fix 2 - Case-Insensitive Language Matching:**

```python
# BEFORE (cocoindex_config.py): Case-sensitive matching failed
if language == "Rust":  # Missed "rust", "RUST" variations
    from .language_handlers.rust_visitor import analyze_rust_code

# AFTER: Robust case-insensitive matching
lang_lower = language.lower() if language else ""
if lang_lower == "rust":
    from .language_handlers.rust_visitor import analyze_rust_code
elif lang_lower in ["cpp", "c++", "cxx"]:
    from .language_handlers.cpp_visitor import analyze_cpp_code
```

**Fix 3 - SQL Query Case Sensitivity:**

```python
# BEFORE (keyword_search_parser_lark.py): Exact case matching
where_parts.append(f"{prefix}{validated_field} = %s")

# AFTER: Case-insensitive for language searches
if validated_field == 'language':
    where_parts.append(f"LOWER({prefix}{validated_field}) = LOWER(%s)")
```

**Fix 4 - Language Detection Corrections:**

- **C++**: Database stores as `"C++"` not `"cpp"` - updated test expectations
- **JavaScript**: Missing `tree-sitter-javascript` dependency - installed and configured
- **Haskell**: Missing `'success': True` field - added to analyzer output

**Verification Results:**
After fixes, tests showed rich metadata extraction:

```json
{
  "filename": "tmp/rust_example_1.rs",
  "language": "Rust",
  "functions": ["new", "is_adult", "fibonacci", "main"],
  "classes": [],
  "analysis_method": "rust_ast_visitor"  // ✅ Proper analysis
}
```

**Key Discoveries:**

- **Backend vs Frontend Analysis**: The backend (postgres_backend.py) wasn't using the same multi-language analysis as the frontend
- **Language Case Sensitivity**: Database languages don't always match user input case (`"C++"` vs `"cpp"`)
- **Dependency Management**: Missing tree-sitter language parsers cause silent fallbacks
- **Test Infrastructure**: Need systematic testing across all supported languages
- **Integration Testing**: MCP server tests revealed issues not caught by unit tests

### Case Study 3: Test Result File Management (January 2025)

**Problem:** Test result files had different timestamps, making it hard to group results from the same test run.

**Investigation:**

- Each test case generated its own timestamp: `basename_python_20250801_133633_124.json`, `ast_visitor_20250801_133633_215.json`
- Made it difficult to identify which 15 files belonged to the same test execution

**Solution - Unified Test Run Timestamps:**

```python
# Generate single timestamp at test start
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

# Pass to all result saving operations
await self._save_search_results(test_name, query, search_data, run_timestamp)
```

**Result:**
All files from same test run now share timestamp: `basename_python_20250801_134416_697.json`, `ast_visitor_20250801_134416_697.json`

**Benefits:**

- Easy identification of test run groups
- Simplified cleanup of test results
- Better debugging workflow for batch test analysis

### Case Study 4: Chunking Method and Error Tracking Implementation (2025)

**Enhancement:** Extended the existing `analysis_method` concept to include comprehensive chunking method tracking and tree-sitter error detection.

**Implementation Details:**

**1. Chunking Method Tracking:**

- **Rust Implementation**: Added `chunking_method` to all chunk creation functions in `/workspaces/rust/rust/src/lib.rs`
- **Python Integration**: Updated `/workspaces/rust/python/cocoindex_code_mcp_server/lang/haskell/haskell_ast_chunker.py` to propagate chunking method metadata
- **Consistent Tracking**: Ensures every chunk includes information about the method used (`ast_recursive`, `regex_fallback`, etc.)

**2. Tree-sitter Error Tracking:**

- **Chunking Errors**: `tree_sitter_chunking_error` tracks when error nodes are encountered during parsing for chunking
- **Analysis Errors**: `tree_sitter_analyze_error` tracks when tree-sitter errors occur during code analysis
- **Rust Integration**: Added error detection based on `node.is_error()` in tree-sitter parsing

**Example Output:**

```json
{
  "analysis_method": "haskell_chunk_visitor",
  "chunking_method": "ast_recursive",
  "tree_sitter_chunking_error": "false",
  "tree_sitter_analyze_error": "false"
}
```

**Debugging Value:**

- **Method Verification**: Confirm that advanced chunking methods (`ast_recursive`) are being used vs fallbacks (`regex_fallback`)
- **Error Detection**: Identify problematic code sections that cause tree-sitter parsing failures
- **Quality Assurance**: Track the success rate of different chunking strategies across the codebase
- **Performance Optimization**: Identify files that consistently trigger fallback methods

**Testing:**

```python
# Test chunking method tracking
from cocoindex_code_mcp_server.lang.haskell.haskell_ast_chunker import EnhancedHaskellChunker, HaskellChunkConfig

clean_code = '''
module Test where
factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)
'''

config = HaskellChunkConfig(max_chunk_size=500)
chunker = EnhancedHaskellChunker(config)
chunks = chunker.chunk_code(clean_code, 'test.hs')

# Expected results:
# chunking_method: "haskell_ast_with_context"
# tree_sitter_chunking_error: "false"
```

**Integration with Existing Debugging:**
The new tracking properties integrate seamlessly with existing CocoIndex debugging workflows:

- SQL queries can filter by chunking method
- Evaluation mode captures detailed method information
- Database inspection reveals chunking strategy distribution
- Error analysis identifies parsing issues at both chunking and analysis phases
