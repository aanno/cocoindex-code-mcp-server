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

### 4. Flow Configuration Issues

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
   - "Using AST chunking extension" vs "Using default recursive splitting"
   - "Using custom language handler extension" vs "Using default language handler"
   - File count: "Adding source: path as 'files'"

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
    JSON_EXTRACT(metadata_json, '$.analysis_method') as method,
    COUNT(*) as count
FROM code_embeddings 
GROUP BY method;

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

## Real-World Case Study

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