# CocoIndex Code MCP Server - Current State

## Summary

Successfully fixed critical issues in the Python AST analysis pipeline and RAG system. The codebase now properly detects class methods, docstrings, decorators, and handles DataSlice objects correctly.

## Major Fixes Completed

### 1. Fixed TODO at cocoindex_config.py:261 ✅

**Problem**: PythonNodeHandler not properly integrated with TreeSitterPythonAnalyzer
**Solution**:

- **File**: `python/cocoindex_code_mcp_server/cocoindex_config.py`
- **Change**: Line 261 replaced direct PythonNodeHandler instantiation with proper TreeSitterPythonAnalyzer integration
- **Code**: `analyzer = TreeSitterPythonAnalyzer(prefer_tree_sitter=True)` instead of `handler = PythonNodeHandler()`

### 2. Fixed Critical .kind vs .type Bug ✅

**Problem**: AST node processing failed due to incorrect attribute access
**Root Cause**: Tree-sitter nodes use `.type` attribute, not `.kind`
**Files Fixed**:

- `python/cocoindex_code_mcp_server/ast_visitor.py`: GenericMetadataVisitor.visit_node()
- `python/cocoindex_code_mcp_server/language_handlers/python_handler.py`: All extract methods
**Result**: Function detection now works correctly, finding all methods including class methods

### 3. Fixed Class Decorator Merging ✅

**Problem**: Class decorators not appearing in global decorators list
**Files Fixed**:

- `python/cocoindex_code_mcp_server/lang/python/python_code_analyzer.py`: Added class decorators to global list
- `python/cocoindex_code_mcp_server/lang/python/tree_sitter_python_analyzer.py`: Enhanced metadata merging
**Result**: Both function and class decorators now properly detected and merged

### 4. Fixed RAG System DataSlice Error ✅

**Problem**: `'DataSlice' object has no attribute 'lower'` and PyTorch meta tensor errors
**File**: `python/cocoindex_code_mcp_server/cocoindex_config.py`
**Solution**: Created proper CocoIndex function pattern:

```python
@cocoindex.op.function()
def embed_text_with_smart_model(text: str, language: str) -> NDArray[np.float32]:
    # Uses cocoindex.functions.SentenceTransformerEmbed() properly

@cocoindex.transform_flow()
def smart_code_to_embedding(text: DataSlice[str], language: DataSlice[str]):
    return text.transform(embed_text_with_smart_model, language)
```

## Test Coverage Created

### Core Integration Tests

- **tests/test_python_handler_integration.py**: Comprehensive TreeSitterPythonAnalyzer integration tests
- **tests/test_debug_ast_nodes.py**: AST node processing debug tests (converted to proper pytest)
- **tests/test_class_decorator_merging.py**: Class decorator detection and merging tests (5 test cases)

### Test Results

- ✅ Function detection: Finds all methods including class methods, private methods, dunder methods
- ✅ Docstring detection: Properly detects and extracts docstrings
- ✅ Decorator detection: Both function and class decorators detected and merged
- ✅ No more .kind/.type attribute errors
- ✅ RAG system no longer crashes with DataSlice errors

## Current Capabilities

### Python Code Analysis

- **Functions**: Detects all function types (regular, async, methods, private, dunder)
- **Classes**: Full class analysis with inheritance and decorators
- **Docstrings**: Proper extraction and detection flags
- **Decorators**: Global list includes both function and class decorators
- **Type Hints**: Detection of return types and parameter annotations
- **Imports**: Comprehensive import analysis
- **Variables**: Module and class-level variable detection

### RAG System

- **Background Updates**: ✅ Fixed DataSlice crashes during flow updates
- **Proper CocoIndex Integration**: ✅ Uses approved patterns and functions
- **Smart Embedding**: ⚠️ Temporarily disabled - requires flow-level restructuring (see below)

## Current Issues

### Smart Embedding Implementation ⚠️

**Status**: Temporarily falls back to default embedding
**Problem**: CocoIndex `SentenceTransformerEmbed` requires static model selection, cannot dynamically choose models per chunk
**Solution Required**: Flow-level restructuring as documented in `docs/cocoindex/smart-embedding.md`

**Proper Pattern** (needs implementation at flow_def level):

```python
# 1. Add language detection per chunk
chunk["language"] = chunk["text"].transform(detect_language)

# 2. Filter by language to create separate DataSlice subsets
python_chunks = chunks.filter(lambda row: row["language"] == "python")
rust_chunks = chunks.filter(lambda row: row["language"] == "rust")

# 3. Embed each subset with appropriate model
python_embeddings = python_chunks["text"].transform(
    cocoindex.functions.SentenceTransformerEmbed(model="microsoft/graphcodebert-base")
)
rust_embeddings = rust_chunks["text"].transform(
    cocoindex.functions.SentenceTransformerEmbed(model="microsoft/unixcoder-base")
)

# 4. Merge results back together
```

This requires changes to the main flow definition, not individual transform functions.

## Next Steps Planned

### 1. RAG Verification Testing 🔄

**Goal**: Verify that improvements (class methods, docstrings, decorators) are accessible through RAG queries
**Approach**:

```python
# Test strategy
def test_rag_improvements():
    # 1. Create sample code with class methods, docstrings, decorators
    # 2. Index it through RAG system
    # 3. Query for specific elements:
    #    - "find class methods with docstrings"
    #    - "find functions with @property decorator"
    #    - "find classes with @dataclass decorator"
    # 4. Verify results include our improvements
```

### 2. Custom vs Default CocoIndex Comparison 🔄

**Goal**: Test that our `language_handlers/python_handler.py` provides better results than default CocoIndex
**Approach Options**:

1. **A/B Testing**: Same code analyzed with/without our custom handler
2. **Feature Coverage**: Compare metadata richness (our handler should detect more)
3. **Benchmark Suite**: Standard Python files analyzed both ways
4. **Specific Improvements**: Test edge cases our handler handles better

**Implementation Ideas**:

```python
def compare_handlers():
    # Option 1: Disable our handler, use default CocoIndex
    # Option 2: Mock/bypass our extensions
    # Option 3: Separate test environment with vanilla CocoIndex
    # Compare: function count, decorator detection, docstring extraction
```

## Technical Architecture

### Key Components

- **TreeSitterPythonAnalyzer**: Enhanced analyzer with dual AST approach
- **PythonNodeHandler**: Tree-sitter specific node processing
- **PythonCodeAnalyzer**: Python AST fallback with rich metadata
- **Smart Embedding**: Language-aware model selection
- **Generic Metadata Visitor**: Framework for pluggable language handlers

### Integration Points

- **cocoindex_config.py**: Main flow definitions and embedding functions
- **ast_visitor.py**: Core AST traversal framework
- **language_handlers/**: Pluggable language-specific processors
- **lang/python/**: Python-specific analyzers and utilities

## Configuration

- **Default Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Smart Models**: GraphCodeBERT (Python/JS), UniXcoder (Rust), etc.
- **Analysis Preference**: Tree-sitter first, Python AST fallback
- **Extensions**: AST chunking, Python handlers, smart embedding all loaded

## Known Limitations

- Smart embedding falls back to default when language detection fails
- Some complex decorator expressions may not parse perfectly
- Very large code files (>100KB) use regex fallback
- Nested class analysis depth limited to prevent recursion

## Files Modified

### Core Fixes

- `python/cocoindex_code_mcp_server/cocoindex_config.py`
- `python/cocoindex_code_mcp_server/ast_visitor.py`
- `python/cocoindex_code_mcp_server/language_handlers/python_handler.py`
- `python/cocoindex_code_mcp_server/lang/python/python_code_analyzer.py`
- `python/cocoindex_code_mcp_server/lang/python/tree_sitter_python_analyzer.py`

### Tests Added

- `tests/test_python_handler_integration.py`
- `tests/test_debug_ast_nodes.py`
- `tests/test_class_decorator_merging.py`

## Success Metrics

- ✅ All pytest tests pass
- ✅ No more RAG system crashes
- ✅ Function detection accuracy: ~100% (finds all expected methods)
- ✅ Decorator detection: Both function and class decorators
- ✅ Docstring detection: Properly flags and extracts content
- ✅ Import compatibility: No breaking changes to existing flows

**Status**: Ready for RAG verification testing and comparative analysis with default CocoIndex implementation.
