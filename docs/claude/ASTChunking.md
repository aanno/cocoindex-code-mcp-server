# AST Chunking in CocoIndex

## 🎯 **Preferred Implementation Pattern (January 2025)**

The current preferred approach for implementing AST chunking in CocoIndex uses the `@op.executor_class()` pattern as demonstrated in:

- `python/cocoindex_code_mcp_server/ast_chunking.py` (Python AST chunking)
- `python/cocoindex_code_mcp_server/lang/haskell/haskell_ast_chunker.py` (Haskell AST chunking)

### **Modern Pattern: @op.executor_class()**

```python
from dataclasses import dataclass
from cocoindex import op
from typing import Any

@dataclass
class MyChunkRow:
    """Typed chunk representation for CocoIndex."""
    content: str
    location: str
    start: int
    end: int
    chunking_method: str

    # Dictionary-style access for backward compatibility
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

class MyChunkSpec(op.FunctionSpec):
    """Function specification for MyChunk operation."""
    max_chunk_size: int = 1800
    chunk_overlap: int = 0
    chunk_expansion: bool = False

@op.executor_class()
class MyChunkExecutor:
    """Executor for MyChunk AST-based code chunking."""
    spec: MyChunkSpec

    def analyze(self, content: Any, language: Any = "MyLanguage") -> type:
        """Analyze method required by CocoIndex to determine return type."""
        return list[MyChunkRow]

    def __call__(self, content: str, language: str = "MyLanguage") -> list[MyChunkRow]:
        """Main chunking function - returns typed chunk structures for CocoIndex."""
        # Implementation here
        chunks = []
        # ... chunking logic ...
        return chunks
```

### **Key Features of Modern Pattern**

1. **Typed Returns**: Use dataclasses with proper typing
2. **Dictionary Compatibility**: Add `__getitem__`, `__contains__`, `get()` for backward compatibility
3. **Analyze Method**: Required by CocoIndex to determine return types
4. **Spec Classes**: Separate configuration from execution logic
5. **Unique Locations**: Generate unique location strings for each chunk

## 🎯 **Legacy Integration Context**

The following sections document the historical integration work with ASTChunk library.

## 📋 **Completed Work**

### ✅ **Phase 1: Analysis & Planning**

1. **Analyzed ASTChunk project** - Comprehensive analysis documented in `ASTChunk.md`
2. **Designed integration strategy** - CocoIndex operation with hybrid chunking approach
3. **Created integration module** - `python/cocoindex_code_mcp_server/
ast_chunking.py` with CocoIndex-compatible interface

### ✅ **Phase 2: Implementation**

1. **Created `ast_chunking.py`** - Main integration module with:
   + `CocoIndexASTChunker` class for AST-based chunking
   + Language mapping (Python, Java, C#, TypeScript)
   + Fallback to existing Haskell chunking
   + CocoIndex operation factory function

2. **Updated `cocoindex_config.py`** - Enhanced flow configuration:
   + Added ASTChunk imports
   + Created `create_hybrid_chunking_operation()` function
   + Modified flow to use hybrid chunking approach
   + Supports both AST-based and regex-based chunking

### ✅ **Phase 3: Dependencies**

1. **Installed ASTChunk dependencies**:
   + tree-sitter and language parsers
   + numpy, pyrsistent
   + All requirements from `astchunk/requirements.txt`

## ✅ **Issues Resolved**

### **Import/Environment Problems - FIXED**

1. **Virtual Environment**: ✅ Dependencies working with `~/.venv/bin/activate`
2. **Circular Import**: ✅ Resolved by making CocoIndex import conditional
3. **Path Issues**: ✅ ASTChunk module path configuration working correctly

### **Technical Solutions Applied**

- Made CocoIndex import conditional to avoid circular dependencies
- Added proper error handling for missing dependencies
- Implemented fallback mechanisms for unsupported languages

## 📁 **Files Modified/Created**

### **New Files**

- `python/cocoindex_code_mcp_server/ast_chunking.py` - Main ASTChunk integration module
- `ASTChunk.md` - Comprehensive analysis and integration plan
- `STATE.md` - This current state document

### **Modified Files**

- `python/cocoindex_code_mcp_server/cocoindex_config.py` - Added hybrid chunking operation and imports

## 🔧 **Technical Architecture**

### **Hybrid Chunking Strategy**

```python
# Flow: Code → Language Detection → Chunking Strategy Selection
if language_supported_by_astchunk:
    use_ast_chunking()  # Python, Java, C#, TypeScript
elif language == "Haskell":
    use_haskell_ast_chunking()  # Our existing implementation
else:
    use_regex_chunking()  # Fallback with custom separators
```

### **Integration Points**

1. **CocoIndex Operation**: `create_hybrid_chunking_operation()`
2. **Language Support**:
   + AST-based: Python, Java, C#, TypeScript
   + Haskell: Our existing tree-sitter implementation
   + Others: Regex-based with custom separators
3. **Metadata Enhancement**: Rich chunk metadata with line numbers, file paths, etc.

## ✅ **Completed Tasks**

### **Integration Complete**

1. **Import Issues**: ✅ RESOLVED
   + Fixed circular import with CocoIndex using conditional imports
   + Virtual environment working properly
   + ASTChunk functionality fully operational

2. **Hybrid Chunking**: ✅ WORKING
   + AST chunking operational for supported languages (Python, Java, C#, TypeScript)
   + Fallback to simple text chunking for unsupported languages
   + Haskell integration ready (falls back to simple chunking when CocoIndex unavailable)

3. **Testing & Validation**: ✅ COMPLETED
   + Created comprehensive test cases for different languages
   + Validated chunk quality and metadata
   + Confirmed fallback mechanisms work correctly

### **Future Enhancements**

1. **Multi-language AST Support** - Extend beyond current languages
2. **Unified AST Processing Framework** - Standardize across all languages (ANALYZED - see below)
3. **Performance Optimization** - Caching and efficient processing
4. **Advanced Features** - Semantic search, code understanding, documentation generation

## 🔗 **Todo List Status**

- [✅] Tasks 1-8: Completed (modular refactoring, analysis)
- [✅] Task 9: Design CocoIndex integration strategy (COMPLETE)
- [✅] Task 10: Implement AST-based chunking operation (COMPLETE)
- [✅] Task 11: Add multi-language AST support (COMPLETE - Python, Java, C#, TypeScript)
- [⏳] Task 12: Create unified AST processing framework (ready for next phase)

## 🎉 **Key Achievements**

1. **Successfully integrated ASTChunk** with CocoIndex architecture
2. **Created hybrid approach** that preserves existing Haskell functionality
3. **Designed extensible system** for future language support
4. **Maintained backwards compatibility** with existing code

## 🎯 **Current Status: COMPLETE**

### **✅ All Issues Resolved**

1. ✅ Virtual environment working properly
2. ✅ Circular import issues resolved with conditional imports
3. ✅ Path configuration working correctly
4. ✅ Multi-language chunking fully tested and validated

### **🚀 Integration Summary**

- **ASTChunk successfully integrated** with CocoIndex
- **Hybrid chunking system** operational for 4+ languages
- **Fallback mechanisms** working for unsupported languages
- **Comprehensive testing** completed and validated
- **Ready for production use** in CocoIndex pipelines

---

## 📊 **Analysis: Unified AST Processing Framework (Task 12)**

### **Current State Assessment**

**✅ What We Have:**

- **`ast_visitor.py`** - Generic AST visitor framework with tree-sitter support
- **`language_handlers/`** - Pluggable language-specific handlers (currently Python)
- **`ast_chunking.py`** - ASTChunk integration for code chunking
- **Python-specific analyzers** - Comprehensive Python AST analysis with RAG compliance
- **Tree-sitter infrastructure** - Basic framework for multiple languages

**❓ Current Fragmentation:**

- **Multiple AST approaches**: Python AST, tree-sitter, ASTChunk all separate
- **Language-specific silos**: Python has its own analyzer, others would need separate implementations
- **Inconsistent interfaces**: Different ways to analyze different languages
- **Duplicated functionality**: Similar parsing logic across components

### **🎯 Unified Framework Plan (4 Phases)**

#### **Phase 1: Framework Design**

- Create unified interface for all AST processing
- Standardize metadata output across all languages
- Define common abstractions for nodes, positions, relationships
- Design pluggable analyzer system with language handlers

#### **Phase 2: Core Infrastructure**

- Enhance `ast_visitor.py` as the central framework
- Standardize `NodeHandler` protocol for all languages
- Create unified `ASTAnalyzer` class that orchestrates everything
- Implement metadata normalization to RAG-compliant format

#### **Phase 3: Language Integration**

- Refactor Python analyzer to use unified framework
- Add JavaScript/TypeScript handlers using tree-sitter
- Add Java/C# handlers using tree-sitter
- Integrate ASTChunk as one of the analysis backends

#### **Phase 4: Advanced Features**

- Cross-language code understanding
- Unified semantic analysis
- Relationship mapping between different files/languages
- Performance optimization with caching

### **🤔 Assessment: Is This Needed NOW?**

#### **✅ Arguments FOR:**

- Foundation for growth as we add more languages
- Code quality - would eliminate current fragmentation
- Maintainability - easier to maintain unified system
- Consistency - all languages would have same metadata format
- Performance - could optimize across all languages

#### **❌ Arguments AGAINST:**

- Current system works - Python analysis is complete and working
- Over-engineering risk - might add complexity without immediate benefit
- Time investment - significant effort for uncertain immediate value
- Requirements unclear - don't know what other languages we'll actually need
- YAGNI principle - "You Aren't Gonna Need It" - premature optimization

### **💡 Recommendation: DEFER**

**✅ DECISION: Do NOT implement unified AST processing framework now**

**Reasons:**

1. **No immediate need** - Current system meets all requirements
2. **Unknown future requirements** - We don't know what languages we'll actually need
3. **Risk of over-engineering** - Could add complexity without clear benefit
4. **Working system** - Don't break what's working well
5. **Better to wait for real needs** - Implement when we actually need other languages

**📋 What to do instead:**

- ✅ Document the current architecture clearly (THIS DOCUMENT)
- ✅ Create interfaces that could support unification later
- ✅ Keep the door open for future unification
- ✅ Focus on immediate user needs rather than theoretical architecture

**🔄 When to reconsider:**

- When we need 2+ more languages with full analysis
- When maintenance becomes difficult due to fragmentation
- When performance becomes an issue across languages
- When we have clear requirements for cross-language features

### **🎯 Current Priority Context**

- ✅ RAG metadata compliance - COMPLETE
- ✅ Lark parser implementation - COMPLETE
- ✅ Hybrid search working - COMPLETE
- ✅ Python analysis complete - COMPLETE
- ✅ ASTChunk integration - COMPLETE

**Status**: The unified framework is a good idea **in principle**, but not a good idea **right now** given our current state and priorities.

---

**Status**: ✅ **INTEGRATION COMPLETE** - AST chunking fully operational and ready for use!

**Task 12 Status**: ❌ **DEFERRED** - Unified AST framework analysis complete, implementation deferred pending real multi-language requirements.

---

## 🔥 **MAJOR BREAKTHROUGH: Field Promotion & Metadata Flow (January 2025)**

### **Critical Discovery: chunking_method Conflict Resolution**

During investigation of chunking_method values, we discovered fundamental patterns about how CocoIndex handles metadata flow from operations to search results. This breakthrough explains the entire metadata architecture.

#### **The Problem**

- Only 3 chunking_method values appeared in test results: `ast_tree_sitter`, `rust_haskell_ast`, `rust_haskell_ast_with_errors`
- Expected to see `astchunk_library` and other diverse methods from AST chunkers
- Before changes, "astchunk_library" was wrongly overwriting other legitimate values

#### **Root Cause: Metadata Conflicts**

The issue was **dual sources of truth** for the same field:

```python
# Chunk from ASTChunk operation had:
chunk.chunking_method = "astchunk_library"  # From AST chunker

# But metadata_json had:
metadata_json = {"chunking_method": "ast_tree_sitter"}  # From tree-sitter analysis

# Result: Conflicting values in search results
result = {
    "chunking_method": "astchunk_library",  # Direct field
    "metadata_json": {"chunking_method": "ast_tree_sitter"}  # Metadata field
}
```

#### **The Solution: Field Source Separation**

1. **Preserve ALL chunking methods from AST chunkers** (not just "astchunk_library")
2. **Remove chunking_method from metadata_json** to eliminate confusion
3. **Single source of truth**: chunk.chunking_method is the authoritative value

### **Key Implementation Changes**

#### **schemas.py:304**

```python
# BEFORE: Added chunking_method to metadata causing conflicts
validated["chunking_method"] = str(metadata.get("chunking_method", "unknown"))

# AFTER: Removed to avoid confusion
# NOTE: chunking_method removed from metadata to avoid confusion - it comes from AST chunkers only
```

#### **cocoindex_config.py: Multiple Lines**

```python
# BEFORE: Metadata included chunking_method assignments
"chunking_method": preserve_chunking_method if preserve_chunking_method else "unknown_chunking",

# AFTER: All chunking_method removed from metadata creation
# NOTE: chunking_method removed from metadata - it comes from AST chunkers only
```

### **Critical Patterns Discovered**

#### **1. Automatic Field Promotion**

ALL fields in metadata_json are automatically promoted to top-level search result fields:

```python
# If metadata_json contains:
{"analysis_method": "python_ast", "custom_field": "value"}

# Search results automatically get:
{
    "analysis_method": "python_ast",  # Promoted from metadata_json
    "custom_field": "value",         # Promoted from metadata_json
    "metadata_json": {"analysis_method": "python_ast", "custom_field": "value"}
}
```

#### **2. Dataclass to Field Conversion**

AST operations return typed dataclasses that get converted to result fields:

```python
@dataclass
class ASTChunkRow:
    content: str
    chunking_method: str  # This becomes a result field

# CocoIndex automatically converts:
chunk = ASTChunkRow(content="code", chunking_method="astchunk_library")
# To result field:
result["chunking_method"] = "astchunk_library"
```

#### **3. Conflict Avoidance Pattern**

**❌ ANTI-PATTERN**: Same field in both direct collection and metadata_json

```python
code_embeddings.collect(
    chunking_method=chunk["chunking_method"],  # Direct field
    metadata_json={"chunking_method": "different_value"}  # CONFLICTS!
)
```

**✅ BEST PRACTICE**: Choose single source per field

```python
code_embeddings.collect(
    chunking_method=chunk["chunking_method"],  # From AST chunkers only
    metadata_json={"analysis_method": "..."}   # Other fields only
)
```

### **Testing Results: Perfect Success**

**Before Fix** (Problematic):

```
📄 File: python_minor_errors.py
   chunking_method: 'astchunk_library'
   metadata chunking_method: 'ast_tree_sitter'  ← CONFLICTING VALUES
```

**After Fix** (Correct):

```
📄 File: python_minor_errors.py
   chunking_method: 'astchunk_library'
   metadata chunking_method: 'none'  ← NO CONFUSION!
```

**Diverse chunking methods now preserved**:

- `astchunk_library` (4 occurrences) - from ASTChunk library
- `ast_tree_sitter` (16 occurrences) - from tree-sitter analysis
- `rust_haskell_regex_fallback_3` - from Rust Haskell implementation
- `rust_haskell_error_recovery` - from Rust error handling

### **Universal Metadata Flow Patterns**

#### **Pattern 1: Properties in metadata_json**

```python
# Add properties to metadata_json in collector logic:
metadata_json = {
    "analysis_method": "python_ast",
    "file_size": len(content),
    "has_tests": "test" in content.lower(),
    "custom_property": calculate_value(content)
}
```

#### **Pattern 2: Automatic promotion to results**

```python
# ALL metadata_json fields automatically become result fields
# No additional code needed - CocoIndex handles this automatically
```

#### **Pattern 3: Direct result fields**

```python
# Collect fields directly for immediate result inclusion:
code_embeddings.collect(
    filename=file["filename"],
    chunking_method=chunk["chunking_method"],  # Direct from operation
    functions=extract_functions(content),       # Direct calculation
    metadata_json=metadata_dict                # Bulk metadata
)
```

#### **Pattern 4: Typed operation conversion**

```python
# Operations return dataclasses that get converted to result fields:
@op.executor_class()
class MyOperation:
    def __call__(self, input) -> list[MyDataClass]:
        return [MyDataClass(field1="value", field2="value")]

# Result automatically gets field1 and field2 as top-level properties
```

### **Implications for Future Development**

1. **Metadata Strategy**: Use metadata_json for bulk properties, direct fields for operation outputs
2. **Conflict Prevention**: Never put the same field in both direct collection and metadata_json
3. **Field Promotion**: Leverage automatic promotion - no manual result field creation needed
4. **Typed Operations**: Use dataclasses for structured operation outputs
5. **Single Source of Truth**: Each field should have exactly one authoritative source

### **This Breakthrough Enables**

- **Rich metadata collection** without schema changes
- **Automatic result field promotion** from metadata_json
- **Conflict-free field management** with clear source separation
- **Typed operation integration** with automatic field conversion
- **Flexible development patterns** for metadata experimentation

### **Documentation Impact**

This discovery has been documented in:

- `docs/cocoindex/flow-and-types.md` - Complete metadata flow patterns
- This file - ASTChunk integration context
- Test files - Validation of the patterns

**Status**: 🔥 **BREAKTHROUGH COMPLETE** - Fundamental metadata flow patterns discovered and documented!
