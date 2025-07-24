# TODO - CocoIndex Code MCP Server

## Current Status - Multi-Language Smart Embedding Implementation COMPLETE ✅

### Major Achievement: Smart Embedding Working 🚀
- **Language-aware model selection**: GraphCodeBERT, UniXcode, fallback models properly routing based on language
- **Flow-level filtering**: Implemented using CocoIndex DataSlice.filter() for proper smart embedding
- **Comprehensive testing**: 51/51 tests passing in tests/smart_embedding/ directory
- **Multi-language capabilities baseline**: Established what works out-of-the-box across 8 languages

### Baseline Results from Multi-Language Testing 📊
```
Language     Model                Analysis Method         Fields   Status
Python       graphcodebert-base   tree_sitter+python_ast    6      ✅ Rich AST analysis
Rust         unixcoder-base       basic                     0      ⚠️  Text-only analysis  
JavaScript   graphcodebert-base   basic                     0      ⚠️  Text-only analysis
TypeScript   unixcoder-base       basic                     0      ⚠️  Text-only analysis
Java         graphcodebert-base   basic                     0      ⚠️  Text-only analysis
Go           graphcodebert-base   basic                     0      ⚠️  Text-only analysis
C++          graphcodebert-base   basic                     0      ⚠️  Text-only analysis
Haskell      all-MiniLM-L6-v2     basic                     0      ⚠️  Text-only analysis
```

**Key Finding**: Massive metadata extraction gap - only Python has AST-level analysis with rich metadata (functions, classes, imports, decorators, complexity scores). All other languages use basic text analysis with zero structured metadata.

## Next Session Priority: Haskell AST Enhancement 🎯

### Immediate Tasks (Start of Next Session)
1. **🔍 Analyze language handler architecture**
   - Find Python AST visitor implementation at `language_handlers/python_handler.py` 
   - Understand AST visitor pattern used for rich metadata extraction
   - Identify extension points for new language handlers

2. **🛠️ Implement Haskell AST Visitor**
   - Create `language_handlers/haskell_handler.py` using AST visitor pattern
   - Target metadata extraction: functions, data types, imports, type signatures, complexity
   - Goal: Match Python's 6 metadata fields with Haskell-specific analysis

3. **✅ Test Haskell Enhancement**  
   - Run multi-language baseline test to verify improvement
   - Target: Haskell moving from 0 fields (basic) to 4-6 fields (AST analysis)
   - Validate AST visitor support enabled

### Secondary Tasks
4. **🧪 Integration Testing**
   - Create end-to-end MCP integration test similar to `test_mcp_integration_http.py`
   - Validate smart embedding working in actual RAG queries

5. **🧹 Code Quality**
   - Investigate dual Python analyzer situation (`python_handler.py` vs `tree_sitter_python_analyzer.py`)
   - Mark deprecation if one is redundant

### Future Architecture (Lower Priority)
6. **📊 Qdrant Abstraction Layer**
   - Plan VectorStoreBackend interface design
   - Prepare for multi-database support

## Technical Implementation Notes 📝

### Files Successfully Modified
- **`src/cocoindex-code-mcp-server/cocoindex_config.py`**: Smart embedding implementation with LANGUAGE_MODEL_GROUPS
- **`tests/smart_embedding/`**: Complete test suite (test_language_model_mapping.py, test_embedding_functions.py, test_rag_integration.py)
- **`tests/lang/test_multi_language_*.py`**: Baseline assessment framework  
- **`tests/fixtures/`**: Multi-language test files

### Architecture Patterns Identified
- **Flow-level filtering**: `DataSlice.filter()` method for language-aware embedding selection
- **Model group mapping**: LANGUAGE_MODEL_GROUPS dict with model->languages mapping
- **Transform flows**: `@cocoindex.transform_flow()` decorators for embedding functions
- **AST visitor pattern**: Python shows rich metadata extraction via AST analysis

### Key Configuration
```python
LANGUAGE_MODEL_GROUPS = {
    'graphcodebert': {
        'model': 'microsoft/graphcodebert-base',
        'languages': {'python', 'java', 'javascript', 'php', 'ruby', 'go', 'c', 'c++'}
    },
    'unixcoder': {
        'model': 'microsoft/unixcoder-base', 
        'languages': {'rust', 'typescript', 'tsx', 'c#', 'kotlin', 'scala', 'swift', 'dart'}
    },
    'fallback': {
        'model': 'sentence-transformers/all-MiniLM-L6-v2',
        'languages': set()  # Catches all others including Haskell
    }
}
```

## Memory Context for Next Session 🧠
Stored in gw-memory with tags `cocoindex-code-mcp-server`, `smart-embedding`, `haskell-enhancement`:
- Smart embedding implementation complete and working
- Multi-language baseline showing Python superiority in metadata extraction  
- Next focus: Haskell AST visitor to match Python's analysis quality
- Architecture analysis showing AST visitor pattern as key to rich metadata

---
*Session ended 2025-07-24: Smart embedding complete, ready for Haskell AST enhancement tomorrow*

## Previous Analysis (Background Context)

[Previous TODO content about CocoIndex RAG Architecture Analysis continues below for reference...]

---

### Original Architecture Analysis (Background)

Based on analysis of `docs/instructions/cocoindex-rag-architecture.md`, `docs/instructions/queries-in-cocoindex.md`, `docs/vectordb/hybrid-search-with-pgvector-vs-qdrant.md` and our current implementation using RAG search, several architectural improvements have been identified to better align with best practices for vector database abstraction and maintainability.

#### Current Implementation Analysis

**✅ Strengths Found**
1. **Hybrid Search Implementation**: We have a working `HybridSearchEngine` with vector similarity + keyword filtering
2. **PostgreSQL + pgvector Backend**: Functional PostgreSQL-based vector search using pgvector extension
3. **MCP Server Integration**: Working MCP server with proper tool definitions and transport
4. **Advanced Keyword Parser**: Sophisticated Lark-based parser with `value_contains` operator support
5. **Live Update System**: File monitoring and incremental indexing capabilities
6. **CocoIndex Integration**: Proper CocoIndex flow integration for embedding and chunking

**❌ Gaps Identified Against Architecture Document**

1. **Vector Database Abstraction Layer** (HIGH PRIORITY)
   - Current: Direct PostgreSQL/pgvector integration in `hybrid_search.py`
   - Needed: Backend adapter pattern with `VectorStoreBackend` interface

2. **Export Schema Standardization** (MEDIUM PRIORITY)  
   - Current: Ad-hoc metadata structure throughout codebase
   - Needed: Normalized schema with `ChunkMetadata` TypedDict

3. **Query Layer Abstraction** (MEDIUM PRIORITY)
   - Current: Raw SQL queries mixed with business logic
   - Needed: Portable query layer with `ChunkQuery` abstraction

4. **Qdrant Integration Preparation** (HIGH PRIORITY)
   - Current: No Qdrant support, PostgreSQL-only
   - Needed: Qdrant client integration, memory-mapping, payload indexing

This background analysis provides context for future architectural improvements, but the immediate focus is on completing Haskell AST enhancement to achieve metadata extraction parity across languages.