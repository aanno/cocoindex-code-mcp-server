# embedding metadata

## Overview

Our MCP server supports 'smart embeddings', where the embedding model used depends on the programming language of the code being processed (see docs/claude/Embedding-Selection.md).

## Implementation Status: ✅ COMPLETED

The `embedding_model` metadata field has been successfully implemented to enable safe vector comparisons across different embedding models.

### What Was Implemented

1. **Database Column**: Added `embedding_model` text column to store which embedding model was used for each chunk
2. **Flow Integration**: Modified CocoIndex flow to populate `embedding_model` field for both smart and default embedding modes
3. **Search Filtering**: Updated search code to filter by `embedding_model` to ensure we only compare vectors from the same model
4. **API Enhancement**: Search API now accepts either `language` OR `embedding_model` parameter to specify filtering

### Key Components

- **Helper Functions** (cocoindex_config.py):
  - `get_embedding_model_name()`: Maps model group to actual model identifier
  - `get_default_embedding_model_name()`: Returns default model for non-smart embedding mode
  - `language_to_embedding_model()`: Maps programming language to appropriate embedding model

- **Backend Integration**:
  - VectorStoreBackend interface includes `embedding_model` parameter
  - PostgresBackend filters SQL queries by `embedding_model`
  - HybridSearchEngine resolves language→embedding_model automatically

- **Search Priority**: embedding_model > language > default

### Verified Results

The implementation has been tested and verified:

- ✅ Column `embedding_model` exists in database schema
- ✅ Data correctly populated with model identifiers
- ✅ Smart embeddings working with multiple models (all 768D):
  - `sentence-transformers/all-mpnet-base-v2` (default/fallback) - 768D
  - `microsoft/graphcodebert-base` (Python, Java, JavaScript, etc.) - 768D
  - `microsoft/unixcoder-base` (Rust, TypeScript, C#, etc.) - 768D

### Critical Constraint

**You cannot compare embedding vectors created with different models!** The `embedding_model` field ensures all vector similarity searches are filtered to only compare embeddings from the same model.

## Vector Dimension Standardization: 768D

**Date:** 2025-10-08

### Decision: Switch to 768D Embeddings

All embedding models now produce **768-dimensional vectors** for consistency and quality:

**Previous Configuration (384D mixed):**

- ❌ `sentence-transformers/all-MiniLM-L6-v2`: 384D (fallback)
- ❌ `microsoft/graphcodebert-base`: 768D → projected to 384D
- ❌ `microsoft/unixcoder-base`: 768D → projected to 384D

**New Configuration (768D unified):**

- ✅ `sentence-transformers/all-mpnet-base-v2`: 768D native (fallback)
- ✅ `microsoft/graphcodebert-base`: 768D native
- ✅ `microsoft/unixcoder-base`: 768D native

### Rationale

1. **Correctness:** Using models at native resolution avoids projection mismatch bugs
2. **Quality:** all-mpnet-base-v2 has significantly better semantic quality than all-MiniLM-L6-v2
3. **Simplicity:** Single dimension (768D) easier to reason about and debug
4. **Standards:** 768D is standard for BERT-based models, pgvector handles efficiently

### Trade-offs

**Benefits:**

- ✅ No information loss from dimension projection
- ✅ Better semantic search quality
- ✅ Consistent dimensions across all models
- ✅ Native model resolution

**Costs:**

- ❌ 2x storage increase (768D vs 384D)
- ❌ Marginally slower similarity search (negligible with HNSW index)
- ❌ Slower embedding generation for fallback model

### Migration Requirements

**Database Migration:**

- pgvector automatically adjusts to vector dimensions on first insert
- Requires full `--rescan` to regenerate all embeddings with new models
- Old 384D data will be replaced with new 768D embeddings

**Breaking Change:**

- Yes - all existing embeddings must be regenerated
- No backward compatibility with 384D embeddings

## Search Parameter Requirements

### Vector Search: REQUIRED `language` OR `embedding_model`

**Why Required:**

- Query must be embedded using a specific model
- Cannot search across all models (would require 3 separate embeddings + merge results)
- Must filter database to only search embeddings from the same model

**Implementation:**

- User MUST provide `language` OR `embedding_model` parameter
- Parameter determines which embedding model to use for query embedding
- Query is embedded ONCE with the appropriate model
- Database filtered to only search embeddings from that model

**Example:**

```python
# Search Python code using GraphCodeBERT embeddings
results = search_engine.vector_search(
    query="fibonacci algorithm implementation",
    language="Python"  # Will use microsoft/graphcodebert-base
)

# Or specify model directly
results = search_engine.vector_search(
    query="fibonacci algorithm implementation",
    embedding_model="microsoft/graphcodebert-base"
)
```

### Keyword Search: OPTIONAL `language`/`embedding_model`

**Why Optional:**

- Keyword search doesn't use embeddings (metadata filtering only)
- Users can filter by language in the keyword query itself: `"language:python OR language:rust"`
- Allows general metadata queries across all languages

**Implementation:**

- `language` and `embedding_model` parameters are OPTIONAL
- No validation required
- Allows flexible cross-language metadata searches

**Example:**

```python
# Search across all languages
results = search_engine.keyword_search(
    query="functions:fibonacci AND has_classes:true"
)

# Or filter to specific language
results = search_engine.keyword_search(
    query="functions:fibonacci AND language:Python"
)
```

### Hybrid Search: REQUIRED `language` OR `embedding_model`

**Why Required:**

- Combines vector similarity with keyword filtering
- Must filter by embedding_model for valid vector comparisons
- Same reasoning as vector search

**Implementation:**

- User MUST provide `language` OR `embedding_model` parameter
- Parameter determines embedding model for both query embedding and database filtering
- Ensures only embeddings from same model are compared

**Example:**

```python
# Hybrid search within Python code
results = search_engine.hybrid_search(
    vector_query="complex algorithm implementation",
    keyword_query="has_type_hints:true",
    language="Python"  # Required!
)
```

### Error Messages

**Missing required parameter:**

```python
ValueError: Either 'language' or 'embedding_model' parameter is required for search.
This ensures you only get results from the appropriate embedding model.
Examples: language='Python', embedding_model='microsoft/graphcodebert-base'
```

## TODO List

- [x] Implement `embedding_model` database column
- [x] Update CocoIndex flow to populate `embedding_model`
- [x] Add backend filtering by `embedding_model`
- [x] Fix `--rescan` flag for PostgreSQL lowercase table names
- [x] Verify smart embeddings working in production
- [ ] Add validation to vector_search requiring `language` OR `embedding_model`
- [ ] Add validation to hybrid_search requiring `language` OR `embedding_model`
- [ ] Keep keyword_search with optional `language`/`embedding_model`
- [ ] Update embedding model selection for query embedding in vector search
- [ ] Update test fixtures to add `language` parameter
- [ ] Test validation with missing parameters
- [ ] Update integration tests
