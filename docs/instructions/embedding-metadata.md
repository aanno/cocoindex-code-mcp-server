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
- ✅ Smart embeddings working with multiple models:
  - `sentence-transformers/all-MiniLM-L6-v2` (default/fallback)
  - `microsoft/graphcodebert-base` (Python, Java, JavaScript, etc.)
  - `microsoft/unixcoder-base` (Rust, TypeScript, C#, etc.)

### Critical Constraint

**You cannot compare embedding vectors created with different models!** The `embedding_model` field ensures all vector similarity searches are filtered to only compare embeddings from the same model.
