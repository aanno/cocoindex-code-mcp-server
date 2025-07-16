# RAG experiments

`/workspaces/rust/main.py` is simply a copy of             
`/workspaces/rust/cocoindex/examples/code_embedding`

The target of this project is to have an RAG for code development as MCP server.
There is a MCP server example at `/workspaces/rust/code-index-mcp`.

## State

### CocoIndex Framework

- Hybrid Rust/Python architecture with powerful data transformation capabilities
- Built-in vector embedding support (SentenceTransformer + Voyage AI)
- Incremental processing with PostgreSQL + pgvector backend
- Code chunking and embedding pipeline already implemented

### Existing MCP Server

- Basic file indexing and search functionality
- Advanced search with ripgrep/ugrep integration
- Language-specific analysis (Python, Java, JavaScript, etc.)
- No semantic search or RAG capabilities

## Plan

1. Enhanced Code Embedding Pipeline
   - Leverage CocoIndex's code_embedding_flow with improvements:
   - Better chunking strategies (function/class boundaries)
  - Multiple embedding models (code-specific models like CodeBERT)
  - Metadata enrichment (function signatures, dependencies, etc.)
2. Semantic Search Integration
   - Add vector similarity search to existing MCP server
   - Hybrid search combining exact matches + semantic similarity
   - Context-aware retrieval based on code relationships
3. RAG-Enhanced Code Analysis
   - Contextual code explanations using retrieved similar code
   - Pattern recognition and best practices suggestions
   - Cross-reference detection and dependency analysis

## cocoindex

* [build from sources](https://cocoindex.io/docs/about/contributing)
* [installation with pip](https://cocoindex.io/docs/getting_started/installation)
* [quickstart](https://cocoindex.io/docs/getting_started/quickstart)
* [cli](https://cocoindex.io/docs/core/cli)
* https://github.com/cocoindex-io/cocoindex

## code_embedding

* cocoindex/examples/code_embedding
* [blog post](https://cocoindex.io/blogs/index-code-base-for-rag/)
