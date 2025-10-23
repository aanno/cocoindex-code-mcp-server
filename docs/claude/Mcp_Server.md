# CocoIndex RAG MCP Server Documentation

## Overview

The CocoIndex RAG MCP Server provides hybrid search capabilities for code retrieval through the Model Context Protocol (MCP). It combines vector similarity search with keyword metadata filtering to enable sophisticated code discovery and analysis.

## Status: ✅ PRODUCTION READY

The MCP server is fully functional and successfully integrated with Claude Desktop. All tools are properly advertised and working.

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension
- CocoIndex installed and configured

### Starting the Server

```bash
# HTTP mode (recommended for Claude Desktop integration)
python -m cocoindex_code_mcp_server.main_mcp_server.py --port 3033 /workspaces/rust

# Stdio mode (for direct MCP client integration)
python -m cocoindex_code_mcp_server.main_mcp_server.py /workspaces/rust
```

### Claude Desktop Configuration

Add to `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cocoindex-rag": {
      "command": "pnpm",
      "args": [
        "dlx",
        "supergateway",
        "--streamableHttp",
        "http://localhost:3033/mcp"
      ]
    }
  }
}
```

## Architecture

### Core Components

1. **MCP Server** (`main_mcp_server.py`)
   + Built on mcp-1.12.0 SDK
   + Supports both HTTP/StreamableHTTP and stdio transports
   + Graceful shutdown and error handling

2. **Hybrid Search Engine** (`hybrid_search.py`)
   + Combines vector similarity with keyword metadata filtering
   + PostgreSQL + pgvector backend
   + Configurable scoring weights

3. **CocoIndex Integration** (`cocoindex_config.py`)
   + 30+ programming languages supported
   + Language-specific chunking and analysis
   + Embedding generation with SentenceTransformer

### Transport Modes

#### HTTP/StreamableHTTP (Recommended)

- **Advantages**: Clean separation of concerns, remote access, better error handling
- **Usage**: `--port 3033` argument
- **Integration**: Via supergateway proxy for Claude Desktop

#### Stdio (Legacy)

- **Usage**: Default mode without `--port` argument
- **Limitations**: stdio stream conflicts, harder debugging

## Available Tools

### 1. `hybrid_search` (search-hybrid)

Combines vector similarity and keyword metadata filtering.

**Parameters:**

- `vector_query` (required): Text for semantic similarity search
- `keyword_query` (required): Metadata filtering query
- `top_k` (optional): Number of results (default: 10)
- `vector_weight` (optional): Weight for vector score (default: 0.7)
- `keyword_weight` (optional): Weight for keyword score (default: 0.3)

**Example:**

```json
{
  "vector_query": "function to parse JSON",
  "keyword_query": "language:python AND function_name:parse",
  "top_k": 5
}
```

### 2. `vector_search` (search-vector)

Pure semantic similarity search using embeddings.

**Parameters:**

- `query` (required): Text to search for
- `top_k` (optional): Number of results (default: 10)

### 3. `keyword_search` (search-keyword)

Pure metadata-based filtering using Lark parser.

**Parameters:**

- `query` (required): Keyword query (e.g., `function_name:parse AND language:python`)
- `top_k` (optional): Number of results (default: 10)

### 4. `analyze_code` (code-analyze)

Extract metadata from code for indexing.

**Parameters:**

- `code` (required): Code content to analyze
- `file_path` (required): File path for context
- `language` (optional): Programming language (auto-detected if not provided)

### 5. `get_embeddings` (code-embeddings)

Generate embeddings for text using the configured model.

**Parameters:**

- `text` (required): Text to generate embeddings for

## Available Resources

### 1. `cocoindex://search/stats`

Database and search performance statistics.

### 2. `cocoindex://search/config`

Current hybrid search configuration and settings.

### 3. `cocoindex://database/schema`

Database table structure and schema information.

## Implementation History

### Problem Resolution

**Original Issue**: Claude Desktop couldn't see advertised tools despite successful MCP protocol communication.

**Root Cause**: The MCP server was including `null` fields (`title`, `outputSchema`, `annotations`, `meta`) in tool definitions, which confused Claude Desktop's tool discovery mechanism.

**Solution**: Modified tool serialization to use `model_dump(mode='json', exclude_none=True)` to remove null fields from JSON responses.

**Code Changes:**

```python
# Before (with null fields)
"result": {"tools": [tool.model_dump(mode='json') for tool in tools]}

# After (clean format)
"result": {"tools": [tool.model_dump(mode='json', exclude_none=True) for tool in tools]}
```

### Testing and Validation

#### Integration Tests

- **Location**: `tests/test_mcp_integration_http_e2e.py`
- **Coverage**: Full MCP protocol compliance via HTTP JSON-RPC
- **Status**: ✅ All tests passing

#### Protocol Validation

- **Initialize**: ✅ Returns proper capabilities and server info
- **Tools/list**: ✅ Returns 5 tools with clean schemas
- **Resources/list**: ✅ Returns 3 resources
- **Tools/call**: ✅ All tools execute successfully
- **Error handling**: ✅ Invalid requests handled gracefully

#### Claude Desktop Integration

- **Connection**: ✅ Server shows as connected
- **Tool Discovery**: ✅ All 5 tools visible and usable
- **Search Functionality**: ✅ Hybrid search working correctly

## Performance Considerations

### Database Optimization

- PostgreSQL with pgvector for efficient vector operations
- Proper indexing on metadata fields
- Connection pooling for concurrent requests

### Embedding Model

- Uses SentenceTransformer `all-MiniLM-L6-v2` (384 dimensions)
- Cached embeddings to avoid recomputation
- Batch processing for multiple queries

### Live Updates

- Optional file system monitoring
- Configurable polling intervals
- Background processing to avoid blocking MCP operations

## Security

### Input Validation

- SQL injection prevention through parameterized queries
- Code analysis sandbox isolation
- Resource path validation

### Authentication

- Currently no authentication (local development)
- Database credentials via environment variables
- Future: API key authentication for production

## Troubleshooting

### Common Issues

1. **Tools not visible in Claude Desktop**
   + Check server is running on correct port
   + Verify supergateway configuration
   + Restart Claude Desktop after config changes

2. **Database connection errors**
   + Verify PostgreSQL is running
   + Check environment variables (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD)
   + Ensure pgvector extension is installed

3. **Search returns no results**
   + Check if code index is populated
   + Verify embedding model is loaded
   + Review search query syntax

### Debug Tools

- **Debug script**: `debug_mcp_responses.py` - Tests MCP protocol communication
- **Format checker**: `check_tool_format.py` - Validates tool response format
- **Simulation test**: `test_claude_desktop_simulation.py` - Simulates Claude Desktop interaction

## Development

### Key Dependencies

- `mcp>=1.12.0` - Model Context Protocol SDK
- `psycopg[binary]>=3.1.0` - PostgreSQL adapter
- `pgvector>=0.2.0` - Vector similarity support
- `sentence-transformers` - Embedding generation
- `lark>=1.1.0` - Keyword query parsing

### Code Structure

```
python/cocoindex_code_mcp_server/
├── main_mcp_server.py              # Main MCP server implementation
├── hybrid_search.py           # Search engine core
├── cocoindex_config.py        # CocoIndex flow configuration
├── keyword_search_parser_lark.py  # Query parser
├── lang/python/               # Language-specific analyzers
└── tests/                     # Test suite
```

### Extension Points

- **New search algorithms**: Extend `HybridSearchEngine`
- **Additional languages**: Add analyzers in `lang/` directory
- **Custom metadata**: Extend code analysis pipeline
- **New tools**: Add MCP tool handlers in `main_mcp_server.py`

## Future Enhancements

### Planned Features

- Authentication and authorization
- Multi-tenant support
- Advanced query syntax
- Real-time collaboration features
- Integration with more vector databases

### Performance Improvements

- Distributed search across multiple databases
- Caching layer for frequent queries
- Async processing pipeline
- Query optimization

## References

- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- [CocoIndex Documentation](../cocoindex/)
- [PostgreSQL pgvector Documentation](https://github.com/pgvector/pgvector)
- [SentenceTransformers Documentation](https://www.sbert.net/)
