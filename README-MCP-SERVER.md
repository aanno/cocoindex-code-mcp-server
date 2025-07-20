# CocoIndex RAG MCP Server

A Model Context Protocol (MCP) server that provides hybrid search capabilities combining vector similarity and keyword metadata search for code retrieval using CocoIndex.

## Features

### MCP Tools
- **hybrid_search** - Combine vector similarity and keyword metadata filtering
- **vector_search** - Pure vector similarity search  
- **keyword_search** - Pure keyword metadata search
- **analyze_code** - Code analysis and metadata extraction
- **get_embeddings** - Generate embeddings for text

### MCP Resources
- **search_stats** - Database and search performance statistics
- **search_config** - Current hybrid search configuration
- **database_schema** - Database table structure information

## Prerequisites

1. **Python 3.11+** with required dependencies:
   ```bash
   # Install MCP server dependencies
   pip install -e ".[mcp-server]"
   
   # Or install test dependencies if you want to run tests
   pip install -e ".[mcp-server,test]"
   ```

2. **PostgreSQL with pgvector** extension installed

3. **CocoIndex** embedded and configured:
   ```bash
   cd ../../cocoindex
   maturin develop
   ```

4. **Database with indexed code** (using CocoIndex pipeline)

## Configuration

Set environment variables for database connection:

```bash
export DB_HOST=localhost
export DB_PORT=5432  
export DB_NAME=cocoindex
export DB_USER=postgres
export DB_PASSWORD=password
```

## Usage

### Testing
Run the test suite to verify functionality:
```bash
# From project root, run all MCP server tests
python -m pytest tests/test_mcp_server.py -v

# Run only MCP server marked tests  
python -m pytest tests/test_mcp_server.py -m mcp_server -v

# Run specific test classes
python -m pytest tests/test_mcp_server.py::TestMCPServerBasics -v
```

### Starting the Server
```bash
python start_mcp_server.py
```

Or directly:
```bash
python mcp_server.py
```

### Using with Claude Code

Add to your Claude Code MCP configuration:

```json
{
  "cocoindex-rag": {
    "command": "python",
    "args": ["/path/to/cocoindex-code-mcp-server/mcp_server.py"],
    "env": {
      "DB_HOST": "localhost",
      "DB_NAME": "cocoindex", 
      "DB_USER": "postgres",
      "DB_PASSWORD": "password"
    }
  }
}
```

## Example Queries

### Hybrid Search
```json
{
  "tool": "hybrid_search",
  "arguments": {
    "vector_query": "function to parse JSON data",
    "keyword_query": "function_name:parse AND language:python",
    "top_k": 5,
    "vector_weight": 0.7,
    "keyword_weight": 0.3
  }
}
```

### Vector Search
```json
{
  "tool": "vector_search", 
  "arguments": {
    "query": "error handling in async functions",
    "top_k": 10
  }
}
```

### Keyword Search
```json
{
  "tool": "keyword_search",
  "arguments": {
    "query": "class_name:DatabaseManager AND function_name:connect",
    "top_k": 5
  }
}
```

## Architecture

The MCP server integrates with existing CocoIndex components:

- **HybridSearchEngine** - Core search combining vector + keyword
- **KeywordSearchParser** - Lark-based query parser with advanced operators
- **PostgreSQL + pgvector** - Vector database backend
- **CocoIndex pipeline** - Code embedding and analysis

## Supported Keyword Operators

- `AND`, `OR`, `NOT` - Boolean logic
- `==`, `!=`, `<`, `>`, `<=`, `>=` - Comparison operators  
- `value_contains` - Substring matching
- Field targeting: `function_name:parse`, `language:python`

## Troubleshooting

1. **Import Errors**: Ensure CocoIndex is installed via `maturin develop`
2. **Database Connection**: Check environment variables and PostgreSQL service
3. **Missing Dependencies**: Install via `pip install -e ".[mcp-server,test]"`
4. **Test Failures**: Run `python -m pytest tests/test_mcp_server.py -v` for diagnostics

## Integration

This MCP server is designed to work with:
- Claude Code CLI (`claude.ai/code`)
- Any MCP-compatible client
- Existing CocoIndex RAG pipeline
- PostgreSQL databases with pgvector