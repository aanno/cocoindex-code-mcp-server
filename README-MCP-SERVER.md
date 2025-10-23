# CocoIndex RAG MCP Server

✅ **PRODUCTION READY** - Fully functional MCP server for code search and analysis.

A Model Context Protocol (MCP) server that provides hybrid search capabilities combining vector similarity and keyword metadata search for code retrieval using CocoIndex. Successfully integrated with Claude Desktop and other MCP clients.

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
python -m pytest tests/test_mcp_server.py -m main_mcp_server -v

# Run specific test classes
python -m pytest tests/test_mcp_server.py::TestMCPServerBasics -v
```

### Starting the Server

```bash
# Basic usage (live updates enabled by default, 60s polling)
python start_mcp_server.py

# Custom path
python start_mcp_server.py /path/to/code

# Multiple paths
python start_mcp_server.py /path/to/code1 /path/to/code2

# Disable live updates
python start_mcp_server.py --no-live

# Custom polling interval
python start_mcp_server.py --poll 30

# Or run directly
python main_mcp_server.py
```

### Claude Desktop Integration (Recommended)

1. **Start the MCP server in HTTP mode:**

   ```bash
   python -m cocoindex_code_mcp_server.main_mcp_server.py --port 3033 /workspaces/rust
   ```

2. **Add to Claude Desktop configuration** (`~/.config/Claude/claude_desktop_config.json`):

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

3. **Restart Claude Desktop** - Tools will appear automatically in the interface.

### Using with Claude Code (Legacy)

Add to your Claude Code MCP configuration:

```json
{
  "cocoindex-rag": {
    "command": "python",
    "args": ["/path/to/cocoindex_code_mcp_server/main_mcp_server.py"],
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
  "tool": "search-hybrid",
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
  "tool": "search-vector",
  "arguments": {
    "query": "error handling in async functions",
    "top_k": 10
  }
}
```

### Keyword Search

```json
{
  "tool": "search-keyword",
  "arguments": {
    "query": "class_name:DatabaseManager AND function_name:connect",
    "top_k": 5
  }
}
```

## RAG-Enhanced Development

This MCP server demonstrates significant advantages for LLM-based development workflows:

### 🚀 Development Velocity Advantages

#### Context-Aware Code Discovery

- **Semantic Understanding**: Vector search finds conceptually related code even without exact keyword matches
- **Smart Navigation**: LLMs can explore codebases intelligently using natural language queries
- **Cross-Language Insights**: Discover similar patterns across different programming languages

#### Intelligent Code Analysis

- **Pattern Recognition**: LLMs can identify design patterns, anti-patterns, and architectural decisions
- **Impact Analysis**: Understanding how changes affect related components through semantic relationships
- **Learning from Examples**: Find implementation examples for specific use cases or algorithms

#### Enhanced Problem-Solving

- **Contextual Debugging**: Find similar error patterns and their solutions across the codebase
- **API Discovery**: Locate relevant functions and their usage patterns through natural language queries
- **Documentation Generation**: Automatically generate docs by understanding code semantics

### 🎯 Real-World Development Benefits

#### Onboarding & Knowledge Transfer

```python
# Instead of manually grep-ing through thousands of files
# LLM can ask: "How is authentication handled in this codebase?"
result = hybrid_search(
    vector_query="user authentication login security",
    keyword_query="language:python and value_contains(code, 'auth')"
)
```

#### Feature Development

```python
# Discover existing patterns before implementing new features
# "Show me how database connections are managed"
result = vector_search("database connection pool management")
```

#### Code Reviews & Refactoring

```python
# Find all similar implementations that might need refactoring
# "Find all error handling patterns using try-catch"
result = hybrid_search(
    vector_query="error handling exception try catch",
    keyword_query="value_contains(code, 'try') and language:python"
)
```

### 💡 Why RAG Beats Traditional Search

| Traditional Search | RAG-Enhanced Search |
|------------------|-------------------|
| Exact keyword matching | **Semantic understanding** |
| Syntactic queries only | **Natural language queries** |
| No context awareness | **Contextual relationships** |
| Manual code exploration | **AI-guided discovery** |
| Static documentation | **Dynamic code understanding** |

### 🔧 Development Workflow Integration

#### IDE Integration

The MCP server works with Claude Code, Claude Desktop, and any MCP-compatible client, bringing RAG capabilities directly into development environments.

#### CI/CD Enhancement

Automated code analysis and documentation generation based on semantic understanding of code changes.

#### Team Collaboration

Shared understanding of codebase architecture and patterns through natural language queries that work consistently across team members.

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

## Performance & Troubleshooting

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

### Advanced Troubleshooting

#### Tools Not Visible in Claude Desktop

1. **Check server is running:**

   ```bash
   curl -X POST http://localhost:3033/mcp -H "Content-Type: application/json" \
     -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
   ```

2. **Verify supergateway connection:**

   ```bash
   echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}' | \
   pnpm dlx supergateway --streamableHttp "http://localhost:3033/mcp" --logLevel debug
   ```

3. **Check Claude Desktop config file location:**
   + File: `~/.config/Claude/claude_desktop_config.json` (NOT `.mcp.json`)
   + Restart Claude Desktop after changes

#### Database Connection Issues

1. **Verify PostgreSQL is running and accessible**
2. **Check environment variables are set correctly**
3. **Ensure pgvector extension is installed:**

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

#### Search Returns No Results

1. **Check if code index is populated**
2. **Verify embedding model is loaded**
3. **Review search query syntax**

For more detailed troubleshooting, see `docs/claude/mcp_server_Development.md`.

## Documentation

- **Main Documentation**: `docs/claude/main_mcp_server.md`
- **Development Gotchas**: `docs/claude/mcp_server_Development.md`
- **Test Suite**: `tests/test_mcp_integration_http_e2e.py`
