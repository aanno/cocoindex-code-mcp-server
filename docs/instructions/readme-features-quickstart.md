# Readme: Features and Quickstart

The following are some sketch instructions and features for using the cocoindex_code_mcp_server package. They should be merged into README.md. Before and after README.md should be sanatized by using gw-memory and looking into the code, either directly or (preferred) by using the running RAG MCP server.

## Quickstart

1. Clone this repository.

   ```bash
   git clone --recursive https://github.com/aanno/cocoindex-code-mcp-server.git
   ```

2. Install the package from PyPI or build from source using maturin.
3. In one terminal of your local machine, start the pgvector database:

   ```bash
   cd cocoindex-code-mcp-server
   ./scripts/cocoindex-postgresql.sh
   ```

4. In another terminal, start the cocoindex_code_mcp_server:

   ```bash
   cd cocoindex-code-mcp-server
   python -m cocoindex_code_mcp_server.main_mcp_server --rescan --port 3033 <path_to_code_directory>
   ```

5. The server will index the code in the specified directory and start serving requests.
   This will take some time. It is ready when you see something like:
   'CodeEmbedding.files (batch update): 1505 source rows NO CHANGE'
6. You can now use the RAG that is running at `http://localhost:3033` as streaming HTTP MCP server.
   For example, with claude code, use the following snippet within "mcpServers" in you .mcp.json file:

   ```json
        "vscode-mcp-server": {
            "command": "pnpm",
            "args": [
                "dlx",
                "mcp-remote@next",
                "http://localhost:3000/mcp"
            ]
        },
   ```

### Command line arguments

Fill this section, see python/cocoindex_code_mcp_server/main_mcp_server.py
This should be a table with argument, type, default, and description.

## Features

* Uses cocoindex as the embedding and vector database backend (currently pgvector with PostgreSQL)
* Supports code indexing and retrieval for multiple programming languages (see below)
* Supports streaming HTTP MCP server for real-time code retrieval
* code change detection and incremental indexing
* tree-sitter based chunking and parsing for better code understanding
* uses multiple embedding models depending on language (see smart embedding below)
* supports keyword, vector, and hybrid search

### Supported Languages

Please fill that section, see CONST_LANGUAGE_MAPPINGS in python/cocoindex_code_mcp_server/mappers.py
This should be a table, with languages, file extensions, embedding model used, if astchunk is used, if rust is used, and remarks.

### Smart Embedding

Please fill that section, see docs/cocoindex/smart-embedding.md and docs/claude/Embedding-Selection.md

## Development

### Contributing

Contributions are welcome! Please open issues and pull requests on the GitHub repository.

### Code

* Uses astchunk for supported languages for better code chunking (see ./astchunk, docs/claude/ASTChunk.md, and docs/claude/ASTChunking.md)
* Uses python tree-sitter bindings for parsing code (see python/cocoindex_code_mcp_server/tree_sitter_parser.py) for most languages
* Uses **rust** tree-sitter bindings for Haskell code parsing and semantic analysis (see rust/src/lib.rs)
* mypy type annotations for better code quality and IDE support
* extensive testing using pytest (see tests/ directory)
