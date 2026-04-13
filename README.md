# CocoIndex Code MCP Server

A Model Context Protocol (MCP) server that provides a RAG (Retrieval Augmented Generation) tool with hybrid search capabilities combining vector similarity and keyword metadata search for code retrieval. Built on the [CocoIndex](https://cocoindex.io) data transformation framework with specialized support for multiple programming languages.

This RAG MCP server enables AI tools (LLMs) to retrieve relevant code snippets from large codebases efficiently and in real-time, leveraging CocoIndex's incremental indexing, tree-sitter based chunking, and smart language-specific embeddings. It enhances the performance of code generation, code completion, and code understanding by virtually enlarging the context window available to the AI models.

Currently uses PostgreSQL + pgvector as the vector database backend, but can be adapted to other backends supported by CocoIndex.

## Table of Contents

- [Quickstart](#quickstart)
- [Command Line Arguments](#command-line-arguments)
- [Features](#features)
- [Supported Languages](#supported-languages)
- [Smart Embedding](#smart-embedding)
- [Development](#development)
- [Contributing](#contributing)

## Quickstart

### 1. Clone the Repository (optional)

```bash
git clone --recursive https://github.com/aanno/cocoindex-code-mcp-server.git
cd cocoindex-code-mcp-server
```

Checking out the sources is _not_ strictly necessary if you just want to use the MCP server, as it can be installed
from PyPI. However, there are some scripts e.g. for starting the pgvector database that are missing from the PyPI
package.

### 2. Install

Build from source using maturin:

```bash
# Install dependencies from PyPI
uv sync
uv sync --all-extras

# And build from source
maturin develop
```

Or simple install from PyPI:

```bash
pip install cocoindex-code-mcp-server
```

I provide native wheels for many systems (including Linux, Windows and MacOS) on PyPI, so no build should be necessary
in most cases. cocoindex-code-mcp-server needs Python 3.11+ (and I prefer to build abi3 wheels for better
compatibility).

### 3. Start the PostgreSQL Database

In one terminal on your local machine, start the pgvector database:

```bash
cd cocoindex-code-mcp-server
./scripts/cocoindex-postgresql.sh
# Maybe you need to install pgvector extension once
./scripts/install-pgvector.py
```

Using the scripts is optional, however you need a running PostgreSQL + pgvector database for the MCP server to work.

### 4. Configure the MCP Server (DB Connection)

cocoindex_code_mcp_server uses the `COCOINDEX_DATABASE_URL` environment variable to connect to the database.
It reads the `.env` file in the current directory if present. You can copy the provided `.env.template` to `.env` and
adjust the connection string if needed.

The current directory does not need to be the directory that you want to scan (see section 'Command Line Arguments'
below for details).

```bash
cp .env.template .env
```

### 5. Start the MCP Server

In another terminal, start the cocoindex_code_mcp_server:

```bash
cd cocoindex-code-mcp-server
python -m cocoindex_code_mcp_server.main_mcp_server --rescan --port 3033 <path_to_code_directory>
```

The server will index the code in the specified directory and start serving requests. This will take some time. It is ready when you see something like:

```text
CodeEmbedding.files (batch update): 1505 source rows NO CHANGE
```

The PyPI package does provide starting server with `cocoindex-code-mcp-server <options> <root-source-dir>`. Remember
that you need a running PostgreSQL + pgvector database for this to work.

### 6. Use the MCP Server

You can now use the RAG server running at `http://localhost:3033` as a streaming HTTP MCP server. For example, with Claude Code, use the following snippet within `"mcpServers"` in your `.mcp.json` file:

```json
{
  "cocoindex-rag": {
    "command": "pnpm",
    "args": [
      "dlx",
      "mcp-remote@next",
      "http://localhost:3033/mcp"
    ]
  }
}
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `paths` | positional | - | Path(s) to code directory/directories to index (can specify multiple) |
| `--paths` | option | - | Alternative way to specify paths (can use multiple times) |
| `--no-live` | flag | false | Disable live update mode |
| `--poll` | int | 60 | Polling interval in seconds for live updates |
| `--default-embedding` | flag | false | Use default CocoIndex embedding instead of smart embedding |
| `--default-chunking` | flag | false | Use default CocoIndex chunking instead of tree-sitter/AST chunking |
| `--default-language-handler` | flag | false | Use default CocoIndex language handling |
| `--chunk-factor-percent` | int | 100 | Chunk size scaling factor as percentage (100=default, <100=smaller, >100=larger) |
| `--port` | int | 3000 | Port to listen on for HTTP |
| `--log-level` | string | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `--json-response` | flag | false | Enable JSON responses instead of SSE streams |
| `--rescan` | flag | false | Clear database and tracking tables before starting to force re-indexing |
| `--include PATTERN` | option | - | Gitignore-style pattern for files to index (e.g. `*.nix`). **Replaces** the built-in include list entirely. Can be repeated. |
| `--exclude PATTERN` | option | - | Gitignore-style pattern for files to exclude (e.g. `**/tests/**`). Added on top of any `.gitignore`-derived exclusions. Can be repeated. |
| `--no-gitignore` | flag | false | Disable automatic exclusion of files matched by `.gitignore` files found in the scan tree. |

### File filtering

#### Include patterns

By default the server uses a broad built-in include list (~70 source-code file patterns, all in `**/<pattern>` form so they match at any directory depth). When `--include` is given, it **replaces** the built-in include list entirely — only the patterns you specify will be indexed.

#### Exclude patterns and `.gitignore` support

`.gitignore` files found anywhere in the scan tree are **automatically respected** and take priority:

- **With `.gitignore`**: the gitignore-derived patterns replace the built-in exclude list. They are the authoritative source of exclusions for the project.
- **Without `.gitignore`**: the built-in exclude list (common build artefacts, hidden directories, dependency folders, etc.) is used as a fallback.

In both cases, patterns given via `--exclude` are appended on top. Use `--no-gitignore` to disable `.gitignore` processing entirely (the built-in fallback list then applies as usual).

#### Pattern conversion

All patterns follow [gitignore rules](https://git-scm.com/docs/gitignore) and are automatically converted to the globset format used by CocoIndex:

| Gitignore pattern | Converted to | Meaning |
|---|---|---|
| `target/` | `**/target` + `**/target/**` | Directory named `target` at any depth, and all files inside it |
| `*.log` | `**/*.log` | Any `.log` file at any depth |
| `/dist` | `dist` | Only the root-level `dist` entry |
| `**/node_modules` | `**/node_modules` | Already explicit, left as-is |

Negation (`!`) is not supported and will be warned and skipped.

#### Stale-result filtering

When patterns are **narrowed between runs** (without `--rescan`), the database may still contain indexed entries from the previous broader scan. To prevent stale results from appearing in queries, the server applies a **post-query path filter** to all three search types (keyword, vector, hybrid). Any result whose path no longer matches the current include/exclude patterns is silently dropped and logged at `INFO` level. This means narrowing your patterns takes effect immediately without requiring `--rescan`.

Note: broadening patterns works in the opposite direction — new files are picked up by CocoIndex's incremental indexer on the next update pass.

### Examples

```bash
# Index a single directory with live updates
python -m cocoindex_code_mcp_server.main_mcp_server /path/to/code

# Index multiple directories
python -m cocoindex_code_mcp_server.main_mcp_server /path/to/code1 /path/to/code2

# Force re-indexing with custom port
python -m cocoindex_code_mcp_server.main_mcp_server --rescan --port 3033 /path/to/code

# Disable live updates (one-time indexing)
python -m cocoindex_code_mcp_server.main_mcp_server --no-live /path/to/code

# Custom chunk size (50% smaller chunks)
python -m cocoindex_code_mcp_server.main_mcp_server --chunk-factor-percent 50 /path/to/code

# Index only Nix and Dhall files (replaces built-in include list)
python -m cocoindex_code_mcp_server.main_mcp_server --include '*.nix' --include '*.dhall' /path/to/code

# Exclude test directories and lock files (on top of .gitignore / built-in excludes)
python -m cocoindex_code_mcp_server.main_mcp_server --exclude '**/tests/**' --exclude '*.lock' /path/to/code

# Disable .gitignore-based exclusion (built-in fallback list takes over)
python -m cocoindex_code_mcp_server.main_mcp_server --no-gitignore /path/to/code
```

## Features

- **CocoIndex Backend**: Uses [CocoIndex](https://cocoindex.io) as the embedding and vector database backend with PostgreSQL + pgvector
- **Multiple Language Support**: Specialized support for 20+ programming languages with language-specific parsers and embeddings
- **Streaming HTTP MCP Server**: Real-time code retrieval via Model Context Protocol over HTTP
- **Code Change Detection**: Incremental indexing with automatic detection of file changes
- **Tree-sitter Chunking**: Advanced code parsing and chunking using tree-sitter AST for better code understanding
- **Smart Embedding**: Multiple embedding models automatically selected based on programming language (see [Smart Embedding](#smart-embedding))
- **Hybrid Search**: Combines vector similarity search with keyword/metadata filtering for precise results
  + **Vector Search**: Semantic similarity using language-specific code embeddings
  + **Keyword Search**: Exact matching on metadata fields (functions, classes, imports, etc.)
  + **Hybrid Search**: Weighted combination of both approaches with configurable weights

## Supported Languages

The server supports multiple programming languages with varying levels of integration:

| Language | Extensions | Embedding Model | AST Chunking | Tree-sitter | Remarks |
|----------|------------|-----------------|--------------|-------------|---------|
| **Python** | `.py` | GraphCodeBERT | ✅ astchunk | ✅ python | Custom (not using visitor), <br/>metadata extraction: `language_handlers/python_handler.py`, <br/>analyser: `lang/python/tree_sitter_python_analyzer.py`, <br/>(fallback: `lang/python/python_code_analyzer.py`), <br/>TODO: unify this with visitor approach |
| **Rust** | `.rs` | UniXcoder | ? | ✅ rust | Full metadata support with specialized visitor: `language_handlers/rust_visitor.py` |
| **JavaScript** | `.js`, `.mjs`, `.cjs` | GraphCodeBERT | ?astchunk? | ✅ javascript | Full metadata support with specialized visitor: `language_handlers/javascript_visitor.py` |
| **TypeScript** | `.ts` | UniXcoder | ✅ astchunk | ✅ typescript | Extends javascript visitor: `language_handlers/typescript_visitor.py` |
| **TSX** | `.tsx` | UniXcoder | ✅ astchunk | ?typescript? | ?see typescript? |
| **Java** | `.java` | GraphCodeBERT | ✅ astchunk | ✅ java | Full metadata support with specialized visitor: `language_handlers/java_visitor.py` |
| **Kotlin** | `.kt`, `.kts` | UniXcoder | ? | ✅ kotlin | Full metadata support with specialized visitor: `language_handlers/kotlin_visitor.py` |
| **C** | `.c`, `.h` | GraphCodeBERT | ? | ✅ c | Full metadata support with specialized visitor: `language_handlers/c_visitor.py` |
| **C++** | `.cpp`, `.cc`, `.cxx`,`.hpp` | GraphCodeBERT | ? | ✅ cpp | Extends C visitor: `language_handlers/cpp_visitor.py` |
| **C#** | `.cs` | UniXcoder | ✅ astchunk | ❌ | Tree-sitter parsing/chunking only |
| **Haskell** | `.hs`, `.lhs` | all-mpnet-base-v2 | ✅ | ✅ | Custom maturin extension with specialized visitor, <br/>chunker: `lang/haskell/haskell_ast_chunker.py`, <br/>metadata extraction: `language_handlers/haskell_handler.py` |
| **Other Languages** | see `mappers.py` | all-mpnet-base-v2 | ❌ | ❌ ?regex? | cocoindex defaults (baseline) |

### Legend

- **Embedding Model**: The embedding model automatically selected for the language
- **AST Chunking**: Advanced chunking using [ASTChunk](https://github.com/codelion/astchunk) or custom implementations (based on ideas from ASTChunk and using tree-sitter for the language).
- **Tree-sitter**: Language has tree-sitter parser configured for AST analysis. (python tree-sitter bindings, except for Haskell which uses a Maturin/Rust extension based on rust bindings cargos `tree-sitter` and `tree-sitter-haskell`.)
- **Remarks**: Additional notes about support level
- **Other Languages**: Files recognized but only basic text embedding and chunking applied (cocoindex defaults). <br/>
  This includes: Go, PHP, Ruby, Swift, Scala, Dart, CSS, HTML, JSON, Markdown, YAML, TOML, SQL, R, Fortran, Pascal, XML

## Smart Embedding

The server uses **language-aware code embeddings** that automatically select the optimal embedding model based on the programming language. This approach provides better semantic understanding of code compared to generic text embeddings.

### How It Works

The smart embedding system uses different specialized models optimized for different programming languages:

1. **GraphCodeBERT** (`microsoft/graphcodebert-base`)
   + **Optimized for:** Python, Java, JavaScript, PHP, Ruby, Go, C, C++
   + Pre-trained on code from these languages with graph-based code understanding
   + Best for languages with explicit structure and common patterns

2. **UniXcoder** (`microsoft/unixcoder-base`)
   + **Optimized for:** Rust, TypeScript, C#, Kotlin, Scala, Swift, Dart
   + Unified cross-lingual model for multiple languages
   + Best for modern statically-typed languages

3. **Fallback Model** (`sentence-transformers/all-mpnet-base-v2`)
   + Used for: Languages not specifically supported by code models
   + General-purpose text embedding for broader language support
   + 768-dimensional embeddings matching code-specific models

### Automatic Selection

The embedding model is automatically selected based on file extension:

```python
# Example: Python file automatically uses GraphCodeBERT
file: main.py → language: python → model: microsoft/graphcodebert-base

# Example: Rust file automatically uses UniXcoder
file: lib.rs → language: rust → model: microsoft/unixcoder-base

# Example: Haskell file uses fallback model
file: Main.hs → language: haskell → model: sentence-transformers/all-mpnet-base-v2
```

### Benefits

- **Better Code Understanding**: Code-specific models understand programming constructs better than generic text models
- **Language-Specific Optimization**: Each language gets embeddings from models trained on that language
- **Consistent Search Quality**: Similar code snippets in the same language produce similar embeddings
- **Zero Configuration**: Automatic model selection requires no manual configuration

### Implementation Details

The smart embedding system is implemented as an external wrapper around CocoIndex's `SentenceTransformerEmbed` function, located in `python/cocoindex_code_mcp_server/smart_code_embedding.py`. This approach:

- Does not modify CocoIndex source code
- Uses CocoIndex as a pure dependency
- Provides drop-in compatibility with existing workflows
- Can be easily updated independently

For more technical details, see:

- [`docs/claude/Embedding-Selection.md`](docs/claude/Embedding-Selection.md)
- [`docs/cocoindex/smart-embedding.md`](docs/cocoindex/smart-embedding.md)

## Development

### Prerequisites

- Rust (latest stable version)
- Python 3.11+
- Maturin (build tool for Python extensions in Rust)
- PostgreSQL with pgvector extension
- Tree-sitter language parsers (automatically installed via pyproject.toml)

### Run tests

```bash
# Run tests to verify installation
pytest -c pytest.ini tests/

uv run --with pytest --with pytest-asyncio --with pytest-mock pytest tests/test_pattern_utils.py -v --tb=short 2>&1 | tail -20
```

### Code Quality

The project uses mypy for type checking. Use the provided scripts:

```bash
# Type check main source code
./scripts/mypy-check.sh

# Type check tests
./scripts/mypy-check-tests.sh
```

### Project Structure

- **`python/cocoindex_code_mcp_server/`**: Main MCP server implementation
  + `main_mcp_server.py`: MCP server entry point
  + `cocoindex_config.py`: CocoIndex flow configuration
  + `smart_code_embedding.py`: Language-aware embedding selection
  + `mappers.py`: Language and field mappings
  + `tree_sitter_parser.py`: Tree-sitter parsing utilities
  + `db/`: Database abstraction layer
    - `pgvector/`: PostgreSQL + pgvector backend
  + `lang/`: Language-specific handlers
    - `python/`: Python code analyzer
    - `haskell/`: Haskell support (via Rust extension)
- **`tests/`**: Pytest test suite
- **`docs/`**: Documentation
  + `claude/`: Development notes and architecture docs
  + `cocoindex/`: CocoIndex-specific documentation
  + `instructions/`: Task instructions and guides
- **`rust/`**: Rust components
  + `src/lib.rs`: Haskell tree-sitter Rust extension
- **`astchunk/`**: ASTChunk submodule for advanced code chunking

### Running Tests

```bash
# Run all tests
pytest -c pytest.ini tests/

# Run specific test file
pytest -c pytest.ini tests/test_hybrid_search_integration.py

# Run with coverage
pytest -c pytest.ini tests/ --cov=python/cocoindex_code_mcp_server --cov-report=html
```

## Contributing

Contributions are welcome! Please open issues and pull requests on the [GitHub repository](https://github.com/aanno/cocoindex-code-mcp-server).

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run type checking: `./scripts/mypy-check.sh`
5. Run tests: `pytest tests/`
6. Submit a pull request

### Areas for Contribution

- Additional language support (parsers, embeddings, chunking)
- Enhanced metadata extraction for existing languages
- Performance optimizations
- Documentation improvements
- Bug fixes and issue resolution

## License

AGPL-3.0 or later

## Links

- **CocoIndex Framework**: <https://cocoindex.io>
- **GitHub Repository**: <https://github.com/aanno/cocoindex-code-mcp-server>
- **Model Context Protocol**: <https://modelcontextprotocol.io>
- **ASTChunk**: <https://github.com/codelion/astchunk>

## Acknowledgments

Built on top of the excellent [CocoIndex](https://cocoindex.io) framework for incremental data transformation and the [Model Context Protocol](https://modelcontextprotocol.io) for AI tool integration.
