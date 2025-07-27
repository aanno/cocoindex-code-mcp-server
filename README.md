# RAG experiments

The target of this project is to have an RAG for code development as MCP server.
There is a MCP server example at `/workspaces/rust/code-index-mcp`.

## State

### CocoIndex Framework

- Hybrid Rust/Python architecture with powerful data transformation capabilities
- Incremental processing with PostgreSQL + pgvector backend
- Code chunking and embedding pipeline already implemented
- **NEW**: [Hybrid search](docs/claude/Hybrid-Search.md) with vector similarity and exact matches
  + [Enhanced metadata](docs/cocoindex/metadata.md) for python
- **NEW**: Haskell support
  + Haskell tree-sitter support via custom maturin extension
  + Haskell-specific code chunking inspired by ASTChunk
    * see [Haskell-ASTChunking](docs/claude/Haskell-ASTChunking.md)
- **NEW**: Multiple path support for indexing and searching across multiple directories
- **NEW**: Enhanced code chunking by integration with ASTChunk
  + see [ASTChunk_integration](docs/claude/ASTChunk_integration.md)
  + see [ASTChunk](docs/claude/ASTChunk.md)
- **NEW**: automatic language-aware code embeddings selection
  + see [Embedding-Selection](docs/claude/Embedding-Selection.md)
  + see [Language-Aware-Embeddings](docs/cocoindex/language-aware-embeddings.md)

### tree-sitter

- CocoIndex uses tree-sitter, but only the grammars built in
- [List of grammars](https://github.com/tree-sitter/tree-sitter/wiki/List-of-parsers)  
- If you want support for other languages, additional parsers should be included
- **NEW**: Enhanced tree-sitter implementation with specialized AST visitors for major languages
- **IMPROVED**: Eliminated tree-sitter warnings in MCP server logs through proper parser implementation
- **ENHANCED**: Comprehensive language support with inheritance patterns (C++ → C, TypeScript → JavaScript)
- This has been implemented for: **Haskell** (custom extension), **C**, **C++**, **Rust**, **Java**, **JavaScript**, **TypeScript**, **Kotlin** (specialized visitors) + **Python**, **C#** (configured parsers)

### Existing MCP Server

- Basic file indexing and search functionality
- Advanced search with ripgrep/ugrep integration
- Language-specific analysis (Python, Java, JavaScript, etc.)
- No semantic search or RAG capabilities

### Haskell Support Implementation

This project now includes full Haskell support for CocoIndex through:

1. **Custom Maturin Extension**: A Rust-based Python extension (`haskell-tree-sitter`) that provides:
   - Native tree-sitter-haskell parsing capabilities
   - Python bindings for Haskell AST manipulation
   - Custom separator patterns for optimal code chunking

2. **Integration with CocoIndex**: Enhanced `src/main_interactive_query.py` with:
   - Haskell language detection (`.hs`, `.lhs` files)
   - Custom chunking parameters optimized for Haskell code
   - Tree-sitter-aware code splitting using our custom extension

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

## Building the Project

### Prerequisites

- Rust (latest stable version)
- Python 3.11+
- Maturin (`pip install maturin`)
- Tree-sitter language parsers (automatically installed via pyproject.toml)
  - `tree-sitter>=0.23.0,<0.24.0` (downgraded for compatibility)
  - `tree-sitter-python>=0.23.6`
  - `tree-sitter-c>=0.23.0`
  - `tree-sitter-cpp>=0.23.0`
  - `tree-sitter-rust>=0.23.0`
  - `tree-sitter-java>=0.23.5`
  - `tree-sitter-javascript>=0.23.0`
  - `tree-sitter-typescript>=0.23.2`
  - `tree-sitter-kotlin>=0.23.0`
  - `tree-sitter-c-sharp>=0.23.1`

### Build Steps

```bash
# 1. Build and install the haskell-tree-sitter extension
maturin develop

# 2. Install test dependencies (optional)
pip install -e ".[test]"

# 3. Run tests to verify installation
python run_tests.py

# Alternative test runners:
python run_tests.py --runner unittest
python run_tests.py --runner pytest      # requires: pip install pytest
python run_tests.py --runner coverage    # requires: pip install pytest-cov
```

### Development Build

For development with automatic rebuilding:

```bash
# Build in development mode with automatic rebuilding
maturin develop --reload

# Run specific test suites
python -m unittest tests.test_haskell_parsing -v
python -m unittest tests.test_cocoindex_integration -v
```

### Test Infrastructure

The project includes comprehensive test coverage:

- **`tests/test_haskell_parsing.py`**: Tests the core tree-sitter Haskell parsing functionality
- **`tests/test_cocoindex_integration.py`**: Tests integration with CocoIndex pipeline  
- **`tests/test_cli_args.py`**: Tests command-line argument parsing and path handling
- **`tests/test_multiple_paths.py`**: Tests multiple path processing and source management
- **`tests/fixtures/`**: Sample Haskell files for testing
- **`run_tests.py`**: Unified test runner with multiple backend support

Test categories:
- Unit tests for tree-sitter parsing
- Integration tests for CocoIndex compatibility
- Command-line interface tests
- Multiple path processing tests
- Configuration validation tests
- Edge case and error handling tests

## Language Support

This project enhances CocoIndex with additional language support. Here's the current status:

### 🚀 Recent Improvements

**NEW**: Specialized AST visitors have been implemented for major programming languages, providing enhanced metadata extraction and 100% function recall rates:

- **C**: Comprehensive visitor extracting functions, structs, enums, typedefs
- **C++**: Inherits from C visitor, adds classes, namespaces, templates  
- **Rust**: Extracts functions, structs, enums, traits, impls, modules
- **Java**: Extracts methods, constructors, classes, interfaces, packages
- **JavaScript**: Extracts functions, classes, variables, imports, exports
- **TypeScript**: Inherits from JavaScript, adds interfaces, types, enums, namespaces
- **Kotlin**: Extracts functions, classes, interfaces, objects, data classes
- **Haskell**: Enhanced visitor with improved function detection (57% → 100% recall)

These implementations follow inheritance patterns where applicable (C++ extends C, TypeScript extends JavaScript) and include comprehensive baseline testing infrastructure.

### Supported via CocoIndex Tree-sitter (Built-in)

These languages are natively supported by CocoIndex without additional configuration:

| Language | Extensions | Support Level | AST Node Handler | Tree-sitter Implementation |
|----------|------------|---------------|------------------|----------------------------|
| **C** | `.c` | Full tree-sitter | **✅ Specialized Visitor** | **✅ Fully Configured** |
| **C++** | `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp` | Full tree-sitter | **✅ Specialized Visitor** | **✅ Fully Configured** |
| **C#** | `.cs` | Full tree-sitter | Not implemented | **✅ Fully Configured** |
| **CSS** | `.css`, `.scss` | Full tree-sitter | Not implemented | Available but not configured |
| **Fortran** | `.f`, `.f90`, `.f95`, `.f03` | Full tree-sitter | Not implemented | Available but not configured |
| **Go** | `.go` | Full tree-sitter | Not implemented | Available but not configured |
| **HTML** | `.html`, `.htm` | Full tree-sitter | Not implemented | Available but not configured |
| **Java** | `.java` | Full tree-sitter | **✅ Specialized Visitor** | **✅ Fully Configured** |
| **JavaScript** | `.js`, `.mjs`, `.cjs` | Full tree-sitter | **✅ Specialized Visitor** | **✅ Fully Configured** |
| **JSON** | `.json` | Full tree-sitter | Not implemented | Available but not configured |
| **Kotlin** | `.kt`, `.kts` | Full tree-sitter | **✅ Specialized Visitor** | **✅ Fully Configured** |
| **Markdown** | `.md`, `.mdx` | Full tree-sitter | Not implemented | Available but not configured |
| **Pascal** | `.pas`, `.dpr` | Full tree-sitter | Not implemented | Available but not configured |
| **PHP** | `.php` | Full tree-sitter | Not implemented | Available but not configured |
| **Python** | `.py`, `.pyi` | Full tree-sitter | **✅ Available** | **✅ Fully Configured** |
| **R** | `.r`, `.R` | Full tree-sitter | Not implemented | Available but not configured |
| **Ruby** | `.rb` | Full tree-sitter | Not implemented | Available but not configured |
| **Rust** | `.rs` | Full tree-sitter | **✅ Specialized Visitor** | **✅ Fully Configured** |
| **Scala** | `.scala` | Full tree-sitter | Not implemented | Available but not configured |
| **SQL** | `.sql` | Full tree-sitter | Not implemented | Available but not configured |
| **Swift** | `.swift` | Full tree-sitter | Not implemented | Available but not configured |
| **TOML** | `.toml` | Full tree-sitter | Not implemented | Available but not configured |
| **TypeScript** | `.ts` | Full tree-sitter | **✅ Specialized Visitor** | **✅ Fully Configured** |
| **TSX** | `.tsx` | Full tree-sitter | **✅ Specialized Visitor** | **✅ Fully Configured** |
| **XML** | `.xml` | Full tree-sitter | Not implemented | Available but not configured |
| **YAML** | `.yaml`, `.yml` | Full tree-sitter | Not implemented | Available but not configured |

### Enhanced Tree-sitter Support (This Project)

Languages with additional tree-sitter support added by this project:

| Language | Extensions | Support Level | Implementation | AST Node Handler |
|----------|------------|---------------|----------------|------------------|
| **Haskell** | `.hs`, `.lhs` | Full tree-sitter | Custom maturin extension | **✅ Specialized Visitor** |

### Ad-hoc Pattern-based Support (Custom Languages)

Languages supported through regex-based chunking patterns:

| Language | Extensions | Support Level | Implementation | AST Node Handler |
|----------|------------|---------------|----------------|------------------|
| **Shell** | `.sh`, `.bash` | Pattern-based | Custom regex separators | N/A |
| **Makefile** | `Makefile`, `.makefile` | Pattern-based | Custom regex separators | N/A |
| **CMake** | `.cmake`, `CMakeLists.txt` | Pattern-based | Custom regex separators | N/A |
| **Dockerfile** | `Dockerfile`, `.dockerfile` | Pattern-based | Custom regex separators | N/A |
| **Gradle** | `.gradle` | Pattern-based | Custom regex separators | N/A |
| **Maven** | `pom.xml` | Pattern-based | Custom regex separators | N/A |
| **Config** | `.ini`, `.cfg`, `.conf` | Pattern-based | Custom regex separators | N/A |

### Adding New Language Support

1. **For languages with existing tree-sitter grammars**: Create a maturin extension similar to the Haskell implementation
2. **For languages without tree-sitter support**: Add custom language specifications with appropriate regex patterns
3. **Extend existing support**: Modify separator patterns in the custom language configurations

## Usage

### Running the Code Embedding Pipeline

```bash
# Set up environment variables
export COCOINDEX_DATABASE_URL="postgresql://user:pass@localhost/dbname"

# Run with default path (cocoindex directory)
python src/main_interactive_query.py

# Index a specific directory
python src/main_interactive_query.py /path/to/your/code

# Index multiple directories (fully supported!)
python src/main_interactive_query.py /path/to/code1 /path/to/code2

# Alternative syntax with explicit --paths argument
python src/main_interactive_query.py --paths /path/to/your/code

# Show help and usage examples
python src/main_interactive_query.py --help
```

## ✅ Multiple Path Support

CocoIndex natively supports multiple sources per flow! When multiple paths are specified, each directory is added as a separate source and processed in parallel. Search results will indicate which source each file came from.

The pipeline will automatically detect and properly chunk supported languages using either tree-sitter parsing or pattern-based chunking, enabling better semantic search and RAG capabilities.

### Technical Architecture

The implementation leverages CocoIndex's native multi-source capabilities:
- Flow Builder: Adds multiple sources to the same flow
- Data Scope: Each source gets its own namespace
- Collector: Unified collection from all sources
- Export: Single database table with source identification
- Search: Enhanced to show source information

This solution is much more robust than the initial single-source approach and takes full advantage of CocoIndex's 
designed capabilities forhandling multiple data sources efficiently.
