# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Instructions

If you want to run something directly from `./src`, this is the pattern:
For running `src/cocoindex_code_mcp_server/main_mcp_server.py` do:

```bash
python -m cocoindex_code_mcp_server.main_mcp_server
```

Our code is mypy compatible! Use the type checking script:

```bash
./scripts/mypy-check.sh # and
./scripts/mypy-check-tests.sh
```

- If you are technically stuck or unsure about the next step, ask for help.
- cocoindex is a complex beast, so don't hesitate to ask for clarification or guidance.
- Tests should be pytest at ./tests (NOT at ./src/cocoindex_code_mcp_server/tests).
- Tests should use pytest and pytest plugins only (i.e. don't use unittest).
- You MUST use our own RAG (MCP server 'cocoindex-rag') each time before using grep or search.
- Never try to start/stop our RAG MCP server, just ask, I will do it for you.
- Use gw-memory to store and retrieve information about the codebase.
  + Tag all entries with 'cocoindex_code_mcp_server' to indicate they are related to the code MCP server.
  + After you have been started, it is a good idea to retrieve what's has been stored lately, so you have the latest context.

## MCP tool usage

Don't use tool 'vscode-mcp-server - execute_shell_command_code (MCP)' because of
issues. Instead, use bash directly.

For editing file, use tool 'Opened changes in Visual Studio Code'. This is 
much better than tool 'update'. But if you use tool 'update', don't forget to 
use tool 'filesystem - read_text_file (MCP)' before that. Otherwise you get the
following error: File has not been read yet. Read it first before writing to it.

## DB connections

DB connection properties could be found in .env and should be loaded with load_dotenv.
There are many examples how to do this throughout our code.

## Project Overview

This repository contains an RAG retrieval and query application specialized on code based on CocoIndex,
a high-performance data transformation framework for AI workloads.

See [Code-Structure](docs/claude/Code-Structure.md) how the application code is structured.

Additonal files here (but only for reference how things work, i.e. not as application code dependencies):

1. **cocoindex/**: The main CocoIndex project - a hybrid Rust/Python framework for building data transformation pipelines
2. **code-index-mcp/**: An MCP (Model Context Protocol) server for code indexing and analysis
3. **quickstart/**: Example scripts for getting started with CocoIndex

## Common Development Commands

### CocoIndex (Main Project)

**Build and Development:**
```bash
cd cocoindex
maturin develop                    # Build Rust extension and install Python package
cargo build                       # Build Rust components only
cargo test                        # Run Rust tests
pytest python/cocoindex/tests/     # Run Python tests
```

**Code Quality:**
```bash
cargo fmt                         # Format Rust code
ruff format python/               # Format Python code
mypy python/cocoindex/           # Type check Python code
ruff check python/               # Lint Python code
```

**Pre-commit Hooks:**
```bash
pre-commit install               # Install pre-commit hooks
pre-commit run --all-files      # Run all pre-commit checks
```

### Code Index MCP Server

```bash
cd code-index-mcp
python -m pip install -e .      # Install in development mode
python run.py                    # Run the MCP server
```

## Architecture Overview

### CocoIndex Hybrid Architecture

CocoIndex uses a hybrid Rust/Python architecture with clear separation of concerns:

**Rust Core (`src/`):**
- **`base/`**: Core data structures, schemas, and type definitions
- **`builder/`**: Flow analysis and execution plan generation
- **`execution/`**: Runtime execution engine with incremental processing
- **`ops/`**: Pluggable operation system with sources, targets, and functions
- **`llm/`**: LLM provider integrations (OpenAI, Anthropic, Gemini, etc.)
- **`py/`**: Python-Rust interop layer using PyO3

**Python Interface (`python/cocoindex/`):**
- **`flow.py`**: Main flow definition API and dataflow programming interface
- **`cli.py`**: Command-line interface for running flows
- **`sources.py`**: Data source definitions (S3, Azure Blob, Google Drive, etc.)
- **`targets.py`**: Data target definitions (Postgres, Neo4j, Qdrant, etc.)
- **`functions.py`**: Data transformation functions (embedding, LLM extraction, etc.)

### Key Concepts

**Dataflow Programming Model:**
- Users define transformations as dataflows using Python decorators (`@cocoindex.flow_def`)
- Each transformation creates new fields from input fields without mutation
- System tracks data lineage and enables incremental processing

**Incremental Processing:**
- Core engine tracks data dependencies and only recomputes changed portions
- Uses fingerprinting and memoization for efficient updates
- Supports live updates with minimal recomputation

**Pluggable Operations:**
- Three types of operations: Sources (data input), Functions (transformation), Targets (output)
- Operations are defined in Rust with Python bindings
- Registry system allows dynamic operation loading

**Multi-Database Support:**
- Supports vector databases (Qdrant), graph databases (Neo4j), and relational databases (Postgres)
- Unified interface for different storage backends

## Development Workflow

1. **Rust Changes**: When modifying Rust code, run `maturin develop` to rebuild the Python extension
2. **Python Changes**: Python code changes are immediately available
3. **Testing**: Run both Rust (`cargo test`) and Python (`pytest`) tests
4. **Pre-commit**: The project uses pre-commit hooks for code quality checks

## Important Files

- **`Cargo.toml`**: Rust dependencies and build configuration
- **`pyproject.toml`**: Python package configuration and dependencies
- **`.pre-commit-config.yaml`**: Pre-commit hook configuration
- **`ruff.toml`**: Python linting configuration
- **`python/cocoindex/__init__.py`**: Main Python API exports

## Development Dependencies

- **Rust**: Edition 2024, minimum version 1.88
- **Python**: Minimum version 3.11
- **Maturin**: For building Python extensions from Rust
- **Pre-commit**: For code quality checks
