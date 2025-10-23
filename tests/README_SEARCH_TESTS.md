# Search Test Organization

## Overview

This document describes the comprehensive test organization for all search functionality in the CocoIndex project, including hybrid search, vector-only search, and keyword-only search. The test suite provides extensive coverage across multiple programming languages with easy configuration management.

## Test Structure

### Core Search Test Files

#### Main Search Test Files

- **`test_hybrid_search.py`** - Hybrid search combining vector similarity and keyword filtering (17 tests)
- **`test_full_text_search.py`** - Vector-only semantic search functionality (15 tests)
- **`test_keyword_search.py`** - Keyword-only metadata filtering search (16 tests)

#### Configuration Management

- **`search_config.py`** - `SearchTestConfig` class for unified test configuration

#### Engine and Parser Tests

- **`test_hybrid_search_keyword_parser.py`** - Tests for keyword search parser functionality
- **`test_hybrid_search_engine.py`** - Tests for the hybrid search engine
- **`test_hybrid_search_integration.py`** - Integration tests for the complete workflow
- **`conftest.py`** - Shared pytest fixtures and configuration

### Test Fixtures

- **`fixtures/hybrid_search.jsonc`** - 17 test cases for hybrid search validation across multiple languages
- **`fixtures/full_text_search.jsonc`** - 15 test cases for vector-only search validation
- **`fixtures/keyword_search.jsonc`** - 16 test cases for keyword-only search validation

## Search Test Configuration System

### SearchTestConfig Class

All three main search test files use a unified configuration system via the `SearchTestConfig` class that mirrors command-line options from `main_mcp_server.py`:

```python
from tests.search_config import SearchTestConfig

# Create configuration with defaults optimized for testing
config = SearchTestConfig(
    paths=["/workspaces/rust"],        # --paths
    no_live=True,                      # --no-live (disabled)
    default_embedding=True,            # --default-embedding (enabled)
    default_chunking=False,            # --default-chunking (disabled)
    default_language_handler=False,    # --default-language-handler (disabled)
    chunk_factor_percent=100,          # --chunk-factor-percent
    log_level="DEBUG",                 # --log-level
    poll_interval=30                   # --poll-interval
)

# Apply configuration to infrastructure
async with CocoIndexTestInfrastructure(**config.to_infrastructure_kwargs()) as infrastructure:
    # Run tests with configured infrastructure
```

### Configuration Features

- **Unified Settings**: Consistent configuration across all search test types
- **Debug Logging**: Built-in configuration logging with visual indicators
- **Infrastructure Integration**: Direct conversion to infrastructure parameters
- **Flexible Defaults**: Optimized defaults for testing while maintaining configurability

### Configuration Output Example

When tests run, you'll see configuration details:

```
🔧 Search Test Configuration:
  📁 Paths: ['/workspaces/rust']
  🔴 Live updates: DISABLED
  🎯 Default embedding: ENABLED
  🎯 Default chunking: DISABLED
  🎯 Default language handler: DISABLED
  📏 Chunk factor: 100%
  📊 Log level: DEBUG
```

### Test Organization with Pytest Markers

The tests are organized using pytest markers for easy filtering and execution:

#### Search Type Markers

- `@pytest.mark.hybrid_search` - Hybrid search functionality tests
- `@pytest.mark.vector_search` - Vector-only (semantic) search tests
- `@pytest.mark.keyword_search` - Keyword-only (metadata) search tests
- `@pytest.mark.keyword_parser` - Keyword search parser tests
- `@pytest.mark.search_engine` - Search engine tests

#### Test Category Markers

- `@pytest.mark.unit` - Unit tests (isolated component testing)
- `@pytest.mark.integration` - Integration tests (multi-component testing)
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.external` - Tests requiring external dependencies
- `@pytest.mark.standalone` - Tests that can run independently

## Running Tests

### Run All Search Tests

```bash
pytest ./tests/search/ -v
```

### Run Tests by Search Type

```bash
# Run all hybrid search tests
pytest -m "hybrid_search"

# Run vector-only search tests
pytest -m "vector_search"

# Run keyword-only search tests
pytest -m "keyword_search"

# Run keyword parser tests
pytest -m "keyword_parser"
```

### Run Unit Tests Only

```bash
pytest -m "unit"
```

### Run Tests by Combination

```bash
pytest -m "unit and keyword_parser"
pytest -m "integration and hybrid_search"
```

### Run Specific Test Files

```bash
# Core search functionality tests
pytest tests/search/test_hybrid_search.py
pytest tests/search/test_full_text_search.py
pytest tests/search/test_keyword_search.py

# Engine and parser tests
pytest tests/search/test_hybrid_search_keyword_parser.py
pytest tests/search/test_hybrid_search_engine.py
pytest tests/search/test_hybrid_search_integration.py
```

### Run with Verbose Output

```bash
pytest -v tests/test_hybrid_search_keyword_parser.py
```

## Test Coverage

### Core Search Tests

#### Hybrid Search (`test_hybrid_search.py`)

- **17 tests total** covering multi-language code search
- Language-specific searches (Python, Rust, Java, JavaScript, TypeScript, C++, C, Kotlin, Haskell)
- Cross-language pattern searches (fibonacci implementations, class definitions)
- Metadata validation (functions, classes, complexity, analysis methods)
- Vector + keyword query combination testing
- Test fixture: `fixtures/hybrid_search.jsonc`

#### Vector-Only Search (`test_full_text_search.py`)

- **15 tests total** covering semantic code understanding
- Semantic similarity searches across programming concepts
- Programming paradigm searches (OOP, functional, concurrent)
- Domain-specific searches (database operations, error handling, design patterns)
- Cross-language concept matching using embeddings only
- Test fixture: `fixtures/full_text_search.jsonc`

#### Keyword-Only Search (`test_keyword_search.py`)

- **16 tests total** covering metadata-based filtering
- Language-specific metadata filtering
- Boolean logic testing (AND, OR combinations)
- Metadata field validation (has_classes, functions, complexity)
- Filename pattern matching
- Promoted metadata validation
- Test fixture: `fixtures/keyword_search.jsonc`

### Engine and Parser Tests

#### Keyword Search Parser (`test_hybrid_search_keyword_parser.py`)

- **29 tests total** (28 passed, 1 skipped)
- Basic condition parsing (field:value, exists(field))
- Quoted value handling
- Boolean operators (and, or)
- Operator precedence
- Parentheses grouping
- SQL WHERE clause generation
- Complex real-world query examples

#### Search Engine (`test_hybrid_search_engine.py`)

- Engine initialization and configuration
- Vector-only search
- Keyword-only search
- Hybrid search combining both
- Result formatting (JSON and readable)
- SQL query generation
- Error handling

#### Integration Tests (`test_hybrid_search_integration.py`)

- Main entry point argument parsing
- Complete workflow testing
- Configuration management
- Performance characteristics
- Error handling scenarios

## Running Search Tests

### Quick Start Commands

```bash
# Run all search tests with configuration logging
pytest ./tests/search/ -v -s

# Run individual search test types
pytest tests/search/test_hybrid_search.py -v -s
pytest tests/search/test_full_text_search.py -v -s
pytest tests/search/test_keyword_search.py -v -s

# Run with specific test methods
pytest tests/search/test_hybrid_search.py::TestMCPDirect::test_hybrid_search_validation -v -s
```

### Configuration Customization

To customize test configuration, modify the `SearchTestConfig` instantiation in test files:

```python
# Custom configuration example
config = SearchTestConfig(
    paths=["/custom/path"],
    log_level="INFO",
    chunk_factor_percent=50,
    default_embedding=False
)
```

## Test Status and Results

### Working Tests ✅

- **Core Search Tests**: All three search types with comprehensive coverage (48 total tests)
  + Hybrid search: 17 tests across multiple programming languages
  + Vector search: 15 tests for semantic code understanding
  + Keyword search: 16 tests for metadata-based filtering
- **Configuration System**: SearchTestConfig provides unified, easy-to-use configuration
- **Test Infrastructure**: CocoIndex infrastructure with proper setup/teardown
- **Test Results**: Automatic saving to organized `/test-results/search-{type}/` directories
- **Fixture Loading**: JSONC test fixtures properly parsed and executed

### Test Result Organization

Test results are automatically saved with timestamps:

- `/test-results/search-hybrid/` - Hybrid search results combining vector + keyword
- `/test-results/search-vector/` - Vector-only semantic search results
- `/test-results/search-keyword/` - Keyword-only metadata filter results

### Configuration Features ✨

- **Easy Setup**: Default configuration optimized for testing environment
- **Visual Feedback**: Configuration logging with emojis and clear formatting
- **Flexible Overrides**: Customize any setting while keeping sensible defaults
- **Infrastructure Integration**: Direct parameter mapping to CocoIndex infrastructure

## Database Comparison Enhancement

### Enhanced Test Analysis (2025-08-20)

All three search test methods now include comprehensive **database comparison functionality** that analyzes test failures by comparing search results with actual PostgreSQL database contents:

#### Key Features

- **Root Cause Analysis**: Identifies exactly why tests fail by comparing expected results with database reality
- **Automatic Database Queries**: Connects to PostgreSQL and analyzes indexed code records
- **Detailed Discrepancy Reports**: Shows specific mismatches (e.g., "Expected complexity_score > 2, but max found in DB is 0")
- **Sample Record Display**: Shows actual database record content to understand what's indexed

#### Enhanced Test Files

- **`tests/db_comparison.py`** - Core database comparison utility with `DatabaseComparator` class
- **`tests/common.py`** - Enhanced with database comparison for all search test functions
- **All search test files** - Now provide detailed database analysis on failures

#### Example Enhanced Error Output

```
🔍 Database Comparison Analysis (KEYWORD SEARCH):
  ❌ Expected complexity_score > 2.0, but max found in DB is 0
  ❌ Expected non-empty functions, but 33 records have empty functions
📋 Database has 50 matching records
  Sample DB record: complexity_score=0, has_classes=true, language=Haskell, functions=''
```

#### Critical Issues Identified

1. **Complexity Score Calculation**: All records show complexity_score=0 instead of expected values
2. **Function Extraction Failures**: Many records have empty function names despite containing functions
3. **Language-Specific Extraction Issues**: Rust, Java, JavaScript, TypeScript function parsing problems
4. **Analysis Method Problems**: Various tree-sitter parsers not extracting metadata correctly

## Best Practices

### Running Tests with Database Analysis

1. Use `-v -s` flags for verbose output and database comparison visibility
2. Test individual search types when debugging specific functionality
3. Check `/test-results/` directories for detailed search result analysis
4. Review database comparison reports to identify indexing pipeline issues
5. Modify `SearchTestConfig` defaults for custom test environments

### Configuration Management

1. Keep default configuration optimized for common testing scenarios
2. Override specific parameters when testing edge cases
3. Use debug logging to verify configuration is applied correctly
4. Maintain consistency across all three search test types

## Test Results and Debugging

### Test Result Directories

Test results are automatically saved to organized directories:

- `/test-results/search-hybrid/` - Hybrid search test results
- `/test-results/search-vector/` - Vector-only search test results
- `/test-results/search-keyword/` - Keyword-only search test results

Each test result file contains:

- Test name and timestamp
- Original query parameters
- Complete search results with metadata
- Execution details for debugging

This organization provides flexibility for developers to run targeted test suites based on their needs while maintaining comprehensive coverage of all search functionality.
