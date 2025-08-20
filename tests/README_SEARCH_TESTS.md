# Search Test Organization

## Overview

This document describes the test organization for all search functionality in the CocoIndex project, including hybrid search, vector-only search, and keyword-only search.

## Test Structure

### Test Files

#### Core Search Tests
- **`test_hybrid_search.py`** - Tests for hybrid search functionality using test fixtures
- **`test_full_text_search.py`** - Tests for vector-only (semantic) search functionality  
- **`test_keyword_search.py`** - Tests for keyword-only (metadata) search functionality

#### Engine and Parser Tests
- **`test_hybrid_search_keyword_parser.py`** - Tests for keyword search parser functionality
- **`test_hybrid_search_engine.py`** - Tests for the hybrid search engine
- **`test_hybrid_search_integration.py`** - Integration tests for the complete workflow
- **`conftest.py`** - Shared pytest fixtures and configuration

### Test Fixtures
- **`fixtures/hybrid_search.jsonc`** - Test cases for hybrid search validation
- **`fixtures/full_text_search.jsonc`** - Test cases for vector-only search validation
- **`fixtures/keyword_search.jsonc`** - Test cases for keyword-only search validation

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

## Configuration

### pytest.ini
The project includes a `pytest.ini` file with:
- Marker definitions
- Test path configuration
- Default options for strict marker checking

### conftest.py
Shared fixtures include:
- Mock database components
- Sample search results
- Mock embedding functions
- Test environment setup

## Test Status

### Working Tests ✅
- **Core Search Tests**: All three search types have comprehensive test coverage
- **Keyword Parser**: All core functionality working (28/29 tests passing)  
- **Test Infrastructure**: CocoIndex infrastructure setup and teardown working
- **Test Results**: Automatic saving to `/test-results/search-{type}/` directories
- **Fixture Loading**: JSONC test fixtures properly parsed and executed

### Known Issues ⚠️
- Some engine tests require CocoIndex dependencies that have circular import issues
- Complex nested parentheses parsing has one edge case (marked as skipped)
- External dependency tests are automatically skipped when dependencies unavailable
- Some backend integration tests need database mocking improvements

## Best Practices

### Writing New Tests
1. Use appropriate pytest markers
2. Follow the fixture pattern established in conftest.py
3. Mock external dependencies appropriately
4. Write clear, descriptive test names
5. Group related tests in classes

### Test Organization
- Unit tests should be isolated and fast
- Integration tests can have external dependencies
- Use markers consistently for filtering
- Keep test files focused on specific components

## Migration from unittest

The tests were successfully migrated from unittest to pytest format:
- Converted `self.assert*` to `assert` statements
- Replaced `setUp()` with pytest fixtures
- Added pytest markers for organization
- Maintained all test functionality

## Example Usage

```bash
# Run all search tests
pytest ./tests/search/ -v

# Run only fast unit tests
pytest -m "unit and not slow"

# Run all search type tests
pytest -m "hybrid_search or vector_search or keyword_search"

# Run hybrid search integration tests
pytest -m "integration and hybrid_search"

# Run keyword parser tests with verbose output
pytest -m "keyword_parser" -v

# Run all tests except external dependency tests
pytest -m "not external"

# Run specific test classes
pytest tests/search/test_hybrid_search.py::TestMCPDirect
pytest tests/search/test_full_text_search.py::TestFullTextSearch
pytest tests/search/test_keyword_search.py::TestKeywordSearch
pytest tests/search/test_hybrid_search_keyword_parser.py::TestKeywordSearchParser
```

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