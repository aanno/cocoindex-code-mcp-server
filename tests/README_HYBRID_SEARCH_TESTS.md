# Hybrid Search Test Organization

## Overview

This document describes the test organization for the hybrid search functionality in the CocoIndex project.

## Test Structure

### Test Files

- **`test_hybrid_search_keyword_parser.py`** - Tests for keyword search parser functionality
- **`test_hybrid_search_engine.py`** - Tests for the hybrid search engine
- **`test_hybrid_search_integration.py`** - Integration tests for the complete workflow
- **`conftest.py`** - Shared pytest fixtures and configuration

### Test Organization with Pytest Markers

The tests are organized using pytest markers for easy filtering and execution:

#### Hybrid Search Specific Markers
- `@pytest.mark.hybrid_search` - All hybrid search functionality tests
- `@pytest.mark.keyword_parser` - Keyword search parser tests
- `@pytest.mark.search_engine` - Hybrid search engine tests

#### Test Category Markers
- `@pytest.mark.unit` - Unit tests (isolated component testing)
- `@pytest.mark.integration` - Integration tests (multi-component testing)
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.external` - Tests requiring external dependencies
- `@pytest.mark.standalone` - Tests that can run independently

## Running Tests

### Run All Hybrid Search Tests
```bash
pytest -m "hybrid_search"
```

### Run Unit Tests Only
```bash
pytest -m "unit"
```

### Run Keyword Parser Tests
```bash
pytest -m "keyword_parser"
```

### Run Tests by Combination
```bash
pytest -m "unit and keyword_parser"
pytest -m "integration and hybrid_search"
```

### Run Specific Test Files
```bash
pytest tests/test_hybrid_search_keyword_parser.py
pytest tests/test_hybrid_search_engine.py
pytest tests/test_hybrid_search_integration.py
```

### Run with Verbose Output
```bash
pytest -v tests/test_hybrid_search_keyword_parser.py
```

## Test Coverage

### Keyword Search Parser (`test_hybrid_search_keyword_parser.py`)
- **29 tests total** (28 passed, 1 skipped)
- Basic condition parsing (field:value, exists(field))
- Quoted value handling
- Boolean operators (and, or)
- Operator precedence
- Parentheses grouping
- SQL WHERE clause generation
- Complex real-world query examples

### Search Engine (`test_hybrid_search_engine.py`)
- Engine initialization and configuration
- Vector-only search
- Keyword-only search  
- Hybrid search combining both
- Result formatting (JSON and readable)
- SQL query generation
- Error handling

### Integration Tests (`test_hybrid_search_integration.py`)
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
- **Keyword Parser**: All core functionality working (28/29 tests passing)
- **Basic Integration**: Argument parsing and configuration tests
- **Test Infrastructure**: Pytest markers and organization working

### Known Issues ⚠️
- Some tests require CocoIndex dependencies that have circular import issues
- Complex nested parentheses parsing has one edge case (marked as skipped)
- External dependency tests are automatically skipped when dependencies unavailable

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
# Run only fast unit tests
pytest -m "unit and not slow"

# Run hybrid search integration tests
pytest -m "integration and hybrid_search"

# Run keyword parser tests with verbose output
pytest -m "keyword_parser" -v

# Run all tests except external dependency tests
pytest -m "not external"

# Run a specific test class
pytest tests/test_hybrid_search_keyword_parser.py::TestKeywordSearchParser
```

This organization provides flexibility for developers to run targeted test suites based on their needs while maintaining comprehensive coverage of the hybrid search functionality.