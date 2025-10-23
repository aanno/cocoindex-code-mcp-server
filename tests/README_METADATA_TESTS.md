# Python Metadata and JSON Formatting Tests

This directory contains comprehensive tests for the Python metadata extraction and JSON formatting functionality.

## Test Files

### `test_python_metadata.py`

**Unit tests for Python code metadata analysis**

- Tests the `python_code_analyzer.py` module
- Covers function detection, class analysis, import extraction, decorator detection
- Tests type hints, async functions, complexity scoring, and more
- Includes comprehensive real-world code examples
- Can be run standalone or with pytest

**Usage:**

```bash
# With pytest (recommended)
pytest tests/test_python_metadata.py -v

# Standalone execution
python tests/test_python_metadata.py
```

### `test_json_formatting.py`

**Unit tests for JSON output formatting**

- Tests the improved `format_results_as_json()` function
- Verifies that `code` and `metadata_json` fields are human-readable
- Ensures no JSON escaping of newlines, quotes, and tabs in code fields
- Tests proper JSON formatting for other fields

**Usage:**

```bash
# With pytest (recommended)
pytest tests/test_json_formatting.py -v

# Standalone execution
python tests/test_json_formatting.py
```

### `test_metadata_search.py`

**Integration tests for hybrid search with metadata**

- Tests the complete hybrid search functionality with Python metadata
- Requires running database with indexed code
- Tests vector search, keyword search, and combined hybrid search
- Verifies metadata extraction in search results

**Usage:**

```bash
# With pytest (integration tests)
pytest tests/test_metadata_search.py -m integration -v

# Standalone execution (manual demonstration)
python tests/test_metadata_search.py
```

## Test Categories

### Unit Tests

- `test_python_metadata.py` - Tests core metadata analysis
- `test_json_formatting.py` - Tests output formatting

### Integration Tests

- `test_metadata_search.py` - Tests full search pipeline

## Requirements

### For Unit Tests

- Python 3.11+
- Standard library modules (ast, json, typing, etc.)
- Optional: pytest for better test running

### For Integration Tests

- Full application setup with CocoIndex
- Database connection (PostgreSQL + pgvector)
- Environment variable: `COCOINDEX_DATABASE_URL`
- Indexed code in the database

## Running All Tests

```bash
# Run all unit tests
pytest tests/test_python_metadata.py tests/test_json_formatting.py -v

# Run integration tests (requires database)
pytest tests/test_metadata_search.py -m integration -v

# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -m "not integration" -v  # Unit tests only
```

## Test Coverage

### Python Metadata Analysis (`test_python_metadata.py`)

- ✅ Function detection (simple, async, decorated)
- ✅ Class detection (inheritance, methods, properties)
- ✅ Import analysis (various import styles)
- ✅ Decorator detection (@property, @dataclass, custom)
- ✅ Type hints detection
- ✅ Complexity scoring
- ✅ Docstring detection
- ✅ Variable analysis (module-level, class variables)
- ✅ Private/dunder method categorization
- ✅ Real-world code examples
- ✅ Edge cases and error handling
- ✅ JSON serialization compatibility

### JSON Formatting (`test_json_formatting.py`)

- ✅ Code field formatting (no JSON escaping)
- ✅ Metadata JSON field formatting (readable JSON)
- ✅ Normal field formatting (standard JSON)
- ✅ Empty results handling
- ✅ Mixed content formatting

### Integration Tests (`test_metadata_search.py`)

- ✅ Python async function search
- ✅ Class definition search
- ✅ Pure keyword search
- ✅ Metadata field presence verification
- ✅ JSON serialization in real results

## Key Improvements Tested

1. **Human-Readable Output**: Code and metadata_json fields are now formatted without JSON escaping
2. **Rich Python Metadata**: Comprehensive extraction of functions, classes, imports, decorators, etc.
3. **Robust Analysis**: Handles edge cases, syntax errors, and various Python constructs
4. **Integration**: Full end-to-end testing of search with metadata

## Expected Test Results

### Unit Tests

- Should pass completely when python_code_analyzer module is available
- Comprehensive coverage of metadata extraction features
- Fast execution (< 1 second for all unit tests)

### Integration Tests

- Require database connection and indexed code
- May skip if environment not properly set up
- Test real search functionality with actual data

## Troubleshooting

### Import Errors

If you see import errors, ensure:

- You're running from the correct directory
- The `python/` directory is in your Python path
- All required modules are available

### Database Connection Issues

For integration tests:

- Ensure `COCOINDEX_DATABASE_URL` is set
- Database should be running and accessible
- Code should be indexed in the database

### Missing Dependencies

```bash
# Install test dependencies
pip install pytest python-dotenv psycopg-pool
```

## Future Enhancements

- [ ] Add tests for other programming languages
- [ ] Performance benchmarks for metadata extraction
- [ ] More sophisticated code complexity metrics
- [ ] Additional output format tests
- [ ] Mock database tests for integration scenarios

## Contributing

When adding new metadata features:

1. Add corresponding unit tests in `test_python_metadata.py`
2. Update integration tests if needed
3. Ensure all tests pass before submitting changes
4. Update this README with new test coverage
