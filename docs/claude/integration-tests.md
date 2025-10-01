# Integration Testing Guide

This guide explains how to run integration tests for the CocoIndex Code MCP Server and verify results against the codebase and PostgreSQL database.

## Current Status

**As of 2025-10-01:** 12/15 keyword search tests passing (80%)

Major test fixture issues were identified and fixed:
- Case sensitivity (database uses Title Case: `Python`, `Rust`)
- Wrong filenames in test expectations
- Overly strict metadata requirements
- Obsolete `chunking_method` values

See [Integration Test Results](integration-test-results.md) for detailed analysis.

## Overview

Integration tests validate the complete RAG pipeline:
1. **Code Analysis** - Tree-sitter parsing and metadata extraction
2. **AST Chunking** - Semantic code chunking
3. **Database Storage** - PostgreSQL with pgvector
4. **Keyword Search** - Metadata-based filtering
5. **Vector Search** - Semantic similarity search
6. **Hybrid Search** - Combined keyword + vector search

## Test Categories

### 1. Keyword Search Tests
**Location:** `tests/search/test_keyword_search.py`

Tests metadata-based filtering without vector similarity:
- Language filters (`language:python`, `language:rust`)
- Function name searches (`functions:fibonacci`)
- Class filtering (`has_classes:true`)
- Boolean operators (`language:rust OR language:c`)
- Promoted metadata validation

### 2. Vector Search Tests
**Location:** `tests/search/test_vector_search.py`

Tests semantic similarity search using embeddings.

### 3. Hybrid Search Tests
**Location:** `tests/search/test_hybrid_search.py`

Tests combined keyword filtering + vector similarity ranking.

## Running Tests

### Basic Test Execution

```bash
# Run all keyword search tests
pytest -c pytest.ini ./tests/search/test_keyword_search.py

# Run specific test
pytest -c pytest.ini ./tests/search/test_keyword_search.py::test_python_language_filter

# Run with verbose output
pytest -c pytest.ini ./tests/search/test_keyword_search.py -v

# Run all search tests
pytest -c pytest.ini ./tests/search/
```

### Clean Results Before Testing

```bash
# Clear previous test results
rm -r test-results/search-keyword/*
rm -r test-results/search-vector/*
rm -r test-results/search-hybrid/*

# Run tests
pytest -c pytest.ini ./tests/search/test_keyword_search.py
```

### Test Configuration

Tests use fixtures from:
- `tests/fixtures/keyword_search.jsonc` - Keyword search test cases
- `tests/fixtures/vector_search.jsonc` - Vector search test cases
- `tests/fixtures/hybrid_search.jsonc` - Hybrid search test cases

Test code files:
- `tmp/` - Sample code files in various languages (Python, Rust, Java, C, C++, JavaScript, TypeScript, Kotlin, Haskell)

## Analyzing Test Results

### 1. Check JSON Output Files

Test results are written to `test-results/search-keyword/`:

```bash
# List all test results
ls -lh test-results/search-keyword/

# View specific test result
cat test-results/search-keyword/python_language_filter_*.json | jq .

# Count results for each test
for f in test-results/search-keyword/*.json; do
    echo "$(basename $f): $(jq '.search_results.total_results' $f) results"
done
```

### 2. Verify Against Test Fixtures

Compare actual results against expected results in `tests/fixtures/keyword_search.jsonc`:

```bash
# Extract expected vs actual
python3 << 'EOF'
import json
import glob

fixture = json.load(open('tests/fixtures/keyword_search.jsonc'))
for test in fixture['tests']:
    test_name = test['name']
    expected_min = test['expected_results']['min_results']

    # Find result file
    result_files = glob.glob(f'test-results/search-keyword/{test_name}_*.json')
    if result_files:
        result = json.load(open(result_files[0]))
        actual = result['search_results']['total_results']
        status = '✅' if actual >= expected_min else '❌'
        print(f"{status} {test_name}: expected >={expected_min}, got {actual}")
EOF
```

### 3. Verify Against Database

Connect to PostgreSQL to verify data consistency:

```bash
# Check database connection
python3 << 'EOF'
import psycopg
import os

db_url = "postgres://cocoindex:cocoindex@host.docker.internal/cocoindex"
conn = psycopg.connect(db_url)
cur = conn.cursor()

# List tables
cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name LIKE '%code%'
    ORDER BY table_name;
""")
print("Database tables:")
for row in cur.fetchall():
    print(f"  - {row[0]}")

# Count records
cur.execute("SELECT COUNT(*) FROM keywordsearchtest_code_embeddings;")
print(f"\nTotal test records: {cur.fetchone()[0]}")

cur.close()
conn.close()
EOF
```

### 4. Verify Search Results Match Database

Test that search results match what's in the database:

```bash
python3 << 'EOF'
import psycopg
import json
import glob

db_url = "postgres://cocoindex:cocoindex@host.docker.internal/cocoindex"
conn = psycopg.connect(db_url)
cur = conn.cursor()

# Example: Verify boolean OR test
print("Verifying: language:rust OR language:c")
print("="*80)

# Query database
cur.execute("""
    SELECT filename, language, COUNT(*) as chunks
    FROM keywordsearchtest_code_embeddings
    WHERE language IN ('Rust', 'C')
    GROUP BY filename, language
    ORDER BY filename;
""")
db_results = cur.fetchall()
print("\nDatabase results:")
for row in db_results:
    print(f"  {row[0]} ({row[1]}): {row[2]} chunks")

# Load test results
result_files = glob.glob('test-results/search-keyword/boolean_logic_or_*.json')
if result_files:
    result = json.load(open(result_files[0]))
    test_results = result['search_results']['results']
    print(f"\nTest results: {len(test_results)} chunks returned")

    # Count by file
    from collections import Counter
    files = Counter([r['filename'] for r in test_results])
    for filename, count in sorted(files.items()):
        print(f"  {filename}: {count} chunks")

cur.close()
conn.close()
EOF
```

### 5. Check Metadata Quality

Verify promoted metadata fields are populated correctly:

```bash
python3 << 'EOF'
import psycopg

db_url = "postgres://cocoindex:cocoindex@host.docker.internal/cocoindex"
conn = psycopg.connect(db_url)
cur = conn.cursor()

# Check metadata coverage
cur.execute("""
    SELECT
        COUNT(*) as total,
        COUNT(CASE WHEN analysis_method IS NOT NULL AND analysis_method != 'unknown' THEN 1 END) as has_analysis,
        COUNT(CASE WHEN chunking_method IS NOT NULL THEN 1 END) as has_chunking,
        COUNT(CASE WHEN functions IS NOT NULL THEN 1 END) as has_functions,
        COUNT(CASE WHEN classes IS NOT NULL THEN 1 END) as has_classes,
        COUNT(CASE WHEN tree_sitter_analyze_error = FALSE THEN 1 END) as analyze_success,
        COUNT(CASE WHEN tree_sitter_chunking_error = FALSE THEN 1 END) as chunk_success
    FROM keywordsearchtest_code_embeddings;
""")
result = cur.fetchone()
print("Metadata Coverage:")
print(f"  Total records: {result[0]}")
print(f"  Has analysis_method: {result[1]} ({result[1]/result[0]*100:.1f}%)")
print(f"  Has chunking_method: {result[2]} ({result[2]/result[0]*100:.1f}%)")
print(f"  Has functions field: {result[3]} ({result[3]/result[0]*100:.1f}%)")
print(f"  Has classes field: {result[4]} ({result[4]/result[0]*100:.1f}%)")
print(f"  Analyze success: {result[5]} ({result[5]/result[0]*100:.1f}%)")
print(f"  Chunking success: {result[6]} ({result[6]/result[0]*100:.1f}%)")

# Check by language
cur.execute("""
    SELECT language,
           COUNT(*) as total,
           COUNT(CASE WHEN tree_sitter_analyze_error = FALSE THEN 1 END) as success
    FROM keywordsearchtest_code_embeddings
    GROUP BY language
    ORDER BY language;
""")
print("\nBy Language:")
for row in cur.fetchall():
    success_rate = row[2]/row[1]*100
    status = "✅" if success_rate == 100 else "⚠️"
    print(f"  {status} {row[0]:15s}: {row[2]}/{row[1]} success ({success_rate:.0f}%)")

cur.close()
conn.close()
EOF
```

### 6. Verify Against Source Code

Compare database content against actual source files:

```bash
python3 << 'EOF'
import psycopg
import os

db_url = "postgres://cocoindex:cocoindex@host.docker.internal/cocoindex"
conn = psycopg.connect(db_url)
cur = conn.cursor()

# Example: Check python_example_1.py
print("Verifying python_example_1.py")
print("="*80)

# Get from database
cur.execute("""
    SELECT filename, functions, classes, has_classes, length(code) as code_length
    FROM keywordsearchtest_code_embeddings
    WHERE filename = 'python_example_1.py'
    ORDER BY code_length DESC;
""")
db_results = cur.fetchall()
print("\nDatabase records:")
for row in db_results:
    print(f"  Chunk: {row[4]} chars")
    print(f"    Functions: {row[1]}")
    print(f"    Classes: {row[2]}")
    print(f"    Has classes: {row[3]}")

# Check actual file
file_path = 'tmp/python_example_1.py'
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    print(f"\nActual file: {len(content)} chars")
    print(f"  Contains 'fibonacci': {('fibonacci' in content)}")
    print(f"  Contains 'MathUtils': {('MathUtils' in content)}")

cur.close()
conn.close()
EOF
```

## Database Schema

The test table `keywordsearchtest_code_embeddings` has these key fields:

**Core Fields:**
- `filename` - Source file name
- `location` - Character range (int8range)
- `language` - Programming language
- `code` - Actual code chunk content
- `embedding` - Vector embedding (pgvector)

**Promoted Metadata Fields:**
- `analysis_method` - Analyzer used (python_code_analyzer, rust_ast_visitor, etc.)
- `chunking_method` - Chunking strategy (ast_tree_sitter, etc.)
- `functions` - Space-separated function names
- `classes` - Space-separated class names
- `imports` - Import statements
- `complexity_score` - Code complexity metric
- `has_type_hints` - Boolean flag
- `has_async` - Boolean flag
- `has_classes` - Boolean flag

**Error Tracking:**
- `tree_sitter_analyze_error` - Analysis failed
- `tree_sitter_chunking_error` - Chunking failed
- `success` - Overall success flag
- `parse_errors` - Number of parse errors

**Detailed Metadata (JSON):**
- `metadata_json` - Full metadata as text/JSON
- `function_details` - JSON array of function details
- `class_details` - JSON array of class details
- `data_type_details` - JSON array of type details

## Database Connection

Connection details in `.env`:
```bash
COCOINDEX_DATABASE_URL=postgres://cocoindex:cocoindex@host.docker.internal/cocoindex
```

Connect with psycopg:
```python
import psycopg
conn = psycopg.connect("postgres://cocoindex:cocoindex@host.docker.internal/cocoindex")
```

## Test Data Setup

Tests use sample files in `tmp/`:
- Python: `python_example_1.py`, `sample.py`, `python_minor_errors.py`
- Rust: `rust_example_1.rs`
- Java: `java_example_1.java`, `my/packages/structure/Main1.java`
- C: `c_example_1.c`
- C++: `cpp_example_1.cpp`
- JavaScript: `javascript_example_1.js`
- TypeScript: `typescript_example_1.ts`
- Kotlin: `kotlin_example_1.kt` (multiple locations)
- Haskell: `HaskellExample1.hs`, `haskell_buggy_example_1.hs`, `haskell_minor_errors.hs`

## Common Issues

### 1. Database Connection Fails
```bash
# Check if database is accessible
psql postgres://cocoindex:cocoindex@host.docker.internal/cocoindex -c "SELECT 1;"
```

### 2. Test Results Directory Missing
```bash
# Create directories
mkdir -p test-results/search-keyword
mkdir -p test-results/search-vector
mkdir -p test-results/search-hybrid
```

### 3. Fixtures Not Loading
```bash
# Verify fixture files exist and are valid JSON
jq . tests/fixtures/keyword_search.jsonc
```

### 4. Python Import Errors
```bash
# Ensure test dependencies are installed
pip install pytest psycopg python-dotenv
```

## Test Workflow

1. **Clear old results**: `rm -r test-results/search-keyword/*`
2. **Run tests**: `pytest -c pytest.ini ./tests/search/test_keyword_search.py`
3. **Check JSON outputs**: Review files in `test-results/search-keyword/`
4. **Verify against fixtures**: Compare actual vs expected results
5. **Verify against database**: Query PostgreSQL to confirm consistency
6. **Verify against source**: Compare with actual code files in `tmp/`
7. **Document issues**: Record any failures or unexpected results

## Continuous Integration

For CI/CD pipelines:

```bash
#!/bin/bash
set -e

# Setup
export COCOINDEX_DATABASE_URL="postgres://cocoindex:cocoindex@host.docker.internal/cocoindex"

# Clean previous results
rm -rf test-results/search-*/*

# Run tests
pytest -c pytest.ini ./tests/search/ -v --tb=short

# Verify results
python3 scripts/verify_test_results.py

# Generate report
python3 scripts/generate_test_report.py > test-results/summary.txt
```

## See Also

- [Integration Test Results](integration-test-results.md) - Latest test run results
- [Hybrid Search](Hybrid_Search.md) - Search implementation details
- [Flow Debug](Flow-Debug.md) - Debugging CocoIndex flows
- [MCP Server](Mcp_Server.md) - MCP server architecture
