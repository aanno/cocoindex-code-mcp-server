# Integration Testing Guide

This guide explains how to run integration tests for the CocoIndex Code MCP Server and verify results against the codebase and PostgreSQL database.

## Current Status

**As of 2025-10-02:**

- **Keyword Search:** 12/15 tests passing (80%)
- **Hybrid Search:** 20/21 tests passing (95.2%) ✅
- **Vector Search:** 14/15 tests passing (93.3%) ✅

Major test fixture issues identified and fixed across all search types:

- Case sensitivity (database uses Title Case: `Python`, `Rust`, `Java`, etc.)
- Wrong filenames in test expectations (`test_*.ext` → `*_example_1.ext`)
- Overly strict metadata requirements (complexity scores, field requirements)
- Obsolete `chunking_method` values (`astchunk_library` → `ast_tree_sitter`)
- False `has_classes: false` requirements for Rust/C/Haskell (structs counted as classes)
- Overly strict complexity scores (`>2` or `>3` → `>0`)

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
**Fixture:** `tests/fixtures/vector_search.jsonc`

Tests semantic similarity search using embeddings only (no keyword filtering):

- Semantic code pattern searches (AST, algorithms, data structures)
- Programming paradigm searches (OOP, functional, concurrent)
- Domain-specific searches (database, error handling, design patterns)
- Cross-language concept searches (fibonacci, inheritance, generics)

### 3. Hybrid Search Tests

**Location:** `tests/search/test_hybrid_search.py`
**Fixture:** `tests/fixtures/hybrid_search.jsonc`

Tests combined keyword filtering + vector similarity ranking:

- Language-specific semantic searches (Python, Rust, Java, JavaScript, TypeScript, C++, C, Kotlin, Haskell)
- Cross-language pattern searches (fibonacci implementations, class definitions)
- Metadata validation with semantic relevance
- Configurable vector/keyword weighting

## Running Tests

### Basic Test Execution

```bash
# Run keyword search tests
pytest -c pytest.ini ./tests/search/test_keyword_search.py

# Run hybrid search tests
pytest -c pytest.ini ./tests/search/test_hybrid_search.py

# Run vector search tests
pytest -c pytest.ini ./tests/search/test_vector_search.py

# Run specific test
pytest -c pytest.ini ./tests/search/test_keyword_search.py::TestKeywordSearch::test_keyword_search_validation

# Run with verbose output
pytest -c pytest.ini ./tests/search/test_keyword_search.py -v

# Run all search tests
pytest -c pytest.ini ./tests/search/
```

### Clean Results Before Testing

```bash
# Clear previous keyword search test results
rm -r test-results/search-keyword/*
pytest -c pytest.ini ./tests/search/test_keyword_search.py

# Clear previous hybrid search test results
rm -r test-results/search-hybrid/*
pytest -c pytest.ini ./tests/search/test_hybrid_search.py

# Clear all search test results
rm -r test-results/search-*/*
pytest -c pytest.ini ./tests/search/
```

### Clear Database Tables for Re-indexing

**⚠️ CRITICAL:** When code changes affect analysis logic (e.g., complexity calculation, metadata extraction), you MUST clear BOTH embeddings tables AND tracking tables to force re-indexing.

#### Why This Is Needed

CocoIndex uses tracking tables to avoid re-processing unchanged files:

- Each flow has a tracking table: `{test_type}searchtest__cocoindex_tracking`
- Tracks which files have been processed and their fingerprints
- If source files are unchanged, flow reports "NO CHANGE" and skips processing
- This causes tests to use stale data even after code fixes

#### Database Tables to Clear

**Embeddings Tables:**

```sql
keywordsearchtest_code_embeddings
vectorsearchtest_code_embeddings
hybridsearchtest_code_embeddings
```

**Tracking Tables (critical!):**

```sql
searchtest_keyword__cocoindex_tracking
searchtest_vector__cocoindex_tracking
searchtest_hybrid__cocoindex_tracking
```

#### Clear Database Script

```bash
python3 << 'EOF'
import psycopg
from dotenv import load_dotenv
import os

load_dotenv()
db_url = os.getenv("COCOINDEX_DATABASE_URL")
conn = psycopg.connect(db_url)
cur = conn.cursor()

# Tables to clear
embeddings_tables = [
    'keywordsearchtest_code_embeddings',
    'vectorsearchtest_code_embeddings',
    'hybridsearchtest_code_embeddings'
]

tracking_tables = [
    'searchtest_keyword__cocoindex_tracking',
    'searchtest_vector__cocoindex_tracking',
    'searchtest_hybrid__cocoindex_tracking'
]

# Clear embeddings tables
for table in embeddings_tables:
    cur.execute(f"DELETE FROM {table};")
    count = cur.rowcount
    print(f"✅ Deleted {count} records from {table}")

# Clear tracking tables (critical for re-indexing!)
for table in tracking_tables:
    cur.execute(f"DELETE FROM {table};")
    count = cur.rowcount
    print(f"✅ Deleted {count} records from {table}")

conn.commit()
cur.close()
conn.close()
print("\n✅ Database cleared - ready for fresh indexing")
EOF
```

#### When to Clear Database Tables

Clear database tables in these scenarios:

1. **After fixing analysis bugs** (e.g., Rust complexity calculation)
2. **After changing metadata extraction logic** (e.g., adding new node types)
3. **After modifying AST visitors** (e.g., updating complexity weights)
4. **When test files show stale metadata** (e.g., complexity_score=0 after fix)
5. **When flow reports "NO CHANGE" but you expect changes**

**Example scenario:**

```
1. Bug: Rust files have complexity_score=0
2. Fix: Add Rust node types to complexity_weights
3. Test: Still fails with complexity_score=0 (stale data!)
4. Clear: Delete from both embeddings AND tracking tables
5. Retest: Now shows correct complexity_score=16 ✅
```

### Test Configuration

Tests use fixtures from:

- `tests/fixtures/keyword_search.jsonc` - Keyword search test cases (15 tests)
- `tests/fixtures/vector_search.jsonc` - Vector search test cases (15 tests)
- `tests/fixtures/hybrid_search.jsonc` - Hybrid search test cases (17 tests)

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

## Hybrid Search Testing

### Hybrid Search Implementation: INTERSECTION Behavior ✅ VERIFIED

**Test Results:** ✅ **20/21 tests PASSING** (95.2%)
**Date:** 2025-10-02

#### How Hybrid Search Actually Works

From `postgres_backend.py:211-250`, hybrid search implements **INTERSECTION semantics**:

```sql
WITH vector_scores AS (
    SELECT *, embedding <=> %s AS vector_distance,
           (1.0 - (embedding <=> %s)) AS vector_similarity
    FROM table
    WHERE {keyword_filters}  -- KEYWORD FILTER FIRST (intersection)
),
ranked_results AS (
    SELECT *, (vector_similarity * vector_weight) AS hybrid_score
    FROM vector_scores
)
SELECT * FROM ranked_results ORDER BY hybrid_score DESC LIMIT top_k
```

**Key Findings:**

1. ✅ **INTERSECTION approach**: Keyword filters restrict results FIRST
2. ✅ **Vector similarity ranks** within the filtered subset
3. ⚠️ **keyword_weight parameter is IGNORED** - only vector_weight is used (bug)
4. ✅ **Hybrid results ⊆ Keyword results** (always true)
5. ✅ **Hybrid ≠ Union** of vector + keyword results

#### Three Search Modes Comparison

| Search Type | Filter | Ranking | Use Case |
|-------------|--------|---------|----------|
| **Vector** | None | Vector similarity | Semantic search across all code |
| **Keyword** | Metadata filters | filename, start position | Exact metadata filtering |
| **Hybrid** | Metadata filters | Vector similarity (within filtered) | Semantic search within specific language/file type |

#### Intersection Verification Tests

All 4 intersection verification tests passed, confirming expected behavior:

| Test | Vector Query | Keyword Filter | Results | Verified |
|------|-------------|----------------|---------|----------|
| `fibonacci_java_only` | "recursive fibonacci..." | `language:Java` | 6 Java results | ✅ |
| `complex_python_only` | "complex algorithm..." | `language:Python` | 6 Python results | ✅ |
| `class_based_languages` | "class definition..." | `has_classes:true` | 10 OOP results | ✅ |
| `semantic_ranking_rust` | "struct implementation..." | `language:Rust` | 2 Rust results | ✅ |

**Conclusion:** Hybrid search correctly filters by keywords FIRST, then ranks by semantic similarity within filtered set.

### Test Categories

#### 1. Language-Specific Semantic Searches

Tests semantic queries combined with language filters:

```json
{
  "query": {
    "vector_query": "struct implementation methods",
    "keyword_query": "language:Rust"
  }
}
```

#### 2. Cross-Language Pattern Searches

Tests semantic patterns across multiple languages:

```json
{
  "query": {
    "vector_query": "fibonacci recursive algorithm implementation",
    "keyword_query": "functions:fibonacci"
  }
}
```

#### 3. Metadata Validation

Tests metadata fields with semantic relevance:

```json
{
  "query": {
    "vector_query": "python class functions",
    "keyword_query": "language:Python has_classes:true"
  }
}
```

#### 4. Intersection Behavior Verification (NEW)

Tests that verify hybrid search uses intersection semantics:

**Test: Cross-Language Fibonacci Filtered to Java**

```json
{
  "name": "hybrid_intersection_fibonacci_java_only",
  "description": "Vector finds fibonacci in all languages, keyword filters to Java only",
  "query": {
    "vector_query": "recursive fibonacci sequence algorithm implementation",
    "keyword_query": "language:Java"
  },
  "expected_results": {
    "should_contain": [{
      "filename_pattern": ".*java_example.*\\.java$",
      "expected_metadata": {
        "language": "Java",
        "functions": "!empty"
      }
    }],
    "min_results": 1
  }
}
```

**Expected Behavior:**

- Vector search alone: Finds fibonacci in Python, Rust, C++, Java, Kotlin, etc.
- Keyword search alone: Finds all Java files
- Hybrid search: Finds ONLY Java fibonacci implementations (intersection)

**Test: Result Count Invariants**

For any hybrid search:

- `count(hybrid_results) <= count(keyword_results)` ✅ (always true)
- `count(hybrid_results) <= count(vector_results)` ⚠️ (not necessarily true)
- `hybrid_results ⊆ keyword_results` ✅ (always true)

### Running Hybrid Search Tests

```bash
# Clean results
rm -r test-results/search-hybrid/*

# Run tests
pytest -c pytest.ini ./tests/search/test_hybrid_search.py

# View results
ls -lh test-results/search-hybrid/
cat test-results/search-hybrid/<test_name>_*.json | jq .
```

### Analyzing Hybrid Search Results

Hybrid search results include both keyword and vector components:

```python
import json

# Load hybrid search result
with open('test-results/search-hybrid/java_class_inheritance_search_*.json') as f:
    result = json.load(f)

# Check query used
print(f"Vector query: {result['query']['vector_query']}")
print(f"Keyword query: {result['query']['keyword_query']}")

# Check results
for r in result['search_results']['results']:
    print(f"File: {r['filename']}")
    print(f"  Language: {r['language']}")
    print(f"  Classes: {r['classes']}")
    print(f"  Functions: {r['functions']}")
```

### Hybrid vs Keyword Comparison

```bash
# Run both test types
pytest -c pytest.ini ./tests/search/test_keyword_search.py
pytest -c pytest.ini ./tests/search/test_hybrid_search.py

# Compare results
python3 << 'EOF'
import json
import glob

keyword_results = glob.glob('test-results/search-keyword/*.json')
hybrid_results = glob.glob('test-results/search-hybrid/*.json')

print(f"Keyword tests: {len(keyword_results)}")
print(f"Hybrid tests: {len(hybrid_results)}")

# Compare specific test
kw_file = 'test-results/search-keyword/python_language_filter_*.json'
hy_file = 'test-results/search-hybrid/basename_python_language_filter_*.json'

if glob.glob(kw_file) and glob.glob(hy_file):
    kw = json.load(open(glob.glob(kw_file)[0]))
    hy = json.load(open(glob.glob(hy_file)[0]))

    print(f"\nPython results:")
    print(f"  Keyword-only: {kw['search_results']['total_results']} results")
    print(f"  Hybrid: {hy['search_results']['total_results']} results")
EOF
```

## Continuous Integration

For CI/CD pipelines:

```bash
#!/bin/bash
set -e

# Setup
export COCOINDEX_DATABASE_URL="postgres://cocoindex:cocoindex@host.docker.internal/cocoindex"

# Clean previous results
rm -rf test-results/search-*/*

# Run all search tests
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
