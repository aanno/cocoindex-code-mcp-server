# Hybrid Search for CocoIndex

## Overview

The Hybrid Search system provides an advanced alternative entry point for CocoIndex that combines **vector similarity search** with **keyword metadata filtering**. This dual-prompt system allows users to perform sophisticated queries that leverage both semantic understanding and precise metadata criteria.

## Architecture

### Components

1. **`main_hybrid_search.py`** - Alternative main entry point with enhanced configuration
2. **`hybrid_search.py`** - Core hybrid search engine implementation
3. **`keyword_search_parser_lark.py`** - Advanced keyword search syntax parser

### Key Features

- **Dual Prompt System**: Separate inputs for vector and keyword queries
- **Hybrid Search Logic**: Combines both query types with AND logic
- **Advanced Keyword Syntax**: Supports boolean operators, field matching, and existence checks
- **Live Updates**: Background monitoring with configurable polling (default: enabled, 60s)
- **Smart Output**: Automatic JSON formatting for complex results, readable format for simple results
- **CocoIndex Integration**: Uses CocoIndex's native hybrid search capabilities

## Usage

### Command Line Interface

```bash
# Basic usage with defaults (live updates ON, 60s polling)
python -m cocoindex_code_mcp_server.main_hybrid_search.py

# Custom path and polling interval
python -m cocoindex_code_mcp_server.main_hybrid_search.py /path/to/code --poll 30

# Multiple paths
python -m cocoindex_code_mcp_server.main_hybrid_search.py /path1 /path2 --poll 45

# Disable live updates
python -m cocoindex_code_mcp_server.main_hybrid_search.py --no-live

# Explicit paths argument
python -m cocoindex_code_mcp_server.main_hybrid_search.py --paths /path/to/code1 /path/to/code2
```

### Command Line Options

- `paths` (positional): Code directory paths to index (default: "cocoindex")
- `--paths`: Alternative way to specify paths explicitly
- `--no-live`: Disable live update mode (live updates enabled by default)
- `--poll SECONDS`: Polling interval in seconds for live updates (default: 60)

### Interactive Search Interface

The system prompts for two types of queries:

1. **Vector Query** (semantic search): Natural language text for similarity matching
2. **Keyword Query** (metadata filtering): Structured query for metadata criteria

Both queries are combined with AND logic to produce filtered, ranked results.

## Keyword Search Syntax

### Basic Field Matching

```
field:value
```

Examples:

- `language:python` - Match files with language "python"
- `filename:main_interactive_query.py` - Match files named "main_interactive_query.py"
- `source_name:files_0` - Match specific source

### Quoted Values

```
field:"quoted value"
```

Examples:

- `filename:"test file.py"` - Match files with spaces in names
- `language:"C++"` - Match language with special characters

### Existence Checks

```
exists(field)
```

Examples:

- `exists(embedding)` - Match records that have an embedding
- `exists(source_name)` - Match records with a source name

### Value Contains Checks

```
value_contains(field, "search_string")
```

Search for records where a field's value contains a specific substring. This performs case-insensitive partial matching using SQL `ILIKE`.

Examples:

- `value_contains(filename, "test")` - Match files with "test" anywhere in the filename
- `value_contains(code, "async")` - Match code chunks containing "async" in the code content
- `value_contains(language, "script")` - Match languages containing "script" (e.g., "JavaScript", "TypeScript")

### Boolean Operators

#### AND Operator

```
condition1 and condition2
```

Examples:

- `language:python and filename:main_interactive_query.py`
- `exists(embedding) and language:rust`

#### OR Operator

```
condition1 or condition2
```

Examples:

- `language:python or language:rust`
- `filename:main_interactive_query.py or filename:app.py`

### Grouping with Parentheses

```
(condition1 or condition2) and condition3
```

Examples:

- `(language:python or language:rust) and exists(embedding)`
- `filename:main_interactive_query.py and (language:python or language:go)`

### General Text Search

If no field is specified, the system performs a general text search across the code content:

```
python function
```

This searches for "python function" within the actual code content.

## Query Examples

### Vector + Keyword Combinations

1. **Find Authentication in Python**
   + Vector Query: `authentication login user verification`
   + Keyword Query: `language:python and exists(embedding)`

2. **Error Handling Patterns in Rust/Go**
   + Vector Query: `error handling exception try catch`
   + Keyword Query: `(language:rust or language:go) and exists(embedding)`

3. **Database Connection Code**
   + Vector Query: `database connection pool connect establish`
   + Keyword Query: `exists(embedding) and (language:python or language:java)`

4. **Test Files with Specific Patterns**
   + Vector Query: `unit test mock assert expect`
   + Keyword Query: `filename:test and language:python`

5. **Search for Async Functions**
   + Vector Query: `asynchronous function async await`
   + Keyword Query: `value_contains(code, "async") and language:python`

6. **Find Configuration Files**
   + Vector Query: `configuration settings config parameters`
   + Keyword Query: `value_contains(filename, "config") and exists(embedding)`

### Advanced Keyword Queries

1. **Multiple Language Support**

   ```
   (language:python or language:rust or language:go) and exists(embedding)
   ```

2. **Specific File Patterns**

   ```
   (filename:main_interactive_query.py or filename:app.py) and language:python
   ```

3. **Source-Specific Search**

   ```
   source_name:files_0 and language:rust and exists(embedding)
   ```

4. **Value Contains Search**

   ```
   value_contains(filename, "test") and language:python
   ```

5. **Complex Value Contains with Boolean Logic**

   ```
   (value_contains(code, "async") or value_contains(code, "await")) and language:python
   ```

## Output Formats

### JSON Output (Complex Data)

When results contain complex nested data structures (like detailed location information), the system automatically outputs JSON:

```json
[
  {
    "filename": "python/cocoindex_code_mcp_server/main_interactive_query.py",
    "language": "Python",
    "code": "def authenticate_user(username, password):\n    ...",
    "score": 0.856,
    "start": {"line": 45, "column": 0},
    "end": {"line": 52, "column": 4},
    "source": "files_0",
    "score_type": "hybrid"
  }
]
```

### Readable Output (Simple Data)

For simpler results, the system uses human-readable formatting:

```
📊 Found 3 results:

1. [0.856] (hybrid) python/auth.py [files_0] (Python) (L45-L52)
   def authenticate_user(username, password):
   ---

2. [0.743] (vector) python/login.py (Python) (L12-L18)
   def login_handler(request):
   ---
```

## Technical Implementation

### Search Types

1. **Vector Search Only**: When only vector query is provided
2. **Keyword Search Only**: When only keyword query is provided
3. **Hybrid Search**: When both queries are provided (combined with AND logic)

### Scoring System

- **Vector Score**: Cosine similarity score (1.0 - distance)
- **Keyword Score**: Binary match (1.0 for matches)
- **Hybrid Score**: Weighted combination (default: 70% vector, 30% keyword)

### Database Integration

The system integrates with PostgreSQL + pgvector:

- Uses `embedding <=> vector` for cosine similarity
- Supports complex WHERE clauses for metadata filtering
- Leverages PostgreSQL's full SQL capabilities for advanced queries

### Live Update System

- **Background Monitoring**: File system events and polling
- **Incremental Updates**: Only reprocesses changed files
- **Non-Blocking**: Search continues while updates happen in background
- **Configurable Polling**: Adjustable interval (default: 60 seconds)

## Configuration

### Default Settings

- **Live Updates**: Enabled by default
- **Polling Interval**: 60 seconds
- **Search Results**: 10 results maximum
- **Vector Weight**: 0.7 (70%)
- **Keyword Weight**: 0.3 (30%)

### Environment Variables

The system uses the same environment configuration as the main CocoIndex system:

- `COCOINDEX_DATABASE_URL`: PostgreSQL connection string
- Other CocoIndex environment variables for embeddings, API keys, etc.

## Performance Considerations

### Indexing Performance

- **Initial Build**: Full indexing on first run
- **Incremental Updates**: Only changed files are reprocessed
- **Background Processing**: Non-blocking updates during search

### Search Performance

- **Vector Search**: Leverages pgvector indexes for fast similarity search
- **Keyword Filtering**: Uses PostgreSQL indexes on metadata fields
- **Hybrid Queries**: Optimized with PostgreSQL query planning

### Memory Usage

- **Streaming Results**: Results processed as they arrive
- **Connection Pooling**: Efficient database connection management
- **Minimal Footprint**: Live updater runs in background with low overhead

## Error Handling

### Common Issues

1. **Database Connection**: Clear error messages for connection failures
2. **Invalid Syntax**: Helpful parsing error messages for keyword queries
3. **Empty Results**: Informative messages when no matches found
4. **Interrupted Updates**: Graceful handling of Ctrl+C during live updates

### Recovery Mechanisms

- **Automatic Reconnection**: Database connection resilience
- **Partial Results**: Return available results even if some queries fail
- **Graceful Degradation**: Fall back to simpler queries when complex ones fail

## Integration with CocoIndex

### Flow Integration

The hybrid search system seamlessly integrates with CocoIndex flows:

- Uses the same `code_embedding_flow` configuration
- Leverages existing chunking and embedding logic
- Supports all CocoIndex source types and configurations

### Extension Points

- **Custom Embeddings**: Can use different embedding models
- **Additional Metadata**: Supports custom metadata fields
- **Source Integration**: Works with any CocoIndex source (S3, Azure, etc.)

## Implementing New Operators

The keyword search system is extensible and supports adding new operators. Here's how to implement new operators like `value_contains`:

### Development Process

1. **Update Grammar** (`python/grammars/keyword_search.lark`):

   ```lark
   // Add new operator rule
   my_new_operator: "my_operator" "(" FIELD "," value ")"

   // Add to conditions
   ?condition: field_condition
             | exists_condition
             | value_contains_condition
             | my_new_operator        // Add here
             | "(" or_expr ")"
   ```

2. **Add SearchCondition Field** (`keyword_search_parser_lark.py`):

   ```python
   @dataclass
   class SearchCondition:
       field: str
       value: str
       is_exists_check: bool = False
       is_value_contains_check: bool = False
       is_my_new_operator_check: bool = False  # Add new field
   ```

3. **Implement Transformer** (`keyword_search_parser_lark.py`):

   ```python
   def my_new_operator(self, items):
       """Transform my_operator(field, value) condition."""
       field, value = items
       return SearchCondition(
           field=str(field),
           value=str(value),
           is_my_new_operator_check=True
       )
   ```

4. **Add SQL Generation** (`build_sql_where_clause`):

   ```python
   elif condition.is_my_new_operator_check:
       # Generate appropriate SQL
       where_parts.append(f"{prefix}{condition.field} ~ %s")  # Example: regex
       params.append(f"^{condition.value}")
   ```

5. **Write Tests** (`tests/test_my_operator.py`):

   ```python
   def test_my_operator_parsing():
       parser = KeywordSearchParser()
       result = parser.parse('my_operator(field, "value")')
       assert result.conditions[0].is_my_new_operator_check is True
   ```

### Example: Adding a Range Operator

For a `range(field, min, max)` operator:

```python
# 1. Grammar addition
range_condition: "range" "(" FIELD "," value "," value ")"

# 2. SearchCondition extension
is_range_check: bool = False
range_min: str = ""
range_max: str = ""

# 3. SQL generation
elif condition.is_range_check:
    where_parts.append(f"CAST({prefix}{condition.field} AS INTEGER) BETWEEN %s AND %s")
    params.extend([condition.range_min, condition.range_max])
```

### Testing New Operators

Use the existing RAG system to test implementations:

```python
# Test via MCP
result = mcp__cocoindex-rag__hybrid_search(
    vector_query="example search",
    keyword_query='my_operator(field, "value") and language:python'
)
```

## Future Enhancements

### Planned Features

1. **Fuzzy Matching**: Approximate string matching in keyword queries
2. **Range Queries**: Numeric range filtering (e.g., `range(line_count, 10, 100)`)
3. **Regex Support**: Regular expression matching (e.g., `regex_match(filename, ".*test.*")`)
4. **Saved Queries**: Ability to save and reuse complex queries
5. **Query History**: Track and replay previous searches

### API Endpoints

Future versions may include REST API endpoints:

- `POST /search/hybrid` - Programmatic hybrid search
- `GET /search/history` - Query history
- `POST /search/saved` - Saved query management

## Troubleshooting

### Common Problems

1. **No Results Found**
   + Check if indexing completed successfully
   + Verify keyword syntax is correct
   + Try simpler queries to narrow down issues

2. **Slow Queries**
   + Reduce result limit
   + Simplify keyword conditions
   + Check database indexes

3. **Live Updates Not Working**
   + Verify file permissions
   + Check polling interval settings
   + Look for error messages in output

### Debug Mode

For debugging, you can examine the generated SQL queries by modifying the search engine to print query details.

## Best Practices

### Query Design

1. **Start Simple**: Begin with basic queries and add complexity gradually
2. **Use Specific Fields**: Prefer field-specific searches over general text search
3. **Combine Strategically**: Use both vector and keyword queries for best results
4. **Test Incrementally**: Test keyword syntax separately before combining

### Performance Optimization

1. **Use Indexes**: Ensure database has appropriate indexes for your queries
2. **Limit Results**: Use reasonable result limits for interactive use
3. **Monitor Resources**: Watch database and memory usage during live updates
4. **Tune Polling**: Adjust polling intervals based on your use case

### Maintenance

1. **Regular Updates**: Keep the index updated with live monitoring
2. **Monitor Logs**: Check for errors during background updates
3. **Database Maintenance**: Regular PostgreSQL maintenance for optimal performance
4. **Backup Strategy**: Include both code and database in backup plans
