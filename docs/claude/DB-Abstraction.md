# Database Abstraction Layer - Usage and Extension Guide

## Overview

The CocoIndex MCP Server now features a complete database abstraction layer that allows seamless switching between different vector database backends (PostgreSQL + pgvector, Qdrant) while maintaining a unified API. This abstraction enables easy extension to additional database implementations.

**Status**: Phase 2 implementation complete ✅ with comprehensive schema standardization, query abstraction, and backend factory patterns.

## Key Features

- **Unified Interface**: Single API for all vector database operations
- **Schema Standardization**: Type-safe metadata structures across all backends
- **Query Abstraction**: Database-agnostic query building and execution
- **Backend Factory**: Easy backend switching via configuration
- **MCP Protocol Compliance**: Full JSON Schema support for all endpoints

## Architecture Components

### 1. Backend Interface (`backends/__init__.py`)

The core `VectorStoreBackend` interface defines the contract all database implementations must follow:

```python
class VectorStoreBackend:
    def upsert(self, embeddings: List[List[float]], metadata: List[ChunkMetadata]) -> None
    def query(self, embedding: List[float], top_k: int, filters: Optional[Dict] = None) -> List[SearchResult]
    def configure(self, **options: Any) -> None
    def get_info(self) -> BackendInfo
```

### 2. Schema Definitions (`schemas.py`)

Standardized metadata structure using TypedDict for full mypy compliance:

```python
class ChunkMetadata(TypedDict):
    filename: str
    language: str
    code: str
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity: int
    start_line: int
    end_line: int
    chunk_id: str
    embedding_model: str
```

### 3. Query Abstraction (`query_abstraction.py`)

Fluent API for building complex queries:

```python
query = (create_query()
         .text("async database connection")
         .hybrid_search(vector_weight=0.8)
         .where_language("Python")
         .with_type_hints()
         .limit(20)
         .build())
```

### 4. Field Mapping (`mappers.py`)

Backend-specific field mapping handles differences in data storage formats:

- **PostgresFieldMapper**: JSONB + individual columns
- **QdrantFieldMapper**: Unified payload structure
- **Dynamic mapping**: Automatic mapper selection based on backend type

## Current Backend Implementations

### PostgreSQL Backend (`backends/postgres_backend.py`)

**Status**: ✅ Complete with abstraction integration

Features:

- pgvector extension for vector similarity search
- JSONB metadata storage with GIN indexing
- Full-text search integration with tsvector
- Connection pooling support
- Optimized hybrid search (vector + keyword)

Usage:

```python
from src.cocoindex_code_mcp_server.backends import create_backend

backend = create_backend("postgres",
                        connection_pool=pool,
                        table_name="code_chunks")
```

### Qdrant Backend (`backends/qdrant_backend.py`)

**Status**: 🏗️ Skeleton implementation ready for development

Prepared features:

- Memory-mapped payload optimization
- Advanced filtering with payload indexing
- Collection management
- High-performance HNSW vector search

Usage:

```python
backend = create_backend("qdrant",
                        host="localhost",
                        port=6333,
                        collection_name="code_chunks")
```

## MCP Protocol Integration

### JSON Schema Support

Following MCP best practices, all endpoints use strict JSON Schema validation:

**Tool Endpoints** (both input and output schemas):

```json
{
  "name": "search_code",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "language": {"type": "string"},
      "top_k": {"type": "integer", "maximum": 50}
    },
    "required": ["query"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "results": {
        "type": "array",
        "items": {"$ref": "#/definitions/SearchResult"}
      }
    }
  }
}
```

**Resource Endpoints** (output schema only):

```json
{
  "name": "code_metadata_schema",
  "schema": {
    "type": "object",
    "properties": {
      "filename": {"type": "string"},
      "language": {"type": "string"},
      "functions": {"type": "array", "items": {"type": "string"}}
    }
  }
}
```

## Usage Examples

### Basic Search

```python
from src.cocoindex_code_mcp_server.query_abstraction import simple_search

# Simple text search
query = simple_search("async function", top_k=10)
results = await executor.execute(query)
```

### Advanced Filtered Search

```python
from src.cocoindex_code_mcp_server.query_abstraction import create_query

# Complex query with filters
query = (create_query()
         .text("database connection pool")
         .hybrid_search(vector_weight=0.8, keyword_weight=0.2)
         .where_language("Python")
         .where_complexity_greater_than(5)
         .with_functions_containing("connect")
         .limit(15)
         .build())

results = await executor.execute(query)
```

### Backend Switching

```python
from src.cocoindex_code_mcp_server.backends import create_backend

# PostgreSQL backend
pg_backend = create_backend("postgres", connection_pool=pool)

# Qdrant backend (when available)
qdrant_backend = create_backend("qdrant", host="localhost", port=6333)

# Same query works with both backends
executor_pg = QueryExecutor(pg_backend)
executor_qdrant = QueryExecutor(qdrant_backend)
```

## Extending to New Database Implementations

### Step 1: Implement Backend Interface

Create a new backend class in `backends/`:

```python
# backends/milvus_backend.py
from .base import VectorStoreBackend

class MilvusBackend(VectorStoreBackend):
    def __init__(self, host: str, port: int, collection_name: str):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        # Initialize Milvus connection

    def upsert(self, embeddings: List[List[float]], metadata: List[ChunkMetadata]) -> None:
        # Convert metadata using MilvusFieldMapper
        mapper = create_mapper("milvus")
        milvus_data = [mapper.to_backend_format(meta) for meta in metadata]
        # Implement Milvus-specific upsert logic

    def query(self, embedding: List[float], top_k: int, filters: Optional[Dict] = None) -> List[SearchResult]:
        # Implement Milvus-specific query logic
        # Convert results back using mapper.from_backend_format()
        pass
```

### Step 2: Create Field Mapper

Add a mapper for backend-specific field handling:

```python
# mappers.py - Add new mapper class

class MilvusFieldMapper(FieldMapper):
    def to_backend_format(self, metadata: ChunkMetadata) -> Dict[str, Any]:
        # Convert ChunkMetadata to Milvus payload format
        return {
            "filename": metadata["filename"],
            "language": metadata["language"],
            "metadata_json": {
                "functions": metadata["functions"],
                "classes": metadata["classes"],
                # ... other fields
            }
        }

    def from_backend_format(self, data: Dict[str, Any]) -> ChunkMetadata:
        # Convert Milvus result back to ChunkMetadata
        return ChunkMetadata(
            filename=data["filename"],
            language=data["language"],
            functions=data["metadata_json"]["functions"],
            # ... other fields
        )
```

### Step 3: Register in Factory

Update the backend factory to include the new implementation:

```python
# backends/__init__.py - Update registry

BACKEND_REGISTRY["milvus"] = MilvusBackend

def create_backend(backend_type: str, **kwargs) -> VectorStoreBackend:
    if backend_type == "milvus":
        return MilvusBackend(**kwargs)
    # ... existing backends
```

### Step 4: Add Query Optimizations

If the new backend supports special query optimizations:

```python
# query_abstraction.py - Add backend-specific optimizations

class QueryOptimizer:
    def optimize_for_milvus(self, query: ChunkQuery) -> Dict[str, Any]:
        # Milvus-specific query optimizations
        optimized = {}
        if query.get("hybrid_search"):
            # Use Milvus hybrid search features
            optimized["search_params"] = {"nprobe": 32}
        return optimized
```

### Step 5: Add Tests

Create comprehensive tests for the new backend:

```python
# tests/backends/test_milvus_backend.py

def test_milvus_upsert():
    backend = MilvusBackend(host="localhost", port=19530)
    metadata = [create_test_metadata()]
    embeddings = [[0.1, 0.2, 0.3]]

    backend.upsert(embeddings, metadata)
    # Verify data was stored correctly

def test_milvus_query():
    backend = MilvusBackend(host="localhost", port=19530)
    results = backend.query([0.1, 0.2, 0.3], top_k=5)

    assert len(results) <= 5
    assert all(isinstance(r, SearchResult) for r in results)
```

## Query Changes for New Backends

### Database-Specific Optimizations

Different backends may require different query strategies:

**PostgreSQL**: Leverage SQL capabilities

```python
def optimize_for_postgres(self, query: ChunkQuery) -> str:
    sql = """
    WITH vector_scores AS (
        SELECT *, embedding <=> %s AS distance
        FROM {table_name}
        WHERE language = %s
    )
    SELECT * FROM vector_scores
    ORDER BY distance ASC LIMIT %s
    """
    return sql
```

**Qdrant**: Use advanced filtering

```python
def optimize_for_qdrant(self, query: ChunkQuery) -> Dict:
    return {
        "vector": query["embedding"],
        "filter": {
            "must": [
                {"key": "language", "match": {"value": query["language"]}}
            ]
        },
        "limit": query["top_k"]
    }
```

### Capability-Aware Queries

Use the backend capability system to adapt queries:

```python
def build_query(self, backend_info: BackendInfo, query: ChunkQuery):
    if BackendCapability.FULL_TEXT_SEARCH in backend_info.capabilities:
        # Use advanced text search features
        query["full_text"] = self.optimize_text_search(query["text"])
    else:
        # Fall back to basic keyword matching
        query["keyword_filter"] = self.simple_keyword_filter(query["text"])
```

## Classes to Implement for New Backends

### Required Classes

1. **Backend Implementation**
   + `CustomBackend(VectorStoreBackend)`: Main backend interface
   + Location: `backends/custom_backend.py`

2. **Field Mapper**
   + `CustomFieldMapper(FieldMapper)`: Data format conversion
   + Location: `mappers.py` (add to existing file)

3. **Query Optimizer** (optional)
   + `CustomQueryOptimizer`: Backend-specific optimizations
   + Location: `query_abstraction.py` (extend existing optimizer)

### Optional Classes

4. **Connection Manager**
   + `CustomConnectionManager`: Connection pooling/management
   + Location: `backends/custom_backend.py`

5. **Schema Manager**
   + `CustomSchemaManager`: Collection/table schema management
   + Location: `backends/custom_backend.py`

6. **Migration Helper**
   + `CustomMigrationHelper`: Data migration utilities
   + Location: `migrations/custom_migration.py`

## Best Practices

### Type Safety

- Always use proper type hints and validate with mypy
- Leverage TypedDict for metadata structures
- Use Pydantic models for complex schemas

### Error Handling

- Implement graceful fallbacks for unsupported features
- Provide clear error messages with backend context
- Log performance metrics for query optimization

### Testing

- Write comprehensive unit tests for each backend
- Include integration tests with real data
- Test migration paths between backends

### MCP Compliance

- Define JSON schemas for all tool inputs and outputs
- Use resource endpoints for metadata schemas
- Follow MCP protocol naming conventions

## Performance Considerations

### Backend Selection Guidelines

**PostgreSQL + pgvector**: Best for

- Small to medium datasets (<1M vectors)
- Rich metadata querying with SQL
- ACID compliance requirements
- Full-text search integration

**Qdrant**: Best for

- Large datasets (>10M vectors)
- High-performance vector similarity search
- Memory-constrained environments
- Advanced filtering requirements

**Extension targets**:

- **Milvus**: Enterprise features, GPU acceleration
- **Weaviate**: GraphQL API, semantic search
- **Pinecone**: Managed service, auto-scaling

### Optimization Strategies

1. **Connection Pooling**: Reuse database connections
2. **Batch Operations**: Group upserts and queries
3. **Index Optimization**: Backend-specific index tuning
4. **Caching**: Cache frequently accessed metadata
5. **Async Operations**: Use async/await for I/O operations

## Migration Support

The abstraction layer includes migration utilities for switching backends:

```python
from src.cocoindex_code_mcp_server.migration import migrate_backend

# Migrate from PostgreSQL to Qdrant
migrate_backend(
    source_backend=postgres_backend,
    target_backend=qdrant_backend,
    batch_size=1000
)
```

## Future Roadmap

### Phase 3 Extensions (Optional)

- Advanced chunking strategy selection
- Multi-backend hybrid queries
- Real-time index updates
- Performance monitoring dashboard

### Additional Backend Targets

- **Chroma**: Open-source embedding database
- **Vespa**: Large-scale search and recommendation
- **Azure Cognitive Search**: Cloud-native search
- **Elasticsearch**: Traditional search with vector support

This abstraction layer provides a solid foundation for extending the CocoIndex MCP Server to support any vector database while maintaining clean, type-safe, and MCP-compliant interfaces.

## Adding New Fields to the Database Schema

### Overview

The CocoIndex MCP Server uses a sophisticated field mapping system that automatically handles database schema evolution. Adding new fields is streamlined through the `CONST_FIELD_MAPPINGS` in `mappers.py`, which serves as the single source of truth for all database fields.

### General Process for Adding Fields

#### Step 1: Update Field Mappings

Add your new field to `CONST_FIELD_MAPPINGS` in `python/cocoindex_code_mcp_server/mappers.py`:

```python
CONST_FIELD_MAPPINGS = {
    # ... existing fields ...
    "your_new_field": "your_new_field",  # Database column name
}
```

#### Step 2: Add Extraction Function (if from metadata)

If your field comes from `metadata_json`, create an extraction function in `python/cocoindex_code_mcp_server/cocoindex_config.py`:

```python
@cocoindex.op.function()
def extract_your_new_field(metadata_json: str) -> str:
    """Extract your_new_field from metadata JSON."""
    return extract_string_field(metadata_json, "your_new_field", "default_value")
```

#### Step 3: Add Field Transformation (if needed)

If your field needs processing, add the transformation in the chunk processing section:

```python
# Around line 1420 in cocoindex_config.py
chunk["your_new_field"] = chunk["extracted_metadata"].transform(extract_your_new_field)
```

#### Step 4: Automatic Collection

The automated collection system will automatically include your field in the database operations. No manual updates to the `collect()` call are needed - this is handled dynamically via `CONST_FIELD_MAPPINGS`.

### PostgreSQL-Specific Implementation

PostgreSQL backend includes additional safeguards for schema evolution through dynamic column introspection.

#### Column Introspection and Caching

The PostgreSQL backend automatically detects available database columns and caches this information:

```python
# In postgres_backend.py
@cached(column_cache)
def _get_table_columns(pool: ConnectionPool, table_name: str) -> Set[str]:
    """Get available columns from database with caching (60s TTL)."""
    # Queries information_schema.columns for actual database schema
```

#### Dynamic Field Filtering

The backend only selects fields that actually exist in the database:

```python
def _build_select_clause(self, include_distance: bool = False) -> Tuple[str, List[str]]:
    """Build SELECT clause dynamically using only available DB columns."""
    available_columns = self._get_available_columns()

    # Filter CONST_SELECTABLE_FIELDS to only those that exist
    for field in CONST_SELECTABLE_FIELDS:
        if field in available_columns:
            fields.append(field)
        else:
            missing_fields.append(field)

    # Log warnings for missing expected fields
    if new_missing_fields:
        logger.warning(f"Expected columns missing: {sorted(new_missing_fields)}")
```

#### PostgreSQL Field Addition Workflow

1. **Add to CONST_FIELD_MAPPINGS**: Your field is now tracked by the system
2. **Run your application**: The backend detects the missing column and logs a warning
3. **Add database column**: Use migration to add the actual column to PostgreSQL
4. **Column automatically included**: On next run, the column introspection detects the new field and includes it in queries

#### Example: Adding a `complexity_rating` Field

```python
# Step 1: Add to mappers.py
CONST_FIELD_MAPPINGS = {
    # ... existing fields ...
    "complexity_rating": "complexity_rating",
}

# Step 2: Add extraction function (if from metadata)
@cocoindex.op.function()
def extract_complexity_rating_field(metadata_json: str) -> int:
    """Extract complexity_rating from metadata JSON."""
    return extract_int_field(metadata_json, "complexity_rating", 0)

# Step 3: Add transformation (if needed)
chunk["complexity_rating"] = chunk["extracted_metadata"].transform(extract_complexity_rating_field)

# Step 4: Add PostgreSQL column (migration)
ALTER TABLE codeembedding__code_embeddings
ADD COLUMN complexity_rating INTEGER DEFAULT 0;
```

#### Migration Handling

For PostgreSQL, you'll need to add the actual column to your database schema:

```sql
-- Add the new column with appropriate type and default
ALTER TABLE codeembedding__code_embeddings
ADD COLUMN your_new_field VARCHAR(255) DEFAULT '';

-- Add index if needed for query performance
CREATE INDEX IF NOT EXISTS idx_your_new_field
ON codeembedding__code_embeddings(your_new_field);
```

#### Benefits of This Approach

1. **Graceful Degradation**: Application continues working even when database schema is behind
2. **Warning System**: Clear logs about missing expected columns
3. **Automatic Integration**: New columns are automatically detected and used once added
4. **Caching**: Column introspection is cached (60s TTL) for performance
5. **Case Handling**: Automatic handling of PostgreSQL's lowercase table name conventions

This system ensures robust database schema evolution while maintaining backward compatibility and providing clear feedback about schema mismatches.
