#!/usr/bin/env python3

"""
Integration tests for Phase 2: Schema & Query Standardization.

Tests the interaction between schemas, mappers, and query abstraction.
"""

import pytest
from typing import Dict, Any
import numpy as np

from src.cocoindex_code_mcp_server.schemas import (
    ChunkMetadata, QueryFilter, FilterOperator, SearchResultType,
    validate_chunk_metadata, create_default_chunk_query
)
from src.cocoindex_code_mcp_server.mappers import (
    PostgresFieldMapper, QdrantFieldMapper, MapperFactory, ResultMapper
)
from src.cocoindex_code_mcp_server.query_abstraction import (
    QueryBuilder, create_query, simple_search, find_functions_in_language
)


class TestSchemas:
    """Test schema definitions and validation."""
    
    def test_chunk_metadata_validation_success(self):
        """Test successful metadata validation."""
        raw_metadata = {
            "filename": "test.py",
            "language": "Python", 
            "location": "test.py:1-10",
            "code": "def test(): pass",
            "functions": ["test"],
            "classes": [],
            "imports": [],
            "complexity_score": 1,
            "has_type_hints": False,
            "has_async": False,
            "has_classes": False
        }
        
        validated = validate_chunk_metadata(raw_metadata)
        
        assert validated["filename"] == "test.py"
        assert validated["language"] == "Python"
        assert validated["functions"] == ["test"]
        assert validated["complexity_score"] == 1
        assert validated["has_type_hints"] is False
    
    def test_chunk_metadata_validation_missing_required(self):
        """Test validation failure with missing required fields."""
        raw_metadata = {
            "code": "def test(): pass"
            # Missing filename, language, location
        }
        
        with pytest.raises(ValueError, match="Required field 'filename' missing"):
            validate_chunk_metadata(raw_metadata)
    
    def test_chunk_metadata_validation_invalid_types(self):
        """Test validation failure with invalid field types."""
        raw_metadata = {
            "filename": "test.py",
            "language": "Python",
            "location": "test.py:1-10", 
            "functions": "not_a_list"  # Should be list
        }
        
        with pytest.raises(ValueError, match="'functions' field must be a list"):
            validate_chunk_metadata(raw_metadata)
    
    def test_create_default_chunk_query(self):
        """Test default query creation."""
        query = create_default_chunk_query(text="test search", top_k=5)
        
        assert query["text"] == "test search"
        assert query["top_k"] == 5
        assert query["filters"] == []
        assert query["filter_logic"] == "AND"
        assert query["vector_weight"] == 0.7
        assert query["keyword_weight"] == 0.3


class TestPostgresFieldMapper:
    """Test PostgreSQL field mapping."""
    
    def test_to_backend_format(self):
        """Test conversion to PostgreSQL format."""
        mapper = PostgresFieldMapper()
        metadata: ChunkMetadata = {
            "filename": "test.py",
            "language": "Python",
            "location": "test.py:1-10",
            "code": "def test(): pass",
            "functions": ["test"],
            "classes": [],
            "complexity_score": 1,
            "metadata_json": {"extra": "data"}
        }
        
        pg_format = mapper.to_backend_format(metadata)
        
        assert pg_format["filename"] == "test.py"
        assert pg_format["language"] == "Python"
        assert pg_format["functions"] == ["test"]
        assert pg_format["complexity_score"] == 1
        # JSONB field should be JSON string
        assert '"extra": "data"' in pg_format["metadata_json"]
    
    def test_from_backend_format(self):
        """Test conversion from PostgreSQL format."""
        mapper = PostgresFieldMapper()
        pg_row = {
            "filename": "test.py",
            "language": "Python", 
            "location": "test.py:1-10",
            "code": "def test(): pass",
            "functions": ["test"],
            "classes": [],
            "complexity_score": 1,
            "metadata_json": '{"extra": "data"}'  # JSON string
        }
        
        metadata = mapper.from_backend_format(pg_row)
        
        assert metadata["filename"] == "test.py"
        assert metadata["functions"] == ["test"]
        assert metadata["complexity_score"] == 1
        assert metadata["metadata_json"] == {"extra": "data"}  # Parsed back to dict
    
    def test_map_query_filter_equals(self):
        """Test query filter mapping for equals operator."""
        mapper = PostgresFieldMapper()
        query_filter = QueryFilter(
            field="language",
            operator=FilterOperator.EQUALS,
            value="Python"
        )
        
        sql_fragment = mapper.map_query_filter(query_filter)
        assert sql_fragment == "language = %s"
    
    def test_map_query_filter_in(self):
        """Test query filter mapping for IN operator."""
        mapper = PostgresFieldMapper()
        query_filter = QueryFilter(
            field="language",
            operator=FilterOperator.IN,
            value=["Python", "JavaScript"]
        )
        
        sql_fragment = mapper.map_query_filter(query_filter)
        assert sql_fragment == "language IN (%s, %s)"
    
    def test_map_query_filter_jsonb_contains(self):
        """Test JSONB contains operation."""
        mapper = PostgresFieldMapper()
        query_filter = QueryFilter(
            field="metadata_json",
            operator=FilterOperator.CONTAINS,
            value={"key": "value"}
        )
        
        sql_fragment = mapper.map_query_filter(query_filter)
        assert sql_fragment == "metadata_json @> %s::jsonb"
    
    def test_build_insert_query(self):
        """Test INSERT query building."""
        mapper = PostgresFieldMapper()
        metadata: ChunkMetadata = {
            "filename": "test.py",
            "language": "Python",
            "location": "test.py:1-10"
        }
        
        query, params = mapper.build_insert_query("test_table", metadata)
        
        assert "INSERT INTO test_table" in query
        assert "ON CONFLICT (filename, location)" in query
        assert "DO UPDATE SET" in query
        assert len(params) == 3  # filename, language, location


class TestQdrantFieldMapper:
    """Test Qdrant field mapping."""
    
    def test_to_backend_format(self):
        """Test conversion to Qdrant payload format."""
        mapper = QdrantFieldMapper()
        metadata: ChunkMetadata = {
            "filename": "test.py",
            "language": "Python",
            "location": "test.py:1-10",
            "functions": ["test"],
            "embedding": np.array([0.1, 0.2, 0.3], dtype=np.float32)
        }
        
        payload = mapper.to_backend_format(metadata)
        
        assert payload["filename"] == "test.py"
        assert payload["functions"] == ["test"]
        # Embedding should be excluded from payload
        assert "embedding" not in payload
    
    def test_from_backend_format(self):
        """Test conversion from Qdrant point format."""
        mapper = QdrantFieldMapper()
        qdrant_point = {
            "payload": {
                "filename": "test.py",
                "language": "Python",
                "functions": ["test"],
                "complexity_score": 1
            }
        }
        
        metadata = mapper.from_backend_format(qdrant_point)
        
        assert metadata["filename"] == "test.py"
        assert metadata["functions"] == ["test"]
        assert metadata["complexity_score"] == 1
    
    def test_map_query_filter_range(self):
        """Test range filter mapping."""
        mapper = QdrantFieldMapper()
        query_filter = QueryFilter(
            field="complexity_score",
            operator=FilterOperator.GREATER_THAN,
            value=5
        )
        
        qdrant_filter = mapper.map_query_filter(query_filter)
        
        expected = {"key": "complexity_score", "range": {"gt": 5}}
        assert qdrant_filter == expected
    
    def test_build_search_filters_and_logic(self):
        """Test building search filters with AND logic."""
        mapper = QdrantFieldMapper()
        filters = [
            QueryFilter("language", FilterOperator.EQUALS, "Python"),
            QueryFilter("complexity_score", FilterOperator.GREATER_THAN, 5)
        ]
        
        result = mapper.build_search_filters(filters, "AND")
        
        assert "must" in result
        assert len(result["must"]) == 2


class TestMapperFactory:
    """Test mapper factory."""
    
    def test_create_postgres_mapper(self):
        """Test creating PostgreSQL mapper."""
        mapper = MapperFactory.create_mapper("postgres")
        assert isinstance(mapper, PostgresFieldMapper)
    
    def test_create_qdrant_mapper(self):
        """Test creating Qdrant mapper.""" 
        mapper = MapperFactory.create_mapper("qdrant")
        assert isinstance(mapper, QdrantFieldMapper)
    
    def test_create_unknown_mapper(self):
        """Test error for unknown mapper type."""
        with pytest.raises(ValueError, match="Unknown backend type"):
            MapperFactory.create_mapper("unknown")


class TestQueryBuilder:
    """Test query builder functionality."""
    
    def test_basic_query_building(self):
        """Test basic query construction."""
        query = (create_query()
                .text("test search")
                .hybrid_search()
                .limit(5)
                .build())
        
        assert query["text"] == "test search"
        assert query["top_k"] == 5
        assert query["vector_weight"] == 0.7
        assert query["keyword_weight"] == 0.3
    
    def test_filter_chaining(self):
        """Test filter chaining."""
        query = (create_query()
                .keyword_search()
                .where_language("Python")
                .where_complexity_greater_than(5)
                .with_type_hints()
                .filter_logic_and()
                .build())
        
        filters = query["filters"]
        assert len(filters) == 3
        
        # Check language filter
        assert filters[0].field == "language"
        assert filters[0].operator == FilterOperator.EQUALS
        assert filters[0].value == "Python"
        
        # Check complexity filter
        assert filters[1].field == "complexity_score"
        assert filters[1].operator == FilterOperator.GREATER_THAN
        assert filters[1].value == 5
        
        # Check type hints filter
        assert filters[2].field == "has_type_hints"
        assert filters[2].operator == FilterOperator.EQUALS
        assert filters[2].value is True
        
        assert query["filter_logic"] == "AND"
    
    def test_convenience_functions(self):
        """Test convenience query functions."""
        # Test simple search
        query1 = simple_search("test", top_k=20)
        assert query1["text"] == "test"
        assert query1["top_k"] == 20
        
        # Test language-specific search
        query2 = find_functions_in_language("Python", top_k=15)
        assert query2["top_k"] == 15
        filters = query2["filters"]
        assert len(filters) == 2  # language + has_functions


class TestResultMapper:
    """Test result mapping utilities."""
    
    def test_from_postgres_result(self):
        """Test mapping from PostgreSQL result."""
        pg_row = {
            "filename": "test.py",
            "language": "Python",
            "code": "def test(): pass",
            "location": "test.py:1-10",
            "start": 1,
            "end": 10,
            "source_name": "test_source"
        }
        
        result = ResultMapper.from_postgres_result(
            pg_row, 
            score=0.95, 
            score_type=SearchResultType.HYBRID_COMBINED
        )
        
        assert result.filename == "test.py"
        assert result.language == "Python"
        assert result.score == 0.95
        assert result.score_type == SearchResultType.HYBRID_COMBINED
        assert result.source == "test_source"
    
    def test_from_qdrant_result(self):
        """Test mapping from Qdrant result."""
        qdrant_point = {
            "payload": {
                "filename": "test.py",
                "language": "Python", 
                "code": "def test(): pass",
                "location": "test.py:1-10",
                "start": 1,
                "end": 10,
                "source_name": "test_source"
            }
        }
        
        result = ResultMapper.from_qdrant_result(
            qdrant_point,
            score=0.88,
            score_type=SearchResultType.VECTOR_SIMILARITY
        )
        
        assert result.filename == "test.py"
        assert result.language == "Python"
        assert result.score == 0.88
        assert result.score_type == SearchResultType.VECTOR_SIMILARITY


if __name__ == "__main__":
    pytest.main([__file__])