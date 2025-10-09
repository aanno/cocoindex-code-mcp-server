#!/usr/bin/env python3
"""
Integration tests db_abstraction: QueryExecutor with backend integration.
"""

import asyncio
from typing import List

import pytest

from cocoindex_code_mcp_server.query_abstraction import QueryBuilder, QueryExecutor
from cocoindex_code_mcp_server.schemas import (
    ChunkMetadata,
    FilterOperator,
    SearchResult,
    SearchResultType,
)


class TestQueryExecutorIntegration:
    """Test QueryExecutor integration with real backends."""

    def test_query_executor_with_mock_backend(self, mocker):
        """Test QueryExecutor with a mocked backend."""
        # Create mock backend
        mock_backend = mocker.Mock()

        # Mock SearchResult with proper ChunkMetadata
        mock_metadata: ChunkMetadata = {
            "filename": "test.py",
            "language": "Python",
            "location": "test.py:1-10",
            "code": "def hello(): pass",
            "start": 1,
            "end": 10,
            "source_name": "test",
            "functions": ["hello"],
            "classes": [],
            "imports": [],
            "complexity_score": 1,
            "has_type_hints": False,
            "has_async": False,
            "has_classes": False,
            "metadata_json": {"test": "data"}
        }

        mock_result = SearchResult(
            filename="test.py",
            language="Python",
            code="def hello(): pass",
            location="test.py:1-10",
            start=1,
            end=10,
            score=0.95,
            score_type=SearchResultType.VECTOR_SIMILARITY,
            source="test",
            metadata=mock_metadata
        )

        # Mock backend methods
        mock_backend.vector_search.return_value = [mock_result]
        mock_backend.keyword_search.return_value = [mock_result]
        mock_backend.hybrid_search.return_value = [mock_result]

        # Create QueryExecutor with mock embedding function
        def mock_embedding_func(text: str) -> list[float]:
            # Return a simple mock embedding
            return [0.1] * 384

        executor = QueryExecutor(mock_backend, embedding_func=mock_embedding_func)

        # Test vector search
        query = QueryBuilder().text("test query").vector_search().limit(5).build()
        results = asyncio.run(executor.execute(query))

        assert len(results) == 1
        assert results[0].filename == "test.py"
        assert results[0].metadata is not None
        assert results[0].metadata["functions"] == ["hello"]

        # Verify backend was called correctly
        mock_backend.vector_search.assert_called_once()

    def test_query_builder_fluent_interface(self):
        """Test QueryBuilder creates proper ChunkQuery objects."""
        # Test vector search query
        vector_query = (QueryBuilder()
                        .text("find functions")
                        .vector_search()
                        .limit(10)
                        .build())

        assert vector_query.get("text") == "find functions"
        query_type = vector_query.get("query_type")
        assert query_type is not None and query_type.value == "vector"  # QueryType enum
        assert vector_query.get("top_k") == 10

        # Test hybrid search query
        hybrid_query = (QueryBuilder()
                        .text("async functions")
                        .filter_by("language", FilterOperator.EQUALS, "Python")
                        .hybrid_search(vector_weight=0.7, keyword_weight=0.3)
                        .limit(20)
                        .build())

        assert hybrid_query.get("text") == "async functions"
        query_type = hybrid_query.get("query_type")
        assert query_type is not None and query_type.value == "hybrid"
        assert hybrid_query.get("top_k") == 20
        filters = hybrid_query.get("filters", [])
        assert len(filters) == 1
        assert filters[0].field == "language"
        assert filters[0].value == "Python"

    def test_schema_search_result_metadata_compatibility(self):
        """Test that SchemaSearchResult handles ChunkMetadata properly."""
        from cocoindex_code_mcp_server.query_abstraction import SchemaSearchResult

        # Create ChunkMetadata
        metadata: ChunkMetadata = {
            "filename": "example.py",
            "language": "Python",
            "location": "example.py:5-15",
            "code": "class Example: pass",
            "start": 5,
            "end": 15,
            "source_name": "test_source",
            "functions": [],
            "classes": ["Example"],
            "imports": ["os", "sys"],
            "complexity_score": 2,
            "has_type_hints": True,
            "has_async": False,
            "has_classes": True,
            "metadata_json": {"ast_nodes": ["ClassDef"]}
        }

        # Create SchemaSearchResult
        result = SchemaSearchResult(
            filename="example.py",
            language="Python",
            code="class Example: pass",
            location="example.py:5-15",
            start=5,
            end=15,
            score=0.88,
            score_type=SearchResultType.HYBRID_COMBINED,
            source="test_source",
            metadata=metadata
        )

        # Verify metadata is preserved and accessible
        assert result.metadata is not None
        assert result.metadata.get("classes") == ["Example"]
        assert result.metadata.get("has_classes") == True
        assert result.metadata.get("complexity_score") == 2
        metadata_json = result.metadata.get("metadata_json", {})
        assert metadata_json.get("ast_nodes") == ["ClassDef"]


@pytest.mark.asyncio
async def test_async_query_execution(mocker):
    """Test async QueryExecutor execution."""
    # Mock backend with async-compatible methods
    mock_backend = mocker.Mock()

    mock_result = SearchResult(
        filename="async_test.py",
        language="Python",
        code="async def test(): await asyncio.sleep(1)",
        location="async_test.py:1-1",
        start=1,
        end=1,
        score=0.92,
        score_type=SearchResultType.VECTOR_SIMILARITY,
        source="async_test",
        metadata={
            "filename": "async_test.py",
            "language": "Python",
            "has_async": True,
            "functions": ["test"]
        }
    )

    mock_backend.vector_search.return_value = [mock_result]

    # Mock embedding function for async test
    def mock_embedding_func(text: str) -> list[float]:
        return [0.2] * 384

    executor = QueryExecutor(mock_backend, embedding_func=mock_embedding_func)
    query = QueryBuilder().text("async function").vector_search().build()

    results: List[SearchResult] = await executor.execute(query)

    assert len(results) == 1
    metadata = results[0].metadata
    if metadata is not None:
        assert metadata.get("has_async") == True
        functions = metadata.get("functions", [])
        assert "test" in functions


if __name__ == "__main__":
    pytest.main([__file__])
