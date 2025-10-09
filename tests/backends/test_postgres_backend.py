#!/usr/bin/env python3

"""
Tests for PostgreSQL backend implementation.
"""

from typing import Any, Dict, List

import numpy as np
import pytest

from cocoindex_code_mcp_server.backends import QueryFilters, SearchResult
from cocoindex_code_mcp_server.backends.postgres_backend import PostgresBackend
from cocoindex_code_mcp_server.keyword_search_parser_lark import (
    SearchCondition,
)
from cocoindex_code_mcp_server.schemas import SearchResultType


def build_mock_row(data: Dict[str, Any], fields: List[str]) -> tuple:
    """
    Build a database result tuple in the correct column order.

    Args:
        data: Dictionary mapping field names to values
        fields: List of field names in the expected order

    Returns:
        Tuple of values in the same order as fields
    """
    return tuple(data.get(field) for field in fields)


@pytest.fixture
def mock_pool(mocker):
    """Create a mock PostgreSQL connection pool."""
    pool = mocker.MagicMock()

    # Set up connection context manager
    mock_conn = mocker.MagicMock()
    pool.connection.return_value.__enter__.return_value = mock_conn
    pool.connection.return_value.__exit__.return_value = None

    # Set up cursor context manager
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value.__exit__.return_value = None

    return pool, mock_conn, mock_cursor


@pytest.fixture
def postgres_backend(mock_pool, mocker):
    """Create a PostgresBackend with mocked dependencies."""
    pool, mock_conn, mock_cursor = mock_pool

    # Mock _get_table_columns to return expected columns
    # These are the columns that tests expect to be available
    mock_columns = {
        'filename', 'language', 'code', 'embedding', 'start', 'end',
        'source_name', 'location', 'functions', 'classes', 'imports',
        'complexity_score', 'has_type_hints', 'has_async', 'has_classes',
        'metadata_json', 'analysis_method', 'chunking_method',
        'tree_sitter_analyze_error', 'tree_sitter_chunking_error'
    }
    mocker.patch(
        'cocoindex_code_mcp_server.backends.postgres_backend._get_table_columns',
        return_value=mock_columns
    )

    return PostgresBackend(pool=pool, table_name="test_embeddings"), mock_conn, mock_cursor


@pytest.mark.unit
@pytest.mark.backend
class TestPostgresBackend:
    """Test PostgreSQL backend implementation."""

    def test_initialization(self, mocker):
        """Test backend initialization."""
        pool = mocker.MagicMock()
        backend = PostgresBackend(pool=pool, table_name="custom_table")

        assert backend.pool == pool
        assert backend.table_name == "custom_table"

    def test_vector_search(self, mocker, postgres_backend):
        """Test vector similarity search."""
        mock_register = mocker.patch('cocoindex_code_mcp_server.backends.postgres_backend.register_vector')

        backend, mock_conn, mock_cursor = postgres_backend

        # Build mock data dictionary
        mock_data = {
            'filename': 'test.py',
            'language': 'Python',
            'code': 'def test():',
            'location': 'line:0#0',
            'start': {'line': 1},
            'end': {'line': 3},
            'source_name': 'files',
            'functions': [],
            'classes': [],
            'imports': [],
            'complexity_score': 0,
            'has_type_hints': False,
            'has_async': False,
            'has_classes': False,
            'metadata_json': {},
            'analysis_method': 'tree_sitter',
            'chunking_method': 'ast_chunking',
            'tree_sitter_analyze_error': 'false',
            'tree_sitter_chunking_error': 'false',
        }

        # Mock fetchall to dynamically build row based on actual available_fields
        def mock_fetchall_fn():
            # Get the actual field order from the backend's last _build_select_clause call
            # This is determined by the SELECT clause that was built
            _, actual_fields = backend._build_select_clause(include_distance=True, distance_alias="distance")
            mock_row = build_mock_row(mock_data, actual_fields) + (0.2,)  # distance
            return [mock_row]

        mock_cursor.fetchall.side_effect = mock_fetchall_fn

        # Test vector search
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        results = backend.vector_search(query_vector, top_k=5)

        # Verify register_vector was called
        mock_register.assert_called_once_with(mock_conn)

        # Verify SQL query
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args
        sql_query = call_args[0][0]
        query_params = call_args[0][1]

        # Check that query includes expected components
        assert "SELECT" in sql_query
        assert "filename" in sql_query
        assert "language" in sql_query
        assert "code" in sql_query
        assert "embedding <=> %s AS distance" in sql_query
        assert "FROM test_embeddings" in sql_query
        assert "ORDER BY distance" in sql_query
        assert "LIMIT %s" in sql_query

        np.testing.assert_array_equal(query_params[0], query_vector)
        assert query_params[1] == 5

        # Verify results
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.filename == "test.py"
        assert result.language == "Python"
        assert result.code == "def test():"
        assert result.score == 0.8  # 1.0 - 0.2
        assert result.score_type == SearchResultType.VECTOR_SIMILARITY

    def test_keyword_search(self, mocker, postgres_backend):
        """Test keyword/metadata search."""
        mock_build_where = mocker.patch('cocoindex_code_mcp_server.backends.postgres_backend.build_sql_where_clause')

        backend, mock_conn, mock_cursor = postgres_backend

        # Mock WHERE clause builder
        mock_build_where.return_value = ("language = %s", ["Python"])

        # Build mock data
        mock_data = {
            'filename': 'test.py',
            'language': 'Python',
            'code': 'def test():',
            'location': 'line:0#0',
            'start': {'line': 1},
            'end': {'line': 3},
            'source_name': 'files',
            'functions': [],
            'classes': [],
            'imports': [],
            'complexity_score': 0,
            'has_type_hints': False,
            'has_async': False,
            'has_classes': False,
            'metadata_json': {},
            'analysis_method': 'tree_sitter',
            'chunking_method': 'ast_chunking',
            'tree_sitter_analyze_error': 'false',
            'tree_sitter_chunking_error': 'false',
        }

        # Mock fetchall to dynamically build row based on actual available_fields
        def mock_fetchall_fn():
            _, actual_fields = backend._build_select_clause()
            mock_row = build_mock_row(mock_data, actual_fields) + (0.0,)  # distance
            return [mock_row]

        mock_cursor.fetchall.side_effect = mock_fetchall_fn

        # Test keyword search
        filters = QueryFilters(conditions=[SearchCondition(field="language", value="Python")])
        results = backend.keyword_search(filters, top_k=5)

        # Verify WHERE clause builder was called
        mock_build_where.assert_called_once()

        # Verify SQL query
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args
        sql_query = call_args[0][0]
        query_params = call_args[0][1]

        # Check query structure (can't check exact SELECT due to dynamic columns)
        assert "SELECT" in sql_query
        assert "0.0 as distance" in sql_query
        assert "FROM test_embeddings" in sql_query
        assert "WHERE language = %s" in sql_query
        assert "ORDER BY filename, start" in sql_query
        assert "LIMIT %s" in sql_query

        assert query_params == ["Python", 5]

        # Verify results
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.filename == "test.py"
        assert result.language == "Python"
        assert result.score_type == SearchResultType.KEYWORD_MATCH
        assert result.score == 1.0

    def test_hybrid_search(self, mocker, postgres_backend):
        """Test hybrid search combining vector and keyword."""
        mock_build_where = mocker.patch('cocoindex_code_mcp_server.backends.postgres_backend.build_sql_where_clause')
        mock_register = mocker.patch('cocoindex_code_mcp_server.backends.postgres_backend.register_vector')

        backend, mock_conn, mock_cursor = postgres_backend

        # Mock WHERE clause builder
        mock_build_where.return_value = ("language = %s", ["Python"])

        # Build mock data
        mock_data = {
            'filename': 'test.py',
            'language': 'Python',
            'code': 'def test():',
            'location': 'line:0#0',
            'start': {'line': 1},
            'end': {'line': 3},
            'source_name': 'files',
            'functions': [],
            'classes': [],
            'imports': [],
            'complexity_score': 0,
            'has_type_hints': False,
            'has_async': False,
            'has_classes': False,
            'metadata_json': {},
            'analysis_method': 'tree_sitter',
            'chunking_method': 'ast_chunking',
            'tree_sitter_analyze_error': 'false',
            'tree_sitter_chunking_error': 'false',
        }

        # Mock fetchall to dynamically build row based on actual available_fields
        def mock_fetchall_fn():
            _, actual_fields = backend._build_select_clause()
            # Hybrid search returns: fields + vector_distance + vector_similarity + hybrid_score + hybrid_score (at end)
            # The CTE adds vector_distance, vector_similarity, and hybrid_score
            mock_row = build_mock_row(mock_data, actual_fields) + (0.2, 0.8, 0.75, 0.75)
            return [mock_row]

        mock_cursor.fetchall.side_effect = mock_fetchall_fn

        # Test hybrid search
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        filters = QueryFilters(conditions=[SearchCondition(field="language", value="Python")])
        results = backend.hybrid_search(
            query_vector=query_vector,
            filters=filters,
            top_k=5,
            vector_weight=0.6,
            keyword_weight=0.4
        )

        # Verify register_vector was called
        mock_register.assert_called_once_with(mock_conn)

        # Verify WHERE clause builder was called
        mock_build_where.assert_called_once()

        # Verify SQL query
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args
        sql_query = call_args[0][0]
        query_params = call_args[0][1]

        assert "WITH vector_scores AS" in sql_query
        assert "embedding <=> %s" in sql_query
        assert "WHERE language = %s" in sql_query
        assert "ORDER BY hybrid_score DESC" in sql_query
        assert "LIMIT %s" in sql_query

        # Verify parameters: [query_vector, query_vector] + where_params + [vector_weight, keyword_weight, top_k]
        assert len(query_params) == 6
        np.testing.assert_array_equal(query_params[0], query_vector)
        np.testing.assert_array_equal(query_params[1], query_vector)
        assert query_params[2] == "Python"
        assert query_params[3] == 0.6  # vector_weight
        assert query_params[4] == 0.4  # keyword_weight
        assert query_params[5] == 5    # top_k

        # Verify results
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.filename == "test.py"
        assert result.language == "Python"
        assert result.score_type == SearchResultType.HYBRID_COMBINED
        assert result.score == 0.75

    def test_get_table_info(self, postgres_backend):
        """Test table information retrieval."""
        backend, mock_conn, mock_cursor = postgres_backend

        # Mock database schema results
        mock_cursor.fetchall.side_effect = [
            # Columns query
            [
                ("filename", "character varying", "NO"),
                ("language", "character varying", "NO"),
                ("embedding", "vector", "YES")
            ],
            # Indexes query
            [
                ("idx_embedding_cosine",
                 "CREATE INDEX idx_embedding_cosine ON test_embeddings USING ivfflat (embedding vector_cosine_ops)"),
                ("idx_language", "CREATE INDEX idx_language ON test_embeddings (language)")
            ]
        ]

        # Mock row count
        mock_cursor.fetchone.return_value = [1234]

        # Test table info
        info = backend.get_table_info()

        # Verify multiple queries were executed
        assert mock_cursor.execute.call_count == 3

        # Verify returned info structure
        assert info["backend_type"] == "postgres"
        assert info["table_name"] == "test_embeddings"
        assert info["row_count"] == 1234

        assert len(info["columns"]) == 3
        assert info["columns"][0]["name"] == "filename"
        assert info["columns"][0]["type"] == "character varying"
        assert info["columns"][0]["nullable"] is False

        assert len(info["indexes"]) == 2
        assert info["indexes"][0]["name"] == "idx_embedding_cosine"

    def test_close(self, mocker, postgres_backend):
        """Test backend cleanup."""
        backend, _, _ = postgres_backend

        # Add close method to pool mock
        setattr(backend.pool, 'close', mocker.Mock())

        # Test close
        backend.close()

        # Verify pool close was called
        getattr(backend.pool, 'close').assert_called_once()

    def test_close_no_close_method(self, postgres_backend):
        """Test backend cleanup when pool has no close method."""
        backend, _, _ = postgres_backend

        # Remove close method from pool mock
        if hasattr(backend.pool, 'close'):
            delattr(backend.pool, 'close')

        # Test close (should not raise exception)
        backend.close()

    def test_format_result_with_python_metadata(self, postgres_backend):
        """Test result formatting with Python metadata fields from database."""
        backend, _, _ = postgres_backend

        # Test result formatting with metadata fields from database
        # These fields would have been populated during indexing
        available_fields = ["filename", "language", "code", "start", "end", "source_name",
                            "functions", "classes", "imports", "complexity_score", "has_type_hints"]
        mock_data = {
            'filename': 'test.py',
            'language': 'Python',
            'code': 'def test():',
            'start': {"line": 1},
            'end': {"line": 3},
            'source_name': 'files',
            'functions': ["test_func"],
            'classes': ["TestClass"],
            'imports': ["os", "sys"],
            'complexity_score': 5,
            'has_type_hints': True
        }
        row = build_mock_row(mock_data, available_fields) + (0.2,)  # distance

        result = backend._format_result(row, available_fields=available_fields, score_type="vector")

        # Verify result structure
        assert isinstance(result, SearchResult)
        assert result.filename == "test.py"
        assert result.language == "Python"
        assert result.metadata is not None
        assert result.metadata["functions"] == ["test_func"]
        assert result.metadata["classes"] == ["TestClass"]
        assert result.metadata["complexity_score"] == 5

    def test_format_result_minimal_fields(self, postgres_backend):
        """Test result formatting with minimal database fields (no metadata)."""
        backend, _, _ = postgres_backend

        # Test result formatting with only core fields, no metadata fields
        available_fields = ["filename", "language", "code", "start", "end", "source_name"]
        mock_data = {
            'filename': 'test.py',
            'language': 'Python',
            'code': 'def test():',
            'start': {"line": 1},
            'end': {"line": 3},
            'source_name': 'files'
        }
        row = build_mock_row(mock_data, available_fields) + (0.2,)  # distance

        result = backend._format_result(row, available_fields=available_fields, score_type="vector")

        # Verify result structure - should still work with minimal fields
        assert isinstance(result, SearchResult)
        assert result.filename == "test.py"
        assert result.language == "Python"
        assert result.code == "def test():"
        assert result.metadata is not None
        # Core fields should be in metadata
        assert result.metadata["filename"] == "test.py"
        assert result.metadata["language"] == "Python"

    def test_format_result_non_python(self, postgres_backend):
        """Test result formatting for non-Python code."""
        backend, _, _ = postgres_backend

        # Test result formatting for non-Python language
        available_fields = ["filename", "language", "code", "start", "end", "source_name"]
        mock_data = {
            'filename': 'test.js',
            'language': 'JavaScript',
            'code': 'function test() {}',
            'start': {"line": 1},
            'end': {"line": 3},
            'source_name': 'files'
        }
        row = build_mock_row(mock_data, available_fields) + (0.3,)  # distance

        result = backend._format_result(row, available_fields=available_fields, score_type="vector")

        # Verify result structure
        assert isinstance(result, SearchResult)
        assert result.filename == "test.js"
        assert result.language == "JavaScript"
        assert result.code == "function test() {}"
        assert result.metadata is not None
        assert result.metadata["filename"] == "test.js"
        assert result.metadata["language"] == "JavaScript"

    def test_build_where_clause(self, mocker, postgres_backend):
        """Test QueryFilters to SQL WHERE clause conversion."""
        backend, _, _ = postgres_backend

        # Create test filters
        filters = QueryFilters(conditions=[SearchCondition(field="language", value="Python")])

        # Test WHERE clause building (this tests the mock search group creation)
        mock_build = mocker.patch('cocoindex_code_mcp_server.backends.postgres_backend.build_sql_where_clause')
        mock_build.return_value = ("language = %s", ["Python"])

        where_clause, params = backend._build_where_clause(filters)

        # Verify mock was called with properly structured search group
        mock_build.assert_called_once()
        search_group = mock_build.call_args[0][0]
        assert hasattr(search_group, 'conditions')
        assert len(search_group.conditions) == 1
        condition = search_group.conditions[0]
        assert condition.field == "language"
        assert condition.value == "Python"

        assert where_clause == "language = %s"
        assert params == ["Python"]
