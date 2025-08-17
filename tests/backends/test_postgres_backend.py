#!/usr/bin/env python3

"""
Tests for PostgreSQL backend implementation.
"""

from unittest.mock import MagicMock, Mock, patch
import numpy as np
import pytest

from cocoindex_code_mcp_server.backends.postgres_backend import PostgresBackend
from cocoindex_code_mcp_server.backends import SearchResult, QueryFilters
from cocoindex_code_mcp_server.schemas import SearchResultType
from cocoindex_code_mcp_server.keyword_search_parser_lark import SearchCondition, SearchGroup
from typing import Tuple


@pytest.fixture
def mock_pool():
    """Create a mock PostgreSQL connection pool."""
    pool = MagicMock()
    
    # Set up connection context manager
    mock_conn = MagicMock()
    pool.connection.return_value.__enter__.return_value = mock_conn
    pool.connection.return_value.__exit__.return_value = None
    
    # Set up cursor context manager
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value.__exit__.return_value = None
    
    return pool, mock_conn, mock_cursor


@pytest.fixture
def postgres_backend(mock_pool):
    """Create a PostgresBackend with mocked dependencies."""
    pool, mock_conn, mock_cursor = mock_pool
    return PostgresBackend(pool=pool, table_name="test_embeddings"), mock_conn, mock_cursor


@pytest.mark.unit
@pytest.mark.backend
class TestPostgresBackend:
    """Test PostgreSQL backend implementation."""

    def test_initialization(self, mock_pool: Tuple[MagicMock, MagicMock, MagicMock]):
        """Test backend initialization."""
        pool, _, _ = mock_pool
        backend = PostgresBackend(pool=pool, table_name="custom_table")
        
        assert backend.pool == pool
        assert backend.table_name == "custom_table"

    @patch('cocoindex_code_mcp_server.backends.postgres_backend.register_vector')
    def test_vector_search(self, mock_register: MagicMock, postgres_backend: Tuple[PostgresBackend, MagicMock, MagicMock]):
        """Test vector similarity search.
        
        NOTE: This test currently exposes a bug in postgres_backend.py where 
        _format_result() tries to parse distance as float but gets string data.
        The issue is in the column parsing logic - available_fields is empty
        because _get_available_columns() returns no columns in the mock.
        This should be fixed in the source code, not the test.
        """
        backend, mock_conn, mock_cursor = postgres_backend
        
        # Mock database results
        mock_cursor.fetchall.return_value = [
            ("test.py", "Python", "def test():", 0.2, {"line": 1}, {"line": 3}, "files")
        ]
        
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
        
        assert "SELECT filename, language, code, embedding <=> %s AS distance" in sql_query
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

    @patch('cocoindex_code_mcp_server.backends.postgres_backend.build_sql_where_clause')
    def test_keyword_search(self, mock_build_where: MagicMock, postgres_backend: Tuple[PostgresBackend, MagicMock, MagicMock]):
        """Test keyword/metadata search.
        
        NOTE: This test currently exposes a bug in postgres_backend.py where
        _build_select_clause() generates malformed SQL like "SELECT , 0.0 as distance"
        when available_fields is empty. The root cause is _get_available_columns()
        returning no columns in the mock environment.
        This should be fixed in the source code, not the test.
        """
        backend, mock_conn, mock_cursor = postgres_backend
        
        # Mock WHERE clause builder
        mock_build_where.return_value = ("language = %s", ["Python"])
        
        # Mock database results
        mock_cursor.fetchall.return_value = [
            ("test.py", "Python", "def test():", 0.0, {"line": 1}, {"line": 3}, "files")
        ]
        
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
        
        assert "SELECT filename, language, code, 0.0 as distance" in sql_query
        assert "FROM test_embeddings" in sql_query
        assert "WHERE language = %s" in sql_query
        assert "ORDER BY filename, start" in sql_query
        assert "LIMIT %s" in sql_query
        
        assert query_params == ["Python", 5]
        
        # Verify results
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.score_type == SearchResultType.KEYWORD_MATCH
        assert result.score == 1.0

    @patch('cocoindex_code_mcp_server.backends.postgres_backend.register_vector')
    @patch('cocoindex_code_mcp_server.backends.postgres_backend.build_sql_where_clause')
    def test_hybrid_search(self, mock_build_where: MagicMock, mock_register: MagicMock, postgres_backend: Tuple[PostgresBackend, MagicMock, MagicMock]):
        """Test hybrid search combining vector and keyword."""
        backend, mock_conn, mock_cursor = postgres_backend
        
        # Mock WHERE clause builder
        mock_build_where.return_value = ("language = %s", ["Python"])
        
        # Mock database results (hybrid search returns extra hybrid_score column)
        mock_cursor.fetchall.return_value = [
            ("test.py", "Python", "def test():", 0.2, {"line": 1}, {"line": 3}, "files", 0.75)
        ]
        
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
        
        # Verify parameters: [query_vector, query_vector] + where_params + [vector_weight, top_k]
        assert len(query_params) == 5
        np.testing.assert_array_equal(query_params[0], query_vector)
        np.testing.assert_array_equal(query_params[1], query_vector)
        assert query_params[2] == "Python"
        assert query_params[3] == 0.6  # vector_weight
        assert query_params[4] == 5    # top_k
        
        # Verify results
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.score_type == SearchResultType.HYBRID_COMBINED
        assert result.score == 0.75

    def test_get_table_info(self, postgres_backend: Tuple[PostgresBackend, MagicMock, MagicMock]):
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
                ("idx_embedding_cosine", "CREATE INDEX idx_embedding_cosine ON test_embeddings USING ivfflat (embedding vector_cosine_ops)"),
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

    def test_close(self, postgres_backend: Tuple[PostgresBackend, MagicMock, MagicMock]):
        """Test backend cleanup."""
        backend, _, _ = postgres_backend
        
        # Add close method to pool mock
        setattr(backend.pool, 'close', Mock())
        
        # Test close
        backend.close()
        
        # Verify pool close was called
        getattr(backend.pool, 'close').assert_called_once()

    def test_close_no_close_method(self, postgres_backend: Tuple[PostgresBackend, MagicMock, MagicMock]):
        """Test backend cleanup when pool has no close method."""
        backend, _, _ = postgres_backend
        
        # Remove close method from pool mock
        if hasattr(backend.pool, 'close'):
            delattr(backend.pool, 'close')
        
        # Test close (should not raise exception)
        backend.close()

    @patch('cocoindex_code_mcp_server.backends.postgres_backend.analyze_python_code')
    def test_format_result_with_python_metadata(self, mock_analyze: MagicMock, postgres_backend: Tuple[PostgresBackend, MagicMock, MagicMock]):
        """Test result formatting with Python metadata extraction."""
        backend, _, _ = postgres_backend
        
        # Mock Python analysis
        mock_analyze.return_value = {
            "functions": ["test_func"],
            "classes": ["TestClass"],
            "imports": ["os", "sys"],
            "complexity_score": 5,
            "has_type_hints": True,
            "has_async": False,
            "has_classes": True,
            "private_methods": ["_helper"],
            "dunder_methods": ["__init__"],
            "decorators": ["@property"],
            "analysis_method": "python_ast"
        }
        
        # Test result formatting
        row = ("test.py", "Python", "def test():", 0.2, {"line": 1}, {"line": 3}, "files")
        result = backend._format_result(row, score_type="vector")
        
        # Verify Python analysis was called
        mock_analyze.assert_called_once_with("def test():", "test.py")
        
        # Verify result structure
        assert isinstance(result, SearchResult)
        assert result.filename == "test.py"
        assert result.language == "Python"
        assert result.metadata is not None
        assert result.metadata["functions"] == ["test_func"]
        assert result.metadata["classes"] == ["TestClass"]
        assert result.metadata["complexity_score"] == 5

    @patch('cocoindex_code_mcp_server.backends.postgres_backend.analyze_python_code')
    def test_format_result_python_analysis_error(self, mock_analyze: MagicMock, postgres_backend: Tuple[PostgresBackend, MagicMock, MagicMock]):
        """Test result formatting when Python analysis fails."""
        backend, _, _ = postgres_backend
        
        # Mock Python analysis failure
        mock_analyze.side_effect = Exception("Analysis failed")
        
        # Test result formatting
        row = ("test.py", "Python", "def test():", 0.2, {"line": 1}, {"line": 3}, "files")
        result = backend._format_result(row, score_type="vector")
        
        # Verify result structure with fallback metadata
        assert isinstance(result, SearchResult)
        assert result.metadata is not None
        assert result.metadata["functions"] == []
        assert result.metadata["metadata_json"].get("analysis_error") == "Analysis failed"

    def test_format_result_non_python(self, postgres_backend: Tuple[PostgresBackend, MagicMock, MagicMock]):
        """Test result formatting for non-Python code."""
        backend, _, _ = postgres_backend
        
        # Test result formatting for non-Python language
        row = ("test.js", "JavaScript", "function test() {}", 0.3, {"line": 1}, {"line": 3}, "files")
        result = backend._format_result(row, score_type="vector")
        
        # Verify result structure with basic metadata for non-Python code
        assert isinstance(result, SearchResult)
        assert result.filename == "test.js"
        assert result.language == "JavaScript"
        assert result.metadata is not None
        assert result.metadata["functions"] == []
        assert result.metadata["metadata_json"].get("analysis_method") == "none"

    def test_build_where_clause(self, postgres_backend: Tuple[PostgresBackend, MagicMock, MagicMock]):
        """Test QueryFilters to SQL WHERE clause conversion."""
        backend, _, _ = postgres_backend
        
        # Create test filters
        filters = QueryFilters(conditions=[SearchCondition(field="language", value="Python")])
        
        # Test WHERE clause building (this tests the mock search group creation)
        with patch('cocoindex_code_mcp_server.backends.postgres_backend.build_sql_where_clause') as mock_build:
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