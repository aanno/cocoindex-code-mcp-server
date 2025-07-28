#!/usr/bin/env python3

"""
Tests for the hybrid search engine with backend abstraction.
"""

from unittest.mock import MagicMock, Mock, patch
import numpy as np

import pytest

from cocoindex_code_mcp_server.db.pgvector.hybrid_search import (
    HybridSearchEngine,
    format_results_as_json,
    format_results_readable
)
from cocoindex_code_mcp_server.backends import QueryFilters, VectorStoreBackend, SearchResult
from cocoindex_code_mcp_server.keyword_search_parser import SearchCondition, SearchGroup
from numpy import ndarray
from typing import Any, Dict, List


class MockVectorStoreBackend(VectorStoreBackend):
    """Mock backend for testing."""
    
    def __init__(self):
        self.vector_search_calls = []
        self.keyword_search_calls = []
        self.hybrid_search_calls = []
    
    def vector_search(self, query_vector: ndarray, top_k: int=10) -> List[SearchResult]:
        self.vector_search_calls.append((query_vector, top_k))
        return [
            SearchResult(
                filename="test.py",
                language="Python", 
                code="def test():",
                score=0.8,
                start={"line": 1},
                end={"line": 3},
                source="files",
                score_type="vector"
            )
        ]
    
    def keyword_search(self, filters: QueryFilters, top_k: int=10) -> List[SearchResult]:
        self.keyword_search_calls.append((filters, top_k))
        return [
            SearchResult(
                filename="test.py",
                language="Python",
                code="def test():",
                score=1.0,
                start={"line": 1},
                end={"line": 3},
                source="files", 
                score_type="keyword"
            )
        ]
    
    def hybrid_search(self, query_vector: ndarray, filters: QueryFilters, top_k: int=10, vector_weight: float=0.7, keyword_weight: float=0.3) -> List[SearchResult]:
        self.hybrid_search_calls.append((query_vector, filters, top_k, vector_weight, keyword_weight))
        return [
            SearchResult(
                filename="test.py",
                language="Python",
                code="def test():",
                score=0.75,
                start={"line": 1},
                end={"line": 3},
                source="files",
                score_type="hybrid"
            )
        ]
    
    def configure(self, **options):
        pass
    
    def get_table_info(self):
        return {"backend_type": "mock", "table_name": "test_table"}
    
    def close(self):
        pass


@pytest.fixture
def mock_backend():
    """Create a mock vector store backend."""
    return MockVectorStoreBackend()


@pytest.fixture 
def mock_pool():
    """Create a mock database connection pool."""
    pool = MagicMock()
    pool.connection.return_value = MagicMock()
    return pool


@pytest.fixture
def mock_parser():
    """Create a mock keyword search parser."""
    return Mock()


@pytest.fixture
def mock_embedding_func():
    """Create a mock embedding function."""
    mock_func = Mock()
    mock_func.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    return mock_func


@pytest.fixture
def hybrid_engine_with_backend(mock_backend, mock_parser, mock_embedding_func):
    """Create a HybridSearchEngine with mock backend."""
    return HybridSearchEngine(
        backend=mock_backend,
        parser=mock_parser,
        embedding_func=mock_embedding_func
    )


@pytest.fixture
def hybrid_engine_legacy(mock_pool, mock_parser, mock_embedding_func):
    """Create a HybridSearchEngine with legacy pool constructor."""
    with patch('cocoindex_code_mcp_server.backends.BackendFactory.create_backend') as mock_factory:
        mock_factory.return_value = MockVectorStoreBackend()
        return HybridSearchEngine(
            pool=mock_pool,
            table_name="test_table",
            parser=mock_parser,
            embedding_func=mock_embedding_func
        )


@pytest.mark.db_integration
@pytest.mark.search_engine
class TestHybridSearchEngine:
    """Test HybridSearchEngine class with backend abstraction."""

    def test_initialization_with_backend(self, mock_backend: MockVectorStoreBackend, mock_parser: Mock, mock_embedding_func: Mock):
        """Test engine initialization with backend."""
        engine = HybridSearchEngine(
            backend=mock_backend,
            parser=mock_parser,
            embedding_func=mock_embedding_func
        )
        
        assert engine.backend == mock_backend
        assert engine.parser == mock_parser
        assert engine.embedding_func == mock_embedding_func

    @patch('cocoindex_code_mcp_server.backends.BackendFactory.create_backend')
    def test_initialization_legacy_pool(self, mock_factory: MagicMock, mock_pool: MagicMock, mock_parser: Mock, mock_embedding_func: Mock):
        """Test engine initialization with legacy pool parameter."""
        mock_backend = MockVectorStoreBackend()
        mock_factory.return_value = mock_backend
        
        engine = HybridSearchEngine(
            pool=mock_pool,
            table_name="test_table",
            parser=mock_parser,
            embedding_func=mock_embedding_func
        )
        
        # Verify backend factory was called with correct parameters
        mock_factory.assert_called_once_with("postgres", pool=mock_pool, table_name="test_table")
        assert engine.backend == mock_backend

    def test_initialization_error_no_backend_or_pool(self, mock_parser: Mock, mock_embedding_func: Mock):
        """Test engine initialization error when neither backend nor pool provided."""
        with pytest.raises(ValueError, match="Either 'backend' or 'pool' parameter must be provided"):
            HybridSearchEngine(
                parser=mock_parser,
                embedding_func=mock_embedding_func
            )

    @patch('cocoindex_code_mcp_server.db.pgvector.hybrid_search.KeywordSearchParser')
    @patch('cocoindex_code_mcp_server.db.pgvector.hybrid_search.cocoindex.utils.get_target_default_name')
    @patch('cocoindex_code_mcp_server.db.pgvector.hybrid_search.code_to_embedding')
    @patch('cocoindex_code_mcp_server.backends.BackendFactory.create_backend')
    def test_initialization_with_defaults(self, mock_factory: MagicMock, mock_embedding: MagicMock, mock_get_name: MagicMock, mock_parser_class: MagicMock, mock_pool: MagicMock):
        """Test engine initialization with default values."""
        mock_backend = MockVectorStoreBackend()
        mock_factory.return_value = mock_backend
        mock_parser_instance = Mock()
        mock_parser_class.return_value = mock_parser_instance
        mock_get_name.return_value = "default_table"

        engine = HybridSearchEngine(pool=mock_pool)

        mock_factory.assert_called_once_with("postgres", pool=mock_pool, table_name="default_table")
        assert engine.backend == mock_backend
        assert engine.parser == mock_parser_instance

    def test_search_vector_only(self, hybrid_engine_with_backend: HybridSearchEngine, mock_backend: MockVectorStoreBackend, mock_parser: Mock, mock_embedding_func: Mock):
        """Test search with vector query only."""
        # Mock parser to return empty search group
        empty_group = SearchGroup(conditions=[])
        mock_parser.parse.return_value = empty_group

        # Execute search
        results = hybrid_engine_with_backend.search("test query", "", top_k=5)

        # Verify embedding function was called
        mock_embedding_func.assert_called_once_with("test query")

        # Verify parser was called with empty keyword query
        mock_parser.parse.assert_called_once_with("")

        # Verify backend vector_search was called
        assert len(mock_backend.vector_search_calls) == 1
        query_vector, top_k = mock_backend.vector_search_calls[0]
        np.testing.assert_array_equal(query_vector, np.array([0.1, 0.2, 0.3], dtype=np.float32))
        assert top_k == 5

        # Verify results format
        assert len(results) == 1
        result = results[0]
        assert result["filename"] == "test.py"
        assert result["language"] == "Python"
        assert result["code"] == "def test():"
        assert result["score"] == 0.8
        assert result["score_type"] == "vector"

    def test_search_keyword_only(self, hybrid_engine_with_backend: HybridSearchEngine, mock_backend: MockVectorStoreBackend, mock_parser: Mock, mock_embedding_func: Mock):
        """Test search with keyword query only."""
        # Mock parser to return a search group with conditions
        condition = SearchCondition(field="language", value="python")
        search_group = SearchGroup(conditions=[condition])
        mock_parser.parse.return_value = search_group

        # Execute search
        results = hybrid_engine_with_backend.search("", "language:python", top_k=5)

        # Verify embedding function was not called
        mock_embedding_func.assert_not_called()

        # Verify parser was called with keyword query
        mock_parser.parse.assert_called_once_with("language:python")

        # Verify backend keyword_search was called
        assert len(mock_backend.keyword_search_calls) == 1
        filters, top_k = mock_backend.keyword_search_calls[0]
        assert len(filters.conditions) == 1
        assert top_k == 5

        # Verify results format
        assert len(results) == 1
        result = results[0]
        assert result["score_type"] == "keyword"

    def test_search_hybrid(self, hybrid_engine_with_backend: HybridSearchEngine, mock_backend: MockVectorStoreBackend, mock_parser: Mock, mock_embedding_func: Mock):
        """Test hybrid search with both vector and keyword queries."""
        # Mock parser to return a search group with conditions
        condition = SearchCondition(field="language", value="python")
        search_group = SearchGroup(conditions=[condition])
        mock_parser.parse.return_value = search_group

        # Execute search
        results = hybrid_engine_with_backend.search("test query", "language:python", top_k=5, vector_weight=0.6, keyword_weight=0.4)

        # Verify embedding function was called
        mock_embedding_func.assert_called_once_with("test query")

        # Verify parser was called with keyword query
        mock_parser.parse.assert_called_once_with("language:python")

        # Verify backend hybrid_search was called
        assert len(mock_backend.hybrid_search_calls) == 1
        query_vector, filters, top_k, vector_weight, keyword_weight = mock_backend.hybrid_search_calls[0]
        np.testing.assert_array_equal(query_vector, np.array([0.1, 0.2, 0.3], dtype=np.float32))
        assert len(filters.conditions) == 1
        assert top_k == 5
        assert vector_weight == 0.6
        assert keyword_weight == 0.4

        # Verify results format
        assert len(results) == 1
        result = results[0]
        assert result["score_type"] == "hybrid"
        assert result["score"] == 0.75

    def test_search_empty_queries(self, hybrid_engine_with_backend: HybridSearchEngine, mock_parser: Mock):
        """Test search with empty queries."""
        # Mock parser to return empty search group for empty string
        empty_group = SearchGroup(conditions=[])
        mock_parser.parse.return_value = empty_group

        # Execute search with both empty queries
        results = hybrid_engine_with_backend.search("", "", top_k=5)

        # Should return empty results
        assert results == []

    def test_search_result_to_dict_conversion(self, hybrid_engine_with_backend: HybridSearchEngine):
        """Test SearchResult to dict conversion."""
        search_result = SearchResult(
            filename="test.py",
            language="Python",
            code="def test():",
            score=0.85,
            start={"line": 1},
            end={"line": 3},
            source="files",
            score_type="hybrid",
            metadata={"functions": ["test"], "complexity": 5}
        )
        
        result_dict = hybrid_engine_with_backend._search_result_to_dict(search_result)
        
        assert result_dict["filename"] == "test.py"
        assert result_dict["language"] == "Python"
        assert result_dict["code"] == "def test():"
        assert result_dict["score"] == 0.85
        assert result_dict["start"] == {"line": 1}
        assert result_dict["end"] == {"line": 3}
        assert result_dict["source"] == "files"
        assert result_dict["score_type"] == "hybrid"
        assert result_dict["functions"] == ["test"]
        assert result_dict["complexity"] == 5


@pytest.mark.unit
@pytest.mark.search_engine
class TestResultFormatting:
    """Test result formatting functions."""

    def test_format_results_readable_empty(self):
        """Test readable formatting with empty results."""
        results: List[Dict[str, Any]] = []
        output: str = format_results_readable(results)
        assert output == "No results found."

    def test_format_results_readable_single(self):
        """Test readable formatting with single result."""
        results = [
            {
                "filename": "test.py",
                "language": "Python",
                "code": "def test():",
                "score": 0.85,
                "start": {"line": 1},
                "end": {"line": 3},
                "source": "custom_source",
                "score_type": "hybrid"
            }
        ]

        output = format_results_readable(results)

        # Check that it contains expected elements
        assert "📊 Found 1 results:" in output
        assert "[0.850] (hybrid) test.py [custom_source] (Python) (L1-L3)" in output
        assert "def test():" in output
        assert "---" in output

    def test_format_results_readable_multiple(self):
        """Test readable formatting with multiple results."""
        results = [
            {
                "filename": "test1.py",
                "language": "Python",
                "code": "def test1():",
                "score": 0.90,
                "start": {"line": 1},
                "end": {"line": 3},
                "source": "files",
                "score_type": "vector"
            },
            {
                "filename": "test2.py",
                "language": "Python",
                "code": "def test2():",
                "score": 0.80,
                "start": {"line": 5},
                "end": {"line": 7},
                "source": "custom",
                "score_type": "keyword"
            }
        ]

        output = format_results_readable(results)

        # Check that it contains expected elements
        assert "📊 Found 2 results:" in output
        assert "1. [0.900] (vector) test1.py" in output
        assert "2. [0.800] (keyword) test2.py [custom]" in output

    def test_format_results_readable_no_source_info(self):
        """Test readable formatting when source is 'files' or missing."""
        results = [
            {
                "filename": "test.py",
                "language": "Python",
                "code": "def test():",
                "score": 0.85,
                "start": {"line": 1},
                "end": {"line": 3},
                "source": "files",  # Should not show source info
                "score_type": "vector"
            }
        ]

        output = format_results_readable(results)

        # Should not contain source info for "files"
        assert "[files]" not in output
        assert "test.py (Python)" in output

    def test_format_results_readable_with_python_metadata(self):
        """Test readable formatting with Python metadata."""
        results = [
            {
                "filename": "test.py",
                "language": "Python",
                "code": "def test():",
                "score": 0.85,
                "start": {"line": 1},
                "end": {"line": 3},
                "source": "files",
                "score_type": "vector",
                "functions": ["test", "helper"],
                "classes": ["TestClass"],
                "imports": ["os", "sys", "json"],
                "decorators": ["@pytest.fixture"],
                "has_type_hints": True,
                "has_async": False,
                "complexity_score": 15
            }
        ]

        output = format_results_readable(results)

        # Check Python metadata is included
        assert "Functions: test, helper" in output
        assert "Classes: TestClass" in output
        assert "Imports: os, sys, json" in output  # First 3 imports
        assert "Decorators: @pytest.fixture" in output
        assert "typed" in output
        assert "complex(15)" in output


@pytest.mark.unit
@pytest.mark.backend
class TestBackendIntegration:
    """Test backend integration scenarios."""

    @patch('cocoindex_code_mcp_server.backends.BackendFactory.create_backend')
    def test_legacy_constructor_creates_postgres_backend(self, mock_factory: MagicMock, mock_pool: MagicMock):
        """Test that legacy constructor creates PostgreSQL backend."""
        mock_backend = MockVectorStoreBackend()
        mock_factory.return_value = mock_backend
        
        engine = HybridSearchEngine(pool=mock_pool, table_name="custom_table")
        
        mock_factory.assert_called_once_with("postgres", pool=mock_pool, table_name="custom_table")
        assert engine.backend == mock_backend

    def test_backend_constructor_uses_provided_backend(self, mock_backend: MockVectorStoreBackend):
        """Test that backend constructor uses provided backend directly."""
        engine = HybridSearchEngine(backend=mock_backend)
        
        assert engine.backend == mock_backend

    def test_backend_method_delegation(self, hybrid_engine_with_backend: HybridSearchEngine, mock_backend: MockVectorStoreBackend, mock_parser: Mock):
        """Test that search methods properly delegate to backend."""
        # Set up parser mock
        empty_group = SearchGroup(conditions=[])
        mock_parser.parse.return_value = empty_group
        
        # Test vector search
        hybrid_engine_with_backend.search("query", "", top_k=3)
        assert len(mock_backend.vector_search_calls) == 1
        
        # Test keyword search  
        condition_group = SearchGroup(conditions=[SearchCondition(field="lang", value="py")])
        mock_parser.parse.return_value = condition_group
        hybrid_engine_with_backend.search("", "lang:py", top_k=3)
        assert len(mock_backend.keyword_search_calls) == 1
        
        # Test hybrid search
        hybrid_engine_with_backend.search("query", "lang:py", top_k=3)
        assert len(mock_backend.hybrid_search_calls) == 1
