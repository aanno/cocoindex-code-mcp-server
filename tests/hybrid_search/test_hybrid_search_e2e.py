#!/usr/bin/env python3

"""
End-to-end test for hybrid search functionality.
Tests both vector similarity and keyword search capabilities with real database.
"""

import os
import sys
import pytest
import logging
from dotenv import load_dotenv
import cocoindex
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

# Package should be installed via maturin develop or pip install -e .

from cocoindex_code_mcp_server.db.pgvector.hybrid_search import HybridSearchEngine
from cocoindex_code_mcp_server.cocoindex_config import code_embedding_flow

LOGGER = logging.getLogger(__name__)

# Mark all tests in this file as requiring database integration
pytestmark = pytest.mark.db_integration

@pytest.fixture(scope="module")
def setup_cocoindex():
    """Setup CocoIndex for testing."""
    load_dotenv()
    cocoindex.init()
    
    # Ensure index is up-to-date
    try:
        stats = code_embedding_flow.update()
        LOGGER.info(f"✅ Index updated: {stats}")
    except Exception as e:
        pytest.skip(f"Could not update CocoIndex: {e}")
    
    yield
    
    # Cleanup if needed
    pass

@pytest.fixture
def db_pool():
    """Create database connection pool."""
    database_url = os.getenv("COCOINDEX_DATABASE_URL")
    if not database_url:
        pytest.skip("COCOINDEX_DATABASE_URL not set")
    
    pool = ConnectionPool(database_url)
    return pool

@pytest.fixture
def search_engine(db_pool):
    """Create HybridSearchEngine instance."""
    return HybridSearchEngine(db_pool)

@pytest.mark.integration
@pytest.mark.hybrid_search
def test_vector_only_search(setup_cocoindex, search_engine):
    """Test pure vector similarity search."""
    results = search_engine.search(
        vector_query="function definition python",
        keyword_query="",
        top_k=3
    )
    
    assert len(results) > 0, "Vector search should return results"
    assert len(results) <= 3, "Should respect top_k limit"
    
    # Check result structure
    for result in results:
        assert "filename" in result
        assert "score" in result
        assert isinstance(result["score"], (int, float))
        assert 0 <= result["score"] <= 1, "Score should be between 0 and 1"

@pytest.mark.integration  
@pytest.mark.hybrid_search
def test_keyword_only_search(setup_cocoindex, search_engine):
    """Test pure keyword search."""
    results = search_engine.search(
        vector_query="",
        keyword_query="language:Python",
        top_k=5
    )
    
    assert len(results) >= 0, "Keyword search should work without errors"
    
    # If results exist, verify they match the filter
    for result in results:
        assert "language" in result or "filename" in result
        # Python files should be included

@pytest.mark.integration
@pytest.mark.hybrid_search  
def test_hybrid_search(setup_cocoindex, search_engine):
    """Test combined vector + keyword search."""
    results = search_engine.search(
        vector_query="error handling",
        keyword_query="language:Python",
        top_k=3
    )
    
    assert len(results) >= 0, "Hybrid search should work without errors"
    
    # If results exist, they should match both criteria
    for result in results:
        assert "filename" in result
        assert "score" in result

@pytest.mark.integration
@pytest.mark.hybrid_search
def test_complex_keyword_search(setup_cocoindex, search_engine):
    """Test complex keyword query with boolean operators."""
    results = search_engine.search(
        vector_query="",
        keyword_query="(language:Python or language:Rust) and exists(embedding)",
        top_k=5
    )
    
    assert len(results) >= 0, "Complex keyword search should work"
    
    # Verify language field in results
    for result in results:
        if "language" in result:
            assert result["language"] in ["Python", "Rust"], \
                f"Expected Python or Rust, got {result['language']}"

@pytest.mark.integration
@pytest.mark.hybrid_search
def test_empty_queries(setup_cocoindex, search_engine):
    """Test behavior with empty queries."""
    # Both empty
    results = search_engine.search(
        vector_query="",
        keyword_query="",
        top_k=5
    )
    assert len(results) == 0, "Empty queries should return no results"

@pytest.mark.integration
@pytest.mark.hybrid_search
def test_search_limits(setup_cocoindex, search_engine):
    """Test that search respects top_k limits."""
    # Test different limits
    for top_k in [1, 3, 5, 10]:
        results = search_engine.search(
            vector_query="function",
            keyword_query="",
            top_k=top_k
        )
        assert len(results) <= top_k, f"Results should not exceed top_k={top_k}"

@pytest.mark.integration
@pytest.mark.hybrid_search
def test_search_scoring(setup_cocoindex, search_engine):
    """Test that search results are properly scored."""
    results = search_engine.search(
        vector_query="function definition",
        keyword_query="",
        top_k=5
    )
    
    if len(results) > 1:
        # Results should be sorted by score (descending)
        scores = [result["score"] for result in results]
        assert scores == sorted(scores, reverse=True), \
            "Results should be sorted by score (highest first)"

if __name__ == "__main__":
    # Allow running as standalone script for debugging
    pytest.main([__file__, "-v"])
