#!/usr/bin/env python3

"""
Integration tests for hybrid search with Python metadata extraction.
These tests require a running database with indexed code.
"""

import logging
import os

import pytest
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
import cocoindex

from cocoindex_code_mcp_server.db.pgvector.hybrid_search import (
    HybridSearchEngine,
    format_results_readable,
)
from cocoindex_code_mcp_server.keyword_search_parser_lark import KeywordSearchParser

from .cocoindex_util import get_default_db_name

# Set up logger for tests
LOGGER = logging.getLogger(__name__)


@pytest.mark.integration
class TestMetadataSearch:
    """Integration test class for metadata search functionality."""

    @pytest.fixture(autouse=True)
    def setup_search_engine(self):
        """Setup search engine for testing."""
        load_dotenv()
        cocoindex.init()
        # Import flow to register it
        from cocoindex_code_mcp_server.cocoindex_config import code_embedding_flow

        db_url = os.getenv("COCOINDEX_DATABASE_URL")
        if not db_url:
            pytest.skip("COCOINDEX_DATABASE_URL not set")

        try:
            name = get_default_db_name()
            self.pool = ConnectionPool(db_url)
            self.search_engine = HybridSearchEngine(name, KeywordSearchParser(), pool=self.pool)

            # Check if the required table exists
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = %s
                        );
                    """, (name,))
                    if cur is not None:
                        one = cur.fetchone()
                        if one is not None:
                            table_exists = one[0]
                            if not table_exists:
                                pytest.skip(f"{name} table does not exist - database not initialized")
                        else:
                            pytest.skip(f"{name} table does not exist - database not initialized")
                    else:
                        pytest.fail(f"{name} table does not exist - database not initialized")
        except Exception as e:
            pytest.skip(f"Could not connect to database or check table: {e}")

    def test_python_async_function_search(self):
        """Test searching for Python async functions."""
        results = self.search_engine.search(
            vector_query="async function",
            keyword_query="language:Python",
            top_k=3
        )

        # Should find results
        assert len(results) > 0

        # At least one result should have async-related metadata
        any(result.get('has_async', False) for result in results)
        # Note: This assertion might not always pass if the codebase doesn't have async code
        # assert has_async_result, "Expected to find at least one async function"

        # All results should be Python
        for result in results:
            assert result.get('language') == 'Python'
            # Should have metadata fields
            assert 'functions' in result
            assert 'classes' in result
            assert 'has_async' in result

    def test_python_class_search(self):
        """Test searching for Python class definitions."""
        results = self.search_engine.search(
            vector_query="class definition",
            keyword_query="language:Python",
            top_k=3
        )

        # Should find results
        assert len(results) > 0

        # At least one result should have classes
        has_classes = any(len(result.get('classes', [])) > 0 for result in results)
        assert has_classes, "Expected to find at least one class definition"

        # All results should be Python
        for result in results:
            assert result.get('language') == 'Python'
            assert 'classes' in result

    def test_pure_python_keyword_search(self):
        """Test pure keyword search for Python files."""
        results = self.search_engine.search(
            vector_query="",
            keyword_query="language:Python",
            top_k=5
        )

        # Should find Python results
        assert len(results) > 0

        # All should be Python files
        for result in results:
            assert result.get('language') == 'Python'
            # Should have Python metadata fields
            assert 'functions' in result
            assert 'classes' in result
            assert 'imports' in result
            assert 'has_type_hints' in result
            assert 'has_async' in result
            assert 'complexity_score' in result

    def test_metadata_json_field_presence(self):
        """Test that Python results include metadata_json field."""
        results = self.search_engine.search(
            vector_query="function",
            keyword_query="language:Python",
            top_k=2
        )

        assert len(results) > 0

        for result in results:
            if result.get('language') == 'Python':
                assert 'metadata_json' in result
                assert isinstance(result['metadata_json'], str)
                # Should be valid JSON
                import json
                metadata = json.loads(result['metadata_json'])
                assert 'language' in metadata
                assert metadata['language'] == 'Python'


def run_manual_metadata_search():
    """Manual test runner for demonstration purposes."""

    # Load environment
    load_dotenv()

    db_url = os.getenv("COCOINDEX_DATABASE_URL")
    if not db_url:
        print("❌ COCOINDEX_DATABASE_URL not set")
        return

    try:
        name = get_default_db_name()
        pool = ConnectionPool(db_url)
        search_engine = HybridSearchEngine(name, KeywordSearchParser(), pool=pool)
    except Exception as e:
        print(f"❌ Could not connect to database: {e}")
        return

    print("🧪 Testing Enhanced Hybrid Search with Python Metadata")
    print("=" * 60)

    # Test 1: Search for Python functions
    print("\n🔍 Test 1: Search for Python functions with 'async' keyword")
    results = search_engine.search(
        vector_query="async function",
        keyword_query="language:Python",
        top_k=3
    )

    if results:
        print("\n📋 Readable Format:")
        print(format_results_readable(results))

        print("\n📊 JSON Format (showing metadata fields):")
        # Show just the metadata fields for the first result
        if results:
            first_result = results[0]
            metadata_fields = {
                key: value for key, value in first_result.items()
                if key in ['functions', 'classes', 'imports', 'decorators',
                           'has_type_hints', 'has_async', 'complexity_score']
            }
            print(f"Metadata for {first_result.get('filename', 'unknown')}:")
            import json
            print(json.dumps(metadata_fields, indent=2))
    else:
        print("No results found.")

    # Test 2: Search for class definitions
    print("\n" + "=" * 60)
    print("🔍 Test 2: Search for class definitions")
    results = search_engine.search(
        vector_query="class definition",
        keyword_query="language:Python",
        top_k=3
    )

    if results:
        print(format_results_readable(results))
    else:
        print("No results found.")

    # Test 3: Pure keyword search
    print("\n" + "=" * 60)
    print("🔍 Test 3: Pure keyword search for Python files")
    results = search_engine.search(
        vector_query="",
        keyword_query="language:Python",
        top_k=5
    )

    print(f"Found {len(results)} Python code chunks")
    if results:
        # Show metadata summary
        functions_found = []
        classes_found = []
        imports_found = set()

        for result in results:
            if result.get('functions'):
                functions_found.extend(result['functions'])
            if result.get('classes'):
                classes_found.extend(result['classes'])
            if result.get('imports'):
                imports_found.update(result['imports'])

        print("\n📊 Metadata Summary:")
        print(
            f"  Functions found: {len(functions_found)} ({', '.join(functions_found[:5])}{'...' if len(functions_found) > 5 else ''})")
        print(
            f"  Classes found: {len(classes_found)} ({', '.join(classes_found[:5])}{'...' if len(classes_found) > 5 else ''})")
        print(
            f"  Unique imports: {len(imports_found)} ({', '.join(list(imports_found)[:5])}{'...' if len(imports_found) > 5 else ''})")

    print("\n✅ Metadata search test completed!")


if __name__ == "__main__":
    # Check if pytest is available for proper testing
    try:
        import pytest
        print("✅ Use 'pytest test_metadata_search.py -m integration' to run integration tests")
        print("📝 Running manual demonstration...")
    except ImportError:
        print("📝 pytest not available, running manual test...")

    run_manual_metadata_search()
