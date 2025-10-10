#!/usr/bin/env python3

"""
Integration tests for the hybrid search workflow.
"""

import tempfile
import pytest

# Package should be installed via maturin develop or pip install -e .


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.hybrid_search
@pytest.mark.external
@pytest.mark.db_integration
class TestMainHybridSearchIntegration:
    """Integration tests for the main hybrid search entry point."""

    # @patch('cocoindex_code_mcp_server.main_hybrid_search.cocoindex.init')
    # @patch('cocoindex_code_mcp_server.main_hybrid_search.load_dotenv')
    def test_argument_parsing_basic(self, mocker): 
        # mock_load_dotenv: MagicMock, mock_cocoindex_init: MagicMock):
        """Test basic argument parsing."""
        mock_load_dotenv = mocker.patch('cocoindex_code_mcp_server.main_hybrid_search.cocoindex.init')
        mock_cocoindex_init = mocker.patch('cocoindex_code_mcp_server.main_hybrid_search.load_dotenv')
        
        try:
            from cocoindex_code_mcp_server.main_hybrid_search import (
                determine_paths,
                parse_hybrid_search_args,
            )

            # Test default arguments
            mocker.patch('sys.argv', ['main_hybrid_search.py'])
            args = parse_hybrid_search_args()
            paths = determine_paths(args)

            assert paths is None  # Should use default
            assert not args.no_live  # Live updates enabled by default
            assert args.poll == 60  # Default polling interval
        except ImportError:
            pytest.skip("CocoIndex not available in test environment")

    # @patch('cocoindex_code_mcp_server.main_hybrid_search.cocoindex.init')
    # @patch('cocoindex_code_mcp_server.main_hybrid_search.load_dotenv')
    def test_argument_parsing_custom_paths(self, mocker): 
        # mock_load_dotenv: MagicMock, mock_cocoindex_init: MagicMock)
        """Test argument parsing with custom paths."""
        mock_load_dotenv = mocker.patch('cocoindex_code_mcp_server.main_hybrid_search.cocoindex.init')
        mock_cocoindex_init = mocker.patch('cocoindex_code_mcp_server.main_hybrid_search.load_dotenv')

        
        try:
            from cocoindex_code_mcp_server.main_hybrid_search import (
                determine_paths,
                parse_hybrid_search_args,
            )

            # Test with positional paths
            mocker.patch('sys.argv', ['main_hybrid_search.py', '/path1', '/path2'])
            args = parse_hybrid_search_args()
            paths = determine_paths(args)

            assert paths == ['/path1', '/path2']
        except ImportError:
            pytest.skip("CocoIndex not available in test environment")

    # @patch('cocoindex_code_mcp_server.main_hybrid_search.cocoindex.init')
    # @patch('cocoindex_code_mcp_server.main_hybrid_search.load_dotenv')
    def test_argument_parsing_no_live(self, mocker): 
        # mock_load_dotenv: MagicMock, mock_cocoindex_init: MagicMock)
        """Test argument parsing with live updates disabled."""
        mock_load_dotenv = mocker.patch('cocoindex_code_mcp_server.main_hybrid_search.cocoindex.init')
        mock_cocoindex_init = mocker.patch('cocoindex_code_mcp_server.main_hybrid_search.load_dotenv')
        
        try:
            from cocoindex_code_mcp_server.main_hybrid_search import (
                parse_hybrid_search_args,
            )

            mocker.patch('sys.argv', ['main_hybrid_search.py', '--no-live'])
            args = parse_hybrid_search_args()

            assert args.no_live
        except ImportError:
            pytest.skip("CocoIndex not available in test environment")

    # @patch('cocoindex_code_mcp_server.main_hybrid_search.cocoindex.init')
    # @patch('cocoindex_code_mcp_server.main_hybrid_search.load_dotenv')
    def test_argument_parsing_custom_poll(self, mocker): 
        # mock_load_dotenv: MagicMock, mock_cocoindex_init: MagicMock)
        """Test argument parsing with custom polling interval."""
        mock_load_dotenv = mocker.patch('cocoindex_code_mcp_server.main_hybrid_search.cocoindex.init')
        mock_cocoindex_init = mocker.patch('cocoindex_code_mcp_server.main_hybrid_search.load_dotenv')

        try:
            from cocoindex_code_mcp_server.main_hybrid_search import (
                parse_hybrid_search_args,
            )

            mocker.patch('sys.argv', ['main_hybrid_search.py', '--poll', '30'])
            args = parse_hybrid_search_args()

            assert args.poll == 30
        except ImportError:
            pytest.skip("CocoIndex not available in test environment")

    @pytest.mark.skip(reason='Integration test needs complex database mocking')
    # @patch('cocoindex_code_mcp_server.main_hybrid_search.update_flow_config')
    # @patch('cocoindex_code_mcp_server.main_hybrid_search.run_interactive_hybrid_search')
    # @patch('cocoindex_code_mcp_server.main_hybrid_search.cocoindex.init')
    # @patch('cocoindex_code_mcp_server.main_hybrid_search.load_dotenv')
    def test_main_workflow_no_live(self, mocker): 
        # mock_load_dotenv, mock_cocoindex_init, mock_run_interactive, mock_update_config)
        """Test main workflow without live updates."""
        mock_load_dotenv = mocker.patch('cocoindex_code_mcp_server.main_hybrid_search.cocoindex.init')
        mock_cocoindex_init = mocker.patch('cocoindex_code_mcp_server.main_hybrid_search.load_dotenv')
        mock_run_interactive = mocker.patch('cocoindex_code_mcp_server.main_hybrid_search.run_interactive_hybrid_search')
        mock_update_config = mocker.patch('cocoindex_code_mcp_server.main_hybrid_search.update_flow_config')
        
        try:
            from cocoindex_code_mcp_server.main_hybrid_search import main

            # Mock the flow update
            mock_flow = mocker.patch('main_hybrid_search.code_embedding_flow')
            mock_flow.update.return_value = {"processed": 10}

            mocker.patch('sys.argv', ['main_hybrid_search.py', '--no-live'])
            main()

            # Verify configuration was updated
            mock_update_config.assert_called_once()

            # Verify flow update was called
            mock_flow.update.assert_called_once()

            # Verify interactive search was started
            mock_run_interactive.assert_called_once()
        except ImportError:
            pytest.skip("CocoIndex not available in test environment")


@pytest.mark.integration
@pytest.mark.hybrid_search
@pytest.mark.db_integration
class TestWorkflowIntegration:
    """Test the complete workflow integration."""

    @pytest.fixture
    def mock_database_setup(self, mocker):
        """Set up mock database components."""
        mock_pool = mocker.Mock()
        mock_conn = mocker.Mock()
        mock_cursor = mocker.Mock()

        # Set up mock chain
        mock_pool.connection.return_value.__enter__ = mocker.Mock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = mocker.Mock(return_value=None)
        mock_conn.cursor.return_value.__enter__ = mocker.Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = mocker.Mock(return_value=None)

        return mock_pool, mock_conn, mock_cursor

    # @patch('cocoindex_code_mcp_server.db.pgvector.hybrid_search.os.getenv')
    # @patch('cocoindex_code_mcp_server.db.pgvector.hybrid_search.ConnectionPool')
    def test_interactive_hybrid_search_workflow(self, mocker): 
        # mock_pool_class, mock_getenv, mock_database_setup):
        """Test the complete interactive hybrid search workflow."""
        mock_getenv = mocker.patch('cocoindex_code_mcp_server.db.pgvector.hybrid_search.os.getenv')
        mock_pool_class = mocker.patch('cocoindex_code_mcp_server.db.pgvector.hybrid_search.ConnectionPool')
        
        try:
            # Mock environment and database setup
            mock_getenv.return_value = "postgresql://test"
            mock_pool = mocker.Mock()
            mock_conn = mocker.Mock()
            mock_cursor = mocker.Mock()
            mock_pool_class.return_value = mock_pool

            # Mock database results
            mock_cursor.fetchall.return_value = [
                ("test.py", "Python", "def test():", 0.2, {"line": 1}, {"line": 3}, "files", 0.85)
            ]

            # Mock user input
            mock_input = mocker.patch('builtins.input')
            # Simulate user entering queries and then empty query to quit
            mock_input.side_effect = [
                "authentication functions",  # vector query
                "language:python",           # keyword query
                "",                         # empty query to quit
            ]

            from cocoindex_code_mcp_server.db.pgvector.hybrid_search import (
                run_interactive_hybrid_search,
            )

            # Should not raise any exceptions
            run_interactive_hybrid_search()

            # Verify database connection was created
            mock_pool_class.assert_called_once_with("postgresql://test")
        except ImportError:
            pytest.skip("CocoIndex not available in test environment")

    @pytest.mark.skip(reason='Integration test needs register_vector mocking')
    def test_end_to_end_search_pipeline(self, mock_database_setup):
        """Test the end-to-end search pipeline with mocked components."""
        try:
            from cocoindex_code_mcp_server.db.pgvector.hybrid_search import (
                HybridSearchEngine,
            )
            from cocoindex_code_mcp_server.keyword_search_parser_lark import (
                KeywordSearchParser,
            )

            mock_pool, mock_conn, mock_cursor = mock_database_setup

            # Create engine with mocked dependencies
            parser = KeywordSearchParser()
            def embedding_func(x): return [0.1, 0.2, 0.3]  # Simple mock embedding

            engine = HybridSearchEngine(
                table_name="test_embeddings",
                parser=parser,
                pool=mock_pool,
                embedding_func=embedding_func
            )

            # Mock database results for hybrid search
            mock_cursor.fetchall.return_value = [
                ("auth.py", "Python", "def authenticate(user, password):", 0.15,
                 {"line": 10, "column": 0}, {"line": 15, "column": 4}, "files_0", 0.88),
                ("login.py", "Python", "def login_user(username, password):", 0.25,
                 {"line": 5, "column": 0}, {"line": 8, "column": 4}, "files_0", 0.82)
            ]

            # Execute hybrid search
            results = engine.search(
                vector_query="authentication login user verification",
                keyword_query="language:python and exists(embedding)",
                top_k=5
            )

            # Verify results
            assert len(results) == 2

            # Check first result
            result1 = results[0]
            assert result1["filename"] == "auth.py"
            assert result1["language"] == "Python"
            assert result1["score"] == 0.88
            assert result1["score_type"] == "hybrid"

            # Check second result
            result2 = results[1]
            assert result2["filename"] == "login.py"
            assert result2["score"] == 0.82
        except ImportError:
            pytest.skip("CocoIndex not available in test environment")

    @pytest.mark.skip(reason='Integration test needs register_vector mocking')
    def test_complex_query_parsing_and_execution(self, mock_database_setup):
        """Test complex query parsing and execution."""
        try:
            from cocoindex_code_mcp_server.db.pgvector.hybrid_search import (
                HybridSearchEngine,
            )
            from cocoindex_code_mcp_server.keyword_search_parser_lark import (
                KeywordSearchParser,
            )

            mock_pool, mock_conn, mock_cursor = mock_database_setup

            # Create engine with real parser
            parser = KeywordSearchParser()
            def embedding_func(x): return [0.1, 0.2, 0.3]

            engine = HybridSearchEngine(
                table_name="test_embeddings",
                parser=parser,
                pool=mock_pool,
                embedding_func=embedding_func
            )

            # Mock database results
            mock_cursor.fetchall.return_value = [
                ("main_interactive_query.py", "Python", "def main():", 0.3,
                 {"line": 1}, {"line": 10}, "files", 0.75)
            ]

            # Execute search with complex keyword query
            results = engine.search(
                vector_query="main function entry point",
                keyword_query="(filename:main_interactive_query.py or filename:app.py) and language:python and exists(embedding)",
                top_k=10
            )

            # Verify that complex query was parsed and executed
            assert len(results) == 1
            result = results[0]
            assert result["filename"] == "main_interactive_query.py"
            assert result["score_type"] == "hybrid"
        except ImportError:
            pytest.skip("CocoIndex not available in test environment")


@pytest.mark.integration
@pytest.mark.hybrid_search
@pytest.mark.external
@pytest.mark.db_integration
class TestConfigurationIntegration:
    """Test configuration and setup integration."""

    # @patch('cocoindex_code_mcp_server.cocoindex_config.code_embedding_flow')
    def test_flow_configuration_update(self, mocker): 
        # mock_flow: MagicMock)
        """Test that flow configuration is properly updated."""
        mock_flow = mocker.patch('cocoindex_code_mcp_server.cocoindex_config.code_embedding_flow')
        
        try:
            from cocoindex_code_mcp_server.cocoindex_config import update_flow_config

            # Test configuration update
            update_flow_config(
                paths=["/test/path1", "/test/path2"],
                enable_polling=True,
                poll_interval=45
            )

            # Import the global config to verify it was updated
            from cocoindex_code_mcp_server.cocoindex_config import _global_flow_config

            assert _global_flow_config['paths'] == ["/test/path1", "/test/path2"]
            assert _global_flow_config['enable_polling'] == True
            assert _global_flow_config['poll_interval'] == 45
        except ImportError:
            pytest.skip("CocoIndex not available in test environment")

    # @patch('cocoindex_code_mcp_server.cocoindex_config.code_embedding_flow')
    def test_flow_configuration_defaults(self, mocker):
        # mock_flow: MagicMock):
        """Test flow configuration with default values."""
        mock_flow = mocker.patch('cocoindex_code_mcp_server.cocoindex_config.code_embedding_flow')
        
        try:
            from cocoindex_code_mcp_server.cocoindex_config import update_flow_config

            # Test with default values
            update_flow_config()

            from cocoindex_code_mcp_server.cocoindex_config import _global_flow_config

            assert _global_flow_config['paths'] == ["cocoindex"]
            assert _global_flow_config['enable_polling'] == False
            assert _global_flow_config['poll_interval'] == 30
        except ImportError:
            pytest.skip("CocoIndex not available in test environment")


@pytest.mark.unit
@pytest.mark.hybrid_search
@pytest.mark.db_integration
class TestErrorHandling:
    """Test error handling in the integration workflow."""

    def test_search_with_database_error(self, mocker):
        """Test search behavior when database errors occur."""
        try:
            from cocoindex_code_mcp_server.db.pgvector.hybrid_search import (
                HybridSearchEngine,
            )

            # Mock database to raise an exception
            mock_pool = mocker.Mock()
            mock_pool.connection.side_effect = Exception("Database connection failed")

            from cocoindex_code_mcp_server.keyword_search_parser_lark import (
                KeywordSearchParser,
            )
            parser = KeywordSearchParser()
            engine = HybridSearchEngine(
                table_name="test_embeddings",
                parser=parser,
                pool=mock_pool,
                embedding_func=lambda x: [0.1, 0.2, 0.3]
            )

            # Search should raise the database exception
            with pytest.raises(Exception, match="Database connection failed"):
                engine.search("test query", "language:python", top_k=5)
        except ImportError:
            pytest.skip("CocoIndex not available in test environment")

    @pytest.mark.skip(reason='Integration test needs context manager mocking')
    def test_search_with_invalid_keyword_syntax(self, mocker):
        """Test search with invalid keyword syntax."""
        try:
            from cocoindex_code_mcp_server.db.pgvector.hybrid_search import (
                HybridSearchEngine,
            )
            from cocoindex_code_mcp_server.keyword_search_parser_lark import (
                KeywordSearchParser,
            )

            # Use real parser to test actual parsing behavior
            parser = KeywordSearchParser()
            mock_pool = mocker.Mock()

            engine = HybridSearchEngine(
                table_name="test_embeddings",
                parser=parser,
                pool=mock_pool,
                embedding_func=lambda x: [0.1, 0.2, 0.3]
            )

            # Test with malformed query - parser should handle gracefully
            # Invalid syntax like unmatched parentheses
            results = engine.search("test", "((language:python", top_k=5)

            # Should not crash, may return empty results or parse as text search
            assert isinstance(results, list)
        except ImportError:
            pytest.skip("CocoIndex not available in test environment")

    def test_embedding_function_error(self, mocker):
        """Test behavior when embedding function fails."""
        try:
            from cocoindex_code_mcp_server.db.pgvector.hybrid_search import (
                HybridSearchEngine,
            )

            # Mock embedding function that raises an error
            def failing_embedding_func(query):
                raise Exception("Embedding service unavailable")

            mock_pool = mocker.Mock()

            from cocoindex_code_mcp_server.keyword_search_parser_lark import (
                KeywordSearchParser,
            )
            parser = KeywordSearchParser()
            engine = HybridSearchEngine(
                table_name="test_embeddings",
                parser=parser,
                pool=mock_pool,
                embedding_func=failing_embedding_func
            )

            # Vector search should raise the embedding exception
            with pytest.raises(Exception, match="Embedding service unavailable"):
                engine.search("test query", "", top_k=5)
        except ImportError:
            pytest.skip("CocoIndex not available in test environment")


@pytest.mark.slow
@pytest.mark.hybrid_search
@pytest.mark.db_integration
class TestPerformanceCharacteristics:
    """Test performance characteristics of the search system."""

    @pytest.mark.skip(reason='Integration test has JSON formatting issues')
    def test_large_result_set_handling(self):
        """Test handling of large result sets."""
        try:
            from cocoindex_code_mcp_server.db.pgvector.hybrid_search import (
                format_results_as_json,
                format_results_readable,
            )

            # Create a large number of mock results
            large_results = []
            for i in range(1000):
                large_results.append({
                    "filename": f"file_{i}.py",
                    "language": "Python",
                    "code": f"def function_{i}():",
                    "score": 0.8 - (i * 0.0001),  # Decreasing scores
                    "start": {"line": i},
                    "end": {"line": i + 5},
                    "source": "files",
                    "score_type": "vector"
                })

            # Test that formatting functions can handle large datasets
            readable_output = format_results_readable(large_results)
            json_output = format_results_as_json(large_results)

            # Verify outputs are reasonable
            assert "1000 results" in readable_output
            assert isinstance(json_output, str)

            # JSON should be parseable
            import json
            parsed = json.loads(json_output)
            assert len(parsed) == 1000
        except ImportError:
            pytest.skip("CocoIndex not available in test environment")

    def test_query_complexity_handling(self):
        """Test handling of complex queries."""
        from cocoindex_code_mcp_server.keyword_search_parser_lark import (
            KeywordSearchParser,
        )

        parser = KeywordSearchParser()

        # Test very complex nested query
        complex_query = (
            "((language:python or language:rust or language:go) and "
            "(filename:main_interactive_query.py or filename:app.py or filename:server.py)) and "
            "exists(embedding) and "
            "((source_name:files_0 or source_name:files_1) and "
            "(exists(start) and exists(end)))"
        )

        # Should parse without error
        result = parser.parse(complex_query)
        assert result is not None
        assert len(result.conditions) > 0
