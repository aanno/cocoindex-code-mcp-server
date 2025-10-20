#!/usr/bin/env python3

"""
Shared pytest fixtures and configuration for hybrid search tests.
"""

import os

import pytest

# Package should be installed via maturin develop or pip install -e .
# No need to manually add src to path


@pytest.fixture(scope="session")
def setup_test_environment():
    """Set up the test environment."""

    # Add src to path
    # src_path = Path(__file__).parent / "src"
    # / "cocoindex_code_mcp_server"
    # sys.path.insert(0, str(src_path))

    # Package should be installed via maturin develop or pip install -e .
    # Return current working directory instead
    return os.getcwd()


@pytest.fixture
def mock_database_pool(mocker):
    """Create a mock database connection pool with proper context manager support."""
    mock_pool = mocker.Mock()
    mock_conn = mocker.Mock()
    mock_cursor = mocker.Mock()

    # Set up context manager support
    mock_pool.connection.return_value.__enter__ = mocker.Mock(return_value=mock_conn)
    mock_pool.connection.return_value.__exit__ = mocker.Mock(return_value=None)
    mock_conn.cursor.return_value.__enter__ = mocker.Mock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = mocker.Mock(return_value=None)

    return mock_pool, mock_conn, mock_cursor


@pytest.fixture
def sample_search_results():
    """Provide sample search results for testing."""
    return [
        {
            "filename": "auth.py",
            "language": "Python",
            "code": "def authenticate(user, password):",
            "score": 0.88,
            "start": {"line": 10, "column": 0},
            "end": {"line": 15, "column": 4},
            "source": "files_0",
            "score_type": "hybrid"
        },
        {
            "filename": "login.py",
            "language": "Python",
            "code": "def login_user(username, password):",
            "score": 0.82,
            "start": {"line": 5, "column": 0},
            "end": {"line": 8, "column": 4},
            "source": "files_0",
            "score_type": "vector"
        }
    ]


@pytest.fixture
def mock_embedding_function():
    """Create a mock embedding function that returns a consistent vector."""
    def embedding_func(text):
        # Return a simple deterministic vector based on text length
        base_vector = [0.1, 0.2, 0.3]
        # Slightly modify based on text to make it deterministic but different
        modifier = len(text) * 0.01
        return [x + modifier for x in base_vector]

    return embedding_func


@pytest.fixture(autouse=True)
def skip_external_dependencies():
    """Automatically skip tests that require external dependencies if they're not available."""
    pass  # This can be expanded to check for specific dependencies


def pytest_collection_modifyitems(config, items):
    """Modify test items based on markers."""
    # Add skip markers for external dependencies
    for item in items:
        if "external" in item.keywords:
            # Check if external dependencies are available
            try:
                pass
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="CocoIndex not available"))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "hybrid_search: mark test as part of hybrid search functionality"
    )
    config.addinivalue_line(
        "markers", "keyword_parser: mark test as testing keyword search parser"
    )
    config.addinivalue_line(
        "markers", "search_engine: mark test as testing search engine"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external dependencies"
    )
    config.addinivalue_line(
        "markers", "standalone: mark test as runnable in isolation"
    )
    config.addinivalue_line(
        "markers", "main_mcp_server: mark test as testing MCP server functionality"
    )


# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)
