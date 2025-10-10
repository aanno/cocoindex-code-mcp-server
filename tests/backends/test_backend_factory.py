#!/usr/bin/env python3

"""
Tests for Backend Factory functionality.
"""

import pytest

from cocoindex_code_mcp_server.backends import (
    BackendFactory,
    VectorStoreBackend,
)


class MockBackend(VectorStoreBackend):
    """Mock backend for testing factory."""

    def __init__(self, config_param: str = "default"):
        self.config_param = config_param

    def vector_search(self, query_vector, top_k=10, embedding_model=None):
        return []

    def keyword_search(self, filters, top_k=10):
        return []

    def hybrid_search(self, query_vector, filters, top_k=10, vector_weight=0.7, keyword_weight=0.3, embedding_model=None):
        return []

    def configure(self, **options):
        pass

    def get_table_info(self):
        return {"backend_type": "mock"}

    def close(self):
        pass


@pytest.mark.unit
@pytest.mark.backend
class TestBackendFactory:
    """Test Backend Factory functionality."""

    def test_register_backend(self):
        """Test backend registration."""
        # Clear any existing registrations for clean test
        original_backends = BackendFactory._backends.copy()
        BackendFactory._backends.clear()

        try:
            # Register a mock backend
            BackendFactory.register_backend("mock", MockBackend)

            # Verify registration
            assert "mock" in BackendFactory._backends
            assert BackendFactory._backends["mock"] == MockBackend
        finally:
            # Restore original backends
            BackendFactory._backends = original_backends

    def test_create_backend_success(self):
        """Test successful backend creation."""
        # Register mock backend temporarily
        original_backends = BackendFactory._backends.copy()
        BackendFactory._backends["mock"] = MockBackend

        try:
            # Create backend instance
            backend = BackendFactory.create_backend("mock", config_param="test_value")

            # Verify instance
            assert isinstance(backend, MockBackend)
            assert backend.config_param == "test_value"
        finally:
            # Restore original backends
            BackendFactory._backends = original_backends

    def test_create_backend_unknown_type(self):
        """Test error handling for unknown backend type."""
        with pytest.raises(ValueError, match="Unknown backend type 'nonexistent'"):
            BackendFactory.create_backend("nonexistent")

    def test_create_backend_error_message_lists_available(self, mocker):
        """Test that error message lists available backends."""
        # Temporarily set known backends
        original_backends = BackendFactory._backends.copy()
        BackendFactory._backends = {"postgres": mocker.Mock, "qdrant": mocker.Mock}

        try:
            with pytest.raises(ValueError) as exc_info:
                BackendFactory.create_backend("unknown")

            error_message = str(exc_info.value)
            assert "Available: postgres, qdrant" in error_message
        finally:
            # Restore original backends
            BackendFactory._backends = original_backends

    def test_list_backends(self, mocker):
        """Test listing available backends."""
        # Temporarily set known backends
        original_backends = BackendFactory._backends.copy()
        BackendFactory._backends = {"postgres": mocker.Mock, "qdrant": mocker.Mock, "test": mocker.Mock}

        try:
            backends = BackendFactory.list_backends()

            # Verify list contents
            assert isinstance(backends, list)
            assert set(backends) == {"postgres", "qdrant", "test"}
        finally:
            # Restore original backends
            BackendFactory._backends = original_backends

    def test_list_backends_empty(self):
        """Test listing backends when none are registered."""
        # Temporarily clear backends
        original_backends = BackendFactory._backends.copy()
        BackendFactory._backends.clear()

        try:
            backends = BackendFactory.list_backends()

            # Verify empty list
            assert backends == []
        finally:
            # Restore original backends
            BackendFactory._backends = original_backends

    def test_auto_registration_postgres(self, mocker): # mock_postgres_class: MagicMock):
        """Test that PostgreSQL backend is auto-registered."""
        # Import should trigger auto-registration
        from cocoindex_code_mcp_server.backends import BackendFactory
        
        mock_postgres_class = mocker.patch('cocoindex_code_mcp_server.backends.postgres_backend.PostgresBackend')

        # Verify postgres is available
        backends = BackendFactory.list_backends()
        assert "postgres" in backends

    def test_auto_registration_qdrant_not_available(self):
        """Test that Qdrant backend auto-registration handles import errors gracefully."""
        # This tests the try/except block in _auto_register_backends
        # Qdrant should not be available since it's a skeleton with NotImplementedError
        from cocoindex_code_mcp_server.backends import BackendFactory

        # Qdrant might or might not be in the list depending on import success
        # This test mainly ensures no exceptions are raised during import
        backends = BackendFactory.list_backends()
        assert isinstance(backends, list)

    def test_factory_is_singleton_like(self):
        """Test that factory maintains state across imports."""
        from cocoindex_code_mcp_server.backends import BackendFactory as Factory1
        from cocoindex_code_mcp_server.backends import BackendFactory as Factory2

        # Should be the same class
        assert Factory1 is Factory2

        # Should share same backends registry
        assert Factory1._backends is Factory2._backends

    def test_backend_creation_with_complex_config(self):
        """Test backend creation with complex configuration parameters."""
        # Register mock backend temporarily
        original_backends = BackendFactory._backends.copy()

        class ConfigurableBackend(VectorStoreBackend):
            def __init__(self, host="localhost", port=5432, database="test", **kwargs):
                super().__init__(host, port, ConfigurableBackend, kwargs)
                self.database = database

            def vector_search(self, query_vector, top_k=10, embedding_model=None):  # type: ignore[override]
                return []

            def keyword_search(self, filters, top_k=10):
                return []

            def hybrid_search(self, query_vector, filters, top_k=10, vector_weight=0.7, keyword_weight=0.3, embedding_model=None):  # type: ignore[override]
                return []

            def configure(self, **options):
                pass

            def get_table_info(self):
                return {}

            def close(self):
                pass

        BackendFactory._backends["configurable"] = ConfigurableBackend

        try:
            # Create backend with complex config
            backend = BackendFactory.create_backend(
                "configurable",
                host="example.com",
                port=3306,
                database="production",
                ssl_mode="require",
                timeout=30
            )

            # Verify configuration was passed correctly
            assert backend.host == "example.com"
            assert backend.port == 3306
            assert hasattr(backend, 'database') and getattr(backend, 'database') == "production"
            assert backend.extra_config["ssl_mode"] == "require"
            assert backend.extra_config["timeout"] == 30
        finally:
            # Restore original backends
            BackendFactory._backends = original_backends

    def test_backend_registration_overwrites(self):
        """Test that backend registration overwrites existing registrations."""
        original_backends = BackendFactory._backends.copy()

        class FirstBackend(VectorStoreBackend):
            def __init__(self, **kwargs):
                super().__init__("localhost", 5432, FirstBackend, kwargs)
                self.backend_type_label = "first"

            def vector_search(self, query_vector, top_k=10, embedding_model=None): return []  # type: ignore[override]
            def keyword_search(self, filters, top_k=10): return []
            def hybrid_search(self, query_vector, filters, top_k=10, vector_weight=0.7, keyword_weight=0.3, embedding_model=None): return []  # type: ignore[override]
            def configure(self, **options): pass
            def get_table_info(self): return {}
            def close(self): pass

        class SecondBackend(VectorStoreBackend):
            def __init__(self, **kwargs):
                super().__init__("localhost", 5432, SecondBackend, kwargs)
                self.backend_type_label = "second"

            def vector_search(self, query_vector, top_k=10, embedding_model=None): return []  # type: ignore[override]
            def keyword_search(self, filters, top_k=10): return []
            def hybrid_search(self, query_vector, filters, top_k=10, vector_weight=0.7, keyword_weight=0.3, embedding_model=None): return []  # type: ignore[override]
            def configure(self, **options): pass
            def get_table_info(self): return {}
            def close(self): pass

        try:
            # Register first backend
            BackendFactory.register_backend("test", FirstBackend)
            backend1 = BackendFactory.create_backend("test")
            assert backend1.backend_type == FirstBackend

            # Register second backend with same name (should overwrite)
            BackendFactory.register_backend("test", SecondBackend)
            backend2 = BackendFactory.create_backend("test")
            assert backend2.backend_type == SecondBackend

        finally:
            # Restore original backends
            BackendFactory._backends = original_backends
