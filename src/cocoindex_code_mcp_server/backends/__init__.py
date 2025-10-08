#!/usr/bin/env python3

"""
Vector store backend factory and interface definitions.

This module provides the abstraction layer for different vector database backends,
allowing the CocoIndex MCP server to support multiple vector stores through a
unified interface.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Type

import numpy as np
from numpy.typing import NDArray

from cocoindex_code_mcp_server.keyword_search_parser_lark import (
    SearchCondition,
    SearchGroup,
)

# Import our standardized SearchResult from Phase 2 schemas
from ..schemas import SearchResult

LOGGER = logging.getLogger(__name__)


@dataclass
class QueryFilters:
    """Query filters for keyword/metadata search."""
    conditions: List[SearchCondition | SearchGroup]
    operator: str = "AND"  # AND, OR


class VectorStoreBackend(ABC):
    """Abstract base class for vector store backends."""

    def __init__(self, host: str, port: int, backend_type: Type[Any], extra_config: Dict[str, Any]) -> None:
        self._host = host
        self._port = port
        self._extra_config = extra_config
        self._backend_type = backend_type

    @abstractmethod
    def vector_search(
        self,
        query_vector: NDArray[np.float32],
        top_k: int = 10,
        embedding_model: str | None = None
    ) -> List[SearchResult]:
        """Perform pure vector similarity search with optional embedding model filter."""

    @abstractmethod
    def keyword_search(
        self,
        filters: QueryFilters,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Perform pure keyword/metadata search."""

    @abstractmethod
    def hybrid_search(
        self,
        query_vector: NDArray[np.float32],
        filters: QueryFilters,
        top_k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        embedding_model: str | None = None
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword search with optional embedding model filter."""

    @abstractmethod
    def configure(self, **options: Any) -> None:
        """Configure backend-specific options."""

    @abstractmethod
    def get_table_info(self) -> Dict[str, Any]:
        """Get information about the vector store structure."""

    @abstractmethod
    def close(self) -> None:
        """Close backend connections and cleanup resources."""

    @property
    def host(self):
        # return getattr(self, 'host', None)
        return self._host

    @property
    def port(self):
        # return getattr(self, 'port', None)
        return self._port

    @property
    def extra_config(self):
        # return getattr(self, 'extra_config', None)
        return self._extra_config

    @property
    def backend_type(self):
        # return getattr(self, 'backend_type', None)
        return self._backend_type


class BackendFactory:
    """Factory for creating vector store backends."""

    _backends: Dict[str, Type[VectorStoreBackend]] = {}

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[VectorStoreBackend]) -> None:
        """Register a new backend implementation."""
        cls._backends[name] = backend_class

    @classmethod
    def create_backend(cls, backend_type: str, **config: Any) -> VectorStoreBackend:
        """Create a backend instance."""
        if backend_type not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(f"Unknown backend type '{backend_type}'. Available: {available}")

        backend_class = cls._backends[backend_type]
        return backend_class(**config)

    @classmethod
    def list_backends(cls) -> List[str]:
        """List available backend types."""
        return list(cls._backends.keys())


# Auto-register backends when they're imported
def _auto_register_backends() -> None:
    """Automatically register available backends."""
    try:
        from .postgres_backend import PostgresBackend
        BackendFactory.register_backend("postgres", PostgresBackend)
    except ImportError:
        pass

    try:
        from .qdrant_backend import QdrantBackend
        BackendFactory.register_backend("qdrant", QdrantBackend)
    except ImportError:
        pass


# Auto-register on import
_auto_register_backends()


__all__ = [
    "VectorStoreBackend",
    "BackendFactory",
    "SearchResult",
    "QueryFilters"
]
