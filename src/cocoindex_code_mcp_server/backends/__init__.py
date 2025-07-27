#!/usr/bin/env python3

"""
Vector store backend factory and interface definitions.

This module provides the abstraction layer for different vector database backends,
allowing the CocoIndex MCP server to support multiple vector stores through a
unified interface.
"""

from typing import Any, Dict, List, Optional, Protocol, Union, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SearchResult:
    """Standardized search result format."""
    filename: str
    language: str
    code: str
    score: float
    start: Union[int, Dict[str, Any]]
    end: Union[int, Dict[str, Any]]
    source: str
    score_type: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryFilters:
    """Query filters for keyword/metadata search."""
    conditions: List[Dict[str, Any]]
    operator: str = "AND"  # AND, OR


class VectorStoreBackend(ABC):
    """Abstract base class for vector store backends."""
    
    @abstractmethod
    def vector_search(
        self, 
        query_vector: NDArray[np.float32], 
        top_k: int = 10
    ) -> List[SearchResult]:
        """Perform pure vector similarity search."""
        pass
    
    @abstractmethod
    def keyword_search(
        self, 
        filters: QueryFilters, 
        top_k: int = 10
    ) -> List[SearchResult]:
        """Perform pure keyword/metadata search."""
        pass
    
    @abstractmethod
    def hybrid_search(
        self,
        query_vector: NDArray[np.float32],
        filters: QueryFilters,
        top_k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword search."""
        pass
    
    @abstractmethod
    def configure(self, **options: Any) -> None:
        """Configure backend-specific options."""
        pass
    
    @abstractmethod
    def get_table_info(self) -> Dict[str, Any]:
        """Get information about the vector store structure."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close backend connections and cleanup resources."""
        pass


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