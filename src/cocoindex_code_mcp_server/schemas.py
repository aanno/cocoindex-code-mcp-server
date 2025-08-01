#!/usr/bin/env python3

"""
Schema definitions for CocoIndex MCP Server.

This module defines standardized metadata structures and query abstractions
that work across different vector database backends (PostgreSQL, Qdrant, etc.).
All schemas use mypy-compatible TypedDict for static type checking.
"""

from typing import Any, Dict, List, Optional, Union, TypedDict, Literal
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Core Metadata Schema
# =============================================================================

class ChunkMetadata(TypedDict, total=False):
    """
    Standardized metadata structure for code chunks across all backends.
    
    This schema represents the complete metadata available for each code chunk,
    ensuring consistency whether data is stored in PostgreSQL JSONB or 
    Qdrant payload format.
    
    Total=False allows partial metadata when not all fields are available.
    """
    # Core identification fields (always present)
    filename: str
    language: str
    location: str
    
    # Content fields
    code: str
    start: int
    end: int
    
    # Source tracking
    source_name: str
    
    # Extracted semantic metadata
    functions: List[str]
    classes: List[str] 
    imports: List[str]
    complexity_score: int
    has_type_hints: bool
    has_async: bool
    has_classes: bool
    
    # Raw metadata (backend-specific storage)
    metadata_json: Dict[str, Any]
    
    # Vector embedding (when available)
    embedding: Optional[NDArray[np.float32]]


class ExtractedMetadata(TypedDict):
    """
    Structure for metadata extracted from code analysis.
    
    This matches the output from our language handlers and AST analyzers.
    """
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity_score: int
    has_type_hints: bool
    has_async: bool
    has_classes: bool
    decorators_used: List[str]
    analysis_method: str


# =============================================================================
# Query Abstraction
# =============================================================================

class QueryType(Enum):
    """Types of search queries supported."""
    VECTOR = "vector"
    KEYWORD = "keyword" 
    HYBRID = "hybrid"


class FilterOperator(Enum):
    """Operators for metadata filtering."""
    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    LIKE = "LIKE"
    ILIKE = "ILIKE"
    IN = "IN"
    NOT_IN = "NOT IN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    CONTAINS = "CONTAINS"  # For JSONB/payload contains operations


@dataclass
class QueryFilter:
    """Individual filter condition for queries."""
    field: str
    operator: FilterOperator
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value
        }


class ChunkQuery(TypedDict, total=False):
    """
    Database-agnostic query interface for code chunks.
    
    This abstraction allows the same query structure to work across
    PostgreSQL (SQL) and Qdrant (payload filters) backends.
    """
    # Query text for vector search
    text: Optional[str]
    
    # Query type
    query_type: QueryType
    
    # Result limits
    top_k: int
    
    # Metadata filters
    filters: List[QueryFilter]
    filter_logic: Literal["AND", "OR"]
    
    # Hybrid search weights
    vector_weight: float
    keyword_weight: float
    
    # Embedding vector (when doing pure vector search)
    embedding: Optional[NDArray[np.float32]]

# =============================================================================
# Search Results
# =============================================================================

class SearchResultType(Enum):
    """Types of search result scores."""
    VECTOR_SIMILARITY = "vector_similarity"
    KEYWORD_MATCH = "keyword_match"
    HYBRID_COMBINED = "hybrid_combined"
    EXACT_MATCH = "exact_match"


@dataclass  
class SearchResult:
    """Standardized search result across all backends."""
    # Core content
    filename: str
    language: str
    code: str
    location: str
    
    # Position info
    start: Union[int, Dict[str, Any]]
    end: Union[int, Dict[str, Any]]
    
    # Search metadata
    score: float
    score_type: SearchResultType
    source: str
    
    # Full metadata
    metadata: Optional[ChunkMetadata] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "filename": self.filename,
            "language": self.language, 
            "code": self.code,
            "location": self.location,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "score_type": self.score_type.value,
            "source": self.source,
            "metadata": self.metadata
        }


# =============================================================================
# Backend Capability System  
# =============================================================================

class BackendCapability(Enum):
    """Capabilities that backends can support."""
    VECTOR_SEARCH = "search-vector"
    KEYWORD_SEARCH = "search-keyword"
    HYBRID_SEARCH = "search-hybrid"
    FULL_TEXT_SEARCH = "search-full_text" # TODO: not in use?
    JSONB_QUERIES = "jsonb_queries"
    PAYLOAD_INDEXING = "payload_indexing"
    TRANSACTION_SUPPORT = "transaction_support"
    BATCH_OPERATIONS = "batch_operations"


@dataclass
class BackendInfo:
    """Information about a backend's capabilities and configuration."""
    backend_type: str
    capabilities: List[BackendCapability]
    max_vector_dimensions: Optional[int] = None
    supports_metadata_indexing: bool = False
    transaction_support: bool = False
    
    def has_capability(self, capability: BackendCapability) -> bool:
        """Check if backend supports a specific capability."""
        return capability in self.capabilities


# =============================================================================
# Schema Validation
# =============================================================================

def validate_chunk_metadata(metadata: Dict[str, Any]) -> ChunkMetadata:
    """
    Validate and convert raw metadata to ChunkMetadata schema.
    
    Args:
        metadata: Raw metadata dictionary
        
    Returns:
        Validated ChunkMetadata
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = ["filename", "language", "location"]
    
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Required field '{field}' missing from metadata")
    
    # Type validation for critical fields
    if not isinstance(metadata.get("functions", []), list):
        raise ValueError("'functions' field must be a list")
        
    if not isinstance(metadata.get("classes", []), list):
        raise ValueError("'classes' field must be a list")
        
    if not isinstance(metadata.get("imports", []), list):
        raise ValueError("'imports' field must be a list")
    
    # Convert to proper types
    validated: ChunkMetadata = {
        "filename": str(metadata["filename"]),
        "language": str(metadata["language"]),
        "location": str(metadata["location"]),
    }
    
    # Optional fields with defaults
    if "code" in metadata:
        validated["code"] = str(metadata["code"])
    
    validated["functions"] = metadata.get("functions", [])
    validated["classes"] = metadata.get("classes", [])
    validated["imports"] = metadata.get("imports", [])
    validated["complexity_score"] = int(metadata.get("complexity_score", 0))
    validated["has_type_hints"] = bool(metadata.get("has_type_hints", False))
    validated["has_async"] = bool(metadata.get("has_async", False))
    validated["has_classes"] = bool(metadata.get("has_classes", False))
    
    if "start" in metadata:
        validated["start"] = int(metadata["start"])
    if "end" in metadata:
        validated["end"] = int(metadata["end"])
    if "source_name" in metadata:
        validated["source_name"] = str(metadata["source_name"])
    if "metadata_json" in metadata:
        validated["metadata_json"] = metadata["metadata_json"]
    if "embedding" in metadata and metadata["embedding"] is not None:
        validated["embedding"] = metadata["embedding"]
    
    return validated


def create_default_chunk_query(
    text: Optional[str] = None,
    query_type: QueryType = QueryType.HYBRID,
    top_k: int = 10
) -> ChunkQuery:
    """Create a default ChunkQuery with sensible defaults."""
    return ChunkQuery(
        text=text,
        query_type=query_type,
        top_k=top_k,
        filters=[],
        filter_logic="AND",
        vector_weight=0.7,
        keyword_weight=0.3
    )


# =============================================================================
# Export for convenience
# =============================================================================

__all__ = [
    # Core schemas
    "ChunkMetadata",
    "ExtractedMetadata", 
    "ChunkQuery",
    "QueryFilter",
    "SearchResult",
    
    # Enums
    "QueryType",
    "FilterOperator", 
    "SearchResultType",
    "BackendCapability",
    
    # Backend info
    "BackendInfo",
    
    # Validation functions
    "validate_chunk_metadata",
    "create_default_chunk_query"
]
