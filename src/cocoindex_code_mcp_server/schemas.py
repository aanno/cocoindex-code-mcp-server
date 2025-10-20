#!/usr/bin/env python3
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 aanno <aanno@users.noreply.github.com>
#
# This file is part of cocoindex_code_mcp_server from
# https://github.com/aanno/cocoindex-code-mcp-server
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Schema definitions for CocoIndex MCP Server.

This module defines standardized metadata structures and query abstractions
that work across different vector database backends (PostgreSQL, Qdrant, etc.).
All schemas use mypy-compatible TypedDict for static type checking.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

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

    # Analysis tracking metadata (promoted from metadata_json)
    analysis_method: str
    chunking_method: str
    tree_sitter_chunking_error: bool
    tree_sitter_analyze_error: bool

    # Raw metadata (backend-specific storage)
    metadata_json: Dict[str, Any]

    # Vector embedding (when available)
    embedding: Optional[NDArray[np.float32]]

    has_docstrings: bool
    docstring: str

    decorators_used: List[str]
    dunder_methods: List[str]
    private_methods: List[str]

    variables: List[str]
    decorators: List[str]

    function_details: Dict[str, Any]
    class_details: Dict[str, Any]


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
    chunking_method: str
    tree_sitter_chunking_error: bool
    tree_sitter_analyze_error: bool


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
        return {"field": self.field, "operator": self.operator.value, "value": self.value}


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
            "metadata": self.metadata,
        }


# =============================================================================
# Backend Capability System
# =============================================================================


class BackendCapability(Enum):
    """Capabilities that backends can support."""

    VECTOR_SEARCH = "search-vector"
    KEYWORD_SEARCH = "search-keyword"
    HYBRID_SEARCH = "search-hybrid"
    FULL_TEXT_SEARCH = "search-full_text"  # TODO: not in use?
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

    # Analysis tracking metadata with defaults
    validated["analysis_method"] = str(metadata.get("analysis_method", "unknown"))
    # NOTE: chunking_method removed from metadata to avoid confusion - it comes from AST chunkers only
    validated["tree_sitter_chunking_error"] = bool(metadata.get("tree_sitter_chunking_error", False))
    validated["tree_sitter_analyze_error"] = bool(metadata.get("tree_sitter_analyze_error", False))

    validated["has_docstrings"] = bool(metadata.get("has_docstrings", False))
    validated["docstring"] = str(metadata.get("docstring", ""))

    validated["decorators_used"] = metadata.get("decorators_used", [])
    validated["dunder_methods"] = metadata.get("dunder_methods", [])
    validated["private_methods"] = metadata.get("private_methods", [])

    if "function_details" in metadata:
        validated["function_details"] = metadata.get("function_details", [])
    if "class_details" in metadata:
        validated["class_details"] = metadata.get("class_details", [])

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
    text: Optional[str] = None, query_type: QueryType = QueryType.HYBRID, top_k: int = 10
) -> ChunkQuery:
    """Create a default ChunkQuery with sensible defaults."""
    return ChunkQuery(
        text=text,
        query_type=query_type,
        top_k=top_k,
        filters=[],
        filter_logic="AND",
        vector_weight=0.7,
        keyword_weight=0.3,
    )


# =============================================================================
# Promoted Metadata Fields Configuration
# =============================================================================

# GENERALIZED PROMOTION: All fields from metadata_json are automatically promoted
# to top-level in search results. This makes the system flexible and future-proof.
# No need to maintain hardcoded lists of promoted fields.

# Fields that are always extracted as individual database columns
STANDARD_METADATA_FIELDS = [
    "functions",
    "classes",
    "imports",
    "complexity_score",
    "has_type_hints",
    "has_async",
    "has_classes",
    "metadata_json",
]

# Note: Any additional fields in metadata_json will be automatically promoted
# to top-level in search results without needing configuration changes.

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
    "create_default_chunk_query",
    # Configuration constants
    "STANDARD_METADATA_FIELDS",
]
