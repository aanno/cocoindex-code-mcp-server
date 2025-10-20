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
Database-agnostic query abstraction for code chunks.

This module provides a unified query interface that can work across different
vector database backends while maintaining backend-specific optimizations.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .mappers import MapperFactory
from .schemas import ChunkQuery, FilterOperator, QueryFilter, QueryType
from .schemas import SearchResult as SchemaSearchResult
from .schemas import SearchResultType

if TYPE_CHECKING:
    from .backends import VectorStoreBackend


@dataclass
class QueryBuilder:
    """
    Builder pattern for constructing database-agnostic queries.

    Provides a fluent interface for building complex queries that work
    across different vector store backends.
    """

    _text: Optional[str] = None
    _query_type: QueryType = QueryType.HYBRID
    _top_k: int = 10
    _filters: List[QueryFilter] = field(default_factory=list)
    _filter_logic: str = "AND"
    _vector_weight: float = 0.7
    _keyword_weight: float = 0.3
    _embedding: Optional[NDArray[np.float32]] = None

    def text(self, query_text: str) -> "QueryBuilder":
        """Set the text query for vector/semantic search."""
        self._text = query_text
        return self

    def vector_search(self) -> "QueryBuilder":
        """Set query type to pure vector search."""
        self._query_type = QueryType.VECTOR
        return self

    def keyword_search(self) -> "QueryBuilder":
        """Set query type to pure keyword search."""
        self._query_type = QueryType.KEYWORD
        return self

    def hybrid_search(self, vector_weight: float = 0.7, keyword_weight: float = 0.3) -> "QueryBuilder":
        """Set query type to hybrid search with custom weights."""
        self._query_type = QueryType.HYBRID
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        return self

    def limit(self, top_k: int) -> "QueryBuilder":
        """Set maximum number of results to return."""
        self._top_k = top_k
        return self

    def filter_by(self, field: str, operator: FilterOperator, value: Any) -> "QueryBuilder":
        """Add a filter condition."""
        self._filters.append(QueryFilter(field=field, operator=operator, value=value))
        return self

    def where(self, field: str, value: Any) -> "QueryBuilder":
        """Add an equality filter (convenience method)."""
        return self.filter_by(field, FilterOperator.EQUALS, value)

    def where_language(self, language: str) -> "QueryBuilder":
        """Filter by programming language (convenience method)."""
        return self.where("language", language)

    def where_filename_like(self, pattern: str) -> "QueryBuilder":
        """Filter by filename pattern (convenience method)."""
        return self.filter_by("filename", FilterOperator.ILIKE, f"%{pattern}%")

    def where_has_functions(self) -> "QueryBuilder":
        """Filter for chunks that contain functions (convenience method)."""
        return self.filter_by("functions", FilterOperator.IS_NOT_NULL, None)

    def where_complexity_greater_than(self, score: int) -> "QueryBuilder":
        """Filter by minimum complexity score (convenience method)."""
        return self.filter_by("complexity_score", FilterOperator.GREATER_THAN, score)

    def with_type_hints(self) -> "QueryBuilder":
        """Filter for code with type hints (convenience method)."""
        return self.where("has_type_hints", True)

    def with_async_code(self) -> "QueryBuilder":
        """Filter for code with async functions (convenience method)."""
        return self.where("has_async", True)

    #    def where_analysis_method(self, analysis_method: str) -> 'QueryBuilder':
    #        return self.where("analysis_method", analysis_method)

    def where_chunking_method(self, chunking_method: str) -> "QueryBuilder":
        return self.where("chunking_method", chunking_method)

    def where_tree_sitter_analyze_error(self) -> "QueryBuilder":
        return self.where("tree_sitter_analyze_error", True)

    def where_tree_sitter_chunking_error(self) -> "QueryBuilder":
        return self.where("tree_sitter_chunking_error", True)

    def where_has_docstrings(self) -> "QueryBuilder":
        return self.where("has_docstrings", True)

    def where_docstring(self, docstring: str) -> "QueryBuilder":
        return self.where("docstring", docstring)

    def contains_decorator_used(self, decorator_used: str) -> "QueryBuilder":
        return self.filter_by(decorator_used, FilterOperator.IN, "decorators_used")

    def contains_decorator(self, decorator: str) -> "QueryBuilder":
        return self.filter_by(decorator, FilterOperator.IN, "decorators")

    # TODO: dunder_methods, private_methods, variables, function_details, class_details

    def filter_logic_and(self) -> "QueryBuilder":
        """Use AND logic for combining filters."""
        self._filter_logic = "AND"
        return self

    def filter_logic_or(self) -> "QueryBuilder":
        """Use OR logic for combining filters."""
        self._filter_logic = "OR"
        return self

    def with_embedding(self, embedding: NDArray[np.float32]) -> "QueryBuilder":
        """Provide pre-computed embedding for vector search."""
        self._embedding = embedding
        return self

    def build(self) -> ChunkQuery:
        """Build the final ChunkQuery."""
        return ChunkQuery(
            text=self._text,
            query_type=self._query_type,
            top_k=self._top_k,
            filters=self._filters,
            filter_logic=self._filter_logic,  # type: ignore
            vector_weight=self._vector_weight,
            keyword_weight=self._keyword_weight,
            embedding=self._embedding,
        )


class QueryExecutor:
    """
    Executes database-agnostic queries against specific backends.

    Translates ChunkQuery objects into backend-specific operations
    while maintaining consistent result format.
    """

    def __init__(
        self,
        backend: "VectorStoreBackend",
        embedding_func: Optional[Callable[[str], Union[NDArray[np.floating], object]]] = None,
    ):
        """
        Initialize with a specific backend.

        Args:
            backend: Vector store backend to execute queries against
            embedding_func: Optional function to convert text to embeddings
        """
        self.backend = backend
        self.mapper = MapperFactory.create_mapper(backend.__class__.__name__.replace("Backend", "").lower())
        self.embedding_func = embedding_func

    async def execute(self, query: ChunkQuery) -> List[SchemaSearchResult]:
        """
        Execute a query against the backend.

        Args:
            query: Database-agnostic query to execute

        Returns:
            List of search results in standardized format

        Raises:
            ValueError: If query is invalid for the backend
            RuntimeError: If query execution fails
        """
        try:
            if query["query_type"] == QueryType.VECTOR:
                return await self._execute_vector_search(query)
            elif query["query_type"] == QueryType.KEYWORD:
                return await self._execute_keyword_search(query)
            elif query["query_type"] == QueryType.HYBRID:
                return await self._execute_hybrid_search(query)
            else:
                raise ValueError(f"Unknown query type: {query['query_type']}")

        except Exception as e:
            raise RuntimeError(f"Query execution failed: {e}") from e

    async def _execute_vector_search(self, query: ChunkQuery) -> List[SchemaSearchResult]:
        """Execute pure vector similarity search."""
        embedding: Optional[Union[NDArray[np.floating], object]] = query.get("embedding")
        if embedding is None:
            if query.get("text") is None:
                raise ValueError("Vector search requires either embedding or text")
            # Convert text to embedding if embedding function is available
            if self.embedding_func is None:
                raise ValueError("No embedding function provided for text-to-embedding conversion")
            text = query.get("text")
            if text is None:
                raise ValueError("Vector search requires either embedding or text")
            embedding_result = self.embedding_func(text)
            # Handle the embedding function result which could be numpy array or other object
            embedding = embedding_result

        # Use backend's vector search
        # Convert embedding to numpy array if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        backend_results = self.backend.vector_search(query_vector=embedding, top_k=query.get("top_k", 10))

        # Convert backend results to schema results
        results = []
        for backend_result in backend_results:
            schema_result = SchemaSearchResult(
                filename=backend_result.filename,
                language=backend_result.language,
                code=backend_result.code,
                location=backend_result.source,  # Map source to location
                start=backend_result.start,
                end=backend_result.end,
                score=backend_result.score,
                score_type=SearchResultType.VECTOR_SIMILARITY,
                source=backend_result.source,
                metadata=backend_result.metadata,
            )
            results.append(schema_result)

        return results

    async def _execute_keyword_search(self, query: ChunkQuery) -> List[SchemaSearchResult]:
        """Execute pure keyword/metadata search."""
        from .backends import QueryFilters

        # Convert query filters to backend format
        filter_conditions = []
        for qf in query.get("filters", []):
            # Convert our QueryFilter to the backend's SearchCondition format
            # This is a simplified conversion - real implementation would be more robust
            filter_conditions.append(qf)

        query_filters = QueryFilters(
            conditions=filter_conditions,  # type: ignore
            operator=query.get("filter_logic", "AND"),
        )

        backend_results = self.backend.keyword_search(filters=query_filters, top_k=query.get("top_k", 10))

        # Convert backend results to schema results
        results = []
        for backend_result in backend_results:
            schema_result = SchemaSearchResult(
                filename=backend_result.filename,
                language=backend_result.language,
                code=backend_result.code,
                location=backend_result.source,  # Map source to location
                start=backend_result.start,
                end=backend_result.end,
                score=backend_result.score,
                score_type=SearchResultType.KEYWORD_MATCH,
                source=backend_result.source,
                metadata=backend_result.metadata,
            )
            results.append(schema_result)

        return results

    async def _execute_hybrid_search(self, query: ChunkQuery) -> List[SchemaSearchResult]:
        """Execute hybrid search combining vector and keyword search."""
        from .backends import QueryFilters

        embedding = query.get("embedding")
        if embedding is None:
            if query.get("text") is None:
                raise ValueError("Hybrid search requires either embedding or text")
            # Would need to compute embedding from text here
            raise NotImplementedError("Text-to-embedding conversion not implemented yet")

        # Convert filters
        filter_conditions = []
        for qf in query.get("filters", []):
            filter_conditions.append(qf)

        query_filters = QueryFilters(
            conditions=filter_conditions,  # type: ignore
            operator=query.get("filter_logic", "AND"),
        )

        backend_results = self.backend.hybrid_search(
            query_vector=embedding,
            filters=query_filters,
            top_k=query.get("top_k", 10),
            vector_weight=query.get("vector_weight", 0.7),
            keyword_weight=query.get("keyword_weight", 0.3),
        )

        # Convert backend results to schema results
        results = []
        for backend_result in backend_results:
            schema_result = SchemaSearchResult(
                filename=backend_result.filename,
                language=backend_result.language,
                code=backend_result.code,
                location=backend_result.source,  # Map source to location
                start=backend_result.start,
                end=backend_result.end,
                score=backend_result.score,
                score_type=SearchResultType.HYBRID_COMBINED,
                source=backend_result.source,
                metadata=backend_result.metadata,
            )
            results.append(schema_result)

        return results


class QueryOptimizer:
    """
    Optimizes queries for specific backends.

    Applies backend-specific optimizations and query transformations
    to improve performance.
    """

    def __init__(self, backend_type: str):
        """
        Initialize optimizer for specific backend type.

        Args:
            backend_type: Type of backend ("postgres", "qdrant", etc.)
        """
        self.backend_type = backend_type.lower()

    def optimize_query(self, query: ChunkQuery) -> ChunkQuery:
        """
        Optimize query for the target backend.

        Args:
            query: Original query to optimize

        Returns:
            Optimized query
        """
        if self.backend_type == "postgres":
            return self._optimize_for_postgres(query)
        elif self.backend_type == "qdrant":
            return self._optimize_for_qdrant(query)
        else:
            return query  # No optimization for unknown backends

    def _optimize_for_postgres(self, query: ChunkQuery) -> ChunkQuery:
        """Apply PostgreSQL-specific optimizations."""
        optimized = query.copy()

        # PostgreSQL is better at full-text search, adjust hybrid weights
        if query.get("query_type") == QueryType.HYBRID:
            # Favor keyword search slightly more for PostgreSQL
            optimized["vector_weight"] = max(0.6, query.get("vector_weight", 0.7) - 0.1)
            optimized["keyword_weight"] = min(0.4, query.get("keyword_weight", 0.3) + 0.1)

        # Add PostgreSQL-specific filter optimizations
        # e.g., prefer indexed columns for filtering
        return optimized

    def _optimize_for_qdrant(self, query: ChunkQuery) -> ChunkQuery:
        """Apply Qdrant-specific optimizations."""
        optimized = query.copy()

        # Qdrant excels at vector search, adjust weights accordingly
        if query.get("query_type") == QueryType.HYBRID:
            # Favor vector search more for Qdrant
            optimized["vector_weight"] = min(0.8, query.get("vector_weight", 0.7) + 0.1)
            optimized["keyword_weight"] = max(0.2, query.get("keyword_weight", 0.3) - 0.1)

        return optimized


# =============================================================================
# Convenience Functions
# =============================================================================


def create_query() -> QueryBuilder:
    """Create a new query builder."""
    return QueryBuilder()


def simple_search(text: str, top_k: int = 10) -> ChunkQuery:
    """Create a simple hybrid search query."""
    return create_query().text(text).hybrid_search().limit(top_k).build()


def find_functions_in_language(language: str, top_k: int = 10) -> ChunkQuery:
    """Find code chunks with functions in a specific language."""
    return create_query().keyword_search().where_language(language).where_has_functions().limit(top_k).build()


def find_complex_code(min_complexity: int = 5, top_k: int = 10) -> ChunkQuery:
    """Find complex code chunks."""
    return create_query().keyword_search().where_complexity_greater_than(min_complexity).limit(top_k).build()


def find_async_python_code(top_k: int = 10) -> ChunkQuery:
    """Find Python code with async functions."""
    return create_query().keyword_search().where_language("Python").with_async_code().limit(top_k).build()


def semantic_search_with_filters(
    text: str, language: Optional[str] = None, min_complexity: Optional[int] = None, top_k: int = 10
) -> ChunkQuery:
    """Create a semantic search with optional filters."""
    builder = create_query().text(text).hybrid_search().limit(top_k)

    if language:
        builder = builder.where_language(language)

    if min_complexity is not None:
        builder = builder.where_complexity_greater_than(min_complexity)

    return builder.build()


# =============================================================================
# Export for convenience
# =============================================================================

__all__ = [
    "QueryBuilder",
    "QueryExecutor",
    "QueryOptimizer",
    # Convenience functions
    "create_query",
    "simple_search",
    "find_functions_in_language",
    "find_complex_code",
    "find_async_python_code",
    "semantic_search_with_filters",
]
