#!/usr/bin/env python3

"""
PostgreSQL + pgvector backend implementation.

This module wraps the existing PostgreSQL/pgvector functionality from hybrid_search.py
into the standardized VectorStoreBackend interface.
"""

import logging
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
from cachetools import TTLCache, cached
from numpy.typing import NDArray
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from ..keyword_search_parser_lark import build_sql_where_clause, Operator
from ..mappers import CONST_SELECTABLE_FIELDS, PostgresFieldMapper, ResultMapper
from ..schemas import (
    SearchResult,
    SearchResultType,
)
from . import QueryFilters, VectorStoreBackend

logger = logging.getLogger(__name__)

# Cache for database column information (60 second TTL, max 100 entries)
column_cache = TTLCache(maxsize=100, ttl=60)


@cached(column_cache)
def _get_table_columns(pool: ConnectionPool, table_name: str) -> Set[str]:
    """
    Get available columns from database table with caching.

    Args:
        pool: PostgreSQL connection pool
        table_name: Name of the table to inspect

    Returns:
        Set of available column names
    """
    # Query database for available columns
    # PostgreSQL stores table names in lowercase, so try both cases
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # First try with the exact table name provided
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                """,
                (table_name,)
            )
            available_columns = {row[0] for row in cur.fetchall()}

            # If no columns found, try with lowercase table name
            if not available_columns:
                lowercase_table_name = table_name.lower()
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = %s
                    """,
                    (lowercase_table_name,)
                )
                available_columns = {row[0] for row in cur.fetchall()}

                # Return both the columns and indicate if we used lowercase
                if available_columns:
                    logger.info(f"Table name '{table_name}' not found, using lowercase '{lowercase_table_name}'")

    return available_columns


class PostgresBackend(VectorStoreBackend):
    """PostgreSQL + pgvector backend implementation."""

    def __init__(self, pool: ConnectionPool, table_name: str) -> None:
        """
        Initialize PostgreSQL backend.

        Args:
            pool: PostgreSQL connection pool
            table_name: Name of the table containing vector embeddings
        """
        self.pool = pool
        self.table_name = table_name
        self.mapper = PostgresFieldMapper()
        self._columns_warned: Set[str] = set()  # Track which missing columns we've already warned about

    def _get_available_columns(self) -> Set[str]:
        """Get available columns from database with caching."""
        # Use the cached function to get columns
        available_columns = _get_table_columns(self.pool, self.table_name)

        # If no columns found, try with lowercase table name and update instance
        if not available_columns:
            lowercase_table_name = self.table_name.lower()
            available_columns = _get_table_columns(self.pool, lowercase_table_name)

            # If we found columns with lowercase, update the table name for future queries
            if available_columns:
                logger.info(f"Table name '{self.table_name}' not found, using lowercase '{lowercase_table_name}'")
                self.table_name = lowercase_table_name

        return available_columns

    def _build_select_clause(self, include_distance: bool = False,
                             distance_alias: str = "distance") -> Tuple[str, List[str]]:
        """Build SELECT clause dynamically using only available DB columns."""
        available_columns = self._get_available_columns()

        # Filter selectable fields to only those that exist in the database
        fields = []
        available_fields = []
        missing_fields = []

        for field in CONST_SELECTABLE_FIELDS:
            if field in available_columns:
                # Quote PostgreSQL reserved keywords
                if field == "end":
                    fields.append('"end"')
                else:
                    fields.append(field)
                available_fields.append(field)
            else:
                missing_fields.append(field)

        # Log warning for missing fields (only once per field)
        new_missing_fields = set(missing_fields) - self._columns_warned
        if new_missing_fields:
            logger.warning(
                f"Expected columns missing from table '{self.table_name}': {sorted(new_missing_fields)}. "
                f"Query will be restricted to available columns: {sorted(available_fields)}"
            )
            self._columns_warned.update(new_missing_fields)

        if include_distance:
            fields.append(f"embedding <=> %s AS {distance_alias}")

        return ", ".join(fields), available_fields

    def vector_search(
        self,
        query_vector: NDArray[np.float32],
        top_k: int = 10,
        embedding_model: str | None = None
    ) -> List[SearchResult]:
        """
        Perform pure vector similarity search using pgvector.

        CRITICAL: embedding_model parameter ensures we only compare vectors from the same model.
        You cannot compare embedding vectors created with different models!

        Args:
            query_vector: The query embedding vector
            top_k: Number of results to return
            embedding_model: Filter to only search embeddings from this model (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        """
        with self.pool.connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                select_clause, available_fields = self._build_select_clause(
                    include_distance=True, distance_alias="distance")

                # Build WHERE clause for embedding_model filter
                where_clause = ""
                params: List[Union[NDArray[np.float32], str, int]] = [query_vector]
                if embedding_model:
                    where_clause = "WHERE embedding_model = %s"
                    params.append(embedding_model)

                params.append(top_k)

                cur.execute(
                    f"""
                    SELECT {select_clause}
                    FROM {self.table_name}
                    {where_clause}
                    ORDER BY distance
                    LIMIT %s
                    """,
                    tuple(params),
                )
                return [
                    self._format_result(row, available_fields, score_type="vector")
                    for row in cur.fetchall()
                ]

    def keyword_search(
        self,
        filters: QueryFilters,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Perform pure keyword/metadata search using PostgreSQL."""
        # Convert QueryFilters to SQL where clause
        where_clause, params = self._build_where_clause(filters)

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                select_clause, available_fields = self._build_select_clause()
                
                query = f"""
                    SELECT {select_clause}, 0.0 as distance
                    FROM {self.table_name}
                    WHERE {where_clause}
                    ORDER BY filename, start
                    LIMIT %s
                    """
                
                query_params = params + [top_k]
                
                # Log the SQL query for debugging
                logger.info(f"🔍 Keyword Search SQL Query:")
                logger.info(f"   Query: {query}")
                logger.info(f"   Params: {query_params}")
                
                cur.execute(query, query_params)
                results = cur.fetchall()
                
                logger.info(f"📊 Query returned {len(results)} results")
                
                return [
                    self._format_result(row, available_fields, score_type="keyword")
                    for row in results
                ]

    def hybrid_search(
        self,
        query_vector: NDArray[np.float32],
        filters: QueryFilters,
        top_k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        embedding_model: str | None = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and keyword search.

        CRITICAL: embedding_model parameter ensures we only compare vectors from the same model.
        You cannot compare embedding vectors created with different models!

        Args:
            query_vector: The query embedding vector
            filters: Keyword/metadata filters
            top_k: Number of results to return
            vector_weight: Weight for vector similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            embedding_model: Filter to only search embeddings from this model
        """
        where_clause, params = self._build_where_clause(filters)

        # CRITICAL: Add embedding_model filter to WHERE clause
        # This ensures we only compare embeddings from the same model
        if embedding_model:
            where_clause = f"({where_clause}) AND embedding_model = %s"
            params.append(embedding_model)

        with self.pool.connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                # Hybrid search: vector similarity with keyword filtering
                select_clause, available_fields = self._build_select_clause()
                cur.execute(
                    f"""
                    WITH vector_scores AS (
                        SELECT {select_clause}, embedding <=> %s AS vector_distance,
                               (1.0 - (embedding <=> %s)) AS vector_similarity
                        FROM {self.table_name}
                        WHERE {where_clause}
                    ),
                    ranked_results AS (
                        SELECT *,
                               (vector_similarity * %s + %s) AS hybrid_score
                        FROM vector_scores
                    )
                    SELECT *, hybrid_score
                    FROM ranked_results
                    ORDER BY hybrid_score DESC
                    LIMIT %s
                    """,
                    [query_vector, query_vector] + params + [vector_weight, keyword_weight, top_k],
                )
                return [
                    self._format_result(row, available_fields, score_type="hybrid")
                    for row in cur.fetchall()
                ]

    def configure(self, **options: Any) -> None:
        """Configure PostgreSQL-specific options."""
        # For now, configuration is handled through the connection pool
        # Future: Could support connection pool size, query timeouts, etc.

    def get_table_info(self) -> Dict[str, Any]:
        """Get information about the PostgreSQL table structure."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                # Get table schema
                cur.execute(
                    """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (self.table_name,)
                )
                columns = cur.fetchall()

                # Get table size
                cur.execute(
                    f"SELECT COUNT(*) FROM {self.table_name}"
                )
                result = cur.fetchone()
                row_count = result[0] if result else 0

                # Get index information
                cur.execute(
                    """
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = %s
                    """,
                    (self.table_name,)
                )
                indexes = cur.fetchall()

                # Get available columns for this table
                available_columns = {col[0] for col in columns}
                expected_columns = CONST_SELECTABLE_FIELDS
                missing_columns = expected_columns - available_columns

                return {
                    "backend_type": "postgres",
                    "table_name": self.table_name,
                    "row_count": row_count,
                    "columns": [
                        {
                            "name": col[0],
                            "type": col[1],
                            "nullable": col[2] == "YES"
                        }
                        for col in columns
                    ],
                    "indexes": [
                        {
                            "name": idx[0],
                            "definition": idx[1]
                        }
                        for idx in indexes
                    ],
                    "schema_info": {
                        "available_columns": sorted(available_columns),
                        "expected_columns": sorted(expected_columns),
                        "missing_columns": sorted(missing_columns)
                    }
                }

    def close(self) -> None:
        """Close PostgreSQL connection pool."""
        if hasattr(self.pool, 'close'):
            self.pool.close()
        # Clear instance-level tracking
        self._columns_warned.clear()
        # Note: Global column_cache is shared across instances and persists

    def _build_where_clause(self, filters: QueryFilters) -> Tuple[str, List[Any]]:
        """Convert QueryFilters to PostgreSQL WHERE clause."""
        # Create a mock search group compatible with existing parser

        class MockSearchGroup:
            def __init__(self, conditions: List[Any], operator: str = "AND") -> None:
                self.conditions = conditions
                # Convert string to Operator enum for compatibility with build_sql_where_clause
                self.operator = Operator.OR if operator.upper() == "OR" else Operator.AND

        search_group = MockSearchGroup(filters.conditions, filters.operator)
        return build_sql_where_clause(search_group)  # type: ignore

    def _format_result(self, row: Tuple[Any, ...], available_fields: List[str],
                       score_type: str = "vector") -> SearchResult:
        """Format PostgreSQL database row into SearchResult using available field mapping."""
        # Build dictionary from row values using only available fields
        pg_row = {}
        for i, field in enumerate(available_fields):
            if i < len(row):
                pg_row[field] = row[i]

        # Handle distance/score fields based on search type
        if score_type == "hybrid":
            # Row includes hybrid_score at the end
            score = float(row[-1]) if len(row) > len(available_fields) else 1.0
        elif score_type == "vector":
            # Row includes distance field
            distance_idx = len(available_fields)  # distance is after available fields
            if len(row) > distance_idx:
                distance = float(row[distance_idx])
                score = 1.0 - distance
            else:
                score = 1.0
        else:  # keyword
            # Row includes distance field (0.0)
            score = 1.0

        # Ensure required fields have defaults
        pg_row.setdefault("source_name", "default")
        if "location" not in pg_row and "filename" in pg_row and "start" in pg_row and "end" in pg_row:
            pg_row["location"] = f"{
                pg_row['filename']}:{
                pg_row['start']}-{
                pg_row['end']}" if pg_row.get('start') and pg_row.get('end') else pg_row['filename']

        # Convert score_type string to SearchResultType enum
        if score_type == "vector":
            result_type = SearchResultType.VECTOR_SIMILARITY
        elif score_type == "keyword":
            result_type = SearchResultType.KEYWORD_MATCH
        elif score_type == "hybrid":
            result_type = SearchResultType.HYBRID_COMBINED
        else:
            result_type = SearchResultType.VECTOR_SIMILARITY

        # Use ResultMapper to convert to standardized SearchResult
        return ResultMapper.from_postgres_result(pg_row, score, result_type)
