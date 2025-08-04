#!/usr/bin/env python3

"""
PostgreSQL + pgvector backend implementation.

This module wraps the existing PostgreSQL/pgvector functionality from hybrid_search.py
into the standardized VectorStoreBackend interface.
"""

from ast import FunctionDef, FunctionType
import json
from typing import Any, Dict, List, Union, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from . import VectorStoreBackend, QueryFilters
from ..schemas import ChunkMetadata, SearchResult, SearchResultType, validate_chunk_metadata
from ..mappers import PostgresFieldMapper, ResultMapper, CONST_SELECTABLE_FIELDS
from ..keyword_search_parser_lark import build_sql_where_clause
from ..lang.python.python_code_analyzer import analyze_python_code


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
        
    def _build_select_clause(self, include_distance: bool = False, distance_alias: str = "distance") -> str:
        """Build SELECT clause dynamically from single source of truth."""
        # Use all selectable fields from the consolidated configuration
        fields = []
        for field in CONST_SELECTABLE_FIELDS:
            # Quote PostgreSQL reserved keywords
            if field == "end":
                fields.append('"end"')
            else:
                fields.append(field)
        
        if include_distance:
            fields.append(f"embedding <=> %s AS {distance_alias}")
            
        return ", ".join(fields)
    
    def vector_search(
        self, 
        query_vector: NDArray[np.float32], 
        top_k: int = 10
    ) -> List[SearchResult]:
        """Perform pure vector similarity search using pgvector."""
        with self.pool.connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                select_clause = self._build_select_clause(include_distance=True, distance_alias="distance")
                cur.execute(
                    f"""
                    SELECT {select_clause}
                    FROM {self.table_name}
                    ORDER BY distance
                    LIMIT %s
                    """,
                    (query_vector, top_k),
                )
                return [
                    self._format_result(row, score_type="vector") 
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
                select_clause = self._build_select_clause()
                cur.execute(
                    f"""
                    SELECT {select_clause}, 0.0 as distance
                    FROM {self.table_name}
                    WHERE {where_clause}
                    ORDER BY filename, start
                    LIMIT %s
                    """,
                    params + [top_k],
                )
                return [
                    self._format_result(row, score_type="keyword") 
                    for row in cur.fetchall()
                ]
    
    def hybrid_search(
        self,
        query_vector: NDArray[np.float32],
        filters: QueryFilters,
        top_k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword search."""
        where_clause, params = self._build_where_clause(filters)
        
        with self.pool.connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                # Hybrid search: vector similarity with keyword filtering  
                select_clause = self._build_select_clause()
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
                               (vector_similarity * %s) AS hybrid_score
                        FROM vector_scores
                    )
                    SELECT *, hybrid_score
                    FROM ranked_results
                    ORDER BY hybrid_score DESC
                    LIMIT %s
                    """,
                    [query_vector, query_vector] + params + [vector_weight, top_k],
                )
                return [
                    self._format_result(row, score_type="hybrid") 
                    for row in cur.fetchall()
                ]
    
    def configure(self, **options: Any) -> None:
        """Configure PostgreSQL-specific options."""
        # For now, configuration is handled through the connection pool
        # Future: Could support connection pool size, query timeouts, etc.
        pass
    
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
                    ]
                }
    
    def close(self) -> None:
        """Close PostgreSQL connection pool."""
        if hasattr(self.pool, 'close'):
            self.pool.close()
    
    def _build_where_clause(self, filters: QueryFilters) -> Tuple[str, List[Any]]:
        """Convert QueryFilters to PostgreSQL WHERE clause."""
        # Create a mock search group compatible with existing parser
        from ..keyword_search_parser_lark import SearchGroup as LegacySearchGroup
        
        class MockSearchGroup:
            def __init__(self, conditions: List[Any]) -> None:
                self.conditions = conditions
                self.operator = "and"
        
        search_group = MockSearchGroup(filters.conditions)
        return build_sql_where_clause(search_group)  # type: ignore
    
    def _format_result(self, row: Tuple[Any, ...], score_type: str = "vector") -> SearchResult:
        """Format PostgreSQL database row into SearchResult using dynamic field mapping."""
        # Get ordered list of selected fields to map row values
        selectable_fields = list(CONST_SELECTABLE_FIELDS)
        
        # Build dictionary from row values using dynamic field mapping
        pg_row = {}
        for i, field in enumerate(selectable_fields):
            if i < len(row):
                pg_row[field] = row[i]
        
        # Handle distance/score fields based on search type
        if score_type == "hybrid":
            # Row includes hybrid_score at the end
            score = float(row[-1]) if len(row) > len(selectable_fields) else 1.0
        elif score_type == "vector":
            # Row includes distance field 
            distance_idx = len(selectable_fields)  # distance is after selectable fields
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
            pg_row["location"] = f"{pg_row['filename']}:{pg_row['start']}-{pg_row['end']}" if pg_row.get('start') and pg_row.get('end') else pg_row['filename']
        
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
