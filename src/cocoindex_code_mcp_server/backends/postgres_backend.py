#!/usr/bin/env python3

"""
PostgreSQL + pgvector backend implementation.

This module wraps the existing PostgreSQL/pgvector functionality from hybrid_search.py
into the standardized VectorStoreBackend interface.
"""

import json
from typing import Any, Dict, List, Union, Tuple

import numpy as np
from numpy.typing import NDArray
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from . import VectorStoreBackend, SearchResult, QueryFilters
from ..keyword_search_parser_lark import build_sql_where_clause
from ..lang.python.python_code_analyzer import analyze_python_code


class PostgresBackend(VectorStoreBackend):
    """PostgreSQL + pgvector backend implementation."""
    
    def __init__(self, pool: ConnectionPool, table_name: str):
        """
        Initialize PostgreSQL backend.
        
        Args:
            pool: PostgreSQL connection pool
            table_name: Name of the table containing vector embeddings
        """
        self.pool = pool
        self.table_name = table_name
    
    def vector_search(
        self, 
        query_vector: NDArray[np.float32], 
        top_k: int = 10
    ) -> List[SearchResult]:
        """Perform pure vector similarity search using pgvector."""
        with self.pool.connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT filename, language, code, embedding <=> %s AS distance,
                           start, "end", source_name
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
                cur.execute(
                    f"""
                    SELECT filename, language, code, 0.0 as distance,
                           start, "end", source_name
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
                cur.execute(
                    f"""
                    WITH vector_scores AS (
                        SELECT filename, language, code, embedding <=> %s AS vector_distance,
                               start, "end", source_name,
                               (1.0 - (embedding <=> %s)) AS vector_similarity
                        FROM {self.table_name}
                        WHERE {where_clause}
                    ),
                    ranked_results AS (
                        SELECT *,
                               (vector_similarity * %s) AS hybrid_score
                        FROM vector_scores
                    )
                    SELECT filename, language, code, vector_distance, start, "end", source_name, hybrid_score
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
                row_count = cur.fetchone()[0]
                
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
        class MockSearchGroup:
            def __init__(self, conditions: List[Dict[str, Any]]):
                self.conditions = conditions
        
        search_group = MockSearchGroup(filters.conditions)
        return build_sql_where_clause(search_group)
    
    def _format_result(self, row: Tuple[Any, ...], score_type: str = "vector") -> SearchResult:
        """Format PostgreSQL database row into SearchResult."""
        if score_type == "hybrid":
            filename, language, code, vector_distance, start, end, source_name, hybrid_score = row
            score = float(hybrid_score)
        else:
            filename, language, code, distance, start, end, source_name = row
            score = 1.0 - float(distance) if score_type == "vector" else 1.0
        
        # Build base result
        result = SearchResult(
            filename=filename,
            language=language,
            code=code,
            score=score,
            start=start,
            end=end,
            source=source_name or "default",
            score_type=score_type
        )
        
        # Add rich metadata for Python code
        if language == "Python":
            try:
                metadata = analyze_python_code(code, filename)
                result.metadata = {
                    "functions": metadata.get("functions", []),
                    "classes": metadata.get("classes", []),
                    "imports": metadata.get("imports", []),
                    "complexity_score": metadata.get("complexity_score", 0),
                    "has_type_hints": metadata.get("has_type_hints", False),
                    "has_async": metadata.get("has_async", False),
                    "has_classes": metadata.get("has_classes", False),
                    "private_methods": metadata.get("private_methods", []),
                    "dunder_methods": metadata.get("dunder_methods", []),
                    "decorators": metadata.get("decorators", []),
                    "analysis_method": metadata.get("analysis_method", "python_ast"),
                    "metadata_json": json.dumps(metadata, default=str)
                }
            except Exception as e:
                # Fallback: add basic metadata fields even if analysis fails
                result.metadata = {
                    "functions": [],
                    "classes": [],
                    "imports": [],
                    "complexity_score": 0,
                    "has_type_hints": False,
                    "has_async": False,
                    "has_classes": False,
                    "analysis_method": "python_ast",
                    "analysis_error": str(e)
                }
        
        return result