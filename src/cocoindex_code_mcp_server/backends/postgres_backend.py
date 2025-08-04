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
from ..mappers import PostgresFieldMapper, ResultMapper
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
        """Format PostgreSQL database row into SearchResult using new schema system."""
        # Handle different row formats based on score type
        if score_type == "hybrid":
            filename, language, code, vector_distance, start, end, source_name, hybrid_score = row
            score = float(hybrid_score)
        else:
            filename, language, code, distance, start, end, source_name = row
            score = 1.0 - float(distance) if score_type == "vector" else 1.0
        
        # Create PostgreSQL row dict compatible with our mapper
        pg_row = {
            "filename": filename,
            "language": language,
            "code": code,
            "start": start,
            "end": end,
            "source_name": source_name or "default",
            "location": f"{filename}:{start}-{end}" if start and end else filename,
        }
        
        # Add rich metadata using multi-language analysis
        try:
            # Import the extract_code_metadata function to use our multi-language analyzers
            from ..cocoindex_config import extract_code_metadata
            
            # Extract metadata using our multi-language analyzers
            metadata_json_str = extract_code_metadata(code, language, filename)
            analysis_metadata = json.loads(metadata_json_str)
            
            if analysis_metadata is not None:
                # Add analyzed metadata to the row
                pg_row.update({
                    "functions": analysis_metadata.get("functions", []),
                    "classes": analysis_metadata.get("classes", []),
                    "imports": analysis_metadata.get("imports", []),
                    "complexity_score": analysis_metadata.get("complexity_score", 0),
                    "has_type_hints": analysis_metadata.get("has_type_hints", False),
                    "has_async": analysis_metadata.get("has_async", False),
                    "has_classes": analysis_metadata.get("has_classes", False),
                    "metadata_json": json.dumps(analysis_metadata, default=str)
                })
            else:
                # Add default metadata fields
                pg_row.update({
                    "functions": [],
                    "classes": [],
                    "imports": [],
                    "complexity_score": 0,
                    "has_type_hints": False,
                    "has_async": False,
                    "has_classes": False,
                    "metadata_json": json.dumps({"analysis_error": "metadata was None"}),
                    "analysis_method": None,
                    "chunking_method": None,
                    "tree_sitter_analyze_error:": False,
                    "tree_sitter_chunking_error": False,
                    "has_docstrings": False,
                    "docstring": "",
                    "decorators_used": [],
                    "dunder_methods": [],
                    "private_methods": [],
                    "variables": [],
                    "decorators": [],
                    "function_details": json.dumps({"error": "no function_details"}),
                    "class_details": json.dumps({"error": "no class_details"})
                })
        except Exception as e:
            # Add error metadata
            pg_row.update({
                "functions": [],
                "classes": [],
                "imports": [],
                "complexity_score": 0,
                "has_type_hints": False,
                "has_async": False,
                "has_classes": False,
                "metadata_json": json.dumps({"analysis_error": str(e)}),
                "analysis_method": None,
                "chunking_method": None,
                "tree_sitter_analyze_error:": False,
                "tree_sitter_chunking_error": False,
                "has_docstrings": False,
                "docstring": "",
                "decorators_used": [],
                "dunder_methods": [],
                "private_methods": [],
                "variables": [],
                "decorators": [],
                "function_details": json.dumps({"error": str(e)}),
                "class_details": json.dumps({"error": str(e)})
            })
        
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
