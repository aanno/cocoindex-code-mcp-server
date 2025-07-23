#!/usr/bin/env python3

"""
Hybrid search implementation combining vector similarity and keyword metadata search.
"""

import os
import json
from typing import Any, Dict, List, Optional
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
import cocoindex
from cocoindex_config import code_embedding_flow, code_to_embedding
from keyword_search_parser_lark import KeywordSearchParser, build_sql_where_clause
from lang.python.python_code_analyzer import analyze_python_code
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings


class HybridSearchEngine:
    """Hybrid search engine combining vector and keyword search."""
    
    def __init__(self, pool: ConnectionPool, table_name: str = None, parser: KeywordSearchParser = None, embedding_func=None):
        self.pool = pool
        self.parser = parser or KeywordSearchParser()
        self.table_name = table_name or cocoindex.utils.get_target_default_name(
            code_embedding_flow, "code_embeddings"
        )
        self.embedding_func = embedding_func or (lambda q: code_to_embedding.eval(q))
    
    def search(
        self,
        vector_query: str,
        keyword_query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and keyword filtering.
        
        Args:
            vector_query: Text to embed and search for semantic similarity
            keyword_query: Keyword search query for metadata filtering
            top_k: Number of results to return
            vector_weight: Weight for vector similarity score (0-1)
            keyword_weight: Weight for keyword match score (0-1)
            
        Returns:
            List of search results with combined scoring
        """
        # Parse keyword query
        search_group = self.parser.parse(keyword_query)
        
        # Build the SQL query
        if vector_query.strip() and search_group and search_group.conditions:
            # Both vector and keyword search
            return self._hybrid_search(vector_query, search_group, top_k, vector_weight, keyword_weight)
        elif vector_query.strip():
            # Vector search only
            return self._vector_search(vector_query, top_k)
        elif search_group and search_group.conditions:
            # Keyword search only
            return self._keyword_search(search_group, top_k)
        else:
            # No valid query
            return []
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform pure vector similarity search."""
        query_vector = self.embedding_func(query)
        
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
                return [self._format_result(row, score_type="vector") for row in cur.fetchall()]
    
    def _keyword_search(self, search_group, top_k: int) -> List[Dict[str, Any]]:
        """Perform pure keyword/metadata search."""
        where_clause, params = build_sql_where_clause(search_group)
        
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
                return [self._format_result(row, score_type="keyword") for row in cur.fetchall()]
    
    def _hybrid_search(
        self, 
        vector_query: str, 
        search_group, 
        top_k: int, 
        vector_weight: float, 
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and keyword search."""
        query_vector = self.embedding_func(vector_query)
        where_clause, params = build_sql_where_clause(search_group)
        
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
                return [self._format_result(row, score_type="hybrid") for row in cur.fetchall()]
    
    def _format_result(self, row, score_type: str = "vector") -> Dict[str, Any]:
        """Format database row into result dictionary."""
        if score_type == "hybrid":
            filename, language, code, vector_distance, start, end, source_name, hybrid_score = row
            score = float(hybrid_score)
        else:
            filename, language, code, distance, start, end, source_name = row
            score = 1.0 - float(distance) if score_type == "vector" else 1.0
        
        # Build base result
        result = {
            "filename": filename,
            "language": language,
            "code": code,
            "score": score,
            "start": start,
            "end": end,
            "source": source_name or "default",
            "score_type": score_type
        }
        
        # Add rich metadata for Python code
        if language == "Python":
            try:
                metadata = analyze_python_code(code, filename)
                result.update({
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
                    "metadata_json": json.dumps(metadata, default=str)
                })
            except Exception as e:
                # Fallback: add basic metadata fields even if analysis fails
                result.update({
                    "functions": [],
                    "classes": [],
                    "imports": [],
                    "complexity_score": 0,
                    "has_type_hints": False,
                    "has_async": False,
                    "has_classes": False,
                    "analysis_error": str(e)
                })
        
        return result


def format_results_as_json(results: List[Dict[str, Any]], indent: int = 2) -> str:
    """Format search results as JSON string with human-readable code and metadata_json fields."""
    
    def format_single_result(result: Dict[str, Any], indent_level: int = 1) -> str:
        """Format a single result with custom handling for code and metadata_json fields."""
        lines = ["{"]
        
        for i, (key, value) in enumerate(result.items()):
            is_last = i == len(result) - 1
            comma = "" if is_last else ","
            indent_str = "  " * indent_level
            
            if key in ['code', 'metadata_json'] and isinstance(value, str):
                # For code and metadata_json, output the raw string without JSON escaping
                # Use triple quotes to preserve formatting
                formatted_value = f'"""{value}"""'
                lines.append(f'{indent_str}"{key}": {formatted_value}{comma}')
            else:
                # For other fields, use normal JSON formatting
                formatted_value = json.dumps(value, default=str)
                lines.append(f'{indent_str}"{key}": {formatted_value}{comma}')
        
        lines.append("}")
        return "\n".join(lines)
    
    # Format the entire results array
    if not results:
        return "[]"
    
    output_lines = ["["]
    for i, result in enumerate(results):
        is_last = i == len(results) - 1
        comma = "" if is_last else ","
        
        # Format single result with proper indentation
        formatted_result = format_single_result(result, indent_level=2)
        # Indent the entire result block
        indented_result = "\n".join(f"  {line}" for line in formatted_result.split("\n"))
        output_lines.append(f"{indented_result}{comma}")
    
    output_lines.append("]")
    return "\n".join(output_lines)


def format_results_readable(results: List[Dict[str, Any]]) -> str:
    """Format search results in human-readable format."""
    if not results:
        return "No results found."
    
    output = [f"📊 Found {len(results)} results:\n"]
    
    for i, result in enumerate(results, 1):
        source_info = f" [{result['source']}]" if result.get('source') and result['source'] != 'files' else ""
        score_info = f" ({result['score_type']})" if result.get('score_type') else ""
        
        # Basic result info
        output.append(
            f"{i}. [{result['score']:.3f}]{score_info} {result['filename']}{source_info} "
            f"({result['language']}) (L{result['start']['line']}-L{result['end']['line']})"
        )
        
        # Add Python metadata if available
        if result['language'] == 'Python' and any(key in result for key in ['functions', 'classes', 'imports']):
            metadata_parts = []
            
            if result.get('functions'):
                metadata_parts.append(f"Functions: {', '.join(result['functions'])}")
            if result.get('classes'):
                metadata_parts.append(f"Classes: {', '.join(result['classes'])}")
            if result.get('imports'):
                metadata_parts.append(f"Imports: {', '.join(result['imports'][:3])}")  # Show first 3
            if result.get('decorators'):
                metadata_parts.append(f"Decorators: {', '.join(result['decorators'])}")
            
            # Add type hints and async info
            info_parts = []
            if result.get('has_type_hints'):
                info_parts.append("typed")
            if result.get('has_async'):
                info_parts.append("async")
            if result.get('complexity_score', 0) > 10:
                info_parts.append(f"complex({result['complexity_score']})")
            
            if metadata_parts:
                output.append(f"   📝 {' | '.join(metadata_parts)}")
            if info_parts:
                output.append(f"   🏷️  {', '.join(info_parts)}")
        
        output.append(f"   {result['code']}")
        output.append("   ---")
    
    return "\n".join(output)


def get_multiline_input(prompt_text: str) -> str:
    """Get multiline input using prompt_toolkit with Ctrl+Q to finish."""
    bindings = KeyBindings()
    
    @bindings.add('c-q')  # Ctrl+Q
    def _(event):
        event.app.exit(result=event.app.current_buffer.text)
    
    session = PromptSession(key_bindings=bindings, multiline=True)
    
    print(f"{prompt_text} (finish with Ctrl+Q):")
    try:
        result = session.prompt()
        return result.strip()
    except (KeyboardInterrupt, EOFError):
        return ""


def run_interactive_hybrid_search():
    """Run interactive hybrid search mode with dual prompts."""
    # Initialize the database connection pool
    pool = ConnectionPool(os.getenv("COCOINDEX_DATABASE_URL"))
    search_engine = HybridSearchEngine(pool)
    
    print("\n🔍 Interactive Hybrid Search Mode")
    print("Enter two types of queries:")
    print("  1. Vector query: text for semantic similarity search")
    print("  2. Keyword query: metadata search with syntax like 'language:python and exists(embedding)'")
    print("Both queries are combined with AND logic.")
    print("Press Enter with empty vector query to quit.\n")
    
    print("📝 Keyword search syntax:")
    print("  - field:value (e.g., language:python, filename:main_interactive_query.py)")
    print("  - exists(field) (e.g., exists(embedding))")
    print("  - and/or operators (e.g., language:python and filename:main_interactive_query.py)")
    print("  - parentheses for grouping (e.g., (language:python or language:rust) and exists(embedding))")
    print("  - quoted values for spaces (e.g., filename:\"test file.py\")")
    print()
    
    while True:
        try:
            # Get vector query
            vector_query = get_multiline_input("Vector query (semantic search)")
            if not vector_query:
                break
            
            # Get keyword query
            keyword_query = input("Keyword query (metadata filter): ").strip()
            
            # Perform search
            print("\n🔄 Searching...")
            results = search_engine.search(
                vector_query=vector_query,
                keyword_query=keyword_query,
                top_k=10
            )
            
            # Output results - detect if they're JSON-like and format accordingly
            if results:
                # Check if any result contains complex nested data that suggests JSON output
                has_complex_data = any(
                    isinstance(result.get('start'), dict) or isinstance(result.get('end'), dict)
                    for result in results
                )
                
                if has_complex_data:
                    # Output as JSON for complex data
                    print(format_results_as_json(results))
                else:
                    # Output in readable format for simple data
                    print(format_results_readable(results))
            else:
                print("No results found.")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()


if __name__ == "__main__":
    run_interactive_hybrid_search()
