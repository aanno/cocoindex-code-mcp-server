#!/usr/bin/env python3

"""
Interactive query functionality for the code embedding pipeline.
"""

import os
from typing import Any, Dict, List

from .cocoindex_config import code_embedding_flow, code_to_embedding
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

import cocoindex


def search(pool: ConnectionPool, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for code using semantic similarity.

    Args:
        pool: Database connection pool
        query: Search query string
        top_k: Number of results to return

    Returns:
        List of search results with metadata
    """
    # Get the table name, for the export target in the code_embedding_flow above.
    table_name = cocoindex.utils.get_target_default_name(
        code_embedding_flow, "code_embeddings"
    )
    # Evaluate the transform flow defined above with the input query, to get the embedding.
    query_vector = code_to_embedding.eval(query)
    # Run the query and get the results.
    with pool.connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT filename, language, code, embedding <=> %s AS distance, start, "end", source_name
                FROM {table_name} ORDER BY distance LIMIT %s
            """,
                (query_vector, top_k),
            )
            return [
                {
                    "filename": row[0],
                    "language": row[1],
                    "code": row[2],
                    "score": 1.0 - row[3],
                    "start": row[4],
                    "end": row[5],
                    "source": row[6] if len(row) > 6 else "unknown",
                }
                for row in cur.fetchall()
            ]


def run_interactive_query_mode():
    """Run the interactive query mode."""
    # Initialize the database connection pool.
    database_url = os.getenv("COCOINDEX_DATABASE_URL")
    if not database_url:
        raise ValueError("COCOINDEX_DATABASE_URL not found in environment")
    
    pool = ConnectionPool(database_url)
    print("\n🔍 Interactive search mode. Type queries to search the code index.")
    print("Press Enter with empty query to quit.\n")

    # Run queries in a loop to demonstrate the query capabilities.
    while True:
        try:
            query = input("Search query: ")
            if query == "":
                break
            # Run the query function with the database connection pool and the query.
            results = search(pool, query)
            print(f"\n📊 Found {len(results)} results:")
            for result in results:
                source_info = f" [{result['source']}]" if result.get('source') and result['source'] != 'files' else ""
                print(
                    f"[{result['score']:.3f}] {result['filename']}{source_info} ({result['language']}) (L{result['start']['line']}-L{result['end']['line']})"
                )
                print(f"    {result['code']}")
                print("---")
            print()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


def display_search_results(results: List[Dict[str, Any]]):
    """Display search results in a formatted way."""
    print(f"\n📊 Found {len(results)} results:")
    for result in results:
        source_info = f" [{result['source']}]" if result.get('source') and result['source'] != 'files' else ""
        print(
            f"[{result['score']:.3f}] {result['filename']}{source_info} ({result['language']}) (L{result['start']['line']}-L{result['end']['line']})"
        )
        print(f"    {result['code']}")
        print("---")
    print()
