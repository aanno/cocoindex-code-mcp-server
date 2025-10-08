#!/usr/bin/env python3

"""
Hybrid search implementation combining vector similarity and keyword metadata search.
Updated to use the VectorStoreBackend abstraction layer.
"""

import json
import os
from typing import Any, Dict, List, Union

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from psycopg_pool import ConnectionPool

import cocoindex
from cocoindex_code_mcp_server.backends import (
    BackendFactory,
    QueryFilters,
    SearchResult,
    VectorStoreBackend,
)
from cocoindex_code_mcp_server.cocoindex_config import (
    code_embedding_flow,
    code_to_embedding,
    graphcodebert_embedding,
    unixcoder_embedding,
    language_to_embedding_model,
)
from cocoindex_code_mcp_server.keyword_search_parser_lark import (
    KeywordSearchParser,
)


class HybridSearchEngine:
    """Hybrid search engine combining vector and keyword search."""

    def __init__(self, table_name: str, parser: KeywordSearchParser,
                 backend: Union[VectorStoreBackend, None] = None,
                 pool: Union[ConnectionPool, None] = None,
                 embedding_func=None,
                 embedding_model: str | None = None) -> None:
        # Support both new backend interface and legacy direct pool access
        if backend is not None:
            self.backend = backend
        elif pool is not None:
            # Create PostgreSQL backend from legacy pool
            table_name = table_name or cocoindex.utils.get_target_default_name(
                code_embedding_flow, "code_embeddings"
            )
            self.backend = BackendFactory.create_backend("postgres", pool=pool, table_name=table_name)
        else:
            raise ValueError("Either 'backend' or 'pool' parameter must be provided")

        self.parser = parser or KeywordSearchParser()
        self.embedding_func = embedding_func or (lambda q: code_to_embedding.eval(q))

        # CRITICAL: Store embedding model to filter search results
        # You cannot compare vectors from different embedding models!
        # If not provided, use the default transformer model
        from cocoindex_code_mcp_server.cocoindex_config import DEFAULT_TRANSFORMER_MODEL
        self.embedding_model = embedding_model or DEFAULT_TRANSFORMER_MODEL

    @property
    def pool(self):
        """Access to the database connection pool via backend."""
        return getattr(self.backend, 'pool', None)

    @property
    def table_name(self):
        """Access to the table name via backend."""
        return getattr(self.backend, 'table_name', None)

    def _get_embedding_function(self, embedding_model: str):
        """
        Select the appropriate embedding function based on the model name.

        Args:
            embedding_model: The embedding model identifier

        Returns:
            Callable that embeds text using the specified model
        """
        # Map model names to embedding functions
        if 'graphcodebert' in embedding_model.lower():
            return lambda q: graphcodebert_embedding.eval(q)
        elif 'unixcoder' in embedding_model.lower():
            return lambda q: unixcoder_embedding.eval(q)
        else:
            # Default to sentence-transformers model
            return lambda q: code_to_embedding.eval(q)

    def search(
        self,
        vector_query: str,
        keyword_query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        language: str | None = None,
        embedding_model: str | None = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and keyword filtering.

        Args:
            vector_query: Text to embed and search for semantic similarity (empty string for keyword-only)
            keyword_query: Keyword search query for metadata filtering
            top_k: Number of results to return
            vector_weight: Weight for vector similarity score (0-1)
            keyword_weight: Weight for keyword match score (0-1)
            language: Programming language to filter by (e.g., "Python", "Rust") - REQUIRED if vector_query is used
            embedding_model: Specific embedding model to filter by (e.g., "microsoft/graphcodebert-base") - REQUIRED if vector_query is used

        Returns:
            List of search results with combined scoring

        Note:
            - For vector or hybrid search (when vector_query is not empty): You MUST provide either `language` OR `embedding_model`
            - For keyword-only search (when vector_query is empty): language/embedding_model are OPTIONAL
            - If `embedding_model` is provided, it takes precedence over `language`
            - If `language` is provided, it will be mapped to the appropriate embedding model

        Raises:
            ValueError: If neither `language` nor `embedding_model` is provided when vector_query is used
        """
        # Parse keyword query
        search_group = self.parser.parse(keyword_query)

        # Convert search group to QueryFilters format
        filters = None
        if search_group and search_group.conditions:
            filters = QueryFilters(conditions=search_group.conditions)

        # Use backend abstraction for search operations
        if vector_query.strip() and filters:
            # Both vector and keyword search - REQUIRE language or embedding_model
            if language is None and embedding_model is None:
                raise ValueError(
                    "Either 'language' or 'embedding_model' parameter is required for search. "
                    "This ensures you only get results from the appropriate embedding model. "
                    "Examples: language='Python', embedding_model='microsoft/graphcodebert-base'"
                )

            # Resolve embedding model to use for filtering
            model_to_use = embedding_model or language_to_embedding_model(language)
            embedding_func_to_use = self._get_embedding_function(model_to_use)

            # Hybrid search with embedding
            query_vector = embedding_func_to_use(vector_query)
            results = self.backend.hybrid_search(
                query_vector=query_vector,
                filters=filters,
                top_k=top_k,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                embedding_model=model_to_use  # CRITICAL: Filter by resolved embedding model
            )
        elif vector_query.strip():
            # Vector search only - REQUIRE language or embedding_model
            if language is None and embedding_model is None:
                raise ValueError(
                    "Either 'language' or 'embedding_model' parameter is required for search. "
                    "This ensures you only get results from the appropriate embedding model. "
                    "Examples: language='Python', embedding_model='microsoft/graphcodebert-base'"
                )

            # Resolve embedding model to use for filtering
            model_to_use = embedding_model or language_to_embedding_model(language)
            embedding_func_to_use = self._get_embedding_function(model_to_use)

            # Vector-only search
            query_vector = embedding_func_to_use(vector_query)
            results = self.backend.vector_search(
                query_vector=query_vector,
                top_k=top_k,
                embedding_model=model_to_use  # CRITICAL: Filter by resolved embedding model
            )
        elif filters:
            # Keyword search only (no embedding model filter needed)
            results = self.backend.keyword_search(filters=filters, top_k=top_k)
        else:
            # No valid query
            return []

        # Convert SearchResult objects to dict format for backward compatibility
        return [self._search_result_to_dict(result) for result in results]

    def _search_result_to_dict(self, result: SearchResult) -> Dict[str, Any]:
        """Convert SearchResult to dict format for backward compatibility."""
        result_dict = {
            "filename": result.filename,
            "language": result.language,
            "code": result.code,
            "score": result.score,
            "start": result.start,
            "end": result.end,
            "source": result.source,
            "score_type": result.score_type
        }

        # Add metadata fields if available
        if result.metadata:
            result_dict.update(result.metadata)

        return result_dict


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

    session: PromptSession = PromptSession(key_bindings=bindings, multiline=True)

    print(f"{prompt_text} (finish with Ctrl+Q):")
    try:
        result = session.prompt()
        return result.strip()
    except (KeyboardInterrupt, EOFError):
        return ""


def run_interactive_hybrid_search() -> None:
    """Run interactive hybrid search mode with dual prompts."""
    # Initialize the database connection pool
    url = os.getenv("COCOINDEX_DATABASE_URL")
    if url is None:
        raise ValueError("COCOINDEX_DATABASE_URL environment variable must be set")

    pool = ConnectionPool(url)
    # Use legacy constructor for backward compatibility
    table_name = cocoindex.utils.get_target_default_name(
        code_embedding_flow, "code_embeddings"
    )
    parser = KeywordSearchParser()
    search_engine = HybridSearchEngine(
        table_name=table_name,
        pool=pool,
        parser=parser
    )

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
