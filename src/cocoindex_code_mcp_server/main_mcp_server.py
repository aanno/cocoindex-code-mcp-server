#!/usr/bin/env python3

"""
CocoIndex RAG MCP Server - FIXED Implementation

A Model Context Protocol (MCP) server that provides hybrid search capabilities
combining vector similarity and keyword metadata search for code retrieval.

This implementation follows the official MCP SDK patterns using StreamableHTTPSessionManager.
"""

import contextlib
import json
import logging
import os
import signal
import sys
import threading
from collections.abc import AsyncIterator
from typing import List, Optional

import click
import mcp.types as types
from dotenv import load_dotenv
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

import cocoindex

from . import mcp_json_schemas

# Backend abstraction imports
from .backends import BackendFactory, VectorStoreBackend
from .cocoindex_config import (
    code_embedding_flow,
    code_to_embedding,
    run_flow_update,
    update_flow_config,
)

# Local imports
from .db.pgvector.hybrid_search import HybridSearchEngine
from .keyword_search_parser_lark import KeywordSearchParser
from .lang.python.python_code_analyzer import analyze_python_code

try:
    from coverage import Coverage
    HAS_COVERAGE = True
except ImportError:
    HAS_COVERAGE = False
    Coverage = None  # type: ignore

# Import metadata fields from single source of truth
from .mappers import CONST_METADATA_FIELDS

METADATA_FIELDS = list(CONST_METADATA_FIELDS)


@contextlib.asynccontextmanager
async def coverage_context() -> AsyncIterator[Optional[object]]:
    """Context manager for coverage collection during daemon execution."""
    if not HAS_COVERAGE:
        yield None
        return

    import atexit

    if Coverage is None:
        yield None
        return

    cov = Coverage()
    cov.start()

    # Set up cleanup handlers
    def stop_coverage():
        try:
            cov.stop()
            cov.save()
        except Exception as e:
            logger.warning(f"Error stopping coverage: {e}")

    # Register cleanup handlers
    atexit.register(stop_coverage)

    try:
        yield cov
    finally:
        # Stop coverage without interfering with shutdown
        try:
            stop_coverage()
        except Exception as e:
            # Don't let coverage cleanup block shutdown
            logger.warning(f"Coverage cleanup warning: {e}")

# Configure logging
logger = logging.getLogger(__name__)

# Global state
hybrid_search_engine: Optional[HybridSearchEngine] = None
shutdown_event = threading.Event()
background_thread: Optional[threading.Thread] = None


def safe_embedding_function(query: str) -> object:
    """Safe wrapper for embedding function that handles shutdown gracefully."""
    if shutdown_event.is_set():
        # Return a zero vector if shutting down
        try:
            import numpy as np
            return np.zeros(384, dtype=np.float32)
        except ImportError:
            return [0.0] * 384

    try:
        return code_to_embedding.eval(query)
    except RuntimeError as e:
        if "cannot schedule new futures after shutdown" in str(e):
            try:
                import numpy as np
                return np.zeros(384, dtype=np.float32)
            except ImportError:
                return [0.0] * 384
        raise
    except Exception as e:
        logger.warning(f"Embedding function failed: {e}")
        try:
            import numpy as np
            return np.zeros(384, dtype=np.float32)
        except ImportError:
            return [0.0] * 384


def handle_shutdown(signum, frame) -> None:
    """Handle shutdown signals gracefully."""
    logger.info("Shutdown signal received, cleaning up...")
    shutdown_event.set()

    # Wait for background thread to finish if it exists
    global background_thread
    if background_thread and background_thread.is_alive():
        logger.info("Waiting for background thread to finish...")
        background_thread.join(timeout=3.0)
        if background_thread.is_alive():
            logger.warning("Background thread did not finish cleanly")

    logger.info("Cleanup completed")


def get_mcp_tools() -> list[types.Tool]:
    """Get the list of MCP tools with their schemas."""
    return [
        types.Tool(
            name="search-hybrid",
            description="Perform hybrid search combining vector similarity and keyword metadata filtering. Keyword syntax: field:value, exists(field), value_contains(field, 'text'), multiple terms are AND ed, use parentheses for OR.",
            inputSchema=mcp_json_schemas.HYBRID_SEARCH_INPUT_SCHEMA,
            # outputSchema=mcp_json_schemas.HYBRID_SEARCH_OUTPUT_SCHEMA
        ),
        types.Tool(
            name="search-vector",
            description="Perform pure vector similarity search",
            inputSchema=mcp_json_schemas.VECTOR_SEARCH_INPUT_SCHEMA,
        ),
        types.Tool(
            name="search-keyword",
            description="Perform pure keyword metadata search using field:value, exists(field), value_contains(field, 'text') syntax",
            inputSchema=mcp_json_schemas.KEYWORD_SEARCH_INPUT_SCHEMA,
        ),
        types.Tool(
            name="code-analyze",
            description="Analyze code and extract metadata for indexing",
            inputSchema=mcp_json_schemas.CODE_ANALYZE_INPUT_SCHEMA,
        ),
        types.Tool(
            name="code-embeddings",
            description="Generate embeddings for text using the configured embedding model",
            inputSchema=mcp_json_schemas.CODE_EMBEDDINGS_INPUT_SCHEMA,
        ),
        types.Tool(
            name="help-keyword_syntax",
            description="Get comprehensive help and examples for keyword query syntax",
            inputSchema=mcp_json_schemas.EMPTY_JSON_SCHEMA,
        ),
    ]


def get_mcp_resources() -> list[types.Resource]:
    """Get the list of MCP resources."""
    return [
        types.Resource(
            uri=AnyUrl("cocoindex://search/stats"),
            name="search-statistics",
            description="Database and search performance statistics",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("cocoindex://search/config"),
            name="search-configuration",
            description="Current hybrid search configuration and settings",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("cocoindex://database/schema"),
            name="database-schema",
            description="Database table structure and schema information",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("cocoindex://search/examples"),
            name="search:examples",
            description="Categorized examples of keyword query syntax",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("cocoindex://search/grammar"),
            name="search-keyword-grammar",
            description="Lark grammar for keyword search parsing",
            mimeType="text/x-lark",
        ),
        types.Resource(
            uri=AnyUrl("cocoindex://search/operators"),
            name="search-operators",
            description="List of supported search operators and syntax",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("cocoindex://debug/example_resource"),
            name="debug-example_resource",
            description="Simple test resource for debugging",
            mimeType="application/json",
        ),
    ]


@click.command()
@click.argument("paths", nargs=-1)
@click.option("--paths", "explicit_paths", multiple=True, help="Alternative way to specify paths")
@click.option("--no-live", is_flag=True, help="Disable live update mode")
@click.option("--poll", default=60, help="Polling interval in seconds for live updates")
@click.option("--default-embedding", is_flag=True, help="Use default CocoIndex embedding")
@click.option("--default-chunking", is_flag=True, help="Use default CocoIndex chunking")
@click.option("--default-language-handler", is_flag=True, help="Use default CocoIndex language handling")
@click.option("--chunk-factor-percent", default=100,
              help="Chunk size scaling factor as percentage (100=default, <100=smaller, >100=larger)")
@click.option("--port", default=3000, help="Port to listen on for HTTP")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--json-response", is_flag=True, default=False, help="Enable JSON responses instead of SSE streams")
@click.option("--rescan", is_flag=True, default=False, help="Clear database and tracking tables before starting to force re-indexing")
def main(
    paths: tuple,
    explicit_paths: tuple,
    no_live: bool,
    poll: int,
    default_embedding: bool,
    default_chunking: bool,
    default_language_handler: bool,
    chunk_factor_percent: int,
    port: int,
    log_level: str,
    json_response: bool,
    rescan: bool,
) -> int:
    """CocoIndex RAG MCP Server - Model Context Protocol server for hybrid code search."""

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Load environment and initialize CocoIndex
    load_dotenv()
    cocoindex.init()

    # Handle rescan flag - clear database tables to force re-indexing
    if rescan:
        logger.info("🗑️  Rescan mode enabled - clearing database and tracking tables...")
        try:
            import psycopg
            from .cocoindex_config import code_embedding_flow

            # Get database connection
            database_url = os.getenv("DATABASE_URL") or os.getenv("COCOINDEX_DATABASE_URL")
            if not database_url:
                logger.error("❌ DATABASE_URL not found - cannot perform rescan")
                return 1

            # Get table names for the main flow
            # PostgreSQL stores table names in lowercase, so we need to lowercase them for queries
            embeddings_table = cocoindex.utils.get_target_default_name(code_embedding_flow, "code_embeddings").lower()
            tracking_table = f"{code_embedding_flow.name}__cocoindex_tracking".lower()

            logger.info(f"  Clearing embeddings table: {embeddings_table}")
            logger.info(f"  Clearing tracking table:   {tracking_table}")

            # Clear tables using SQL TRUNCATE (faster and resets auto-increment)
            conn = psycopg.connect(database_url)
            cur = conn.cursor()

            try:
                # Use psycopg's sql module for safe identifier quoting
                from psycopg import sql

                # Clear embeddings table
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = %s
                    );
                """, (embeddings_table,))
                exists_result = cur.fetchone()
                if exists_result and exists_result[0]:
                    # Get count before truncating (for logging)
                    cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(embeddings_table)))
                    count_result = cur.fetchone()
                    count = count_result[0] if count_result else 0
                    # TRUNCATE is faster than DELETE and resets auto-increment
                    cur.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(sql.Identifier(embeddings_table)))
                    logger.info(f"  ✅ Truncated {embeddings_table} ({count} records removed)")
                else:
                    logger.info(f"  ⚠️  Table {embeddings_table} does not exist yet")

                # Clear tracking table (critical for re-indexing!)
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = %s
                    );
                """, (tracking_table,))
                exists_result = cur.fetchone()
                if exists_result and exists_result[0]:
                    # Get count before truncating (for logging)
                    cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(tracking_table)))
                    count_result = cur.fetchone()
                    count = count_result[0] if count_result else 0
                    # TRUNCATE is faster than DELETE and resets auto-increment
                    cur.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(sql.Identifier(tracking_table)))
                    logger.info(f"  ✅ Truncated {tracking_table} ({count} records removed)")
                else:
                    logger.info(f"  ⚠️  Table {tracking_table} does not exist yet")

                conn.commit()
                logger.info("✅ Rescan complete - tables cleared, ready for fresh indexing")

            except Exception as e:
                conn.rollback()
                logger.error(f"❌ Failed to clear tables: {e}")
                return 1
            finally:
                cur.close()
                conn.close()

        except Exception as e:
            logger.error(f"❌ Rescan failed: {e}")
            return 1

    # Determine paths to use
    final_paths = None
    if explicit_paths:
        final_paths = list(explicit_paths)
    elif paths:
        final_paths = list(paths)

    # Configure live updates
    live_enabled = not no_live

    # Update flow configuration
    update_flow_config(
        paths=final_paths,
        enable_polling=live_enabled and poll > 0,
        poll_interval=poll,
        use_default_embedding=default_embedding,
        use_default_chunking=default_chunking,
        use_default_language_handler=default_language_handler,
        chunk_factor_percent=chunk_factor_percent
    )

    logger.info("🚀 CocoIndex RAG MCP Server starting...")
    logger.info(f"📁 Paths: {final_paths or ['cocoindex (default)']}")
    logger.info(f"🔴 Live updates: {'ENABLED' if live_enabled else 'DISABLED'}")
    if live_enabled:
        logger.info(f"⏰ Polling interval: {poll} seconds")
    if chunk_factor_percent != 100:
        logger.info(f"📏 Chunk size scaling: {chunk_factor_percent}%")

    # Create the MCP server
    app: Server = Server("cocoindex-rag")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        """List available MCP tools."""
        return get_mcp_tools()

    @app.list_resources()
    async def list_resources() -> list[types.Resource]:
        """List available MCP resources."""
        return get_mcp_resources()

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        """Handle MCP tool calls with proper error handling."""
        global hybrid_search_engine

        try:
            if not hybrid_search_engine:
                raise RuntimeError("Hybrid search engine not initialized. Please check database connection.")

            if name == "search-hybrid":
                result = await perform_hybrid_search(arguments)
            elif name == "search-vector":
                result = await perform_vector_search(arguments)
            elif name == "search-keyword":
                result = await perform_keyword_search(arguments)
            elif name == "code-analyze":
                result = await analyze_code_tool(arguments)
            elif name == "code-embeddings":
                result = await get_embeddings_tool(arguments)
            elif name == "help-keyword_syntax":
                result = await get_keyword_syntax_help_tool(arguments)
            else:
                raise ValueError(f"Unknown tool '{name}'")

            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]

        except Exception as e:
            logger.exception(f"Error executing tool '{name}'")
            # Return proper MCP error dict as per protocol recommendation
            error_response = {
                "error": {
                    "type": "mcp_protocol_error",
                    "code": 32603,
                    "message": str(e)
                }
            }
            return [types.TextContent(
                type="text",
                text=json.dumps(error_response, indent=2, ensure_ascii=False)
            )]

    @app.read_resource()
    async def handle_read_resource(uri: AnyUrl) -> List[ReadResourceContents]:
        """Read MCP resource content."""
        uri_str = str(uri)
        logger.info(f"🔍 Reading resource: '{uri_str}' (type: {type(uri)}, repr: {repr(uri)})")

        if uri_str == "cocoindex://search/stats":
            content = await get_search_stats()
        elif uri_str == "cocoindex://search/config":
            content = await get_search_config()
        elif uri_str == "cocoindex://database/schema":
            content = await get_database_schema()
        elif uri_str == "cocoindex://search/examples":
            content = await get_query_examples()
        elif uri_str == "cocoindex://search/grammar":
            content = await get_search_grammar()
        elif uri_str == "cocoindex://search/operators":
            content = await get_search_operators()
        elif uri_str == "cocoindex://debug/example_resource":
            logger.info("✅ Test resource accessed successfully!")
            content = json.dumps({"message": "Test resource working", "uri": uri_str}, indent=2)
        else:
            logger.error(
                f"❌ Unknown resource requested: '{uri_str}' (available: search/stats, search/config, database/schema, query/examples, search/grammar, search/operators, test/simple)")
            raise McpError(types.ErrorData(
                code=404,
                message=f"Resource not found: {uri_str}"
            ))

        logger.info(f"✅ Successfully retrieved resource: '{uri_str}'")
        return [ReadResourceContents(
            content=content,
            mime_type="application/json" if uri_str != "cocoindex://search/grammar" else "text/x-lark"
        )]

    # Helper function to make SearchResult objects JSON serializable
    def serialize_search_results(results) -> list:
        """Convert SearchResult objects to JSON-serializable dictionaries."""
        from decimal import Decimal
        from enum import Enum

        def make_serializable(obj):
            """Recursively convert objects to JSON-serializable format."""
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, Decimal):
                return float(obj)
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif hasattr(obj, '__dict__'):
                # Convert object to dict
                if hasattr(obj, 'conditions') and hasattr(obj, 'operator'):  # SearchGroup object
                    return {
                        'conditions': make_serializable(obj.conditions),
                        'operator': make_serializable(obj.operator)
                    }
                elif hasattr(obj, 'field') and hasattr(obj, 'value'):  # SearchCondition object
                    return {
                        'field': make_serializable(obj.field),
                        'value': make_serializable(obj.value),
                        'operator': make_serializable(getattr(obj, 'operator', None))
                    }
                elif hasattr(obj, 'filename'):  # SearchResult object
                    result_dict = {
                        'filename': make_serializable(obj.filename),
                        'language': make_serializable(obj.language),
                        'code': make_serializable(obj.code),
                        'location': make_serializable(obj.location),
                        'start': make_serializable(obj.start),
                        'end': make_serializable(obj.end),
                        'score': make_serializable(obj.score),
                        'score_type': make_serializable(obj.score_type),
                        'source': make_serializable(obj.source)
                    }

                    # Add direct metadata fields from SearchResult
                    metadata_fields = list(METADATA_FIELDS)
                    for key in metadata_fields:
                        if hasattr(obj, key):
                            value = getattr(obj, key)
                            result_dict[key] = make_serializable(value)

                    # Extract ALL fields from metadata_json if it exists (generalized promotion)
                    if hasattr(obj, 'metadata_json') and isinstance(obj.metadata_json, dict):
                        metadata_json = obj.metadata_json
                        # Promote all fields from metadata_json to top-level, avoiding conflicts
                        for key, value in metadata_json.items():
                            # Skip if already exists as top-level field to avoid overwriting
                            if key not in result_dict:
                                result_dict[key] = make_serializable(value)

                    return result_dict
                else:
                    # Generic object serialization
                    return {key: make_serializable(value) for key, value in obj.__dict__.items()}
            else:
                # Fallback to string representation
                return str(obj)

        return [make_serializable(result) for result in results]

    # Tool implementation functions
    async def perform_hybrid_search(arguments: dict) -> dict:
        """Perform hybrid search combining vector and keyword search."""
        vector_query = arguments["vector_query"]
        keyword_query = arguments["keyword_query"]
        top_k = arguments.get("top_k", 10)
        vector_weight = arguments.get("vector_weight", 0.7)
        keyword_weight = arguments.get("keyword_weight", 0.3)
        language = arguments.get("language")
        embedding_model = arguments.get("embedding_model")

        try:
            if hybrid_search_engine is not None:
                results = hybrid_search_engine.search(
                    vector_query=vector_query,
                    keyword_query=keyword_query,
                    top_k=top_k,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                    language=language,
                    embedding_model=embedding_model
                )
        except ValueError as e:
            # Handle field validation errors with helpful messages
            error_msg = str(e)
            if "Invalid field" in error_msg:
                from .schema_validator import get_valid_fields_help
                help_text = get_valid_fields_help()
                raise ValueError(f"{error_msg}\n\n{help_text}")
            raise
        except Exception as e:
            # Handle SQL-related errors
            if "column" in str(e) and "does not exist" in str(e):
                from .schema_validator import get_valid_fields_help
                help_text = get_valid_fields_help()
                raise ValueError(f"Database schema error: {e}\n\n{help_text}")
            raise

        return {
            "query": {
                "vector_query": vector_query,
                "keyword_query": keyword_query,
                "top_k": top_k,
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight
            },
            "results": serialize_search_results(results),
            "total_results": len(results)
        }

    async def perform_vector_search(arguments: dict) -> dict:
        """Perform pure vector similarity search."""
        query = arguments["query"]
        top_k = arguments.get("top_k", 10)
        language = arguments.get("language")
        embedding_model = arguments.get("embedding_model")

        if hybrid_search_engine is not None:
            results = hybrid_search_engine.search(
                vector_query=query,
                keyword_query="",
                top_k=top_k,
                vector_weight=1.0,
                keyword_weight=0.0,
                language=language,
                embedding_model=embedding_model
            )

        return {
            "query": query,
            "results": serialize_search_results(results),
            "total_results": len(results)
        }

    async def perform_keyword_search(arguments: dict) -> dict:
        """Perform pure keyword metadata search."""
        query = arguments["query"]
        top_k = arguments.get("top_k", 10)

        if hybrid_search_engine is not None:
            results = hybrid_search_engine.search(
                vector_query="",
                keyword_query=query,
                top_k=top_k,
                vector_weight=0.0,
                keyword_weight=1.0
            )

        return {
            "query": query,
            "results": serialize_search_results(results),
            "total_results": len(results)
        }

    async def analyze_code_tool(arguments: dict) -> dict:
        """Analyze code and extract metadata."""
        code = arguments["code"]
        file_path = arguments["file_path"]
        language = arguments.get("language")

        # Auto-detect language from file extension if not provided
        if not language:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".py":
                language = "python"
            else:
                language = "unknown"

        if language == "python":
            metadata = analyze_python_code(code, file_path)
        else:
            # Basic metadata for unsupported languages
            metadata = {
                "file_path": file_path,
                "language": language,
                "lines_of_code": len(code.splitlines()),
                "char_count": len(code),
                "analysis_type": "basic"
            }

        return {
            "file_path": file_path,
            "language": language,
            "metadata": metadata
        }

    async def get_embeddings_tool(arguments: dict) -> dict:
        """Generate embeddings for text."""
        text = arguments["text"]

        if hybrid_search_engine is not None:
            embedding = hybrid_search_engine.embedding_func(text)

        return {
            "text": text,
            "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
            "dimensions": len(embedding)
        }

    async def get_keyword_syntax_help_tool(_arguments: dict) -> dict:
        """Get comprehensive help and examples for keyword query syntax."""
        return {
            "keyword_query_syntax": {
                "description": "Comprehensive guide to keyword query syntax for searching code metadata",
                "basic_operators": {
                    "field_matching": {
                        "syntax": "field:value",
                        "examples": ["language:python", "has_async:true", 'filename:"test file.py"']
                    },
                    "existence_check": {
                        "syntax": "exists(field)",
                        "examples": ["exists(embedding)", "exists(functions)", "exists(classes)"]
                    },
                    "substring_search": {
                        "syntax": 'value_contains(field, "search_text")',
                        "examples": ['value_contains(code, "async")', 'value_contains(filename, "test")']
                    }
                },
                "boolean_logic": {
                    "AND": "default, i.e. simple separate search terms with spaces",
                    "OR": "parentheses are used to create OR terms",
                    "examples": ['(language:python language:rust) exists(functions)',
                                 '(value_contains(code, "async")) exists(functions)) (value_contains(filename, "test") has_async:true)']
                },
                "available_fields": [
                    "filename", "language", "code", "functions", "classes", "imports",
                    "complexity_score", "has_type_hints", "has_async", "has_classes",
                    "embedding", "start", "end", "source_name", "location", "metadata_json"
                ]
            }
        }

    # Resource implementation functions
    async def get_search_stats() -> str:
        """Get database and search statistics."""
        if not hybrid_search_engine or not hybrid_search_engine.pool:
            return json.dumps({"error": "No database connection"})

        try:
            with hybrid_search_engine.pool.connection() as conn:
                with conn.cursor() as cur:
                    table_name = hybrid_search_engine.table_name
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    total_records = cur.fetchone()[0]

                    stats = {
                        "table_name": table_name,
                        "total_records": total_records,
                        "connection_pool_size": hybrid_search_engine.pool.max_size,
                    }

            return json.dumps(stats, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Failed to get stats: {str(e)}"})

    async def get_search_config() -> str:
        """Get current search configuration."""
        config = {
            "table_name": hybrid_search_engine.table_name if hybrid_search_engine else "unknown",
            "embedding_model": "TODO: language dependent",
            "parser_type": "TODO: lark_keyword_parser",
            "default_weights": {
                "vector_weight": 0.7,
                "keyword_weight": 0.3
            }
        }
        return json.dumps(config, indent=2)

    async def get_database_schema() -> str:
        """Get database schema information."""
        if not hybrid_search_engine or not hybrid_search_engine.pool:
            return json.dumps({"error": "No database connection"})

        try:
            with hybrid_search_engine.pool.connection() as conn:
                with conn.cursor() as cur:
                    table_name = hybrid_search_engine.table_name
                    cur.execute(f"""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns
                        WHERE table_name = '{table_name}'
                        ORDER BY ordinal_position
                    """)
                    columns = cur.fetchall()

                    schema = {
                        "table_name": table_name,
                        "columns": [
                            {"name": col[0], "type": col[1], "nullable": col[2] == "YES"}
                            for col in columns
                        ]
                    }

            return json.dumps(schema, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Failed to get schema: {str(e)}"})

    async def get_query_examples() -> str:
        """Get categorized examples of keyword query syntax."""
        examples = {
            "basic_matching": [
                "language:python",
                "filename:main.py",
                "has_async:true"
            ],
            "existence_checks": [
                "exists(embedding)",
                "exists(functions)",
                "exists(language) AND language:rust"
            ],
            "value_contains": [
                'value_contains(code, "async")',
                'value_contains(filename, "test")',
                'value_contains(functions, "parse") AND language:python'
            ],
            "boolean_and_logic": [
                "language:python has_async:true",
                "language:python exists(embedding)"
            ],
            "boolean_or_logic": [
                "(language:python language:rust)",
                '(value_contains(code, "async") value_contains(code, "await"))'
            ],
            "boolean_logic": [
                "(language:python language:rust) exists(embedding)"
            ]
        }
        return json.dumps(examples, indent=2)

    async def get_search_grammar() -> str:
        """Get the Lark grammar for keyword search parsing."""
        # This is a simplified version of the grammar used by our parser
        grammar = '''
TODO:
include file src/cocoindex_code_mcp_server/grammars/keyword_search.lark here
        '''
        return grammar.strip()

    async def get_search_operators() -> str:
        """Get list of supported search operators and syntax."""
        operators = {
            "description": "Supported operators for keyword search queries",
            "operators": {
                "field_matching": {
                    "syntax": "field:value",
                    "description": "Match field with exact value",
                    "examples": ["language:python", "filename:test.py"]
                },
                "existence_check": {
                    "syntax": "exists(field)",
                    "description": "Check if field exists",
                    "examples": ["exists(functions)", "exists(classes)"]
                },
                "substring_search": {
                    "syntax": 'value_contains(field, "text")',
                    "description": "Check if field contains substring",
                    "examples": ['value_contains(code, "async")', 'value_contains(filename, "test")']
                },
                "boolean_logic": {
                    "parentheses": "parentheses are used to create OR terms, without parentheses multiple terms are AND ed"
                }
            }
        }
        return json.dumps(operators, indent=2)

    # Initialize search engine
    async def initialize_search_engine(backend: VectorStoreBackend):
        """Initialize the hybrid search engine with provided backend."""
        global hybrid_search_engine

        try:
            # Backend handles its own initialization (e.g., pgvector registration)
            # No need for manual register_vector() calls

            # Initialize search engine components
            parser = KeywordSearchParser()

            # Initialize hybrid search engine
            table_name = cocoindex.utils.get_target_default_name(
                code_embedding_flow, "code_embeddings"
            )
            hybrid_search_engine = HybridSearchEngine(
                table_name=table_name,
                parser=parser,
                backend=backend,
                embedding_func=safe_embedding_function
            )

            logger.info("✅ CocoIndex RAG MCP Server initialized successfully with backend abstraction")

        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            raise

    # Background initialization and flow updates
    async def background_initialization():
        """Start flow updates."""
        try:
            # Set up live updates if enabled
            if live_enabled:
                logger.info("🔄 Starting live flow updates...")

                def run_flow_background():
                    """Background thread function for live flow updates."""
                    while not shutdown_event.is_set():
                        try:
                            run_flow_update(live_update=True)
                            if shutdown_event.wait(poll):
                                break
                        except Exception as e:
                            if not shutdown_event.is_set():
                                logger.error(f"Error in background flow update: {e}")
                                if shutdown_event.wait(10):
                                    break

                # Start background flow update
                global background_thread
                background_thread = threading.Thread(target=run_flow_background, daemon=True)
                background_thread.start()
                logger.info("✅ Background flow updates started")
            else:
                logger.info("🔄 Running one-time flow update...")
                run_flow_update(live_update=False)
                logger.info("✅ Flow update completed")

        except Exception as e:
            logger.error(f"❌ Background initialization failed: {e}")

    # Create session manager
    session_manager = StreamableHTTPSessionManager(
        app=app,
        event_store=None,
        json_response=json_response,
        stateless=True,
    )

    async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        """Context manager for session manager."""
        # Get database URL from environment
        database_url = os.getenv("COCOINDEX_DATABASE_URL")
        if not database_url:
            raise ValueError("COCOINDEX_DATABASE_URL not found in environment")

        backend_type = os.getenv("COCOINDEX_BACKEND_TYPE", "postgres").lower()

        # Use coverage context for long-running daemon
        async with coverage_context() as cov:
            if cov:
                logger.info("📊 Coverage collection started")

            # Use backend abstraction for proper cleanup
            async with session_manager.run():
                logger.info("🚀 MCP Server started with StreamableHTTP session manager!")

                # Create backend using factory pattern
                table_name = cocoindex.utils.get_target_default_name(
                    code_embedding_flow, "code_embeddings"
                )

                # Create the appropriate backend
                if backend_type == "postgres":
                    from pgvector.psycopg import register_vector
                    from psycopg_pool import ConnectionPool

                    pool = ConnectionPool(database_url)
                    # Register pgvector extensions
                    with pool.connection() as conn:
                        register_vector(conn)

                    backend = BackendFactory.create_backend(
                        backend_type,
                        pool=pool,
                        table_name=table_name
                    )
                else:
                    # For other backends that might expect connection_string
                    backend = BackendFactory.create_backend(
                        backend_type,
                        connection_string=database_url,
                        table_name=table_name
                    )

                logger.info(f"🔧 Initializing {backend_type} backend...")
                await initialize_search_engine(backend)

                # Initialize background components
                await background_initialization()

                try:
                    yield
                finally:
                    logger.info("🛑 MCP Server shutting down...")
                    shutdown_event.set()
                    if hasattr(backend, 'close'):
                        backend.close()
                    logger.info("🧹 Backend resources cleaned up")
                    if cov:
                        logger.info("📊 Coverage collection will be finalized")

    # Create ASGI application
    starlette_app = Starlette(
        debug=True,
        routes=[
            Mount("/mcp", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )

    # Run the server
    import uvicorn

    logger.info(f"🌐 Starting HTTP MCP server on http://127.0.0.1:{port}/mcp")
    uvicorn.run(starlette_app, host="127.0.0.1", port=port)

    return 0


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
