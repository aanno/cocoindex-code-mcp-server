#!/usr/bin/env python3

"""
CocoIndex RAG MCP Server

A Model Context Protocol (MCP) server that provides hybrid search capabilities
combining vector similarity and keyword metadata search for code retrieval.
"""

import argparse
import asyncio
import json
import os
import sys
import signal
import threading
from typing import Any, Dict, List, Optional, Sequence

import mcp.server.stdio
import mcp.server.sse
import mcp.types as types
from mcp.server import NotificationOptions, Server
from starlette.applications import Starlette
from starlette.routing import Route
import uvicorn
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

# Local imports
from hybrid_search import HybridSearchEngine
from keyword_search_parser_lark import KeywordSearchParser
from lang.python.python_code_analyzer import analyze_python_code
import cocoindex
from cocoindex_config import code_embedding_flow, code_to_embedding, update_flow_config, run_flow_update
from __init__ import LOGGER

try:
    import coverage
except ImportError:
    coverage = None

# Initialize the MCP server
server = Server("cocoindex-rag")

# Global state
hybrid_search_engine: Optional[HybridSearchEngine] = None
connection_pool: Optional[ConnectionPool] = None
shutdown_event = threading.Event()
background_thread: Optional[threading.Thread] = None
_terminating = threading.Event()  # Atomic termination flag
_want_to_save_coverage = False


def safe_embedding_function(query: str):
    """Safe wrapper for embedding function that handles shutdown gracefully."""
    if shutdown_event.is_set() or _terminating.is_set():
        # Return a zero vector if shutting down
        try:
            import numpy as np
            # Default embedding size for all-MiniLM-L6-v2 is 384
            return np.zeros(384, dtype=np.float32)
        except ImportError:
            return [0.0] * 384
    
    try:
        # Use the CocoIndex embedding function
        return code_to_embedding.eval(query)
    except RuntimeError as e:
        if "cannot schedule new futures after shutdown" in str(e):
            # Return zero vector on shutdown
            try:
                import numpy as np
                return np.zeros(384, dtype=np.float32)
            except ImportError:
                return [0.0] * 384
        raise
    except Exception as e:
        print(f"Warning: Embedding function failed: {e}", file=sys.stderr)
        # Return zero vector as fallback
        try:
            import numpy as np
            return np.zeros(384, dtype=np.float32)
        except ImportError:
            return [0.0] * 384


def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    global _terminating
    
    # Check if already terminating to avoid double cleanup
    if _terminating.is_set():
        print("🔴 Force terminating...", file=sys.stderr)
        os._exit(1)
        return
    
    _terminating.set()
    print("\n🛑 Shutdown signal received, cleaning up...", file=sys.stderr)
    shutdown_event.set()
    
    # Wait for background thread to finish if it exists
    global background_thread
    if background_thread and background_thread.is_alive():
        print("⏳ Waiting for background thread to finish...", file=sys.stderr)
        background_thread.join(timeout=3.0)  # Shorter timeout
        if background_thread.is_alive():
            print("⚠️  Background thread did not finish cleanly", file=sys.stderr)
    
    print("✅ Cleanup completed", file=sys.stderr)
    if coverage is not None and _want_to_save_coverage:
        _cov.stop()
        _cov.save()
        print("✅ Coverage data saved to 'coverage'", file=sys.stderr)

    # Use sys.exit instead of os._exit for cleaner shutdown
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)


def parse_mcp_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(
        description="CocoIndex RAG MCP Server - Model Context Protocol server for hybrid code search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_mcp_server.py                         # Use default path (cocoindex) with live updates
  python main_mcp_server.py /path/to/code           # Index single directory
  python main_mcp_server.py /path/to/code1 /path/to/code2  # Index multiple directories
  python main_mcp_server.py --paths /path/to/code   # Explicit paths argument
  
  # Live update configuration (enabled by default)
  python main_mcp_server.py --no-live               # Disable live updates
  python main_mcp_server.py --poll 30               # Custom polling interval (default: 60s)
  
  # Default behavior options (use CocoIndex defaults instead of extensions)
  python main_mcp_server.py --default-embedding     # Use CocoIndex SentenceTransformerEmbed
  python main_mcp_server.py --default-chunking      # Use CocoIndex SplitRecursively  
  python main_mcp_server.py --default-language-handler  # Skip Python-specific handlers

MCP Tools Available:
  - hybrid_search: Combine vector similarity and keyword metadata filtering
  - vector_search: Pure vector similarity search
  - keyword_search: Pure keyword metadata search  
  - analyze_code: Code analysis and metadata extraction
  - get_embeddings: Generate embeddings for text
  - get_keyword_syntax_help: Get comprehensive help for keyword query syntax

MCP Resources Available:
  - search_stats: Database and search performance statistics
  - search_config: Current hybrid search configuration
  - database_schema: Database table structure information
  - query/grammar: Lark grammar definition for keyword syntax
  - query/examples: Categorized example queries
  - database/fields: Available database fields and types
  - query/operators: Detailed operator reference
        """
    )
    
    parser.add_argument(
        "paths", 
        nargs="*", 
        help="Code directory paths to index (default: cocoindex)"
    )
    
    parser.add_argument(
        "--paths",
        dest="explicit_paths",
        nargs="+",
        help="Alternative way to specify paths"
    )
    
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable live update mode (live updates are enabled by default)"
    )
    
    parser.add_argument(
        "--poll",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Polling interval in seconds for live updates (default: 60)"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="enable saving coverage data on exit"
    )

    # Default behavior options (to use default CocoIndex implementation)
    parser.add_argument(
        "--default-embedding",
        action="store_true",
        help="Use default CocoIndex embedding instead of smart code embedding"
    )
    
    parser.add_argument(
        "--default-chunking", 
        action="store_true",
        help="Use default CocoIndex chunking instead of AST-based chunking"
    )
    
    parser.add_argument(
        "--default-language-handler",
        action="store_true", 
        help="Use default CocoIndex language handling instead of Python handler"
    )

    # Transport options
    transport_group = parser.add_mutually_exclusive_group()
    transport_group.add_argument(
        "--port",
        type=int,
        metavar="PORT",
        help="Run as HTTP server on specified port (e.g., --port 8080)"
    )
    transport_group.add_argument(
        "--url",
        type=str,
        metavar="URL", 
        help="Connect to HTTP server at specified URL (e.g., --url http://localhost:8080)"
    )
    
    return parser.parse_args()


def determine_paths(args):
    """Determine which paths to use based on parsed arguments."""
    paths = None
    if args.explicit_paths:
        paths = args.explicit_paths
    elif args.paths:
        paths = args.paths
    
    return paths


def display_mcp_configuration(args, paths):
    """Display the configuration for MCP server."""
    print("🚀 CocoIndex RAG MCP Server", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    
    # Display paths
    if paths:
        if len(paths) == 1:
            print(f"📁 Indexing path: {paths[0]}", file=sys.stderr)
        else:
            print(f"📁 Indexing {len(paths)} paths:", file=sys.stderr)
            for i, path in enumerate(paths, 1):
                print(f"  {i}. {path}", file=sys.stderr)
    else:
        print("📁 Using default path: cocoindex", file=sys.stderr)
    
    # Display mode configuration
    live_enabled = not args.no_live
    if live_enabled:
        print("🔴 Mode: Live updates ENABLED", file=sys.stderr)
        print(f"⏰ Polling interval: {args.poll} seconds", file=sys.stderr)
    else:
        print("🟡 Mode: Live updates DISABLED", file=sys.stderr)
    
    print(file=sys.stderr)
    print("🔧 MCP Tools: hybrid_search, vector_search, keyword_search, analyze_code, get_embeddings", file=sys.stderr)
    print("📊 MCP Resources: search_stats, search_config, database_schema", file=sys.stderr)
    print(file=sys.stderr)


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available MCP resources."""
    return [
        types.Resource(
            uri="cocoindex://search/stats",
            name="Search Statistics",
            description="Database and search performance statistics",
            mimeType="application/json",
        ),
        types.Resource(
            uri="cocoindex://search/config",
            name="Search Configuration", 
            description="Current hybrid search configuration and settings",
            mimeType="application/json",
        ),
        types.Resource(
            uri="cocoindex://database/schema",
            name="Database Schema",
            description="Database table structure and schema information",
            mimeType="application/json",
        ),
        types.Resource(
            uri="cocoindex://query/grammar",
            name="Keyword Search Grammar",
            description="Lark grammar definition for keyword query syntax",
            mimeType="text/x-lark",
        ),
        types.Resource(
            uri="cocoindex://query/examples",
            name="Query Examples",
            description="Categorized examples of keyword query syntax",
            mimeType="application/json",
        ),
        types.Resource(
            uri="cocoindex://database/fields",
            name="Database Fields",
            description="Available database fields and their types for query building",
            mimeType="application/json",
        ),
        types.Resource(
            uri="cocoindex://query/operators",
            name="Query Operators",
            description="Detailed reference for all supported query operators",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read MCP resource content."""
    if uri == "cocoindex://search/stats":
        return await get_search_stats()
    elif uri == "cocoindex://search/config":
        return await get_search_config()
    elif uri == "cocoindex://database/schema":
        return await get_database_schema()
    elif uri == "cocoindex://query/grammar":
        return await get_query_grammar()
    elif uri == "cocoindex://query/examples":
        return await get_query_examples()
    elif uri == "cocoindex://database/fields":
        return await get_database_fields()
    elif uri == "cocoindex://query/operators":
        return await get_query_operators()
    else:
        raise ValueError(f"Unknown resource: {uri}")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available MCP tools."""
    return [
        types.Tool(
            name="hybrid_search",
            description="Perform hybrid search combining vector similarity and keyword metadata filtering. Keyword syntax: field:value, exists(field), value_contains(field, 'text'), AND/OR logic. Examples: 'language:python AND function_name:parse' or '(language:python OR language:rust) AND exists(embedding)'",
            inputSchema={
                "type": "object",
                "properties": {
                    "vector_query": {
                        "type": "string",
                        "description": "Text to embed and search for semantic similarity"
                    },
                    "keyword_query": {
                        "type": "string", 
                        "description": "Keyword search query for metadata filtering. Syntax: field:value, exists(field), value_contains(field, 'text'), AND/OR operators, parentheses for grouping. Examples: 'function_name:parse AND language:python', 'value_contains(code, \"async\") OR exists(class_name)'"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10
                    },
                    "vector_weight": {
                        "type": "number",
                        "description": "Weight for vector similarity score (0-1)",
                        "default": 0.7
                    },
                    "keyword_weight": {
                        "type": "number", 
                        "description": "Weight for keyword match score (0-1)",
                        "default": 0.3
                    }
                },
                "required": ["vector_query", "keyword_query"]
            },
        ),
        types.Tool(
            name="vector_search",
            description="Perform pure vector similarity search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to embed and search for semantic similarity"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            },
        ),
        types.Tool(
            name="keyword_search",
            description="Perform pure keyword metadata search. Supports: field:value, exists(field), value_contains(field, 'text'), AND/OR operators, parentheses grouping. Examples: 'language:python AND exists(function_name)', 'value_contains(filename, \"test\") OR function_name:main'",
            inputSchema={
                "type": "object", 
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword search query. Syntax: field:value, exists(field), value_contains(field, 'text'), AND/OR operators. Examples: 'function_name:parse AND language:python', '(language:python OR language:rust) AND exists(embedding)', 'value_contains(code, \"TODO\")'"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return", 
                        "default": 10
                    }
                },
                "required": ["query"]
            },
        ),
        types.Tool(
            name="analyze_code",
            description="Analyze code and extract metadata for indexing",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code content to analyze"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "File path for context"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (defaults to auto-detect from file_path)"
                    }
                },
                "required": ["code", "file_path"]
            },
        ),
        types.Tool(
            name="get_embeddings",
            description="Generate embeddings for text using the configured embedding model",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to generate embeddings for"
                    }
                },
                "required": ["text"]
            },
        ),
        types.Tool(
            name="get_keyword_syntax_help",
            description="Get comprehensive help and examples for keyword query syntax",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle MCP tool calls."""
    global hybrid_search_engine
    
    if not hybrid_search_engine:
        return [types.TextContent(
            type="text",
            text="Error: Hybrid search engine not initialized. Please check database connection."
        )]
    
    try:
        if name == "hybrid_search":
            result = await perform_hybrid_search(arguments)
        elif name == "vector_search":
            result = await perform_vector_search(arguments)
        elif name == "keyword_search":
            result = await perform_keyword_search(arguments)
        elif name == "analyze_code":
            result = await analyze_code_tool(arguments)
        elif name == "get_embeddings":
            result = await get_embeddings_tool(arguments)
        elif name == "get_keyword_syntax_help":
            result = await get_keyword_syntax_help_tool(arguments)
        else:
            return [types.TextContent(
                type="text",
                text=f"Error: Unknown tool '{name}'"
            )]
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text", 
            text=f"Error executing tool '{name}': {str(e)}"
        )]


async def perform_hybrid_search(arguments: dict) -> dict:
    """Perform hybrid search combining vector and keyword search."""
    vector_query = arguments["vector_query"]
    keyword_query = arguments["keyword_query"]
    top_k = arguments.get("top_k", 10)
    vector_weight = arguments.get("vector_weight", 0.7)
    keyword_weight = arguments.get("keyword_weight", 0.3)
    
    results = hybrid_search_engine.search(
        vector_query=vector_query,
        keyword_query=keyword_query, 
        top_k=top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight
    )
    
    return {
        "query": {
            "vector_query": vector_query,
            "keyword_query": keyword_query,
            "top_k": top_k,
            "vector_weight": vector_weight,
            "keyword_weight": keyword_weight
        },
        "results": results,
        "total_results": len(results)
    }


async def perform_vector_search(arguments: dict) -> dict:
    """Perform pure vector similarity search."""
    query = arguments["query"]
    top_k = arguments.get("top_k", 10)
    
    # Use hybrid search with empty keyword query for pure vector search
    results = hybrid_search_engine.search(
        vector_query=query,
        keyword_query="",
        top_k=top_k,
        vector_weight=1.0,
        keyword_weight=0.0
    )
    
    return {
        "query": query,
        "results": results,
        "total_results": len(results)
    }


async def perform_keyword_search(arguments: dict) -> dict:
    """Perform pure keyword metadata search."""
    query = arguments["query"]
    top_k = arguments.get("top_k", 10)
    
    # Use hybrid search with empty vector query for pure keyword search
    results = hybrid_search_engine.search(
        vector_query="",
        keyword_query=query,
        top_k=top_k,
        vector_weight=0.0,
        keyword_weight=1.0
    )
    
    return {
        "query": query,
        "results": results,
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
    
    # Use the cocoindex embedding function
    embedding = hybrid_search_engine.embedding_func(text)
    
    return {
        "text": text,
        "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
        "dimensions": len(embedding)
    }


async def get_keyword_syntax_help_tool(arguments: dict) -> dict:
    """Get comprehensive help and examples for keyword query syntax."""
    help_content = {
        "keyword_query_syntax": {
            "description": "Comprehensive guide to keyword query syntax for searching code metadata",
            "basic_operators": {
                "field_matching": {
                    "syntax": "field:value",
                    "description": "Match exact field value",
                    "examples": [
                        "language:python",
                        "function_name:main",
                        'filename:"test file.py"'
                    ],
                    "notes": "Use quotes for values containing spaces or special characters"
                },
                "existence_check": {
                    "syntax": "exists(field)",
                    "description": "Check if field exists and has a non-null value",
                    "examples": [
                        "exists(embedding)",
                        "exists(function_name)",
                        "exists(class_name)"
                    ],
                    "notes": "Useful for filtering records with specific extracted metadata"
                },
                "substring_search": {
                    "syntax": 'value_contains(field, "search_text")',
                    "description": "Search for substring within field values",
                    "examples": [
                        'value_contains(code, "async")',
                        'value_contains(filename, "test")',
                        'value_contains(code, "TODO")'
                    ],
                    "notes": "Case-sensitive substring matching. Always use quotes around search text."
                }
            },
            "boolean_logic": {
                "AND_operator": {
                    "syntax": "condition1 AND condition2",
                    "description": "Both conditions must match",
                    "examples": [
                        "language:python AND function_name:main",
                        "exists(embedding) AND language:rust",
                        'value_contains(code, "async") AND language:python'
                    ],
                    "precedence": "Higher than OR"
                },
                "OR_operator": {
                    "syntax": "condition1 OR condition2",
                    "description": "Either condition can match",
                    "examples": [
                        "language:python OR language:rust",
                        "function_name:main OR function_name:init",
                        'value_contains(code, "TODO") OR value_contains(code, "FIXME")'
                    ],
                    "precedence": "Lower than AND"
                },
                "grouping": {
                    "syntax": "(condition1 OR condition2) AND condition3",
                    "description": "Use parentheses to control evaluation order",
                    "examples": [
                        "(language:python OR language:rust) AND exists(function_name)",
                        'exists(embedding) AND (filename:main_interactive_query.py OR filename:lib.rs)',
                        '(value_contains(code, "async") OR value_contains(code, "await")) AND language:python'
                    ],
                    "notes": "Parentheses override default operator precedence"
                }
            },
            "available_fields": {
                "description": "Common fields available for querying (varies by indexed content)",
                "fields": {
                    "filename": "Source code filename (e.g., 'main_interactive_query.py', 'lib.rs')",
                    "language": "Programming language (e.g., 'python', 'rust', 'javascript')",
                    "code": "Full source code content of the chunk",
                    "function_name": "Extracted function/method names from the code chunk",
                    "class_name": "Extracted class names from the code chunk",
                    "embedding": "Vector embedding representation (use exists() to check)",
                    "start_line": "Starting line number in source file",
                    "end_line": "Ending line number in source file"
                },
                "note": "Use the 'get_database_fields' resource to see all available fields for your specific database"
            },
            "complete_examples": {
                "simple_queries": [
                    "language:python",
                    "exists(function_name)",
                    'value_contains(filename, ".test.")',
                    "function_name:main"
                ],
                "intermediate_queries": [
                    "language:python AND exists(function_name)",
                    "language:rust OR language:go",
                    'value_contains(code, "async") AND language:python',
                    "exists(embedding) AND language:javascript"
                ],
                "advanced_queries": [
                    "(language:python OR language:rust) AND exists(function_name)",
                    'exists(embedding) AND (value_contains(code, "test") OR value_contains(filename, "test"))',
                    '(function_name:main OR function_name:init) AND (language:python OR language:rust)',
                    'value_contains(code, "TODO") OR value_contains(code, "FIXME") OR value_contains(code, "HACK")'
                ]
            },
            "syntax_rules": {
                "field_names": "Must start with letter/underscore, can contain letters, numbers, underscores",
                "quoted_values": "Use single or double quotes for values with spaces/special characters",
                "case_sensitivity": "Field names are case-sensitive, operators (AND/OR) are case-insensitive",
                "whitespace": "Whitespace around operators is ignored"
            },
            "common_mistakes": {
                "missing_quotes": {
                    "wrong": "filename:test file.py",
                    "correct": 'filename:"test file.py"',
                    "reason": "Spaces in values require quotes"
                },
                "incorrect_contains_syntax": {
                    "wrong": 'code contains "async"',
                    "correct": 'value_contains(code, "async")',
                    "reason": "Use value_contains() function syntax"
                },
                "operator_precedence": {
                    "wrong": "language:python OR language:rust AND exists(function_name)",
                    "correct": "(language:python OR language:rust) AND exists(function_name)",
                    "reason": "AND has higher precedence than OR; use parentheses for clarity"
                }
            }
        },
        "additional_resources": {
            "mcp_resources": [
                "cocoindex://query/grammar - Lark grammar definition",
                "cocoindex://query/examples - Categorized example queries",
                "cocoindex://database/fields - Available database fields and types",
                "cocoindex://query/operators - Detailed operator reference"
            ],
            "note": "Use MCP resource reading to access detailed documentation and examples"
        }
    }
    
    return help_content


async def get_search_stats() -> str:
    """Get database and search statistics."""
    global connection_pool
    
    if not connection_pool:
        return json.dumps({"error": "No database connection"})
    
    try:
        with connection_pool.connection() as conn:
            with conn.cursor() as cur:
                # Get table stats
                table_name = hybrid_search_engine.table_name
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                total_records = cur.fetchone()[0]
                
                cur.execute(f"""
                    SELECT pg_size_pretty(pg_total_relation_size('{table_name}')) as table_size,
                           pg_size_pretty(pg_relation_size('{table_name}')) as table_data_size
                """)
                size_info = cur.fetchone()
                
                stats = {
                    "table_name": table_name,
                    "total_records": total_records,
                    "table_size": size_info[0],
                    "table_data_size": size_info[1],
                    "connection_pool_size": connection_pool.max_size,
                    "active_connections": connection_pool.get_stats()["requests_num"] if hasattr(connection_pool, "get_stats") else "unknown"
                }
                
        return json.dumps(stats, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get stats: {str(e)}"})


async def get_search_config() -> str:
    """Get current search configuration."""
    config = {
        "table_name": hybrid_search_engine.table_name if hybrid_search_engine else "unknown",
        "embedding_model": "cocoindex default",
        "parser_type": "lark_keyword_parser",
        "supported_operators": ["AND", "OR", "NOT", "value_contains", "==", "!=", "<", ">", "<=", ">="],
        "default_weights": {
            "vector_weight": 0.7,
            "keyword_weight": 0.3
        }
    }
    return json.dumps(config, indent=2)


async def get_database_schema() -> str:
    """Get database schema information."""
    global connection_pool
    
    if not connection_pool:
        return json.dumps({"error": "No database connection"})
    
    try:
        with connection_pool.connection() as conn:
            with conn.cursor() as cur:
                table_name = hybrid_search_engine.table_name
                cur.execute(f"""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position
                """)
                columns = cur.fetchall()
                
                schema = {
                    "table_name": table_name,
                    "columns": [
                        {
                            "name": col[0],
                            "type": col[1], 
                            "nullable": col[2] == "YES",
                            "default": col[3]
                        }
                        for col in columns
                    ]
                }
                
        return json.dumps(schema, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get schema: {str(e)}"})


async def get_query_grammar() -> str:
    """Get Lark grammar for keyword query syntax."""
    import os
    import pathlib
    
    # Find grammar file relative to this script
    grammar_path = pathlib.Path(__file__).parent / "grammars" / "keyword_search.lark"
    
    try:
        with open(grammar_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading grammar file: {str(e)}"


async def get_query_examples() -> str:
    """Get categorized examples of keyword query syntax."""
    examples = {
        "basic_matching": {
            "description": "Simple field matching",
            "examples": [
                "language:python",
                "filename:main_interactive_query.py", 
                'filename:"test file.py"',
                "function_name:parse"
            ]
        },
        "existence_checks": {
            "description": "Check if field exists",
            "examples": [
                "exists(embedding)",
                "exists(function_name)", 
                "exists(language) AND language:rust"
            ]
        },
        "value_contains": {
            "description": "Substring search within field values",
            "examples": [
                'value_contains(code, "async")',
                'value_contains(filename, "test")',
                'value_contains(code, "function") AND language:python'
            ]
        },
        "boolean_logic": {
            "description": "Combining conditions with AND/OR",
            "examples": [
                "language:python AND function_name:main",
                "(language:python OR language:rust) AND exists(embedding)",
                'value_contains(code, "async") OR value_contains(code, "await")'
            ]
        },
        "complex_queries": {
            "description": "Advanced combinations with grouping",
            "examples": [
                '(language:python OR language:javascript) AND value_contains(code, "async")',
                'exists(embedding) AND (filename:main_interactive_query.py OR filename:index.js)',
                '(value_contains(code, "function") OR value_contains(code, "class")) AND language:python'
            ]
        },
        "common_patterns": {
            "description": "Frequently used search patterns",
            "examples": [
                "language:python AND exists(function_name)",
                'value_contains(code, "TODO") OR value_contains(code, "FIXME")',
                "language:rust AND function_name:main",
                'exists(embedding) AND value_contains(filename, ".py")'
            ]
        }
    }
    return json.dumps(examples, indent=2)


async def get_database_fields() -> str:
    """Get available database fields and their types for query building."""
    global connection_pool
    
    if not connection_pool:
        return json.dumps({"error": "No database connection"})
    
    try:
        with connection_pool.connection() as conn:
            with conn.cursor() as cur:
                table_name = hybrid_search_engine.table_name if hybrid_search_engine else "code_embeddings"
                
                # Get column information from database
                cur.execute(f"""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position
                """)
                columns = cur.fetchall()
                
                # Enhanced field descriptions based on our schema
                field_descriptions = {
                    "filename": "Source code filename (e.g., 'main_interactive_query.py', 'lib.rs')",
                    "language": "Programming language (e.g., 'python', 'rust', 'javascript')",  
                    "code": "Full source code content of the chunk",
                    "function_name": "Extracted function/method names from the code chunk",
                    "class_name": "Extracted class names from the code chunk",
                    "embedding": "Vector embedding representation (for similarity search)",
                    "metadata_json": "Additional metadata in JSON format",
                    "id": "Unique identifier for the code chunk",
                    "start_line": "Starting line number in the source file",
                    "end_line": "Ending line number in the source file"
                }
                
                # Common values for enum-like fields
                common_values = {
                    "language": ["python", "rust", "javascript", "typescript", "java", "go", "c", "cpp"],
                    "function_name": ["main", "init", "setup", "test", "parse", "get", "set"],
                    "class_name": ["Class", "Handler", "Builder", "Config", "Test"]
                }
                
                fields = {}
                for col in columns:
                    col_name = col[0]
                    col_type = col[1]
                    nullable = col[2] == "YES"
                    
                    field_info = {
                        "type": col_type,
                        "nullable": nullable,
                        "description": field_descriptions.get(col_name, f"Database field of type {col_type}")
                    }
                    
                    if col_name in common_values:
                        field_info["common_values"] = common_values[col_name]
                    
                    # Add query examples for each field
                    if col_name == "embedding":
                        field_info["query_examples"] = ["exists(embedding)"]
                    elif col_name in ["filename", "code"]:
                        field_info["query_examples"] = [
                            f'{col_name}:"example value"',
                            f'value_contains({col_name}, "search_term")'
                        ]
                    else:
                        field_info["query_examples"] = [f'{col_name}:value']
                        if col_type in ["text", "character varying"]:
                            field_info["query_examples"].append(f'value_contains({col_name}, "search_term")')
                    
                    fields[col_name] = field_info
                
                result = {
                    "table_name": table_name,
                    "fields": fields,
                    "query_help": {
                        "field_matching": "Use field:value or field:\"quoted value\"",
                        "existence": "Use exists(field) to check if field has any value",
                        "contains": "Use value_contains(field, \"text\") for substring search",
                        "boolean": "Combine with AND/OR and use parentheses for grouping"
                    }
                }
                
                return json.dumps(result, indent=2)
                
    except Exception as e:
        return json.dumps({"error": f"Failed to get database fields: {str(e)}"})


async def get_query_operators() -> str:
    """Get detailed reference for all supported query operators."""
    operators = {
        "field_matching": {
            "syntax": "field:value",
            "description": "Match exact field value",
            "examples": [
                "language:python",
                "function_name:main",
                'filename:"test file.py"'
            ],
            "notes": "Use quotes for values with spaces or special characters"
        },
        "existence_check": {
            "syntax": "exists(field)",
            "description": "Check if field exists and has a non-null value", 
            "examples": [
                "exists(embedding)",
                "exists(function_name)",
                "exists(class_name)"
            ],
            "notes": "Useful for filtering records that have specific extracted metadata"
        },
        "value_contains": {
            "syntax": 'value_contains(field, "search_text")',
            "description": "Substring search within field values",
            "examples": [
                'value_contains(code, "async")',
                'value_contains(filename, "test")',
                'value_contains(code, "TODO")'
            ],
            "notes": "Case-sensitive substring matching. Use quotes around search text."
        },
        "boolean_and": {
            "syntax": "condition1 AND condition2",
            "description": "Both conditions must match",
            "examples": [
                "language:python AND function_name:main",
                "exists(embedding) AND language:rust",
                'value_contains(code, "async") AND language:python'
            ],
            "notes": "AND has higher precedence than OR"
        },
        "boolean_or": {
            "syntax": "condition1 OR condition2", 
            "description": "Either condition can match",
            "examples": [
                "language:python OR language:rust",
                "function_name:main OR function_name:init",
                'value_contains(code, "TODO") OR value_contains(code, "FIXME")'
            ],
            "notes": "OR has lower precedence than AND"
        },
        "grouping": {
            "syntax": "(condition1 OR condition2) AND condition3",
            "description": "Control evaluation order with parentheses",
            "examples": [
                "(language:python OR language:rust) AND exists(function_name)",
                'exists(embedding) AND (filename:main_interactive_query.py OR filename:lib.rs)',
                '(value_contains(code, "async") OR value_contains(code, "await")) AND language:python'
            ],
            "notes": "Use parentheses to override default operator precedence"
        }
    }
    
    result = {
        "operators": operators,
        "precedence": {
            "description": "Operator precedence from highest to lowest",
            "order": [
                "parentheses ()",
                "field conditions (field:value, exists(), value_contains())",
                "AND",
                "OR"
            ]
        },
        "syntax_rules": {
            "field_names": "Must start with letter/underscore, can contain letters, numbers, underscores",
            "quoted_values": "Use single or double quotes for values with spaces/special chars",
            "case_sensitivity": "Field names are case-sensitive, operators (AND/OR) are case-insensitive",
            "whitespace": "Whitespace around operators is ignored"
        },
        "common_mistakes": {
            "missing_quotes": "Use quotes around values with spaces: filename:\"test file.py\"",
            "wrong_contains_syntax": 'Use value_contains(field, "text") not field contains "text"',
            "precedence_errors": "Use parentheses to group OR conditions: (a OR b) AND c"
        }
    }
    
    return json.dumps(result, indent=2)


async def initialize_search_engine():
    """Initialize the hybrid search engine and database connection."""
    global hybrid_search_engine, connection_pool
    
    try:
        # Get database URL from environment
        database_url = os.getenv("COCOINDEX_DATABASE_URL")
        if not database_url:
            raise ValueError("COCOINDEX_DATABASE_URL not found in environment")
        
        # Create connection pool with shorter timeout
        connection_pool = ConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=5,
            timeout=10.0  # Shorter timeout
        )
        
        # Register pgvector
        with connection_pool.connection() as conn:
            register_vector(conn)
        
        # Initialize search engine components
        parser = KeywordSearchParser()
        
        # Initialize hybrid search engine
        hybrid_search_engine = HybridSearchEngine(
            pool=connection_pool,
            parser=parser,
            embedding_func=safe_embedding_function
        )
        
        print("CocoIndex RAG MCP Server initialized successfully", file=sys.stderr)
        
    except Exception as e:
        print(f"Failed to initialize search engine: {e}", file=sys.stderr)
        sys.exit(1)


async def background_initialization(live_enabled: bool, poll_interval: int):
    """Perform heavy initialization in the background after MCP server starts."""
    try:
        # Skip pre-loading embedding model to avoid HuggingFace rate limits
        print("⏭️  Skipping embedding model pre-loading (models will load on first use)", file=sys.stderr)
        
        # Pre-initialize database connection synchronously
        print("🔧 Pre-initializing database connection...", file=sys.stderr)
        try:
            # Get database URL from environment
            database_url = os.getenv("COCOINDEX_DATABASE_URL")
            if not database_url:
                print("❌ COCOINDEX_DATABASE_URL not found in environment", file=sys.stderr)
                return
            
            # Test basic connection first
            import psycopg
            test_conn = psycopg.connect(database_url)
            test_conn.close()
            print("✅ Database connection test successful", file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}", file=sys.stderr)
            return
        
        # Initialize the search engine
        await initialize_search_engine()
        
        # Set up live updates if enabled
        if live_enabled:
            print("🔄 Running initial flow update...", file=sys.stderr)
            
            def run_flow_background():
                """Background thread function for live flow updates."""
                while not shutdown_event.is_set() and not _terminating.is_set():
                    try:
                        run_flow_update(live_update=True)
                        # Wait for the polling interval or until shutdown
                        if shutdown_event.wait(poll_interval) or _terminating.is_set():
                            break  # Shutdown requested
                    except Exception as e:
                        if not shutdown_event.is_set() and not _terminating.is_set():
                            print(f"Error in background flow update: {e}", file=sys.stderr)
                            # Wait a bit before retrying to avoid tight error loops
                            if shutdown_event.wait(10) or _terminating.is_set():
                                break
                        else:
                            print("Background flow update stopped due to shutdown", file=sys.stderr)
            
            # Start background flow update
            global background_thread
            background_thread = threading.Thread(target=run_flow_background, daemon=True)
            background_thread.start()
            print("✅ Background flow update started", file=sys.stderr)
        else:
            print("🔄 Running one-time flow update...", file=sys.stderr)
            run_flow_update(live_update=False)
            print("✅ Flow update completed", file=sys.stderr)
            
    except Exception as e:
        print(f"❌ Background initialization failed: {e}", file=sys.stderr)


async def handle_jsonrpc_request(request_data: dict) -> dict:
    """Handle JSON-RPC request and return response."""
    method = request_data.get("method")
    params = request_data.get("params", {})
    request_id = request_data.get("id")
    
    try:
        if method == "initialize":
            # Handle initialize request
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {"listChanged": True},
                    },
                    "serverInfo": {
                        "name": "cocoindex-rag",
                        "version": "1.0.0"
                    }
                }
            }
        elif method == "tools/list":
            # Get tools from server handlers
            tools = await handle_list_tools()
            return {
                "jsonrpc": "2.0", 
                "id": request_id,
                "result": {"tools": [tool.model_dump(mode='json', exclude_none=True) for tool in tools]}
            }
        elif method == "resources/list":
            # Get resources from server handlers
            resources = await handle_list_resources()
            return {
                "jsonrpc": "2.0",
                "id": request_id, 
                "result": {"resources": [resource.model_dump(mode='json', exclude_none=True) for resource in resources]}
            }
        elif method == "resources/read":
            # Read specific resource
            uri = params.get("uri")
            content = await handle_read_resource(uri)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"contents": [{"uri": uri, "text": content}]}
            }
        elif method == "tools/call":
            # Call specific tool
            name = params.get("name")
            arguments = params.get("arguments", {})
            result = await handle_call_tool(name, arguments)
            return {
                "jsonrpc": "2.0", 
                "id": request_id,
                "result": {"content": [content.model_dump(mode='json', exclude_none=True) for content in result]}
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_id, 
            "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
        }


async def shutdown_monitor():
    """Monitor shutdown event and signal shutdown."""
    while not shutdown_event.is_set() and not _terminating.is_set():
        await asyncio.sleep(0.1)
    
    # Just return when shutdown is requested - let the main loop handle cleanup
    return


async def run_http_server(port: int, live_enabled: bool, poll_interval: int):
    """Run MCP server using modern Streamable HTTP transport."""
    print(f"🌐 Starting HTTP MCP server on port {port}...", file=sys.stderr)
    
    # Initialize background components
    await background_initialization(live_enabled, poll_interval)
    
    # Handle /mcp endpoint for Streamable HTTP transport
    async def handle_mcp_endpoint(request):
        """Handle MCP requests using Streamable HTTP transport."""
        from starlette.responses import Response
        
        if request.method == "POST":
            try:
                # Handle JSON-RPC over HTTP POST
                body = await request.body()
                
                # Parse JSON-RPC request
                import json
                request_data = json.loads(body.decode('utf-8'))
                
                # Handle the request directly instead of using server.run
                response = await handle_jsonrpc_request(request_data)
                
                # Return JSON response
                return Response(
                    content=json.dumps(response),
                    media_type="application/json",
                    headers={"Access-Control-Allow-Origin": "*"}
                )
                
            except Exception as e:
                print(f"MCP request error: {e}", file=sys.stderr)
                return Response(
                    content=f'{{"error": "Request failed: {e}"}}',
                    status_code=500,
                    media_type="application/json"
                )
        else:
            return Response(
                content='{"error": "Only POST requests supported"}',
                status_code=405,
                media_type="application/json"
            )
    
    # Create Starlette application
    app = Starlette(
        routes=[
            Route("/mcp", handle_mcp_endpoint, methods=["POST", "OPTIONS"]),
        ]
    )
    
    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="info"
    )
    
    # Run server
    server_instance = uvicorn.Server(config)
    
    # Start background tasks
    shutdown_task = asyncio.create_task(shutdown_monitor())
    
    try:
        print(f"📡 MCP endpoint: http://127.0.0.1:{port}/mcp", file=sys.stderr)
        
        # Run HTTP server
        await server_instance.serve()
        
    except OSError as e:
        if "Address already in use" in str(e) or e.errno == 98:
            print(f"❌ Port {port} is already in use. Please choose a different port or stop the existing server.", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"❌ Network error: {e}", file=sys.stderr)
            sys.exit(1)
    except KeyboardInterrupt:
        print("🛑 Server stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"❌ HTTP server error: {e}", file=sys.stderr)
        shutdown_event.set()
        sys.exit(1)
    finally:
        print("🏁 HTTP MCP server stopped", file=sys.stderr)
        # Ensure clean shutdown
        shutdown_event.set()


async def run_http_client(url: str, live_enabled: bool, poll_interval: int):
    """Run as HTTP client connecting to existing server."""
    print(f"🔗 Connecting to HTTP MCP server at {url}...", file=sys.stderr)
    
    # For now, just print info - this would typically be handled by Claude Code client
    print(f"ℹ️  HTTP client mode not implemented - this should be handled by MCP client", file=sys.stderr)
    print(f"ℹ️  Expected server URL: {url}", file=sys.stderr)
    
    # Keep process alive for testing
    try:
        while not shutdown_event.is_set():
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass


async def main(args):
    """Main entry point for the MCP server."""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Load environment and initialize CocoIndex
    load_dotenv()
    cocoindex.init()
    
    # Determine paths to use
    paths = determine_paths(args)
    
    # Configure live updates (enabled by default)
    live_enabled = not args.no_live
    poll_interval = args.poll
    
    # Update flow configuration
    update_flow_config(
        paths=paths,
        enable_polling=live_enabled and poll_interval > 0,
        poll_interval=poll_interval,
        use_default_embedding=args.default_embedding,
        use_default_chunking=args.default_chunking,
        use_default_language_handler=args.default_language_handler
    )
    
    # Determine transport mode and run accordingly
    if args.port:
        # HTTP Server mode
        display_mcp_configuration(args, paths)
        print(f"🌐 Running as HTTP server on port {args.port}", file=sys.stderr)
        await run_http_server(args.port, live_enabled, poll_interval)
        return
        
    elif args.url:
        # HTTP Client mode  
        display_mcp_configuration(args, paths)
        print(f"🔗 Running as HTTP client connecting to {args.url}", file=sys.stderr)
        await run_http_client(args.url, live_enabled, poll_interval)
        return
        
    else:
        # Stdio mode (backward compatibility) - use temporary workaround
        if not sys.stdin.isatty():
            # MCP stdio mode - skip configuration display to avoid interference
            print("🚀 MCP server starting (stdio mode)...", file=sys.stderr)
        else:
            # Interactive stdio mode
            display_mcp_configuration(args, paths)
            print("🚀 MCP server starting (stdio mode)...", file=sys.stderr)
    
    try:
        # TODO: 
        # mcp.server.stdio.stdio_server() is not suited for streamable HTTP transport
        # we should stick to the example
        # https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/servers/simple-streamablehttp-stateless/mcp_simple_streamablehttp_stateless/server.py
        # Run the MCP server
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            # Start background initialization and shutdown monitor
            init_task = asyncio.create_task(background_initialization(live_enabled, poll_interval))
            shutdown_task = asyncio.create_task(shutdown_monitor())
            
            # Run server with race condition to handle shutdown
            server_task = asyncio.create_task(server.run(
                read_stream,
                write_stream,
                NotificationOptions(
                    tools_changed=True,
                    resources_changed=True,
                ),
            ))
            
            # Wait for either server completion or shutdown
            try:
                done, pending = await asyncio.wait(
                    [server_task, shutdown_task], 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel any pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            except asyncio.CancelledError:
                # Handle cancellation gracefully
                pass
                    
    except Exception as e:
        print(f"❌ MCP server error: {e}", file=sys.stderr)
        shutdown_event.set()
    finally:
        # Ensure cleanup happens
        global background_thread
        if background_thread and background_thread.is_alive():
            print("⏳ Waiting for background thread to finish...", file=sys.stderr)
            background_thread.join(timeout=3.0)
        print("🏁 MCP server stopped", file=sys.stderr)


if __name__ == "__main__":
    try:
        # Parse command line arguments - now safe for HTTP transport
        args = parse_mcp_args()
        _want_to_save_coverage = args.coverage
        LOGGER.info(f"Running with coverage: {_want_to_save_coverage}")
        if coverage is not None and _want_to_save_coverage:
            global _cov
            _cov = coverage.Coverage(auto_data=True)

        asyncio.run(main(args))
    except (KeyboardInterrupt, SystemExit):
        # Graceful shutdown
        pass
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
