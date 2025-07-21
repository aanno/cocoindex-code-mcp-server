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


# Initialize the MCP server
server = Server("cocoindex-rag")

# Global state
hybrid_search_engine: Optional[HybridSearchEngine] = None
connection_pool: Optional[ConnectionPool] = None
shutdown_event = threading.Event()
background_thread: Optional[threading.Thread] = None
_terminating = threading.Event()  # Atomic termination flag


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
  python mcp_server.py                         # Use default path (cocoindex) with live updates
  python mcp_server.py /path/to/code           # Index single directory
  python mcp_server.py /path/to/code1 /path/to/code2  # Index multiple directories
  python mcp_server.py --paths /path/to/code   # Explicit paths argument
  
  # Live update configuration (enabled by default)
  python mcp_server.py --no-live               # Disable live updates
  python mcp_server.py --poll 30               # Custom polling interval (default: 60s)

MCP Tools Available:
  - hybrid_search: Combine vector similarity and keyword metadata filtering
  - vector_search: Pure vector similarity search
  - keyword_search: Pure keyword metadata search  
  - analyze_code: Code analysis and metadata extraction
  - get_embeddings: Generate embeddings for text

MCP Resources Available:
  - search_stats: Database and search performance statistics
  - search_config: Current hybrid search configuration
  - database_schema: Database table structure information
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
    else:
        raise ValueError(f"Unknown resource: {uri}")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available MCP tools."""
    return [
        types.Tool(
            name="hybrid_search",
            description="Perform hybrid search combining vector similarity and keyword metadata filtering",
            inputSchema={
                "type": "object",
                "properties": {
                    "vector_query": {
                        "type": "string",
                        "description": "Text to embed and search for semantic similarity"
                    },
                    "keyword_query": {
                        "type": "string", 
                        "description": "Keyword search query for metadata filtering (e.g., 'function_name:parse AND language:python')"
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
            description="Perform pure keyword metadata search",
            inputSchema={
                "type": "object", 
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword search query (e.g., 'function_name:parse AND language:python')"
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
        # Pre-warm the embedding function to load the model synchronously
        print("🔧 Pre-loading embedding model...", file=sys.stderr)
        try:
            code_to_embedding.eval("test")  # Force model loading
            print("✅ Embedding model loaded successfully", file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to pre-load embedding model: {e}", file=sys.stderr)
            return
        
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


async def main():
    """Main entry point for the MCP server."""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Load environment and initialize CocoIndex
    load_dotenv()
    cocoindex.init()
    
    # Parse command line arguments - now safe for HTTP transport
    args = parse_mcp_args()
    
    # Determine paths to use
    paths = determine_paths(args)
    
    # Configure live updates (enabled by default)
    live_enabled = not args.no_live
    poll_interval = args.poll
    
    # Update flow configuration
    update_flow_config(
        paths=paths,
        enable_polling=live_enabled and poll_interval > 0,
        poll_interval=poll_interval
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
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        # Graceful shutdown
        pass
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)