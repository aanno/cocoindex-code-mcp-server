#!/usr/bin/env python3

"""
CocoIndex RAG MCP Server

A Model Context Protocol (MCP) server that provides hybrid search capabilities
combining vector similarity and keyword metadata search for code retrieval.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

# Local imports
from hybrid_search import HybridSearchEngine
from keyword_search_parser_lark import KeywordSearchParser
from lang.python.python_code_analyzer import analyze_python_code
import cocoindex
from cocoindex_config import code_embedding_flow, code_to_embedding


# Initialize the MCP server
server = Server("cocoindex-rag")

# Global state
hybrid_search_engine: Optional[HybridSearchEngine] = None
connection_pool: Optional[ConnectionPool] = None


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
        # Get database configuration from environment or defaults
        db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "dbname": os.getenv("DB_NAME", "cocoindex"),
            "user": os.getenv("DB_USER", "postgres"), 
            "password": os.getenv("DB_PASSWORD", "password")
        }
        
        # Create connection pool
        connection_pool = ConnectionPool(
            conninfo=f"host={db_config['host']} port={db_config['port']} "
                    f"dbname={db_config['dbname']} user={db_config['user']} "
                    f"password={db_config['password']}",
            min_size=2,
            max_size=10
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
            embedding_func=lambda q: code_to_embedding.eval(q)
        )
        
        print("CocoIndex RAG MCP Server initialized successfully", file=sys.stderr)
        
    except Exception as e:
        print(f"Failed to initialize search engine: {e}", file=sys.stderr)
        sys.exit(1)


async def main():
    """Main entry point for the MCP server."""
    # Initialize the search engine
    await initialize_search_engine()
    
    # Run the MCP server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            NotificationOptions(
                tools_changed=True,
                resources_changed=True,
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())