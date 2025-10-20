#!/usr/bin/env python3
"""
Minimal MCP server for testing what Claude Desktop expects.
"""

import asyncio

import mcp.server.sse
import mcp.server.stdio
import mcp.types as types
import uvicorn
from mcp.server import Server
from starlette.applications import Starlette
from starlette.routing import Route

# Initialize the MCP server
server = Server("minimal-test")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available MCP tools."""
    return [
        types.Tool(
            name="test_tool",
            description="A simple test tool",
            inputSchema={
                "type": "object",
                "properties": {"message": {"type": "string", "description": "Test message"}},
                "required": ["message"],
            },
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle MCP tool calls."""
    if name == "test_tool":
        return [types.TextContent(type="text", text=f"Test response: {arguments.get('message', 'no message')}")]
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_http_server(port: int):
    """Run the MCP server over HTTP using Server-Sent Events (SSE)."""
    print(f"🌐 Starting minimal MCP server on http://127.0.0.1:{port}")

    # Create SSE transport
    sse_transport = mcp.server.sse.SseServerTransport("/messages", "/sse")

    # Create Starlette app with SSE endpoints
    async def handle_sse(request):
        """Handle SSE subscriptions."""
        return await sse_transport.handle_sse_request(request)

    async def handle_messages(request):
        """Handle message requests."""
        return await sse_transport.handle_post_message(request, server)

    # Create Starlette routes
    routes = [
        Route("/sse", handle_sse, methods=["GET"]),
        Route("/messages", handle_messages, methods=["POST"]),
        Route("/mcp", handle_messages, methods=["POST"]),  # Alternative endpoint
    ]

    app = Starlette(routes=routes)

    # Run the server
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="info")
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3034
    asyncio.run(run_http_server(port))
