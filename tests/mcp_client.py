#!/usr/bin/env python3

"""
Common MCP Test Client

A reusable test client for MCP server integration testing.
Based on ai-enhanced-app-dev/mcp-python/tests/mcp_client.py with adaptations
for our CocoIndex RAG MCP server.
"""

import json
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional, cast

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl

logger = logging.getLogger(__name__)


class MCPTestClient:
    """
    Test client for MCP servers supporting both stdio and HTTP transports.

    This client provides a unified interface for testing MCP servers with
    proper connection management and response parsing.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 3033, transport: str = "http"):
        """
        Initialize MCP test client.

        Args:
            host: Server host (default: 127.0.0.1)
            port: Server port (default: 3033 for our cocoindex server)
            transport: Transport type ('http' or 'stdio', default: 'http')
        """
        self.host = host
        self.port = port
        self.transport = transport

        if transport == "http":
            self.server_url = f"http://{host}:{port}/mcp/"
            self.http_client: Optional[httpx.AsyncClient] = httpx.AsyncClient(timeout=30.0)
        else:
            self.server_url = f"http://{host}:{port}/mcp/"
            self.http_client = None

        # For streaming HTTP MCP client
        self.session: Optional[ClientSession] = None
        self._client_context: Any = None
        self._session_context: Optional[ClientSession] = None

    async def connect(self):
        """Connect to the MCP server using the specified transport."""
        if self.transport == "http":
            await self._connect_http()
        else:
            await self._connect_streaming()

    async def _connect_streaming(self):
        """Connect using streaming HTTP transport (official MCP client)."""
        if not self.server_url:
            raise RuntimeError("Server URL not set for streaming transport")

        self._client_context = streamablehttp_client(url=self.server_url, timeout=timedelta(seconds=30))
        read_stream, write_stream, get_session_id = await self._client_context.__aenter__()

        self._session_context = ClientSession(read_stream, write_stream)
        self.session = await self._session_context.__aenter__()
        await self.session.initialize()
        logger.info("Connected to MCP server at %s (streaming)", self.server_url)

    async def _connect_http(self):
        """Connect using direct HTTP transport (for compatibility)."""
        # For HTTP transport, we don't need a persistent connection
        # Just verify the server is accessible
        try:
            if not self.http_client:
                raise RuntimeError("HTTP client not initialized")

            response = await self.http_client.post(
                self.server_url,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            )

            result = self._parse_http_response(response)
            if "error" in result:
                raise Exception(f"MCP Error: {result['error']}")

            logger.info("Connected to MCP server at %s (HTTP)", self.server_url)
        except Exception as e:
            logger.error("Failed to connect to MCP server: %s", e)
            raise

    def _parse_http_response(self, response):
        """Parse HTTP response (either JSON or SSE format)."""
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        response_text = response.text

        # Handle Server-Sent Events format
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            # Parse SSE format
            for line in response_text.split("\n"):
                if line.startswith("data: "):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data.strip():
                        try:
                            return json.loads(data)
                        except json.JSONDecodeError:
                            continue
            raise Exception("No valid JSON data found in SSE response")
        else:
            # Regular JSON response
            return response.json()

    async def close(self):
        """Close the connection."""
        if self.transport == "http" and self.http_client:
            await self.http_client.aclose()
        else:
            # Streaming transport cleanup
            if self._session_context:
                try:
                    await self._session_context.__aexit__(None, None, None)
                except Exception as e:
                    logger.warning("Error closing session context: %s", e)
            if self._client_context:
                try:
                    await self._client_context.__aexit__(None, None, None)
                except Exception as e:
                    logger.warning("Error closing client context: %s", e)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def call_tool(self, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call a tool and return normalized result.

        Returns:
            Dict with keys:
                - isError: bool indicating if there was an error
                - content: List of content items (for compatibility)
                - result: Raw result data
        """
        if self.transport == "http":
            return await self._call_tool_http(tool_name, params)
        else:
            return await self._call_tool_streaming(tool_name, params)

    async def _call_tool_streaming(self, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call tool using streaming transport."""
        if not self.session:
            raise RuntimeError("Not connected to server")

        result = await self.session.call_tool(tool_name, params or {})

        # Normalize result format
        response: Dict[str, Any] = {"isError": getattr(result, "isError", False), "content": [], "result": result}

        if hasattr(result, "content") and result.content:
            for content in result.content:
                if hasattr(content, "text"):
                    response["content"].append(content.text)

        if hasattr(result, "structuredContent") and result.structuredContent:
            response["structuredContent"] = result.structuredContent

        return response

    async def _call_tool_http(self, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call tool using HTTP transport."""
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")

        response = await self.http_client.post(
            self.server_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": params or {}},
            },
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        )

        result = self._parse_http_response(response)
        if "error" in result:
            return {"isError": True, "content": [f"MCP Error: {result['error']['message']}"], "result": result}

        # Parse MCP result format
        mcp_result = result["result"]
        content_items = []

        # MCP returns {"content": [{"type": "text", "text": "..."}]}
        if isinstance(mcp_result, dict) and "content" in mcp_result:
            for item in mcp_result["content"]:
                # Check if the content contains an error response
                try:
                    content_data = json.loads(item["text"])
                    if isinstance(content_data, dict) and "error" in content_data:
                        return {
                            "isError": True,
                            "content": [f"MCP Tool Error: {content_data['error']['message']}"],
                            "result": result,
                        }
                except json.JSONDecodeError:
                    pass  # Not JSON, continue normally

                content_items.append(item["text"])

        return {"isError": False, "content": content_items, "result": mcp_result}

    async def read_resource(self, resource_name_or_uri: str) -> str:
        """
        Read a resource and return text content.

        Args:
            resource_name_or_uri: Either resource name or full URI
        """
        if self.transport == "http":
            return await self._read_resource_http(resource_name_or_uri)
        else:
            return await self._read_resource_streaming(resource_name_or_uri)

    async def _read_resource_streaming(self, resource_name_or_uri: str) -> str:
        """Read resource using streaming transport."""
        if not self.session:
            raise RuntimeError("Not connected to server")

        # If it's just a name, find the URI
        if not resource_name_or_uri.startswith(("http://", "https://", "cocoindex://")):
            resources_result = await self.session.list_resources()
            resource = next((r for r in resources_result.resources if r.name == resource_name_or_uri), None)

            if not resource:
                raise RuntimeError(f"Resource '{resource_name_or_uri}' not found")
            resource_uri = resource.uri
        else:
            resource_uri = AnyUrl(resource_name_or_uri)

        result = await self.session.read_resource(resource_uri)

        if hasattr(result, "contents") and result.contents:
            for content in result.contents:
                if hasattr(content, "text"):
                    return content.text

        return ""

    async def _read_resource_http(self, resource_name_or_uri: str) -> str:
        """Read resource using HTTP transport."""
        # If it's just a name, we need to resolve it to a URI
        if not resource_name_or_uri.startswith(("http://", "https://", "cocoindex://")):
            # List resources to find the URI
            resources = await self.list_resources()
            # For HTTP transport, we get resource objects with attributes
            matching_resource = None
            for resource in resources:
                if hasattr(resource, "name") and resource.name == resource_name_or_uri:
                    matching_resource = resource
                    break

            if not matching_resource:
                raise RuntimeError(f"Resource '{resource_name_or_uri}' not found")
            resource_uri = matching_resource.uri
        else:
            resource_uri = AnyUrl(resource_name_or_uri)

        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")

        response = await self.http_client.post(
            self.server_url,
            json={"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": {"uri": resource_uri}},
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        )

        result = self._parse_http_response(response)
        if "error" in result:
            raise Exception(f"MCP Error: {result['error']}")

        # Extract text content
        if "result" in result and "contents" in result["result"]:
            for content in result["result"]["contents"]:
                if "text" in content:
                    return content["text"]

        return ""

    async def list_tools(self) -> List[Any]:
        """Get list of available tools."""
        if self.transport == "http":
            return await self._list_tools_http()
        else:
            return await self._list_tools_streaming()

    async def _list_tools_streaming(self) -> List[str]:
        """List tools using streaming transport."""
        if not self.session:
            raise RuntimeError("Not connected to server")

        result = await self.session.list_tools()
        return [tool.name for tool in result.tools]

    async def _list_tools_http(self) -> List[Any]:
        """List tools using HTTP transport."""
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")

        response = await self.http_client.post(
            self.server_url,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        )

        result = self._parse_http_response(response)
        if "error" in result:
            raise Exception(f"MCP Error: {result['error']}")

        # Convert MCP result to tool objects for compatibility
        from types import SimpleNamespace

        tools = []
        for tool in result["result"]["tools"]:
            tools.append(
                SimpleNamespace(name=tool["name"], description=tool["description"], inputSchema=tool["inputSchema"])
            )
        return tools

    async def list_resources(self) -> List[Any]:
        """Get list of available resources."""
        if self.transport == "http":
            return await self._list_resources_http()
        else:
            return await self._list_resources_streaming()

    async def _list_resources_streaming(self) -> List[str]:
        """List resources using streaming transport."""
        if not self.session:
            raise RuntimeError("Not connected to server")

        result = await self.session.list_resources()
        return [resource.name for resource in result.resources]

    async def _list_resources_http(self) -> List[Any]:
        """List resources using HTTP transport."""
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")

        response = await self.http_client.post(
            self.server_url,
            json={"jsonrpc": "2.0", "id": 1, "method": "resources/list", "params": {}},
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        )

        result = self._parse_http_response(response)
        if "error" in result:
            raise Exception(f"MCP Error: {result['error']}")

        # Convert MCP result to resource objects for compatibility
        from types import SimpleNamespace

        resources = []
        for resource in result["result"]["resources"]:
            resources.append(
                SimpleNamespace(name=resource["name"], uri=resource["uri"], description=resource["description"])
            )
        return resources

    async def check_server_running(self) -> bool:
        """Check if MCP server is running."""
        try:
            if self.transport == "http":
                if not self.http_client:
                    return False

                # Test with a simple tools/list request
                if not self.http_client:
                    raise RuntimeError("HTTP client not initialized")

                response = await self.http_client.post(
                    self.server_url,
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                    headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                )
                return response.status_code == 200
            else:
                # For streaming transport, create a temporary connection
                temp_client = MCPTestClient(self.host, self.port, "streaming")
                async with temp_client:
                    tools = await temp_client.list_tools()
                    return len(tools) > 0
        except Exception as e:
            logger.debug("Server check error: %s", e)
            return False


# Legacy compatibility wrapper for existing tests
class MCPHTTPClient(MCPTestClient):
    """Legacy wrapper for backward compatibility."""

    def __init__(self, host: str = "127.0.0.1", port: int = 3033):
        super().__init__(host=host, port=port, transport="http")
        # For compatibility with existing tests
        self.base_url = f"http://{host}:{port}/mcp/"
        self.client = self.http_client
        self.session = cast(Optional[ClientSession], self)  # Self-reference for compatibility
        self.name = "cocoindex-rag"

    async def execute_tool(self, tool_name: str, arguments: dict):
        """Execute tool with legacy interface."""
        result = await self.call_tool(tool_name, arguments)

        # Convert to legacy format expected by existing tests
        from types import SimpleNamespace

        if result["isError"]:
            raise Exception(f"MCP Tool Error: {result['content'][0] if result['content'] else 'Unknown error'}")

        # Convert to legacy tuple format: ["content", [SimpleNamespace objects]]
        content_items = []
        for text in result["content"]:
            content_items.append(SimpleNamespace(type="text", text=text))

        return ["content", content_items]

    async def cleanup(self):
        """Legacy cleanup method."""
        await self.close()
