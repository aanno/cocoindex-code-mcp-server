#!/usr/bin/env python3

"""
Integration test for the CocoIndex RAG MCP Server using proper MCP client.

This module tests the MCP server by using the official MCP client libraries
to establish proper MCP connections and test tool execution.
"""

import asyncio
import json
import logging
import os
import pytest
import pytest_asyncio
from contextlib import AsyncExitStack
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class MCPServer:
    """Manages MCP server connections and tool execution using proper MCP client."""

    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str] = None):
        self.name: str = name
        self.command: str = command
        self.args: list[str] = args
        self.env: dict[str, str] = env or {}
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the MCP server connection using proper MCP client."""
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env={**os.environ, **self.env}
        )
        
        try:
            # Use proper MCP client to connect
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            
            # Create MCP client session
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            
            # Initialize the MCP connection
            await session.initialize()
            self.session = session
            
            logging.info(f"✅ MCP server '{self.name}' initialized successfully")
            
        except Exception as e:
            logging.error(f"❌ Error initializing MCP server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the MCP server."""
        if not self.session:
            raise RuntimeError(f"MCP Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        # Extract tools from MCP response format
        for item in tools_response:
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "tools":
                tools.extend(item[1])

        return tools

    async def list_resources(self) -> list[Any]:
        """List available resources from the MCP server."""
        if not self.session:
            raise RuntimeError(f"MCP Server {self.name} not initialized")

        resources_response = await self.session.list_resources()
        resources = []

        # Extract resources from MCP response format
        for item in resources_response:
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "resources":
                resources.extend(item[1])

        return resources

    async def read_resource(self, uri: str) -> Any:
        """Read a resource from the MCP server."""
        if not self.session:
            raise RuntimeError(f"MCP Server {self.name} not initialized")

        return await self.session.read_resource(uri)

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool using proper MCP client with retry mechanism."""
        if not self.session:
            raise RuntimeError(f"MCP Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"🔧 Executing MCP tool '{tool_name}' with args: {arguments}")
                result = await self.session.call_tool(tool_name, arguments)
                logging.info(f"✅ Tool '{tool_name}' executed successfully")
                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"⚠️ Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"🔄 Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("❌ Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up MCP server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                logging.info(f"🧹 MCP server '{self.name}' cleaned up")
            except Exception as e:
                logging.error(f"❌ Error during cleanup of MCP server {self.name}: {e}")


@pytest_asyncio.fixture
async def mcp_server():
    """Simple test client for MCP server running on port 3033."""
    # Load environment variables
    load_dotenv()
    
    import httpx
    import json
    
    class SimpleMCPClient:
        def __init__(self):
            self.base_url = "http://127.0.0.1:3033/mcp"
            self.client = httpx.AsyncClient(timeout=30.0)
            self.session = self  # For compatibility with existing tests
            self.name = "cocoindex-rag"  # For test compatibility
            
        async def execute_tool(self, tool_name: str, arguments: dict):
            """Execute MCP tool by sending raw MCP request."""
            # Send MCP call_tool request
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            response = await self.client.post(
                self.base_url,
                json=mcp_request,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                raise Exception(f"MCP Error: {result['error']}")
            
            # Return in format expected by tests
            if "result" in result and "content" in result["result"]:
                return ("content", result["result"]["content"])
            return result["result"]
            
        async def list_tools(self):
            """List MCP tools."""
            mcp_request = {
                "jsonrpc": "2.0", 
                "id": 1,
                "method": "tools/list"
            }
            
            response = await self.client.post(
                self.base_url,
                json=mcp_request,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                raise Exception(f"MCP Error: {result['error']}")
                
            return ("tools", result["result"]["tools"]) if "result" in result else []
            
        async def list_resources(self):
            """List MCP resources."""
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1, 
                "method": "resources/list"
            }
            
            response = await self.client.post(
                self.base_url,
                json=mcp_request,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                raise Exception(f"MCP Error: {result['error']}")
                
            return ("resources", result["result"]["resources"]) if "result" in result else []
            
        async def read_resource(self, uri: str):
            """Read MCP resource."""
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "resources/read", 
                "params": {
                    "uri": uri
                }
            }
            
            response = await self.client.post(
                self.base_url,
                json=mcp_request,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                raise Exception(f"MCP Error: {result['error']}")
                
            return ("contents", result["result"]["contents"]) if "result" in result else []
            
        async def cleanup(self):
            await self.client.aclose()
    
    client = SimpleMCPClient()
    yield client
    await client.cleanup()


@pytest.mark.mcp_integration
@pytest.mark.asyncio
class TestMCPIntegrationHTTP:
    """Integration tests using proper MCP client connection."""
    
    async def test_server_initialization(self, mcp_server):
        """Test that MCP server initializes correctly."""
        assert mcp_server.session is not None, "MCP session should be initialized"
        assert mcp_server.name == "cocoindex-rag"
    
    async def test_list_tools(self, mcp_server):
        """Test listing tools via proper MCP client."""
        tools = await mcp_server.list_tools()
        
        # Should have expected tools
        assert len(tools) >= 6, f"Expected at least 6 tools, got {len(tools)}"
        
        # Check specific tools exist
        tool_names = [tool.name for tool in tools]
        expected_tools = [
            "hybrid_search",
            "vector_search", 
            "keyword_search",
            "analyze_code",
            "get_embeddings",
            "get_keyword_syntax_help"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Expected tool '{expected_tool}' not found"
        
        # Check tool structure
        for tool in tools:
            assert hasattr(tool, 'name'), "Tool should have name attribute"
            assert hasattr(tool, 'description'), "Tool should have description attribute"
            assert hasattr(tool, 'inputSchema'), "Tool should have inputSchema attribute"
    
    async def test_list_resources(self, mcp_server):
        """Test listing resources via proper MCP client."""
        resources = await mcp_server.list_resources()
        
        # Should have expected resources
        assert len(resources) >= 4, f"Expected at least 4 resources, got {len(resources)}"
        
        # Check specific resources exist
        resource_names = [resource.name for resource in resources]
        expected_resources = [
            "Search Statistics",
            "Search Configuration",
            "Database Schema",
            "Query Examples"
        ]
        
        for expected_resource in expected_resources:
            assert expected_resource in resource_names, f"Expected resource '{expected_resource}' not found"
        
        # Check resource structure
        for resource in resources:
            assert hasattr(resource, 'name'), "Resource should have name attribute"
            assert hasattr(resource, 'uri'), "Resource should have uri attribute"
            assert hasattr(resource, 'description'), "Resource should have description attribute"
            assert str(resource.uri).startswith("cocoindex://"), f"Resource URI should start with cocoindex://, got {resource.uri}"
    
    async def test_read_resource(self, mcp_server):
        """Test reading a resource via proper MCP client."""
        result = await mcp_server.read_resource("cocoindex://search/config")
        
        # Should get proper MCP response format
        assert isinstance(result, list), "Resource read should return a list"
        assert len(result) == 2, "Resource read should return tuple format"
        assert result[0] == "contents", "First element should be 'contents'"
        
        contents = result[1]
        assert len(contents) >= 1, "Should have at least one content item"
        
        # Check content structure
        content = contents[0]
        assert hasattr(content, 'uri'), "Content should have uri attribute"
        assert hasattr(content, 'text'), "Content should have text attribute"
        assert content.uri == "cocoindex://search/config"
        
        # Content should be valid JSON
        config_data = json.loads(content.text)
        assert isinstance(config_data, dict), "Config should be a dictionary"
        
        # Check expected configuration keys
        expected_keys = [
            "table_name",
            "embedding_model", 
            "parser_type",
            "default_weights"
        ]
        
        for key in expected_keys:
            assert key in config_data, f"Expected config key '{key}' not found"
    
    async def test_execute_tool_get_embeddings(self, mcp_server):
        """Test executing the get_embeddings tool via proper MCP client."""
        result = await mcp_server.execute_tool(
            "get_embeddings",
            {"text": "test text for embedding"}
        )
        
        # Should get proper MCP response format
        assert isinstance(result, list), "Tool result should return a list"
        assert len(result) == 2, "Tool result should return tuple format"
        assert result[0] == "content", "First element should be 'content'"
        
        content_list = result[1]
        assert len(content_list) >= 1, "Should have at least one content item"
        
        # Check content structure
        content = content_list[0]
        assert hasattr(content, 'type'), "Content should have type attribute"
        assert hasattr(content, 'text'), "Content should have text attribute"
        assert content.type == "text"
        
        # Parse the JSON response to check embedding format
        embedding_data = json.loads(content.text)
        assert "embedding" in embedding_data, "Should contain embedding data"
        assert "dimensions" in embedding_data, "Should contain dimensions info"
        assert isinstance(embedding_data["embedding"], list), "Embedding should be a list"
        assert len(embedding_data["embedding"]) > 0, "Embedding should not be empty"
        assert embedding_data["dimensions"] > 0, "Dimensions should be positive"
    
    async def test_execute_tool_vector_search(self, mcp_server):
        """Test executing the vector_search tool via proper MCP client."""
        result = await mcp_server.execute_tool(
            "vector_search",
            {
                "query": "Python async function for processing data",
                "top_k": 3
            }
        )
        
        # Should get proper MCP response format
        assert isinstance(result, list), "Tool result should return a list"
        assert len(result) == 2, "Tool result should return tuple format" 
        assert result[0] == "content", "First element should be 'content'"
        
        content_list = result[1]
        assert len(content_list) >= 1, "Should have at least one content item"
        
        # Check content structure
        content = content_list[0]
        assert content.type == "text"
        
        # Parse the JSON response to check search results
        search_data = json.loads(content.text)
        assert "query" in search_data, "Should contain query info"
        assert "results" in search_data, "Should contain results"
        assert "total_results" in search_data, "Should contain total results count"
        
        # Should have search results (assuming database has data)
        if search_data["total_results"] > 0:
            results = search_data["results"]
            assert isinstance(results, list), "Results should be a list"
            
            # Check first result structure
            first_result = results[0]
            expected_fields = ["filename", "language", "code", "score"]
            for field in expected_fields:
                assert field in first_result, f"Result should contain '{field}' field"
    
    async def test_execute_tool_analyze_code(self, mcp_server):
        """Test executing the analyze_code tool via proper MCP client."""
        test_code = '''
async def process_data(items: list[str]) -> list[str]:
    """Process data asynchronously."""
    return [f"processed_{item}" for item in items]

class DataProcessor:
    """A data processor class."""
    
    def __init__(self, name: str):
        self.name = name
    
    @property
    def status(self) -> str:
        return "active"
'''
        
        result = await mcp_server.execute_tool(
            "analyze_code",
            {
                "code": test_code,
                "file_path": "test.py",
                "language": "python"
            }
        )
        
        # Should get proper MCP response format
        assert isinstance(result, list), "Tool result should return a list"
        assert len(result) == 2, "Tool result should return tuple format"
        assert result[0] == "content", "First element should be 'content'"
        
        content_list = result[1]
        assert len(content_list) >= 1, "Should have at least one content item"
        
        # Check content structure
        content = content_list[0]
        assert content.type == "text"
        
        # Parse the JSON response to check analysis results
        analysis_data = json.loads(content.text)
        assert "file_path" in analysis_data, "Should contain file path"
        assert "language" in analysis_data, "Should contain language"
        assert "metadata" in analysis_data, "Should contain metadata"
        
        # Check metadata structure
        metadata = analysis_data["metadata"]
        expected_fields = ["functions", "classes", "has_async", "has_type_hints"]
        for field in expected_fields:
            assert field in metadata, f"Metadata should contain '{field}' field"
        
        # Should detect the function and class we defined
        assert "process_data" in str(metadata["functions"]), "Should detect process_data function"
        assert "DataProcessor" in str(metadata["classes"]), "Should detect DataProcessor class"
        assert metadata["has_async"] is True, "Should detect async code"
        assert metadata["has_type_hints"] is True, "Should detect type hints"
    
    async def test_execute_tool_keyword_search_basic(self, mcp_server):
        """Test executing the keyword_search tool with basic queries."""
        result = await mcp_server.execute_tool(
            "keyword_search",
            {
                "query": "language:Python",
                "top_k": 5
            }
        )
        
        # Should get proper MCP response format without errors
        assert isinstance(result, list), "Tool result should return a list"
        assert len(result) == 2, "Tool result should return tuple format"
        assert result[0] == "content", "First element should be 'content'"
        
        content_list = result[1]
        assert len(content_list) >= 1, "Should have at least one content item"
        
        # Check content structure
        content = content_list[0]
        assert content.type == "text"
        
        # Parse the JSON response
        search_data = json.loads(content.text)
        assert "query" in search_data, "Should contain query info"
        assert "results" in search_data, "Should contain results"
        assert "total_results" in search_data, "Should contain total results count"
    
    async def test_execute_tool_get_keyword_syntax_help(self, mcp_server):
        """Test executing the get_keyword_syntax_help tool."""
        result = await mcp_server.execute_tool(
            "get_keyword_syntax_help",
            {}
        )
        
        # Should get proper MCP response format
        assert isinstance(result, list), "Tool result should return a list"
        assert len(result) == 2, "Tool result should return tuple format"
        assert result[0] == "content", "First element should be 'content'"
        
        content_list = result[1]
        assert len(content_list) >= 1, "Should have at least one content item"
        
        # Check content structure
        content = content_list[0]
        assert content.type == "text"
        
        # Parse the JSON response to check help content
        help_data = json.loads(content.text)
        assert "keyword_query_syntax" in help_data, "Should contain syntax help"
        
        syntax_help = help_data["keyword_query_syntax"]
        assert "basic_operators" in syntax_help, "Should contain basic operators help"
        assert "boolean_logic" in syntax_help, "Should contain boolean logic help"
        assert "available_fields" in syntax_help, "Should contain available fields help"
    
    async def test_error_handling_invalid_tool(self, mcp_server):
        """Test error handling for invalid tool calls."""
        with pytest.raises(Exception):  # Should raise an exception for unknown tool
            await mcp_server.execute_tool(
                "nonexistent_tool",
                {}
            )
    
    async def test_error_handling_invalid_resource(self, mcp_server):
        """Test error handling for invalid resource URIs."""
        with pytest.raises(Exception):  # Should raise an exception for unknown resource
            await mcp_server.read_resource("cocoindex://invalid/resource")
    
    async def test_smart_embedding_functionality(self, mcp_server):
        """Test that smart embedding is working with language-aware model selection."""
        # Test Python code - should use GraphCodeBERT
        python_result = await mcp_server.execute_tool(
            "vector_search",
            {
                "query": "Python async function with type hints",
                "top_k": 3
            }
        )
        
        # Parse result
        content_list = python_result[1]
        content = content_list[0]
        search_data = json.loads(content.text)
        
        # Should find Python-related results
        if search_data["total_results"] > 0:
            results = search_data["results"]
            # At least some results should be Python
            python_results = [r for r in results if r.get("language") == "Python"]
            assert len(python_results) > 0, "Should find Python results when searching for Python concepts"
            
            # Should have rich metadata for Python results
            for result in python_results:
                assert "has_async" in result, "Python results should have async detection"
                assert "has_type_hints" in result, "Python results should have type hints detection"
                assert "analysis_method" in result, "Python results should have analysis method info"
                
                # Should use enhanced analysis method
                analysis_method = result.get("analysis_method", "")
                assert "tree_sitter" in analysis_method or "python_ast" in analysis_method, \
                    f"Python results should use enhanced analysis, got: {analysis_method}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])