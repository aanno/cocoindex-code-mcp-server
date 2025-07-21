"""
Integration test for the CocoIndex RAG MCP Server using direct HTTP requests.

This module tests the MCP server running on port 3033 by sending raw JSON-RPC
requests over HTTP and validating the responses according to MCP protocol.
"""

import json
import pytest
import httpx
import asyncio


@pytest.mark.integration
class TestMCPIntegrationHTTP:
    """Integration tests using direct HTTP JSON-RPC requests."""
    
    SERVER_URL = "http://localhost:3033/mcp"
    
    async def _send_jsonrpc_request(self, method: str, params: dict = None, request_id: int = 1) -> dict:
        """Send a JSON-RPC request to the MCP server."""
        request_data = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.SERVER_URL,
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
    
    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test MCP protocol initialization."""
        response = await self._send_jsonrpc_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        )
        
        # Check JSON-RPC response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        
        # Check MCP initialization result
        result = response["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert "capabilities" in result
        assert "serverInfo" in result
        
        # Check server capabilities
        capabilities = result["capabilities"]
        assert "tools" in capabilities
        assert "resources" in capabilities
        assert capabilities["tools"]["listChanged"] is True
        assert capabilities["resources"]["listChanged"] is True
        
        # Check server info
        server_info = result["serverInfo"]
        assert server_info["name"] == "cocoindex-rag"
        assert server_info["version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Test listing resources via JSON-RPC."""
        response = await self._send_jsonrpc_request("resources/list", {}, 2)
        
        # Check JSON-RPC response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        
        # Check resources
        resources = response["result"]["resources"]
        assert len(resources) == 7
        
        # Check specific resources exist
        resource_names = [r["name"] for r in resources]
        assert "Search Statistics" in resource_names
        assert "Search Configuration" in resource_names
        assert "Database Schema" in resource_names
        
        # Check resource structure
        for resource in resources:
            assert "name" in resource
            assert "uri" in resource
            assert "description" in resource
            assert "mimeType" in resource
            assert str(resource["uri"]).startswith("cocoindex://")
            # Most resources are JSON, but grammar resource is Lark format
            if "grammar" in str(resource["uri"]):
                assert resource["mimeType"] == "text/x-lark"
            else:
                assert resource["mimeType"] == "application/json"
    
    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test listing tools via JSON-RPC."""
        response = await self._send_jsonrpc_request("tools/list", {}, 3)
        
        # Check JSON-RPC response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "result" in response
        
        # Check tools
        tools = response["result"]["tools"]
        assert len(tools) == 6
        
        # Check specific tools exist
        tool_names = [t["name"] for t in tools]
        expected_tools = [
            "hybrid_search",
            "vector_search", 
            "keyword_search",
            "analyze_code",
            "get_embeddings",
            "get_keyword_syntax_help"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names
        
        # Check tool structure
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            
            # Check input schema structure
            schema = tool["inputSchema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema
    
    @pytest.mark.asyncio
    async def test_read_resource(self):
        """Test reading a resource via JSON-RPC."""
        response = await self._send_jsonrpc_request(
            "resources/read",
            {"uri": "cocoindex://search/config"},
            4
        )
        
        # Check JSON-RPC response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 4
        assert "result" in response
        
        # Check resource content
        contents = response["result"]["contents"]
        assert len(contents) == 1
        
        content = contents[0]
        assert content["uri"] == "cocoindex://search/config"
        assert "text" in content
        
        # Content should be valid JSON
        config_data = json.loads(content["text"])
        assert isinstance(config_data, dict)
        
        # Check expected configuration keys
        expected_keys = [
            "table_name",
            "embedding_model", 
            "parser_type",
            "supported_operators",
            "default_weights"
        ]
        
        for key in expected_keys:
            assert key in config_data
    
    @pytest.mark.asyncio
    async def test_call_tool_get_embeddings(self):
        """Test calling the get_embeddings tool."""
        response = await self._send_jsonrpc_request(
            "tools/call",
            {
                "name": "get_embeddings",
                "arguments": {
                    "text": "test text for embedding"
                }
            },
            5
        )
        
        # Check JSON-RPC response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 5
        assert "result" in response
        
        # Check tool result
        result = response["result"]
        assert "content" in result
        
        # Content should be a list with one item
        content = result["content"]
        assert isinstance(content, list)
        assert len(content) == 1
        
        # Content should be text with embedding data
        first_content = content[0]
        assert first_content["type"] == "text"
        assert "text" in first_content
        
        # Parse the JSON response to check embedding format
        embedding_data = json.loads(first_content["text"])
        assert "embedding" in embedding_data
        # Check for either "model" or "dimensions" field (implementation specific)
        assert "dimensions" in embedding_data or "model" in embedding_data
        assert isinstance(embedding_data["embedding"], list)
        assert len(embedding_data["embedding"]) > 0
    
    @pytest.mark.asyncio
    async def test_call_tool_vector_search(self):
        """Test calling the vector_search tool."""
        response = await self._send_jsonrpc_request(
            "tools/call",
            {
                "name": "vector_search",
                "arguments": {
                    "query": "test search query",
                    "top_k": 5
                }
            },
            6
        )
        
        # Check JSON-RPC response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 6
        assert "result" in response
        
        # Check tool result
        result = response["result"]
        assert "content" in result
        
        # Content should be a list with at least one item
        content = result["content"]
        assert isinstance(content, list)
        assert len(content) >= 1
        
        # First content item should be text
        first_content = content[0]
        assert first_content["type"] == "text"
        assert "text" in first_content
    
    @pytest.mark.asyncio
    async def test_call_tool_hybrid_search(self):
        """Test calling the hybrid_search tool."""
        response = await self._send_jsonrpc_request(
            "tools/call",
            {
                "name": "hybrid_search",
                "arguments": {
                    "vector_query": "test search query",
                    "keyword_query": "function_name:parse",
                    "top_k": 5,
                    "vector_weight": 0.7,
                    "keyword_weight": 0.3
                }
            },
            7
        )
        
        # Check JSON-RPC response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 7
        assert "result" in response
        
        # Check tool result
        result = response["result"]
        assert "content" in result
        
        # Content should be a list with at least one item
        content = result["content"]
        assert isinstance(content, list)
        assert len(content) >= 1
        
        # First content item should be text
        first_content = content[0]
        assert first_content["type"] == "text"
        assert "text" in first_content
    
    @pytest.mark.asyncio
    async def test_call_tool_analyze_code(self):
        """Test calling the analyze_code tool."""
        test_code = '''
def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return "Hello, World!"
'''
        
        response = await self._send_jsonrpc_request(
            "tools/call",
            {
                "name": "analyze_code",
                "arguments": {
                    "code": test_code,
                    "file_path": "test.py",
                    "language": "python"
                }
            },
            8
        )
        
        # Check JSON-RPC response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 8
        assert "result" in response
        
        # Check tool result
        result = response["result"]
        assert "content" in result
        
        # Content should be a list with at least one item
        content = result["content"]
        assert isinstance(content, list)
        assert len(content) >= 1
        
        # First content item should be text
        first_content = content[0]
        assert first_content["type"] == "text"
        assert "text" in first_content
    
    @pytest.mark.asyncio
    async def test_error_handling_unknown_method(self):
        """Test error handling for unknown methods."""
        response = await self._send_jsonrpc_request("unknown/method", {}, 9)
        
        # Check JSON-RPC response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 9
        assert "error" in response
        
        # Check error structure
        error = response["error"]
        assert "code" in error
        assert "message" in error
        assert error["code"] == -32601  # Method not found
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_tool(self):
        """Test error handling for invalid tool calls."""
        response = await self._send_jsonrpc_request(
            "tools/call",
            {
                "name": "nonexistent_tool",
                "arguments": {}
            },
            10
        )
        
        # Should get an error response (either as JSON-RPC error or as content)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 10
        
        # Check if it's a proper JSON-RPC error or error returned as content
        if "error" in response:
            # Proper JSON-RPC error
            error = response["error"]
            assert "code" in error
            assert "message" in error
        else:
            # Error returned as content (our server's current behavior)
            assert "result" in response
            content = response["result"]["content"]
            assert len(content) >= 1
            first_content = content[0]
            assert "Error" in first_content["text"] or "error" in first_content["text"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])