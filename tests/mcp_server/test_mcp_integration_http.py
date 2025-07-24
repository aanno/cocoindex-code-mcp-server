"""
Integration test for the CocoIndex RAG MCP Server using direct HTTP requests.

This module tests the MCP server running on port 3033 by sending raw JSON-RPC
requests over HTTP and validating the responses according to MCP protocol.
"""

import json
import pytest
import httpx
import asyncio


@pytest.mark.mcp_integration
@pytest.mark.asyncio
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


    @pytest.mark.asyncio
    async def test_ast_improvements_decorator_detection(self):
        """Test AST improvements: decorator detection through hybrid search."""
        response = await self._send_jsonrpc_request(
            "tools/call",
            {
                "name": "hybrid_search",
                "arguments": {
                    "vector_query": "Python classes with decorators",
                    "keyword_query": "language:python AND exists(decorators)",
                    "top_k": 10
                }
            },
            11
        )
        
        # Check response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 11
        assert "result" in response
        
        # Parse results to verify decorator detection
        content = response["result"]["content"][0]
        results_text = content["text"]
        
        # Should find results with decorators
        assert "decorators" in results_text.lower()
        # Common decorators should be detected
        assert any(decorator in results_text for decorator in ["@dataclass", "@property", "@staticmethod"])

    @pytest.mark.asyncio
    async def test_ast_improvements_class_method_detection(self):
        """Test AST improvements: class method detection."""
        response = await self._send_jsonrpc_request(
            "tools/call",
            {
                "name": "hybrid_search",
                "arguments": {
                    "vector_query": "class methods with docstrings",
                    "keyword_query": "language:python AND exists(function_details)",
                    "top_k": 10
                }
            },
            12
        )
        
        # Check response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 12
        assert "result" in response
        
        # Parse results to verify class method detection
        content = response["result"]["content"][0]
        results_text = content["text"]
        
        # Should find class methods like __init__, from_dict, etc.
        assert "function_details" in results_text.lower()
        # Common class methods should be detected
        assert any(method in results_text for method in ["__init__", "from_dict", "classmethod"])

    @pytest.mark.asyncio
    async def test_ast_improvements_docstring_detection(self):
        """Test AST improvements: docstring detection."""
        response = await self._send_jsonrpc_request(
            "tools/call",
            {
                "name": "keyword_search",
                "arguments": {
                    "query": "has_docstrings:true AND language:python",
                    "top_k": 10
                }
            },
            13
        )
        
        # Check response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 13
        assert "result" in response
        
        # Parse results to verify docstring detection
        content = response["result"]["content"][0]
        results_text = content["text"]
        
        # Should find entries with has_docstrings=true
        assert "has_docstrings" in results_text.lower()
        # Should find actual docstring content
        assert any(indicator in results_text for indicator in ["docstring", "\"\"\"", "description"])

    @pytest.mark.asyncio
    async def test_ast_improvements_private_dunder_methods(self):
        """Test AST improvements: private and dunder method detection."""
        response = await self._send_jsonrpc_request(
            "tools/call",
            {
                "name": "keyword_search",
                "arguments": {
                    "query": "language:python AND (exists(private_methods) OR exists(dunder_methods))",
                    "top_k": 10
                }
            },
            14
        )
        
        # Check response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 14
        assert "result" in response
        
        # Parse results to verify method classification
        content = response["result"]["content"][0]
        results_text = content["text"]
        
        # Should find private methods (starting with _) and dunder methods (__x__)
        assert any(field in results_text for field in ["private_methods", "dunder_methods"])
        # Common patterns should be detected
        assert any(method in results_text for method in ["__init__", "_private", "__str__"])

    @pytest.mark.asyncio
    async def test_ast_improvements_metadata_completeness(self):
        """Test AST improvements: comprehensive metadata fields."""
        response = await self._send_jsonrpc_request(
            "tools/call",
            {
                "name": "vector_search",
                "arguments": {
                    "query": "python function class metadata",
                    "top_k": 5
                }
            },
            15
        )
        
        # Check response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 15
        assert "result" in response
        
        # Parse results to verify comprehensive metadata
        content = response["result"]["content"][0]
        results_text = content["text"]
        
        # Should include comprehensive metadata fields from our improvements
        expected_fields = [
            "functions", "classes", "decorators", "has_decorators", 
            "has_classes", "has_docstrings", "function_details", 
            "class_details", "analysis_method"
        ]
        
        # At least several of these fields should be present
        found_fields = sum(1 for field in expected_fields if field in results_text)
        assert found_fields >= 4, f"Expected at least 4 metadata fields, found {found_fields}"

    @pytest.mark.asyncio
    async def test_ast_improvements_analysis_method_hybrid(self):
        """Test AST improvements: hybrid analysis method detection."""
        response = await self._send_jsonrpc_request(
            "tools/call",
            {
                "name": "keyword_search",
                "arguments": {
                    "query": "analysis_method:tree_sitter OR analysis_method:python_ast",
                    "top_k": 5
                }
            },
            16
        )
        
        # Check response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 16
        assert "result" in response
        
        # Parse results to verify analysis method tracking
        content = response["result"]["content"][0]
        results_text = content["text"]
        
        # Should show our hybrid analysis approach
        assert "analysis_method" in results_text.lower()
        # Should indicate tree-sitter or python_ast analysis
        assert any(method in results_text for method in ["tree_sitter", "python_ast", "hybrid"])

    @pytest.mark.asyncio
    async def test_ast_improvements_regression_prevention(self):
        """Test AST improvements: ensure previous bugs don't return."""
        # Test 1: Verify .type vs .kind fix by searching for functions
        response1 = await self._send_jsonrpc_request(
            "tools/call",
            {
                "name": "keyword_search",
                "arguments": {
                    "query": "language:python AND exists(functions)",
                    "top_k": 5
                }
            },
            17
        )
        
        assert response1["jsonrpc"] == "2.0"
        assert response1["id"] == 17
        assert "result" in response1
        
        # Should find functions without AttributeError
        content1 = response1["result"]["content"][0]
        assert "functions" in content1["text"].lower()
        
        # Test 2: Verify class decorator merging by searching for @dataclass
        response2 = await self._send_jsonrpc_request(
            "tools/call",
            {
                "name": "keyword_search",
                "arguments": {
                    "query": "decorators:dataclass",
                    "top_k": 3
                }
            },
            18
        )
        
        assert response2["jsonrpc"] == "2.0"
        assert response2["id"] == 18
        assert "result" in response2
        
        # Should find dataclass decorators in results
        content2 = response2["result"]["content"][0]
        results_text2 = content2["text"]
        assert any(indicator in results_text2 for indicator in ["dataclass", "decorator"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
