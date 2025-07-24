import json
import pytest
import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any
from mcp import ClientSession, StreamingHTTPServerParameters
import httpx

LOGGING = logging.getLogger(__name__)

@pytest.mark.mcp_integration
@pytest.mark.asyncio
class TestMCPIntegrationHTTP:
    """Integration tests using MCP ClientSession for streaming HTTP."""
    
    SERVER_URL = "http://localhost:3033/mcp"
    
    async def _create_client_session(self) -> tuple[ClientSession, AsyncExitStack]:
        """Create and initialize an MCP client session."""
        exit_stack = AsyncExitStack()
        server_params = StreamingHTTPServerParameters(url=self.SERVER_URL)
        async with httpx.AsyncClient() as client:
            transport = await exit_stack.enter_async_context(client.stream("POST", self.SERVER_URL))
            session = await exit_stack.enter_async_context(ClientSession(transport.response, transport))
            await session.initialize()
            return session, exit_stack

    async def test_server_initialization(self):
        """Test MCP protocol initialization."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            response = await session.initialize(
                protocolVersion="2024-11-05",
                capabilities={"roots": {"listChanged": True}},
                clientInfo={"name": "test-client", "version": "1.0.0"}
            )
            
            assert response["protocolVersion"] == "2024-11-05"
            assert "capabilities" in response
            assert "serverInfo" in response
            assert response["capabilities"]["tools"]["listChanged"] is True
            assert response["capabilities"]["resources"]["listChanged"] is True
            assert response["serverInfo"]["name"] == "cocoindex-rag"
            assert response["serverInfo"]["version"] == "1.0.0"

    async def test_list_resources(self):
        """Test listing resources."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            resources = await session.list_resources()
            
            assert len(resources) == 7
            resource_names = [r.name for r in resources]
            assert "Search Statistics" in resource_names
            assert "Search Configuration" in resource_names
            assert "Database Schema" in resource_names
            
            for resource in resources:
                assert resource.name
                assert resource.uri
                assert resource.description
                assert resource.mimeType
                assert str(resource.uri).startswith("cocoindex://")
                if "grammar" in str(resource.uri):
                    assert resource.mimeType == "text/x-lark"
                else:
                    assert resource.mimeType == "application/json"

    async def test_list_tools(self):
        """Test listing tools."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            tools = await session.list_tools()
            
            assert len(tools) == 6
            tool_names = [t.name for t in tools]
            expected_tools = [
                "hybrid_search", "vector_search", "keyword_search",
                "analyze_code", "get_embeddings", "get_keyword_syntax_help"
            ]
            for expected_tool in expected_tools:
                assert expected_tool in tool_names
                
            for tool in tools:
                assert tool.name
                assert tool.description
                assert tool.inputSchema
                assert tool.inputSchema["type"] == "object"
                assert "properties" in tool.inputSchema
                assert "required" in tool.inputSchema

    async def test_read_resource(self):
        """Test reading a resource."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            contents = await session.read_resources(["cocoindex://search/config"])
            
            assert len(contents) == 1
            content = contents[0]
            assert content.uri == "cocoindex://search/config"
            assert content.text
            config_data = json.loads(content.text)
            assert isinstance(config_data, dict)
            expected_keys = [
                "table_name", "embedding_model", "parser_type",
                "supported_operators", "default_weights"
            ]
            for key in expected_keys:
                assert key in config_data

    async def test_call_tool_get_embeddings(self):
        """Test calling the get_embeddings tool."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            result = await session.call_tool(
                "get_embeddings",
                {"text": "test text for embedding"}
            )
            
            assert isinstance(result.content, list)
            assert len(result.content) == 1
            first_content = result.content[0]
            assert first_content.type == "text"
            assert first_content.text
            embedding_data = json.loads(first_content.text)
            assert "embedding" in embedding_data
            assert "dimensions" in embedding_data or "model" in embedding_data
            assert isinstance(embedding_data["embedding"], list)
            assert len(embedding_data["embedding"]) > 0

    async def test_call_tool_vector_search(self):
        """Test calling the vector_search tool."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            result = await session.call_tool(
                "vector_search",
                {"query": "test search query", "top_k": 5}
            )
            
            assert isinstance(result.content, list)
            assert len(result.content) >= 1
            first_content = result.content[0]
            assert first_content.type == "text"
            assert first_content.text

    async def test_call_tool_hybrid_search(self):
        """Test calling the hybrid_search tool."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            result = await session.call_tool(
                "hybrid_search",
                {
                    "vector_query": "test search query",
                    "keyword_query": "function_name:parse",
                    "top_k": 5,
                    "vector_weight": 0.7,
                    "keyword_weight": 0.3
                }
            )
            
            assert isinstance(result.content, list)
            assert len(result.content) >= 1
            first_content = result.content[0]
            assert first_content.type == "text"
            assert first_content.text

    async def test_call_tool_analyze_code(self):
        """Test calling the analyze_code tool."""
        test_code = '''
def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return "Hello, World!"
'''
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            result = await session.call_tool(
                "analyze_code",
                {"code": test_code, "file_path": "test.py", "language": "python"}
            )
            
            assert isinstance(result.content, list)
            assert len(result.content) >= 1
            first_content = result.content[0]
            assert first_content.type == "text"
            assert first_content.text

    async def test_error_handling_unknown_method(self):
        """Test error handling for unknown methods."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            try:
                await session._send_raw({"method": "unknown/method", "params": {}})
                assert False, "Expected an error for unknown method"
            except Exception as e:
                assert "Method not found" in str(e)

    async def test_error_handling_invalid_tool(self):
        """Test error handling for invalid tool calls."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            try:
                await session.call_tool("nonexistent_tool", {})
                assert False, "Expected an error for invalid tool"
            except Exception as e:
                assert "Error" in str(e) or "error" in str(e).lower()

    async def test_ast_improvements_decorator_detection(self):
        """Test AST improvements: decorator detection through hybrid search."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            result = await session.call_tool(
                "hybrid_search",
                {
                    "vector_query": "Python classes with decorators",
                    "keyword_query": "language:python AND exists(decorators)",
                    "top_k": 10
                }
            )
            
            assert isinstance(result.content, list)
            assert len(result.content) >= 1
            content = result.content[0]
            assert "decorators" in content.text.lower()
            assert any(decorator in content.text for decorator in ["@dataclass", "@property", "@staticmethod"])

    async def test_ast_improvements_class_method_detection(self):
        """Test AST improvements: class method detection."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            result = await session.call_tool(
                "hybrid_search",
                {
                    "vector_query": "class methods with docstrings",
                    "keyword_query": "language:python AND exists(function_details)",
                    "top_k": 10
                }
            )
            
            assert isinstance(result.content, list)
            assert len(result.content) >= 1
            content = result.content[0]
            assert "function_details" in content.text.lower()
            assert any(method in content.text for method in ["__init__", "from_dict", "classmethod"])

    async def test_ast_improvements_docstring_detection(self):
        """Test AST improvements: docstring detection."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            result = await session.call_tool(
                "keyword_search",
                {"query": "has_docstrings:true AND language:python", "top_k": 10}
            )
            
            assert isinstance(result.content, list)
            assert len(result.content) >= 1
            content = result.content[0]
            assert "has_docstrings" in content.text.lower()
            assert any(indicator in content.text for indicator in ["docstring", "\"\"\"", "description"])

    async def test_ast_improvements_private_dunder_methods(self):
        """Test AST improvements: private and dunder method detection."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            result = await session.call_tool(
                "keyword_search",
                {
                    "query": "language:python AND (exists(private_methods) OR exists(dunder_methods))",
                    "top_k": 10
                }
            )
            
            assert isinstance(result.content, list)
            assert len(result.content) >= 1
            content = result.content[0]
            assert any(field in content.text for field in ["private_methods", "dunder_methods"])
            assert any(method in content.text for method in ["__init__", "_private", "__str__"])

    async def test_ast_improvements_metadata_completeness(self):
        """Test AST improvements: comprehensive metadata fields."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            result = await session.call_tool(
                "vector_search",
                {"query": "python function class metadata", "top_k": 5}
            )
            
            assert isinstance(result.content, list)
            assert len(result.content) >= 1
            content = result.content[0]
            expected_fields = [
                "functions", "classes", "decorators", "has_decorators",
                "has_classes", "has_docstrings", "function_details",
                "class_details", "analysis_method"
            ]
            found_fields = sum(1 for field in expected_fields if field in content.text)
            assert found_fields >= 4, f"Expected at least 4 metadata fields, found {found_fields}"

    async def test_ast_improvements_analysis_method_hybrid(self):
        """Test AST improvements: hybrid analysis method detection."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            result = await session.call_tool(
                "keyword_search",
                {"query": "analysis_method:tree_sitter OR analysis_method:python_ast", "top_k": 5}
            )
            
            assert isinstance(result.content, list)
            assert len(result.content) >= 1
            content = result.content[0]
            assert "analysis_method" in content.text.lower()
            assert any(method in content.text for method in ["tree_sitter", "python_ast", "hybrid"])

    async def test_ast_improvements_regression_prevention(self):
        """Test AST improvements: ensure previous bugs don't return."""
        async with AsyncExitStack() as exit_stack:
            session, _ = await self._create_client_session()
            
            # Test 1: Verify .type vs .kind fix
            result1 = await session.call_tool(
                "keyword_search",
                {"query": "language:python AND exists(functions)", "top_k": 5}
            )
            assert isinstance(result1.content, list)
            assert len(result1.content) >= 1
            assert "functions" in result1.content[0].text.lower()
            
            # Test 2: Verify class decorator merging
            result2 = await session.call_tool(
                "keyword_search",
                {"query": "decorators:dataclass", "top_k": 3}
            )
            assert isinstance(result2.content, list)
            assert len(result2.content) >= 1
            assert any(indicator in result2.content[0].text for indicator in ["dataclass", "decorator"])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
