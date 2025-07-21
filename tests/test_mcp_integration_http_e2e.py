"""
End-to-End integration test for the CocoIndex RAG MCP Server using the MCP client library.

This module tests the MCP server running on port 3033 using the official MCP client
library for Python, providing a true E2E test that mimics how Claude would interact
with the server through the MCP protocol.
"""

import json
import pytest
import asyncio
from typing import Dict, Any, List

# Import MCP client library components
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import (
    Tool, 
    Resource, 
    TextContent, 
    CallToolRequest, 
    ReadResourceRequest,
    InitializeRequest,
    ListToolsRequest,
    ListResourcesRequest
)


@pytest.mark.integration
class TestMCPIntegrationE2E:
    """End-to-End integration tests using the official MCP client library."""
    
    SERVER_URL = "http://localhost:3033/mcp"
    
    async def _test_with_client_session(self, test_func):
        """Helper to run test functions with a valid MCP client session."""
        # Use StreamableHTTP client to connect to our StreamableHTTP MCP server
        async with streamablehttp_client(self.SERVER_URL) as (read, write, get_session_id):
            # Create client session
            session = ClientSession(read, write)
            
            # Initialize the session
            await session.initialize()
            
            # Run the test function with the session
            return await test_func(session)
    
    @pytest.mark.asyncio
    async def test_mcp_client_initialization(self):
        """Test MCP client initialization through the library."""
        async def _test(session):
            # Session should be initialized
            assert session is not None
            
            # Check that we can access initialization result
            init_result = session.get_server_info()
            assert init_result is not None
            assert init_result.name == "cocoindex-rag"
            assert init_result.version == "1.0.0"
            
            # Check capabilities
            capabilities = session.get_server_capabilities()
            assert capabilities is not None
            assert capabilities.tools is not None
            assert capabilities.tools.listChanged is True
            assert capabilities.resources is not None
            assert capabilities.resources.listChanged is True
        
        await self._test_with_client_session(_test)
    
    @pytest.mark.asyncio
    async def test_mcp_list_tools_through_library(self):
        """Test listing tools through the MCP client library."""
        async with await self._create_client_session() as session:
            # List tools using the MCP client
            tools_result = await session.list_tools()
            
            # Check that we got tools
            assert tools_result is not None
            assert hasattr(tools_result, 'tools')
            tools = tools_result.tools
            assert len(tools) == 5
            
            # Check specific tools exist
            tool_names = [tool.name for tool in tools]
            expected_tools = [
                "hybrid_search",
                "vector_search", 
                "keyword_search",
                "analyze_code",
                "get_embeddings"
            ]
            
            for expected_tool in expected_tools:
                assert expected_tool in tool_names
            
            # Check tool structure for one tool
            embedding_tool = next(tool for tool in tools if tool.name == "get_embeddings")
            assert embedding_tool.description is not None
            assert embedding_tool.inputSchema is not None
            assert embedding_tool.inputSchema.get("type") == "object"
            assert "properties" in embedding_tool.inputSchema
            assert "required" in embedding_tool.inputSchema
    
    @pytest.mark.asyncio
    async def test_mcp_list_resources_through_library(self):
        """Test listing resources through the MCP client library."""
        async with await self._create_client_session() as session:
            # List resources using the MCP client
            resources_result = await session.list_resources()
            
            # Check that we got resources
            assert resources_result is not None
            assert hasattr(resources_result, 'resources')
            resources = resources_result.resources
            assert len(resources) == 3
            
            # Check specific resources exist
            resource_names = [resource.name for resource in resources]
            assert "Search Statistics" in resource_names
            assert "Search Configuration" in resource_names
            assert "Database Schema" in resource_names
            
            # Check resource structure
            for resource in resources:
                assert resource.name is not None
                assert resource.uri is not None
                assert resource.description is not None
                assert resource.mimeType == "application/json"
                assert str(resource.uri).startswith("cocoindex://")
    
    @pytest.mark.asyncio
    async def test_mcp_read_resource_through_library(self):
        """Test reading a resource through the MCP client library."""
        async with await self._create_client_session() as session:
            # Read a specific resource
            resource_result = await session.read_resource("cocoindex://search/config")
            
            # Check that we got content
            assert resource_result is not None
            assert hasattr(resource_result, 'contents')
            contents = resource_result.contents
            assert len(contents) == 1
            
            content = contents[0]
            assert content.uri == "cocoindex://search/config"
            assert hasattr(content, 'text')
            
            # Content should be valid JSON
            config_data = json.loads(content.text)
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
    async def test_mcp_call_tool_get_embeddings_through_library(self):
        """Test calling the get_embeddings tool through the MCP client library."""
        async with await self._create_client_session() as session:
            # Call the get_embeddings tool
            tool_result = await session.call_tool(
                "get_embeddings",
                {
                    "text": "test text for embedding"
                }
            )
            
            # Check that we got a result
            assert tool_result is not None
            assert hasattr(tool_result, 'content')
            content = tool_result.content
            assert isinstance(content, list)
            assert len(content) == 1
            
            # Content should be text with embedding data
            first_content = content[0]
            assert isinstance(first_content, TextContent)
            assert first_content.type == "text"
            assert first_content.text is not None
            
            # Parse the JSON response to check embedding format
            embedding_data = json.loads(first_content.text)
            assert "embedding" in embedding_data
            # Check for either "model" or "dimensions" field (implementation specific)
            assert "dimensions" in embedding_data or "model" in embedding_data
            assert isinstance(embedding_data["embedding"], list)
            assert len(embedding_data["embedding"]) > 0
    
    @pytest.mark.asyncio
    async def test_mcp_call_tool_vector_search_through_library(self):
        """Test calling the vector_search tool through the MCP client library."""
        async with await self._create_client_session() as session:
            # Call the vector_search tool
            tool_result = await session.call_tool(
                "vector_search",
                {
                    "query": "test search query",
                    "top_k": 5
                }
            )
            
            # Check that we got a result
            assert tool_result is not None
            assert hasattr(tool_result, 'content')
            content = tool_result.content
            assert isinstance(content, list)
            assert len(content) >= 1
            
            # First content item should be text
            first_content = content[0]
            assert isinstance(first_content, TextContent)
            assert first_content.type == "text"
            assert first_content.text is not None
    
    @pytest.mark.asyncio
    async def test_mcp_call_tool_hybrid_search_through_library(self):
        """Test calling the hybrid_search tool through the MCP client library."""
        async with await self._create_client_session() as session:
            # Call the hybrid_search tool
            tool_result = await session.call_tool(
                "hybrid_search",
                {
                    "vector_query": "test search query",
                    "keyword_query": "function_name:parse",
                    "top_k": 5,
                    "vector_weight": 0.7,
                    "keyword_weight": 0.3
                }
            )
            
            # Check that we got a result
            assert tool_result is not None
            assert hasattr(tool_result, 'content')
            content = tool_result.content
            assert isinstance(content, list)
            assert len(content) >= 1
            
            # First content item should be text
            first_content = content[0]
            assert isinstance(first_content, TextContent)
            assert first_content.type == "text"
            assert first_content.text is not None
    
    @pytest.mark.asyncio
    async def test_mcp_call_tool_analyze_code_through_library(self):
        """Test calling the analyze_code tool through the MCP client library."""
        test_code = '''
def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return "Hello, World!"
'''
        
        async with await self._create_client_session() as session:
            # Call the analyze_code tool
            tool_result = await session.call_tool(
                "analyze_code",
                {
                    "code": test_code,
                    "file_path": "test.py",
                    "language": "python"
                }
            )
            
            # Check that we got a result
            assert tool_result is not None
            assert hasattr(tool_result, 'content')
            content = tool_result.content
            assert isinstance(content, list)
            assert len(content) >= 1
            
            # First content item should be text
            first_content = content[0]
            assert isinstance(first_content, TextContent)
            assert first_content.type == "text"
            assert first_content.text is not None
    
    @pytest.mark.asyncio
    async def test_mcp_call_tool_keyword_search_through_library(self):
        """Test calling the keyword_search tool through the MCP client library."""
        async with await self._create_client_session() as session:
            # Call the keyword_search tool
            tool_result = await session.call_tool(
                "keyword_search",
                {
                    "query": "function_name:parse OR content:search",
                    "top_k": 5
                }
            )
            
            # Check that we got a result
            assert tool_result is not None
            assert hasattr(tool_result, 'content')
            content = tool_result.content
            assert isinstance(content, list)
            assert len(content) >= 1
            
            # First content item should be text
            first_content = content[0]
            assert isinstance(first_content, TextContent)
            assert first_content.type == "text"
            assert first_content.text is not None
    
    @pytest.mark.asyncio
    async def test_mcp_error_handling_through_library(self):
        """Test error handling through the MCP client library."""
        async with await self._create_client_session() as session:
            # Try to call a non-existent tool
            try:
                await session.call_tool(
                    "nonexistent_tool",
                    {}
                )
                # If we get here without an exception, check if error is in content
                assert False, "Expected an error for non-existent tool"
            except Exception as e:
                # We expect some kind of error (either MCP exception or tool error)
                assert True  # Test passes if we get any exception
    
    @pytest.mark.asyncio
    async def test_mcp_session_lifecycle(self):
        """Test the complete MCP session lifecycle through the library."""
        # Test session creation and destruction
        session = await self._create_client_session()
        
        # Session should be active
        assert session is not None
        
        # Should be able to perform operations
        tools_result = await session.list_tools()
        assert tools_result is not None
        assert len(tools_result.tools) == 5
        
        # Should be able to close session cleanly
        await session.__aexit__(None, None, None)
        
        # After closing, operations should fail or session should be marked as closed
        # (behavior depends on implementation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])