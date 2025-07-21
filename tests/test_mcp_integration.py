"""
Integration test for the CocoIndex RAG MCP Server using MCP client library.

This module tests the MCP server running on Streamable HTTP transport by connecting
as a real MCP client and testing the full protocol interaction.
"""

import json
import pytest
import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp import types


@pytest.mark.mcp_integration
@pytest.mark.asyncio
class TestMCPIntegration:
    """Integration tests using MCP client library."""
    
    SERVER_URL = "http://localhost:3033/mcp"
    
    @pytest.mark.asyncio
    async def test_server_connection(self):
        """Test basic connection to the MCP server."""
        async with streamablehttp_client(self.SERVER_URL) as (read, write, get_session_id):
            # Test that we can establish a connection
            assert read is not None
            assert write is not None
    
    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test MCP protocol initialization."""
        async with streamablehttp_client(self.SERVER_URL) as (read, write, get_session_id):
            # Initialize the session
            init_request = types.InitializeRequest(
                method="initialize",
                params=types.InitializeRequestParams(
                    protocolVersion="2024-11-05",
                    capabilities=types.ClientCapabilities(
                        roots=types.RootsCapability(listChanged=True),
                        sampling=types.SamplingCapability()
                    ),
                    clientInfo=types.Implementation(
                        name="test-client",
                        version="1.0.0"
                    )
                )
            )
            
            # Send initialize request
            await write.send(init_request)
            
            # Read response
            response = await read.receive()
            assert response is not None
            assert hasattr(response, 'result')
            
            # The server should respond with its capabilities
            result = response.result
            assert hasattr(result, 'capabilities')
            assert hasattr(result, 'serverInfo')
    
    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Test listing resources via MCP protocol."""
        async with streamablehttp_client(self.SERVER_URL) as (read, write, get_session_id):
            # Initialize first
            await self._initialize_session(read, write)
            
            # List resources
            list_resources_request = types.ListResourcesRequest(
                method="resources/list",
                params={}
            )
            
            await write.send(list_resources_request)
            response = await read.receive()
            
            assert response is not None
            assert hasattr(response, 'result')
            resources = response.result.resources
            
            # Should have 3 resources
            assert len(resources) == 3
            
            # Check resource names
            resource_names = [r.name for r in resources]
            assert "Search Statistics" in resource_names
            assert "Search Configuration" in resource_names
            assert "Database Schema" in resource_names
            
            # Check resource URIs
            for resource in resources:
                assert str(resource.uri).startswith("cocoindex://")
                assert resource.mimeType == "application/json"
    
    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test listing tools via MCP protocol."""
        async with streamablehttp_client(self.SERVER_URL) as (read, write, get_session_id):
            # Initialize first
            await self._initialize_session(read, write)
            
            # List tools
            list_tools_request = types.ListToolsRequest(
                method="tools/list",
                params={}
            )
            
            await write.send(list_tools_request)
            response = await read.receive()
            
            assert response is not None
            assert hasattr(response, 'result')
            tools = response.result.tools
            
            # Should have 6 tools
            assert len(tools) == 6
            
            # Check tool names
            tool_names = [t.name for t in tools]
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
            
            # Check that tools have valid schemas
            for tool in tools:
                assert hasattr(tool, 'inputSchema')
                schema = tool.inputSchema
                assert isinstance(schema, dict)
                assert schema["type"] == "object"
                assert "properties" in schema
    
    @pytest.mark.asyncio
    async def test_read_resource(self):
        """Test reading a resource via MCP protocol."""
        async with streamablehttp_client(self.SERVER_URL) as (read, write, get_session_id):
            # Initialize first
            await self._initialize_session(read, write)
            
            # Read search configuration resource
            read_resource_request = types.ReadResourceRequest(
                method="resources/read",
                params=types.ReadResourceRequestParams(
                    uri="cocoindex://search/config"
                )
            )
            
            await write.send(read_resource_request)
            response = await read.receive()
            
            assert response is not None
            assert hasattr(response, 'result')
            contents = response.result.contents
            
            # Should have one content item
            assert len(contents) == 1
            content = contents[0]
            
            # Check content properties
            assert content.uri == "cocoindex://search/config"
            assert content.mimeType == "application/json"
            
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
    async def test_call_tool_vector_search(self):
        """Test calling the vector_search tool."""
        async with streamablehttp_client(self.SERVER_URL) as (read, write, get_session_id):
            # Initialize first
            await self._initialize_session(read, write)
            
            # Call vector search tool
            call_tool_request = types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(
                    name="vector_search",
                    arguments={
                        "query": "test search query",
                        "top_k": 5
                    }
                )
            )
            
            await write.send(call_tool_request)
            response = await read.receive()
            
            assert response is not None
            assert hasattr(response, 'result')
            
            # The tool should return results
            result = response.result
            assert hasattr(result, 'content')
            
            # Content should be a list with at least one item
            content = result.content
            assert isinstance(content, list)
            assert len(content) >= 1
            
            # First content item should be text
            first_content = content[0]
            assert hasattr(first_content, 'type')
            assert first_content.type == "text"
            assert hasattr(first_content, 'text')
    
    @pytest.mark.asyncio
    async def test_call_tool_get_embeddings(self):
        """Test calling the get_embeddings tool."""
        async with streamablehttp_client(self.SERVER_URL) as (read, write, get_session_id):
            # Initialize first
            await self._initialize_session(read, write)
            
            # Call get embeddings tool
            call_tool_request = types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(
                    name="get_embeddings",
                    arguments={
                        "text": "test text for embedding"
                    }
                )
            )
            
            await write.send(call_tool_request)
            response = await read.receive()
            
            assert response is not None
            assert hasattr(response, 'result')
            
            # The tool should return results
            result = response.result
            assert hasattr(result, 'content')
            
            # Content should be a list with one item
            content = result.content
            assert isinstance(content, list)
            assert len(content) == 1
            
            # Content should be text with embedding data
            first_content = content[0]
            assert hasattr(first_content, 'type')
            assert first_content.type == "text"
            assert hasattr(first_content, 'text')
            
            # Parse the JSON response to check embedding format
            embedding_data = json.loads(first_content.text)
            assert "embedding" in embedding_data
            assert "model" in embedding_data
            assert isinstance(embedding_data["embedding"], list)
            assert len(embedding_data["embedding"]) > 0
    
    async def _initialize_session(self, read, write):
        """Helper method to initialize MCP session."""
        init_request = types.InitializeRequest(
            method="initialize",
            params=types.InitializeRequestParams(
                protocolVersion="2024-11-05",
                capabilities=types.ClientCapabilities(
                    roots=types.RootsCapability(listChanged=True),
                    sampling=types.SamplingCapability()
                ),
                clientInfo=types.Implementation(
                    name="test-client",
                    version="1.0.0"
                )
            )
        )
        
        await write.send(init_request)
        response = await read.receive()
        
        # Send initialized notification
        initialized_notification = types.InitializedNotification(
            method="notifications/initialized",
            params={}
        )
        await write.send(initialized_notification)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
