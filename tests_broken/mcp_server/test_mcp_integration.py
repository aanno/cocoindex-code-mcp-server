"""
Integration test for the CocoIndex RAG MCP Server using MCP client library.

This module tests the MCP server running on Streamable HTTP transport by connecting
as a real MCP client and testing the full protocol interaction.
"""

import json

import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl


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
            # Create and initialize client session
            session = ClientSession(read, write)

            # Initialize with simplified parameters
            init_result = await session.initialize()

            # Check initialization result
            assert init_result is not None
            assert hasattr(init_result, "capabilities")
            assert hasattr(init_result, "serverInfo")

            # Check server capabilities
            capabilities = init_result.capabilities
            assert hasattr(capabilities, "tools")
            assert hasattr(capabilities, "resources")

            # Check server info
            server_info = init_result.serverInfo
            assert server_info.name == "cocoindex-rag"
            assert server_info.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Test listing resources via MCP protocol."""
        async with streamablehttp_client(self.SERVER_URL) as (read, write, get_session_id):
            # Create and initialize client session
            session = ClientSession(read, write)
            await session.initialize()

            # List resources
            resources_result = await session.list_resources()

            assert resources_result is not None
            resources = resources_result.resources

            # Should have 7 resources (updated based on HTTP test)
            assert len(resources) == 7

            # Check resource names
            resource_names = [r.name for r in resources]
            assert "Search Statistics" in resource_names
            assert "Search Configuration" in resource_names
            assert "Database Schema" in resource_names

            # Check resource URIs
            for resource in resources:
                assert str(resource.uri).startswith("cocoindex://")
                # Most resources are JSON, but grammar resource is Lark format
                if "grammar" in str(resource.uri):
                    assert resource.mimeType == "text/x-lark"
                else:
                    assert resource.mimeType == "application/json"

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test listing tools via MCP protocol."""
        async with streamablehttp_client(self.SERVER_URL) as (read, write, get_session_id):
            # Create and initialize client session
            session = ClientSession(read, write)
            await session.initialize()

            # List tools
            tools_result = await session.list_tools()

            assert tools_result is not None
            tools = tools_result.tools

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
                "get_keyword_syntax_help",
            ]

            for expected_tool in expected_tools:
                assert expected_tool in tool_names

            # Check that tools have valid schemas
            for tool in tools:
                assert hasattr(tool, "inputSchema")
                schema = tool.inputSchema
                assert isinstance(schema, dict)
                assert schema["type"] == "object"
                assert "properties" in schema

    @pytest.mark.asyncio
    async def test_read_resource(self):
        """Test reading a resource via MCP protocol."""
        async with streamablehttp_client(self.SERVER_URL) as (read, write, get_session_id):
            # Create and initialize client session
            session = ClientSession(read, write)
            await session.initialize()

            # Read search configuration resource
            resource_result = await session.read_resource(AnyUrl("cocoindex://search/config"))

            assert resource_result is not None
            contents = resource_result.contents

            # Should have one content item
            assert len(contents) == 1
            content = contents[0]

            # Check content properties
            assert content.uri == "cocoindex://search/config"
            assert hasattr(content, "text")

            # Content should be valid JSON
            config_data = json.loads(content.text)
            assert isinstance(config_data, dict)

            # Check expected configuration keys
            expected_keys = ["table_name", "embedding_model", "parser_type", "supported_operators", "default_weights"]

            for key in expected_keys:
                assert key in config_data

    @pytest.mark.asyncio
    async def test_call_tool_vector_search(self):
        """Test calling the vector_search tool."""
        async with streamablehttp_client(self.SERVER_URL) as (read, write, get_session_id):
            # Create and initialize client session
            session = ClientSession(read, write)
            await session.initialize()

            # Call vector search tool
            tool_result = await session.call_tool("vector_search", {"query": "test search query", "top_k": 5})

            assert tool_result is not None

            # The tool should return results
            assert hasattr(tool_result, "content")

            # Content should be a list with at least one item
            content = tool_result.content
            assert isinstance(content, list)
            assert len(content) >= 1

            # First content item should be text
            first_content = content[0]
            assert hasattr(first_content, "type")
            assert first_content.type == "text"
            assert hasattr(first_content, "text")

    @pytest.mark.asyncio
    async def test_call_tool_get_embeddings(self):
        """Test calling the get_embeddings tool."""
        async with streamablehttp_client(self.SERVER_URL) as (read, write, get_session_id):
            # Create and initialize client session
            session = ClientSession(read, write)
            await session.initialize()

            # Call get embeddings tool
            tool_result = await session.call_tool("get_embeddings", {"text": "test text for embedding"})

            assert tool_result is not None

            # The tool should return results
            assert hasattr(tool_result, "content")

            # Content should be a list with one item
            content = tool_result.content
            assert isinstance(content, list)
            assert len(content) == 1

            # Content should be text with embedding data
            first_content = content[0]
            assert hasattr(first_content, "type")
            assert first_content.type == "text"
            assert hasattr(first_content, "text")

            # Parse the JSON response to check embedding format
            embedding_data = json.loads(first_content.text)
            assert "embedding" in embedding_data
            # Check for either "model" or "dimensions" field (implementation specific)
            assert "dimensions" in embedding_data or "model" in embedding_data
            assert isinstance(embedding_data["embedding"], list)
            assert len(embedding_data["embedding"]) > 0

    # Helper method no longer needed - using ClientSession.initialize() directly


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
