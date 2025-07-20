"""
Test suite for the CocoIndex RAG MCP Server.

This module contains pytest tests for the MCP server functionality,
including tool schemas, resource handling, and basic server operations.
"""

import json
import sys
from pathlib import Path

import pytest
import mcp.types as types

# The path setup is handled by conftest.py


@pytest.fixture
def mcp_server():
    """Fixture to import and provide the MCP server module."""
    import mcp_server
    return mcp_server


@pytest.mark.mcp_server
@pytest.mark.unit
class TestMCPServerBasics:
    """Test basic MCP server functionality."""

    def test_server_module_import(self, mcp_server):
        """Test that the MCP server module can be imported."""
        assert mcp_server is not None
        assert hasattr(mcp_server, 'server')
        assert hasattr(mcp_server, 'handle_list_resources')
        assert hasattr(mcp_server, 'handle_list_tools')

    def test_server_object_creation(self, mcp_server):
        """Test that the server object is created properly."""
        server = mcp_server.server
        assert server is not None
        # Check that it's an MCP Server instance
        assert str(type(server)).endswith("Server'>")

    @pytest.mark.asyncio
    async def test_list_resources(self, mcp_server):
        """Test that resources are properly defined."""
        resources = await mcp_server.handle_list_resources()
        
        assert isinstance(resources, list)
        assert len(resources) == 3
        
        # Check specific resources exist
        resource_names = [r.name for r in resources]
        assert "Search Statistics" in resource_names
        assert "Search Configuration" in resource_names
        assert "Database Schema" in resource_names
        
        # Check resource URIs are properly formatted
        for resource in resources:
            assert str(resource.uri).startswith("cocoindex://")
            assert resource.mimeType == "application/json"

    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_server):
        """Test that tools are properly defined."""
        tools = await mcp_server.handle_list_tools()
        
        assert isinstance(tools, list)
        assert len(tools) == 5
        
        # Check specific tools exist
        tool_names = [t.name for t in tools]
        expected_tools = [
            "hybrid_search",
            "vector_search", 
            "keyword_search",
            "analyze_code",
            "get_embeddings"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names


@pytest.mark.mcp_server
@pytest.mark.unit
class TestToolSchemas:
    """Test tool input schemas."""

    @pytest.mark.asyncio
    async def test_tool_schemas_valid(self, mcp_server):
        """Test that all tool schemas are valid JSON Schema."""
        tools = await mcp_server.handle_list_tools()
        
        for tool in tools:
            schema = tool.inputSchema
            
            # Basic schema structure
            assert isinstance(schema, dict)
            assert "type" in schema
            assert schema["type"] == "object"
            assert "properties" in schema
            
            # Check required fields exist in properties
            required = schema.get("required", [])
            properties = schema.get("properties", {})
            
            for req_field in required:
                assert req_field in properties, f"Tool {tool.name}: Missing required property '{req_field}'"

    @pytest.mark.asyncio
    async def test_hybrid_search_schema(self, mcp_server):
        """Test hybrid_search tool schema specifically."""
        tools = await mcp_server.handle_list_tools()
        hybrid_search = next(t for t in tools if t.name == "hybrid_search")
        
        schema = hybrid_search.inputSchema
        properties = schema["properties"]
        
        # Check required fields
        assert "vector_query" in properties
        assert "keyword_query" in properties
        assert set(schema["required"]) == {"vector_query", "keyword_query"}
        
        # Check optional fields have defaults
        assert "top_k" in properties
        assert properties["top_k"]["default"] == 10
        assert "vector_weight" in properties
        assert properties["vector_weight"]["default"] == 0.7

    @pytest.mark.asyncio
    async def test_vector_search_schema(self, mcp_server):
        """Test vector_search tool schema specifically."""
        tools = await mcp_server.handle_list_tools()
        vector_search = next(t for t in tools if t.name == "vector_search")
        
        schema = vector_search.inputSchema
        properties = schema["properties"]
        
        # Check required fields
        assert "query" in properties
        assert schema["required"] == ["query"]
        
        # Check optional fields
        assert "top_k" in properties
        assert properties["top_k"]["default"] == 10

    @pytest.mark.asyncio
    async def test_analyze_code_schema(self, mcp_server):
        """Test analyze_code tool schema specifically."""
        tools = await mcp_server.handle_list_tools()
        analyze_code = next(t for t in tools if t.name == "analyze_code")
        
        schema = analyze_code.inputSchema
        properties = schema["properties"]
        
        # Check required fields
        assert "code" in properties
        assert "file_path" in properties
        assert set(schema["required"]) == {"code", "file_path"}
        
        # Check optional fields
        assert "language" in properties


@pytest.mark.mcp_server
@pytest.mark.unit
class TestResourceHandling:
    """Test MCP resource handling."""

    @pytest.mark.asyncio
    async def test_search_config_resource(self, mcp_server):
        """Test reading the search configuration resource."""
        config_content = await mcp_server.get_search_config()
        
        # Should be valid JSON
        config_data = json.loads(config_content)
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
    async def test_search_config_operators(self, mcp_server):
        """Test that search config includes expected operators."""
        config_content = await mcp_server.get_search_config()
        config_data = json.loads(config_content)
        
        operators = config_data["supported_operators"]
        expected_operators = ["AND", "OR", "NOT", "value_contains", "==", "!=", "<", ">", "<=", ">="]
        
        for op in expected_operators:
            assert op in operators

    @pytest.mark.asyncio
    async def test_read_resource_handler(self, mcp_server):
        """Test the general resource reading handler."""
        # Test valid resource
        config_result = await mcp_server.handle_read_resource("cocoindex://search/config")
        assert isinstance(config_result, str)
        
        # Should be valid JSON
        json.loads(config_result)
        
        # Test invalid resource
        with pytest.raises(ValueError, match="Unknown resource"):
            await mcp_server.handle_read_resource("cocoindex://invalid/resource")


@pytest.mark.mcp_server
@pytest.mark.unit
class TestServerConfiguration:
    """Test server configuration and setup."""

    def test_environment_variable_defaults(self, mcp_server):
        """Test that the server has reasonable defaults for configuration."""
        # The server should handle missing environment variables gracefully
        # This is tested indirectly by checking that the server can be imported
        # without throwing errors
        assert mcp_server is not None

    @pytest.mark.asyncio
    async def test_server_handlers_exist(self, mcp_server):
        """Test that all required MCP handlers are defined."""
        # These should not raise AttributeError
        assert callable(mcp_server.handle_list_resources)
        assert callable(mcp_server.handle_list_tools) 
        assert callable(mcp_server.handle_read_resource)
        assert callable(mcp_server.handle_call_tool)
        
        # These should not raise errors when called
        await mcp_server.handle_list_resources()
        await mcp_server.handle_list_tools()


@pytest.mark.mcp_server
@pytest.mark.integration
class TestIntegration:
    """Test integration with existing components."""

    def test_imports_work(self, mcp_server):
        """Test that all required imports work."""
        # If the module imported successfully, all imports worked
        # This includes:
        # - mcp.server.stdio, mcp.types, mcp.server
        # - psycopg_pool, pgvector.psycopg
        # - hybrid_search, keyword_search_parser_lark, etc.
        # - cocoindex, cocoindex_config
        assert mcp_server is not None

    @pytest.mark.asyncio
    async def test_tool_argument_structure(self, mcp_server):
        """Test that tool arguments follow expected structure."""
        tools = await mcp_server.handle_list_tools()
        
        for tool in tools:
            schema = tool.inputSchema
            
            # All tools should accept arguments as objects
            assert schema["type"] == "object"
            
            # All tools should have at least one property
            assert len(schema["properties"]) > 0
            
            # All tools should have at least one required field
            assert len(schema.get("required", [])) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])