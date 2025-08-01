#!/usr/bin/env python3

#   Tests Affected by MockCocoIndex Issue:
#
#   Directly affected (fail when collected together):
#   1. tests/test_chunk_compatibility.py - Uses @cocoindex.op.function() and @cocoindex.flow_def()
#   2. tests/test_minimal_flow.py - Uses @cocoindex.op.function() and @cocoindex.flow_def()
#
#   Status when run individually:
#   - ✅ tests/test_minimal_flow.py - Passes (5/5 tests)
#   - ⚠️ tests/test_chunk_compatibility.py - Mostly passes (3/4 tests), 1 failure due to missing database config
#
#   Root cause: The MockCocoIndex in test_external_embedding_standalone.py patches sys.modules['cocoindex'] but only provides partial mocking.
#   When pytest collects all tests together, the other tests try to use cocoindex attributes that aren't mocked (flow_def, Json, sources,
#   targets, transform_flow).
#
#   Solution complexity: This requires adding comprehensive mocking for all CocoIndex attributes used across the test suite, which is
#   substantial work involving understanding the expected behavior of each mock method.
#
#   Recommendation: The MCP server functionality we fixed is working correctly. The test infrastructure issue is separate and would require
#   significant effort to resolve properly. You may want to decide whether to invest in fixing this test configuration issue or focus on other
#   priorities.
#
# This is getting quite complex - the MockCocoIndex needs many more attributes: Json, flow_def, sources, targets, transform_flow, etc. This
#   is no longer a "quick fix" but rather requires a comprehensive MockCocoIndex implementation.
#
#   This is NOT easy to fix. The test configuration issue requires creating a comprehensive mock of the entire CocoIndex API surface that the
#   tests depend on. This would be a significant effort involving:
#
#   1. Adding all the missing attributes (Json, flow_def, sources, targets, transform_flow, etc.)
#   2. Understanding the expected behavior of each mock method
#   3. Ensuring compatibility across all test scenarios
#
#   I recommend stopping here and letting you decide how to proceed. The individual test (test_minimal_flow.py) works fine when run alone, so
#   the core MCP server functionality we fixed is solid. The test collection issue is a separate infrastructure problem that would require
#   substantial work to resolve properly.

"""
Test minimal CocoIndex flow functionality.

This test validates that basic CocoIndex flows work correctly with custom
Chunk objects. It serves as a regression test for ValueType deserialization
issues and provides a minimal example of CocoIndex flow patterns.
"""

from types import FunctionType
from typing import cast
import pytest
import cocoindex
from cocoindex_code_mcp_server.ast_chunking import Chunk


@cocoindex.op.function()
def simple_chunk_creator(content: str) -> list[Chunk]:
    """Create a simple chunk without complex transformations."""
    return [Chunk(
        content=content,
        metadata={},
        location="test_location",
        start=0,
        end=len(content)
    )]


@cocoindex.flow_def(name="MinimalTest")
def minimal_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope) -> None:
    """Minimal test flow for validating basic CocoIndex functionality."""
    
    # Use a small test file
    data_scope["files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="tests/fixtures/sample.py",
            included_patterns=["*.py"]
        )
    )
    
    # Create simple collector
    simple_embeddings = data_scope.add_collector()
    
    with data_scope["files"].row() as file:
        # Create chunks without complex processing
        file["chunks"] = file["content"].transform(simple_chunk_creator)
        
        with file["chunks"].row() as chunk:
            # Collect basic chunk information
            simple_embeddings.collect(
                filename=file["filename"],
                content=chunk["content"],
                location=chunk["location"],
                start=chunk["start"],
                end=chunk["end"]
            )
    
    simple_embeddings.export(
        "simple_test",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename", "location"]
    )


class TestMinimalFlow:
    """Test cases for minimal CocoIndex flow functionality."""
    
    def test_simple_chunk_creator_function(self):
        """Test that the simple chunk creator function works correctly."""
        test_content = "def test(): pass"
        chunks = cast(FunctionType, simple_chunk_creator)(test_content)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        assert isinstance(chunk, Chunk)
        assert chunk.content == test_content
        assert chunk.location == "test_location"
        assert chunk.start == 0
        assert chunk.end == len(test_content)
        assert chunk.metadata == {}
    
    def test_chunk_with_empty_content(self):
        """Test chunk creation with empty content."""
        chunks = cast(FunctionType, simple_chunk_creator)("")
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        assert chunk.content == ""
        assert chunk.start == 0
        assert chunk.end == 0
    
    def test_chunk_with_unicode_content(self):
        """Test chunk creation with unicode content."""
        unicode_content = "def 测试(): return '🚀'"
        chunks = cast(FunctionType, simple_chunk_creator)(unicode_content)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        assert chunk.content == unicode_content
        assert chunk.end == len(unicode_content)
    
    @pytest.mark.skipif(
        not pytest.importorskip("psycopg", minversion=None),
        reason="PostgreSQL not available for integration test"
    )
    def test_minimal_flow_setup(self):
        """Test that the minimal flow can be set up without errors.
        
        This is a regression test for ValueType deserialization issues.
        """
        # Load environment variables from .env file
        import os
        from dotenv import load_dotenv
        load_dotenv('.env')
        
        cocoindex.init()
        
        # Create a test file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Simple test file\nprint('Hello, World!')\n")
            temp_file = f.name
        
        try:
            # Define test flow with the temporary file
            @cocoindex.flow_def(name="MinimalSetupTest")
            def test_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope) -> None:
                data_scope["files"] = flow_builder.add_source(
                    cocoindex.sources.LocalFile(
                        path=temp_file,
                        included_patterns=["*.py"]
                    )
                )
                
                collector = data_scope.add_collector()
                
                with data_scope["files"].row() as file:
                    file["chunks"] = file["content"].transform(simple_chunk_creator)
                    
                    with file["chunks"].row() as chunk:
                        collector.collect(
                            filename=file["filename"],
                            content=chunk["content"],
                            location=chunk["location"],
                            start=chunk["start"],
                            end=chunk["end"]
                        )
                
                collector.export(
                    "minimal_setup_test",
                    cocoindex.targets.Postgres(),
                    primary_key_fields=["filename", "location"]
                )
            
            # This should not raise ValueType or other deserialization errors
            flow = test_flow
            flow.setup()
            
            # Verify the flow can be updated (basic execution test)
            stats = flow.update()
            assert stats is not None
            
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def test_chunk_metadata_handling(self):
        """Test that chunks handle metadata correctly in flows."""
        
        @cocoindex.op.function()
        def metadata_chunk_creator(content: str) -> list[Chunk]:
            """Create chunks with metadata for testing."""
            return [Chunk(
                content=content,
                metadata={"test_key": "test_value", "line_count": content.count('\n') + 1},
                location=f"content_{hash(content) % 1000}",
                start=0,
                end=len(content)
            )]
        
        test_content = "line1\nline2\nline3"
        chunks = cast(FunctionType, metadata_chunk_creator)(test_content)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        assert chunk.metadata["test_key"] == "test_value"
        assert chunk.metadata["line_count"] == 3
        assert chunk.location.startswith("content_")
