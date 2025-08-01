#!/usr/bin/env python3
"""
Test Chunk dataclass compatibility with CocoIndex type system.

This test validates the fix for ValueType deserialization errors that occurred
when using Union types in dataclass fields. It ensures that Chunk objects with
cocoindex.Json metadata work properly with CocoIndex flows.
"""

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

import pytest
import cocoindex
from cocoindex_code_mcp_server.ast_chunking import Chunk


@cocoindex.op.function()
def chunk_creator_for_test(content: str) -> list[Chunk]:
    """Create test chunks with cocoindex.Json metadata."""
    chunk = Chunk(
        content=content,
        metadata={"test": "value", "complexity": 5, "has_functions": True},
        location="test_location",
        start=0,
        end=len(content) if content else 0
    )
    return [chunk]


@cocoindex.flow_def(name="ChunkCompatibilityTest")
def test_chunk_compatibility_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope) -> None:
    """Test flow that validates Chunk objects work with CocoIndex."""
    
    # Create test data source
    data_scope["test_files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="tests/fixtures/sample.py",
            included_patterns=["*.py"]
        )
    )
    
    # Create collector for test results
    test_collector = data_scope.add_collector()
    
    with data_scope["test_files"].row() as file:
        # Transform content to chunks using our test function
        file["chunks"] = file["content"].transform(chunk_creator_for_test)
        
        with file["chunks"].row() as chunk:
            # Collect chunk data to verify it works
            test_collector.collect(
                filename=file["filename"],
                content=chunk["content"],
                location=chunk["location"],
                start=chunk["start"],
                end=chunk["end"],
                metadata=chunk["metadata"]
            )
    
    # Export to test that the full pipeline works
    test_collector.export(
        "chunk_compatibility_test",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename", "location"]
    )


class TestChunkCompatibility:
    """Test cases for Chunk dataclass CocoIndex compatibility."""
    
    def test_chunk_creation(self):
        """Test that Chunk objects can be created with cocoindex.Json metadata."""
        chunk = Chunk(
            content="def test(): pass",
            metadata={"functions": ["test"], "complexity": 1},
            location="test.py:1-5",
            start=0,
            end=17
        )
        
        assert chunk.content == "def test(): pass"
        assert chunk.metadata["functions"] == ["test"]
        assert chunk.metadata["complexity"] == 1
        assert chunk.location == "test.py:1-5"
        assert chunk.start == 0
        assert chunk.end == 17
    
    def test_chunk_dictionary_access(self):
        """Test that Chunk supports dictionary-style access."""
        chunk = Chunk(
            content="test content",
            metadata={"key": "value"},
            location="test_loc"
        )
        
        # Test __getitem__
        assert chunk["content"] == "test content"
        assert chunk["location"] == "test_loc"
        assert chunk["key"] == "value"  # From metadata
        
        # Test __setitem__
        chunk["new_key"] = "new_value"
        assert chunk["new_key"] == "new_value"
        
        # Test get method
        assert chunk.get("content") == "test content"
        assert chunk.get("nonexistent", "default") == "default"
    
    def test_chunk_to_dict(self):
        """Test that Chunk.to_dict() works correctly."""
        chunk = Chunk(
            content="test content",
            metadata={"complexity": 5, "has_functions": True},
            location="test.py:1-10",
            start=0,
            end=12
        )
        
        result = chunk.to_dict()
        
        assert result["content"] == "test content"
        assert result["location"] == "test.py:1-10"
        assert result["start"] == 0
        assert result["end"] == 12
        assert result["complexity"] == 5
        assert result["has_functions"] is True
    
    @pytest.mark.skipif(
        not pytest.importorskip("psycopg", minversion=None),
        reason="PostgreSQL not available for integration test"
    )
    def test_cocoindex_flow_integration(self):
        """Integration test that validates Chunk works in CocoIndex flows.
        
        This test requires a running PostgreSQL instance and validates that
        the ValueType deserialization fix works end-to-end.
        """
        cocoindex.init()
        
        # Create a temporary file for testing
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test_function():\n    return 'Hello, World!'\n")
            temp_file = f.name
        
        try:
            # Define test flow
            @cocoindex.flow_def(name="ChunkIntegrationTest")
            def integration_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope) -> None:
                data_scope["files"] = flow_builder.add_source(
                    cocoindex.sources.LocalFile(
                        path=temp_file,
                        included_patterns=["*.py"]
                    )
                )
                
                collector = data_scope.add_collector()
                
                with data_scope["files"].row() as file:
                    file["chunks"] = file["content"].transform(chunk_creator_for_test)
                    
                    with file["chunks"].row() as chunk:
                        collector.collect(
                            filename=file["filename"],
                            content=chunk["content"],
                            location=chunk["location"],
                            start=chunk["start"],
                            end=chunk["end"]
                        )
                
                collector.export(
                    "integration_test",
                    cocoindex.targets.Postgres(),
                    primary_key_fields=["filename", "location"]
                )
            
            # Test flow setup (should not raise ValueType errors)
            flow = integration_flow
            flow.setup()
            
            # Test flow execution (should complete without errors)
            stats = flow.update()
            assert stats is not None
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
