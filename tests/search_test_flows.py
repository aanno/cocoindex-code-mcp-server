#!/usr/bin/env python3

"""
Search test flow definitions.

This module provides separate CocoIndex flow definitions for each search test type,
allowing them to use independent tables and avoid conflicts with the main MCP server.
"""

import cocoindex


@cocoindex.flow_def(name="KeywordSearchTest")
def keyword_search_test_flow(
    flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
) -> None:
    """
    Flow definition for keyword search tests.
    Uses table: keywordsearchtest__test_embeddings
    """
    
    # For now, use a simple placeholder that just exports empty data
    # TODO: Implement the actual flow logic similar to code_embedding_flow
    # This would need to replicate the full transformation pipeline from the main flow
    
    # Create a collector for test embeddings
    code_embeddings = data_scope.add_collector()
    
    # Export to separate test table
    code_embeddings.export(
        "test_embeddings",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename", "location", "source_name"],
    )


@cocoindex.flow_def(name="VectorSearchTest")
def vector_search_test_flow(
    flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
) -> None:
    """
    Flow definition for vector search tests.
    Uses table: vectorsearchtest__test_embeddings
    """
    
    # For now, use a simple placeholder that just exports empty data
    # TODO: Implement the actual flow logic similar to code_embedding_flow
    # This would need to replicate the full transformation pipeline from the main flow
    
    # Create a collector for test embeddings
    code_embeddings = data_scope.add_collector()
    
    # Export to separate test table
    code_embeddings.export(
        "test_embeddings",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename", "location", "source_name"],
    )


@cocoindex.flow_def(name="HybridSearchTest")
def hybrid_search_test_flow(
    flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
) -> None:
    """
    Flow definition for hybrid search tests.
    Uses table: hybridsearchtest__test_embeddings
    """
    
    # For now, use a simple placeholder that just exports empty data
    # TODO: Implement the actual flow logic similar to code_embedding_flow  
    # This would need to replicate the full transformation pipeline from the main flow
    
    # Create a collector for test embeddings
    code_embeddings = data_scope.add_collector()
    
    # Export to separate test table
    code_embeddings.export(
        "test_embeddings", 
        cocoindex.targets.Postgres(),
    )


def get_test_table_name(test_type: str) -> str:
    """
    Get the table name for a specific test type.
    
    Args:
        test_type: One of 'keyword', 'vector', 'hybrid'
    
    Returns:
        Table name for the test type
    """
    flow_mapping = {
        'keyword': keyword_search_test_flow,
        'vector': vector_search_test_flow, 
        'hybrid': hybrid_search_test_flow
    }
    
    if test_type not in flow_mapping:
        raise ValueError(f"Unknown test type: {test_type}. Must be one of {list(flow_mapping.keys())}")
    
    flow = flow_mapping[test_type]
    return cocoindex.utils.get_target_default_name(
        flow=flow, target_name="test_embeddings"
    ).lower()