#!/usr/bin/env python3

"""
Search test flow definitions.

This module provides parameterized CocoIndex flow definitions for each search test type,
allowing them to use independent tables and avoid conflicts with the main MCP server.

Uses cocoindex.open_flow with parameterized flow definitions for proper table separation.
"""

import cocoindex
from dataclasses import dataclass


@dataclass
class SearchTestFlowParameters:
    """Parameters for search test flows."""
    source_path: str
    target_table_name: str


def search_test_flow_def(params: SearchTestFlowParameters):
    """
    Create a parameterized flow definition for search tests.
    
    This reuses the same flow logic but allows different source paths and table names.
    """
    def _flow_def(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
        # Import necessary functions from main flow
        from cocoindex_code_mcp_server.cocoindex_config import (
            SOURCE_CONFIG, extract_language, get_chunking_params,
            code_to_embedding, get_default_embedding_model_name, extract_code_metadata, extract_functions_field,
            get_embedding_model_group, get_embedding_model_name,
            graphcodebert_embedding, unixcoder_embedding, fallback_embedding,
            SMART_EMBEDDING_AVAILABLE,
            extract_classes_field, extract_imports_field, extract_complexity_score_field,
            extract_has_type_hints_field, extract_has_async_field, extract_has_classes_field,
            extract_analysis_method_field, extract_chunking_method_field,
            extract_tree_sitter_chunking_error_field, extract_tree_sitter_analyze_error_field,
            extract_decorators_used_field, extract_dunder_methods_field, extract_success_field,
            extract_parse_errors_field, extract_char_count_field, extract_docstring_field,
            extract_nodes_with_errors_field, extract_data_types_field, extract_instances_field,
            extract_type_classes_field, extract_modules_field, extract_has_module_field,
            extract_function_details_field, extract_data_type_details_field,
            extract_structs_field, extract_traits_field, extract_impls_field,
            extract_exports_field, extract_types_field, extract_enums_field,
            extract_namespaces_field, extract_private_methods_field, extract_variables_field,
            extract_decorators_field, extract_class_details_field,
            convert_dataslice_to_string, list_to_space_separated_str, CUSTOM_LANGUAGES
        )
        
        # Configure source with the specified path
        source_config = SOURCE_CONFIG.copy()
        source_config['path'] = params.source_path
        
        data_scope["files"] = flow_builder.add_source(
            cocoindex.sources.LocalFile(**source_config)
        )
        
        code_embeddings = data_scope.add_collector()
        
        with data_scope["files"].row() as file:
            file["language"] = file["filename"].transform(extract_language)
            file["chunking_params"] = file["language"].transform(get_chunking_params)
            
            # Use simple chunking for tests
            raw_chunks = file["content"].transform(
                cocoindex.functions.SplitRecursively(custom_languages=CUSTOM_LANGUAGES),
                language=file["language"],
                chunk_size=file["chunking_params"]["chunk_size"],
                min_chunk_size=file["chunking_params"]["min_chunk_size"],
                chunk_overlap=file["chunking_params"]["chunk_overlap"],
            )
            
            # Work with raw chunks directly - they are already a collection
            # SplitRecursively returns chunks with 'text' field, not 'content'

            # Add model group for smart embedding selection
            if SMART_EMBEDDING_AVAILABLE:
                with raw_chunks.row() as chunk:
                    chunk["model_group"] = file["language"].transform(get_embedding_model_group)

            with raw_chunks.row() as chunk:
                # Use smart embeddings (language-specific models) like main flow
                if SMART_EMBEDDING_AVAILABLE:
                    model_group = chunk["model_group"]
                    if model_group == "graphcodebert":
                        chunk["embedding"] = chunk["text"].call(graphcodebert_embedding)
                    elif model_group == "unixcoder":
                        chunk["embedding"] = chunk["text"].call(unixcoder_embedding)
                    else:  # fallback
                        chunk["embedding"] = chunk["text"].call(fallback_embedding)
                    # Store the actual embedding model name used
                    chunk["embedding_model"] = chunk["model_group"].transform(get_embedding_model_name)
                else:
                    # Fallback if smart embeddings not available
                    chunk["embedding"] = chunk["text"].call(code_to_embedding)
                    chunk["embedding_model"] = chunk["text"].transform(get_default_embedding_model_name)
                chunk["extracted_metadata"] = chunk["text"].transform(
                    extract_code_metadata,
                    language=file["language"],
                    filename=file["filename"],
                    existing_metadata_json=""
                )
                
                # Extract all metadata fields to match main flow schema
                chunk["functions"] = chunk["extracted_metadata"].transform(extract_functions_field)
                chunk["classes"] = chunk["extracted_metadata"].transform(extract_classes_field)
                chunk["imports"] = chunk["extracted_metadata"].transform(extract_imports_field)
                chunk["complexity_score"] = chunk["extracted_metadata"].transform(extract_complexity_score_field)
                chunk["has_type_hints"] = chunk["extracted_metadata"].transform(extract_has_type_hints_field)
                chunk["has_async"] = chunk["extracted_metadata"].transform(extract_has_async_field)
                chunk["has_classes"] = chunk["extracted_metadata"].transform(extract_has_classes_field)
                chunk["analysis_method"] = chunk["extracted_metadata"].transform(extract_analysis_method_field)
                chunk["chunking_method"] = chunk["extracted_metadata"].transform(extract_chunking_method_field)
                chunk["docstring"] = chunk["extracted_metadata"].transform(extract_docstring_field)
                
                # Error tracking fields
                chunk["tree_sitter_chunking_error"] = chunk["extracted_metadata"].transform(extract_tree_sitter_chunking_error_field)
                chunk["tree_sitter_analyze_error"] = chunk["extracted_metadata"].transform(extract_tree_sitter_analyze_error_field)
                chunk["success"] = chunk["extracted_metadata"].transform(extract_success_field)
                chunk["parse_errors"] = chunk["extracted_metadata"].transform(extract_parse_errors_field)
                chunk["char_count"] = chunk["extracted_metadata"].transform(extract_char_count_field)
                
                # Python-specific fields
                chunk["decorators_used"] = chunk["extracted_metadata"].transform(extract_decorators_used_field)
                chunk["dunder_methods"] = chunk["extracted_metadata"].transform(extract_dunder_methods_field)
                chunk["private_methods"] = chunk["extracted_metadata"].transform(extract_private_methods_field)
                chunk["variables"] = chunk["extracted_metadata"].transform(extract_variables_field)
                chunk["decorators"] = chunk["extracted_metadata"].transform(extract_decorators_field)
                chunk["class_details"] = chunk["extracted_metadata"].transform(extract_class_details_field)
                
                # Language-specific fields (will be empty for non-matching languages)
                chunk["nodes_with_errors"] = chunk["extracted_metadata"].transform(extract_nodes_with_errors_field)  # Haskell
                chunk["data_types"] = chunk["extracted_metadata"].transform(extract_data_types_field)  # Haskell
                chunk["instances"] = chunk["extracted_metadata"].transform(extract_instances_field)  # Haskell
                chunk["type_classes"] = chunk["extracted_metadata"].transform(extract_type_classes_field)  # Haskell
                chunk["modules"] = chunk["extracted_metadata"].transform(extract_modules_field)  # Haskell, Rust
                chunk["has_module"] = chunk["extracted_metadata"].transform(extract_has_module_field)  # Haskell
                chunk["function_details"] = chunk["extracted_metadata"].transform(extract_function_details_field)  # Haskell
                chunk["data_type_details"] = chunk["extracted_metadata"].transform(extract_data_type_details_field)  # Haskell
                chunk["structs"] = chunk["extracted_metadata"].transform(extract_structs_field)  # Rust
                chunk["traits"] = chunk["extracted_metadata"].transform(extract_traits_field)  # Rust
                chunk["impls"] = chunk["extracted_metadata"].transform(extract_impls_field)  # Rust
                chunk["exports"] = chunk["extracted_metadata"].transform(extract_exports_field)  # TypeScript/JavaScript
                chunk["types"] = chunk["extracted_metadata"].transform(extract_types_field)  # TypeScript
                chunk["enums"] = chunk["extracted_metadata"].transform(extract_enums_field)  # TypeScript
                chunk["namespaces"] = chunk["extracted_metadata"].transform(extract_namespaces_field)  # TypeScript/JavaScript/C++
                
                code_embeddings.collect(
                    filename=file["filename"],
                    language=file["language"],
                    location=chunk["location"],
                    code=chunk["text"].transform(convert_dataslice_to_string),
                    embedding=chunk["embedding"],
                    embedding_model=chunk["embedding_model"],  # CRITICAL: Store which model was used
                    start=chunk["start"],
                    end=chunk["end"],
                    source_name="files",
                    metadata_json=chunk["extracted_metadata"],

                    # Core metadata fields
                    complexity_score=chunk["complexity_score"],
                    has_type_hints=chunk["has_type_hints"],
                    has_async=chunk["has_async"],
                    has_classes=chunk["has_classes"],
                    analysis_method=chunk["analysis_method"],
                    chunking_method=chunk["chunking_method"],
                    docstring=chunk["docstring"],
                    
                    # Error tracking fields
                    tree_sitter_chunking_error=chunk["tree_sitter_chunking_error"],
                    tree_sitter_analyze_error=chunk["tree_sitter_analyze_error"],
                    success=chunk["success"],
                    parse_errors=chunk["parse_errors"],
                    char_count=chunk["char_count"],
                    
                    # Language-specific boolean fields
                    has_module=chunk["has_module"],
                    
                    # JSON string fields for complex data
                    class_details=chunk["class_details"],
                    function_details=chunk["function_details"],
                    data_type_details=chunk["data_type_details"],
                    
                    # List fields converted to space-separated strings
                    functions=chunk["functions"].transform(list_to_space_separated_str),
                    classes=chunk["classes"].transform(list_to_space_separated_str),
                    imports=chunk["imports"].transform(list_to_space_separated_str),
                    decorators_used=chunk["decorators_used"].transform(list_to_space_separated_str),
                    dunder_methods=chunk["dunder_methods"].transform(list_to_space_separated_str),
                    private_methods=chunk["private_methods"].transform(list_to_space_separated_str),
                    variables=chunk["variables"].transform(list_to_space_separated_str),
                    decorators=chunk["decorators"].transform(list_to_space_separated_str),
                    nodes_with_errors=chunk["nodes_with_errors"].transform(list_to_space_separated_str),
                    data_types=chunk["data_types"].transform(list_to_space_separated_str),
                    instances=chunk["instances"].transform(list_to_space_separated_str),
                    type_classes=chunk["type_classes"].transform(list_to_space_separated_str),
                    modules=chunk["modules"].transform(list_to_space_separated_str),
                    structs=chunk["structs"].transform(list_to_space_separated_str),
                    traits=chunk["traits"].transform(list_to_space_separated_str),
                    impls=chunk["impls"].transform(list_to_space_separated_str),
                    exports=chunk["exports"].transform(list_to_space_separated_str),
                    types=chunk["types"].transform(list_to_space_separated_str),
                    enums=chunk["enums"].transform(list_to_space_separated_str),
                    namespaces=chunk["namespaces"].transform(list_to_space_separated_str),
                )
        
        # Export to the specified target table
        code_embeddings.export(
            "code_embeddings",
            cocoindex.targets.Postgres(table_name=params.target_table_name),
            primary_key_fields=["filename", "location", "source_name"],
            vector_indexes=[
                cocoindex.VectorIndexDef(
                    field_name="embedding", 
                    metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
                )
            ],
        )
    
    return _flow_def


def create_search_test_flow(test_type: str, source_path: str = "tmp"):
    """
    Create a search test flow for the specified test type.
    
    Args:
        test_type: One of 'keyword', 'vector', 'hybrid'
        source_path: Path to the source files (default: "tmp")
    
    Returns:
        Configured CocoIndex flow instance
    """
    # Generate table name based on test type
    table_names = {
        'keyword': 'keywordsearchtest_code_embeddings',
        'vector': 'vectorsearchtest_code_embeddings', 
        'hybrid': 'hybridsearchtest_code_embeddings'
    }
    
    if test_type not in table_names:
        raise ValueError(f"Unknown test type: {test_type}. Must be one of {list(table_names.keys())}")
    
    table_name = table_names[test_type]
    flow_name = f"SearchTest_{test_type.title()}"
    
    # Create parameters for this test flow
    params = SearchTestFlowParameters(
        source_path=source_path,
        target_table_name=table_name
    )
    
    # Open flow with parameterized definition
    flow = cocoindex.open_flow(flow_name, search_test_flow_def(params))
    return flow


def get_test_table_name(test_type: str) -> str:
    """
    Get the table name for a specific test type.
    
    Args:
        test_type: One of 'keyword', 'vector', 'hybrid'
    
    Returns:
        Table name for the test type
    """
    table_names = {
        'keyword': 'keywordsearchtest_code_embeddings',
        'vector': 'vectorsearchtest_code_embeddings', 
        'hybrid': 'hybridsearchtest_code_embeddings'
    }
    
    if test_type not in table_names:
        raise ValueError(f"Unknown test type: {test_type}. Must be one of {list(table_names.keys())}")
    
    return table_names[test_type]


# Create flow instances for each test type
keyword_search_test_flow = None
vector_search_test_flow = None  
hybrid_search_test_flow = None


def get_search_test_flow(test_type: str):
    """Get or create a search test flow for the specified type."""
    global keyword_search_test_flow, vector_search_test_flow, hybrid_search_test_flow
    
    if test_type == 'keyword':
        if keyword_search_test_flow is None:
            keyword_search_test_flow = create_search_test_flow('keyword')
        return keyword_search_test_flow
    elif test_type == 'vector':
        if vector_search_test_flow is None:
            vector_search_test_flow = create_search_test_flow('vector')
        return vector_search_test_flow
    elif test_type == 'hybrid':
        if hybrid_search_test_flow is None:
            hybrid_search_test_flow = create_search_test_flow('hybrid')
        return hybrid_search_test_flow
    else:
        raise ValueError(f"Unknown test type: {test_type}")