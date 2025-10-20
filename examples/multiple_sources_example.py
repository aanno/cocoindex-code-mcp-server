#!/usr/bin/env python3
"""
Example demonstrating multiple sources in a single CocoIndex flow
"""

import cocoindex


@cocoindex.flow_def(name="MultipleSourcesDemo")
def multiple_sources_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope) -> None:
    """
    Example flow demonstrating multiple sources in a single flow.
    This shows how CocoIndex can handle multiple data sources simultaneously.
    """

    # Source 1: Local Python files
    data_scope["python_files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="cocoindex/src",
            included_patterns=["*.py"],
            excluded_patterns=["**/.*", "__pycache__", "*.pyc"],
        ),
        name="python_source",
    )

    # Source 2: Local Rust files
    data_scope["rust_files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="cocoindex/src",
            included_patterns=["*.rs"],
            excluded_patterns=["**/.*", "target", "*.bak"],
        ),
        name="rust_source",
    )

    # Source 3: Configuration files
    data_scope["config_files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="cocoindex",
            included_patterns=["*.toml", "*.yaml", "*.yml"],
            excluded_patterns=["**/.*", "target", "node_modules"],
        ),
        name="config_source",
    )

    # Create collectors for different file types
    code_embeddings = data_scope.add_collector("code_embeddings")
    config_metadata = data_scope.add_collector("config_metadata")

    # Process Python files
    with data_scope["python_files"].row() as py_file:
        py_file["language"] = "python"
        py_file["chunks"] = py_file["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language="python",
            chunk_size=1000,
            chunk_overlap=200,
        )

        with py_file["chunks"].row() as chunk:
            chunk["embedding"] = chunk["text"].transform(
                cocoindex.functions.SentenceTransformerEmbed(model="sentence-transformers/all-MiniLM-L6-v2")
            )
            code_embeddings.collect(
                filename=py_file["filename"],
                language=py_file["language"],
                chunk_text=chunk["text"],
                embedding=chunk["embedding"],
                location=chunk["location"],
            )

    # Process Rust files
    with data_scope["rust_files"].row() as rs_file:
        rs_file["language"] = "rust"
        rs_file["chunks"] = rs_file["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language="rust",
            chunk_size=1000,
            chunk_overlap=200,
        )

        with rs_file["chunks"].row() as chunk:
            chunk["embedding"] = chunk["text"].transform(
                cocoindex.functions.SentenceTransformerEmbed(model="sentence-transformers/all-MiniLM-L6-v2")
            )
            code_embeddings.collect(
                filename=rs_file["filename"],
                language=rs_file["language"],
                chunk_text=chunk["text"],
                embedding=chunk["embedding"],
                location=chunk["location"],
            )

    # Process configuration files differently
    with data_scope["config_files"].row() as config_file:
        config_metadata.collect(
            filename=config_file["filename"],
            file_type="config",
            content_length=config_file["content"].transform(cocoindex.functions.CalculateLength()),
            content_preview=config_file["content"].transform(cocoindex.functions.TruncateText(max_length=500)),
        )

    # Export to different targets
    code_embeddings.export(
        "code_embeddings",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename", "location"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
            )
        ],
    )

    config_metadata.export(
        "config_metadata",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename"],
    )


# Example of multiple separate flows


@cocoindex.flow_def(name="PythonCodeAnalysis")
def python_analysis_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope) -> None:
    """Separate flow focused on Python code analysis"""
    data_scope["python_files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="cocoindex/python",
            included_patterns=["*.py"],
        )
    )

    analysis_results = data_scope.add_collector("python_analysis")

    with data_scope["python_files"].row() as py_file:
        py_file["analysis"] = py_file["content"].transform(
            cocoindex.functions.ExtractByLlm(
                llm_spec=cocoindex.LlmSpec(
                    api_type=cocoindex.LlmApiType.OPENAI,
                    model="gpt-4o-mini",
                ),
                output_type=dict,
                instruction="Analyze this Python code for complexity, patterns, and purpose. Return a JSON object with 'complexity', 'patterns', and 'purpose' fields.",
            )
        )

        analysis_results.collect(
            filename=py_file["filename"],
            complexity=py_file["analysis"]["complexity"],
            patterns=py_file["analysis"]["patterns"],
            purpose=py_file["analysis"]["purpose"],
        )

    analysis_results.export(
        "python_analysis",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename"],
    )


@cocoindex.flow_def(name="RustCodeAnalysis")
def rust_analysis_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope) -> None:
    """Separate flow focused on Rust code analysis"""
    data_scope["rust_files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="cocoindex/src",
            included_patterns=["*.rs"],
        )
    )

    rust_analysis = data_scope.add_collector("rust_analysis")

    with data_scope["rust_files"].row() as rs_file:
        rs_file["analysis"] = rs_file["content"].transform(
            cocoindex.functions.ExtractByLlm(
                llm_spec=cocoindex.LlmSpec(
                    api_type=cocoindex.LlmApiType.OPENAI,
                    model="gpt-4o-mini",
                ),
                output_type=dict,
                instruction="Analyze this Rust code for safety patterns, performance considerations, and module structure. Return a JSON object with 'safety', 'performance', and 'structure' fields.",
            )
        )

        rust_analysis.collect(
            filename=rs_file["filename"],
            safety=rs_file["analysis"]["safety"],
            performance=rs_file["analysis"]["performance"],
            structure=rs_file["analysis"]["structure"],
        )

    rust_analysis.export(
        "rust_analysis",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename"],
    )


if __name__ == "__main__":
    # This demonstrates how you can work with multiple flows
    import cocoindex

    # Initialize CocoIndex
    cocoindex.init()

    # Get all flows
    all_flows = cocoindex.flow.flows()
    print(f"Available flows: {list(all_flows.keys())}")

    # You can update individual flows
    # multiple_sources_flow.update()
    # python_analysis_flow.update()
    # rust_analysis_flow.update()

    # Or update all flows at once
    # cocoindex.flow.update_all_flows(
    #     cocoindex.flow.FlowLiveUpdaterOptions(live_mode=False)
    # )

    print("Flow definitions created successfully!")
