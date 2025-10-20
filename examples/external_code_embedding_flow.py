#!/usr/bin/env python3

"""
Example: Language-Aware Code Embedding with External Wrapper

This demonstrates how to use intelligent code embedding functionality
without modifying CocoIndex source code, using external wrapper functions.
"""

import os
import sys

from smart_code_embedding import (
    create_python_embedding,
    create_rust_embedding,
    create_smart_code_embedding,
    get_supported_extensions,
    get_supported_languages,
)

import cocoindex

# Add our source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import our external smart embedding functionality

# Import CocoIndex (used as external dependency)


def demonstrate_external_embedding():
    """Demonstrate the external embedding approach."""
    print("=== External Language-Aware Code Embedding Demo ===")
    print()

    # Show supported languages and models
    print("Automatic Model Selection:")
    for lang, model in get_supported_languages().items():
        print(f"  {lang:12} → {model}")
    print()

    # Show supported file extensions
    print("File Extension Detection:")
    for ext, lang in get_supported_extensions().items():
        print(f"  {ext:6} → {lang}")
    print()


@cocoindex.flow_def
def external_code_embedding_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """
    Complete code embedding flow using external wrapper functions.

    This flow demonstrates how to use language-aware embeddings without
    modifying CocoIndex source code.
    """
    with flow_builder.read_files(data_scope.input_directory) as file:
        # Filter to code files only
        file = file.filter(
            lambda f: f["extension"] in [".py", ".rs", ".js", ".ts", ".java", ".kt", ".go", ".cpp", ".c", ".rb", ".php"]
        )

        # Chunk code using CocoIndex's built-in Tree-sitter functionality
        file["chunks"] = file["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language=file["extension"],
            chunk_size=1000,
            chunk_overlap=200,
        )

        with file["chunks"].row() as chunk:
            # Method 1: Smart embedding with automatic detection
            chunk["embedding"] = chunk["text"].transform(create_smart_code_embedding(file_extension=file["extension"]))

            # Add metadata about which model was selected
            chunk["embedding_model"] = file["extension"]

        # Store in vector database
        file["chunks"].save(
            cocoindex.targets.QdrantTarget(
                collection_name="external_code_embeddings",
                metadata_fields=["file_path", "extension", "embedding_model"],
            )
        )


@cocoindex.flow_def
def multi_model_embedding_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """
    Advanced flow using multiple embedding models for the same content.

    Creates both code-specific and general-purpose embeddings for enhanced retrieval.
    """
    with flow_builder.read_files(data_scope.input_directory) as file:
        file = file.filter(lambda f: f["extension"] in [".py", ".rs", ".js", ".ts"])

        file["chunks"] = file["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language=file["extension"],
            chunk_size=1000,
            chunk_overlap=200,
        )

        with file["chunks"].row() as chunk:
            # Code-specific embedding (GraphCodeBERT/UniXcode)
            chunk["embedding_code"] = chunk["text"].transform(
                create_smart_code_embedding(file_extension=file["extension"])
            )

            # General-purpose embedding for broader semantic search
            chunk["embedding_general"] = chunk["text"].transform(
                cocoindex.functions.SentenceTransformerEmbed(model="sentence-transformers/all-MiniLM-L6-v2")
            )

        # Store with both embeddings
        file["chunks"].save(
            cocoindex.targets.QdrantTarget(
                collection_name="multi_model_embeddings", metadata_fields=["file_path", "extension"]
            )
        )


def demonstrate_api_patterns():
    """Demonstrate different ways to use the external API."""
    print("=== API Usage Patterns ===")
    print()

    # Pattern 1: Automatic detection from file extension
    print("1. Automatic Detection from File Extension:")
    embedding_func = create_smart_code_embedding(file_extension=".py")
    print("   File: .py → Model: GraphCodeBERT")
    print("   Usage: create_smart_code_embedding(file_extension='.py')")
    print()

    # Pattern 2: Manual language specification
    print("2. Manual Language Specification:")
    embedding_func = create_smart_code_embedding(language="rust")
    print("   Language: rust → Model: UniXcode")
    print("   Usage: create_smart_code_embedding(language='rust')")
    print()

    # Pattern 3: Force specific model
    print("3. Force Specific Model:")
    embedding_func = create_smart_code_embedding(language="python", force_model="microsoft/graphcodebert-base")
    print("   Usage: create_smart_code_embedding(language='python', force_model='...')")
    print()

    # Pattern 4: Language-specific convenience functions
    print("4. Language-Specific Convenience Functions:")
    create_python_embedding()
    create_rust_embedding()
    print("   create_python_embedding() → GraphCodeBERT")
    print("   create_rust_embedding() → UniXcode")
    print()

    # Pattern 5: Custom model arguments
    print("5. Custom Model Arguments:")
    embedding_func = create_smart_code_embedding(language="python", model_args={"device": "cpu", "batch_size": 16})
    print("   Custom args: device, batch_size, etc.")
    print()


def demonstrate_conditional_embedding():
    """Demonstrate conditional embedding based on language."""
    print("=== Conditional Embedding Example ===")
    print()

    def get_embedding_function(file_extension: str):
        """Get embedding function based on file extension with custom logic."""
        if file_extension in [".py", ".js", ".java"]:
            # Use GraphCodeBERT for well-supported languages
            return create_smart_code_embedding(
                file_extension=file_extension, force_model="microsoft/graphcodebert-base"
            )
        elif file_extension in [".rs", ".ts"]:
            # Use UniXcode for these languages
            return create_smart_code_embedding(file_extension=file_extension, force_model="microsoft/unixcoder-base")
        elif file_extension in [".hs", ".ml"]:
            # Use better general model for functional languages
            return cocoindex.functions.SentenceTransformerEmbed(model="sentence-transformers/all-mpnet-base-v2")
        else:
            # Default fallback
            return create_smart_code_embedding(file_extension=file_extension)

    # Test different extensions
    test_extensions = [".py", ".rs", ".js", ".ts", ".hs", ".cpp"]

    print("Custom conditional logic:")
    for ext in test_extensions:
        get_embedding_function(ext)
        print(f"  {ext:6} → Custom logic applied")
    print()


def show_integration_example():
    """Show how this integrates into existing CocoIndex workflows."""
    print("=== Integration with Existing CocoIndex Flows ===")
    print()

    integration_code = """
# Before (using standard SentenceTransformerEmbed)
chunk["embedding"] = chunk["text"].transform(
    cocoindex.functions.SentenceTransformerEmbed(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
)

# After (using external smart embedding)
from cocoindex_code_mcp_server.smart_code_embedding import create_smart_code_embedding

chunk["embedding"] = chunk["text"].transform(
    create_smart_code_embedding(file_extension=file["extension"])
)

# The result: automatic language-aware model selection!
# - Python files → GraphCodeBERT
# - Rust files → UniXcode
# - Other files → appropriate fallback
"""

    print(integration_code)


def main():
    """Main demonstration function."""
    print("External Language-Aware Code Embedding for CocoIndex")
    print("=" * 60)
    print()

    demonstrate_external_embedding()
    demonstrate_api_patterns()
    demonstrate_conditional_embedding()
    show_integration_example()

    print("=== Benefits of External Approach ===")
    print("✓ No modification of CocoIndex source code")
    print("✓ Uses CocoIndex as pure external dependency")
    print("✓ Full compatibility with existing CocoIndex workflows")
    print("✓ Easy to maintain and update independently")
    print("✓ Can be packaged as separate library")
    print()

    print("=== Getting Started ===")
    print("1. Install dependencies:")
    print("   pip install cocoindex[embeddings]")
    print()
    print("2. Import and use:")
    print("   from cocoindex_code_mcp_server.smart_code_embedding import create_smart_code_embedding")
    print("   embedding_func = create_smart_code_embedding(file_extension='.py')")
    print()
    print("3. Use in CocoIndex flows:")
    print("   chunk['embedding'] = chunk['text'].transform(embedding_func)")
    print()


if __name__ == "__main__":
    main()
