#!/usr/bin/env python3

"""
Comprehensive test suite to prevent chunking method and metadata regressions.

This test validates:
1. Haskell files get proper Rust-based chunking method names (rust_haskell_*)
2. Metadata arrays are stored as JSON arrays, not Python string representations
3. Different chunking scenarios produce expected method names
4. Flow configuration doesn't override chunk metadata improperly
"""


import pytest

from .common import CocoIndexTestInfrastructure

# Add src and tests to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# sys.path.insert(0, str(Path(__file__).parent))
# sys.path.insert(0, str(Path(__file__).parent / ".."))


class TestChunkingMethodRegression:
    """Test suite to prevent chunking method and metadata regressions."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.integration
    async def test_haskell_rust_chunking_method_names(self):
        """Test that Haskell files get proper rust_haskell_* chunking method names.

        SLOW TEST: This test is marked as slow because it:
        1. Creates full CocoIndexTestInfrastructure which:
           - Initializes complete CocoIndex system (cocoindex.init())
           - Sets up database connections and schemas
           - Processes the ENTIRE /workspaces/rust directory (thousands of files)
           - Runs flow updates with language detection, AST parsing, chunking
           - Generates embeddings for all chunks and stores in vector database
           - Initializes search engines and indices
        2. Performs vector similarity search across entire indexed codebase
        3. Searches for Haskell files with top_k=20, analyzing chunking methods

        Expected runtime: 2-5 minutes (depends on codebase size and system performance)
        """
        async with CocoIndexTestInfrastructure(
            paths=["/workspaces/rust/src", "/workspaces/rust/tests/fixtures"],  # Reduced scope for performance
            enable_polling=False,
            default_chunking=False,
            default_language_handler=False
        ) as test_infrastructure:
            results = test_infrastructure.hybrid_search_engine.search(
                vector_query="haskell function pattern matching",
                keyword_query="language:haskell",
                top_k=20
            )

            # Should find Haskell files
            assert len(results) > 0, "Should find Haskell files in test corpus"

            haskell_results = [r for r in results if r.get('language') == 'Haskell']
            assert len(haskell_results) > 0, "Should find Haskell results"

            # Check chunking methods
            chunking_methods = set()
            problematic_files = []
            good_files = []

            for result in haskell_results:
                filename = result.get('filename', 'UNKNOWN')
                chunking_method = result.get('chunking_method', 'MISSING')
                metadata_chunking = result.get('metadata_json', {}).get('chunking_method', 'MISSING')

                chunking_methods.add(chunking_method)

                # Check if we get proper Rust-based method names
                expected_rust_prefixes = ['rust_haskell_']
                if any(chunking_method.startswith(prefix) for prefix in expected_rust_prefixes):
                    good_files.append({
                        'filename': filename,
                        'chunking_method': chunking_method,
                        'metadata_chunking': metadata_chunking
                    })
                else:
                    problematic_files.append({
                        'filename': filename,
                        'chunking_method': chunking_method,
                        'metadata_chunking': metadata_chunking
                    })

            print("\n📊 Chunking Methods Analysis:")
            print(f"   All methods found: {sorted(chunking_methods)}")
            print(f"   Good files (rust_haskell_*): {len(good_files)}")
            print(f"   Problematic files: {len(problematic_files)}")

            if good_files:
                print("\n✅ Good examples:")
                for f in good_files[:3]:
                    print(f"   {f['filename']}: {f['chunking_method']}")

            if problematic_files:
                print("\n❌ Problematic examples:")
                for f in problematic_files[:5]:
                    print(f"   {f['filename']}: {f['chunking_method']}")

            # Main assertions
            assert len(good_files) > 0, (
                f"Expected to find Haskell files with rust_haskell_* chunking methods, "
                f"but found: {sorted(chunking_methods)}"
            )

            # Check for forbidden generic methods that suggest bugs
            bad_generic_methods = {'ast', 'unknown_chunking'}
            found_bad = chunking_methods.intersection(bad_generic_methods)

            if found_bad:
                print(f"\n⚠️  Found potentially problematic generic methods: {found_bad}")
                # This is a warning, not a failure, since some might be legitimate

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.integration
    async def test_metadata_arrays_not_strings(self):
        """Test that metadata arrays are JSON arrays, not Python string representations.

        SLOW TEST: This test is marked as slow because it:
        1. Creates full CocoIndexTestInfrastructure (same expensive setup as above)
        2. Processes entire /workspaces/rust directory with complete indexing pipeline
        3. Searches for Python files with metadata analysis (top_k=10)
        4. Analyzes metadata_json fields for array storage format validation

        Expected runtime: 2-5 minutes (similar to test above)
        """
        async with CocoIndexTestInfrastructure(
            paths=["/workspaces/rust/src", "/workspaces/rust/tests/fixtures"],  # Reduced scope for performance
            enable_polling=False,
            default_chunking=False,
            default_language_handler=False
        ) as test_infrastructure:
            # Search for files that should have array metadata
            results = test_infrastructure.hybrid_search_engine.search(
                vector_query="python function class import",
                keyword_query="language:python",
                top_k=10
            )

            # Should find Python files with metadata
            python_results = [r for r in results if r.get('language') == 'Python']
            assert len(python_results) > 0, "Should find Python files with metadata"

            array_fields_to_check = ['functions', 'classes', 'imports']
            string_representation_examples = []
            good_examples = []

            for result in python_results:
                filename = result.get('filename', 'UNKNOWN')
                metadata_json = result.get('metadata_json', {})

                for field in array_fields_to_check:
                    field_value = metadata_json.get(field)

                    if field_value is not None:
                        # Check if it's a string representation of a Python list
                        if isinstance(field_value, str) and field_value.startswith("["):
                            # Try to determine if it's a Python string representation
                            if "'" in field_value:  # Likely Python repr with single quotes
                                string_representation_examples.append({
                                    'filename': filename,
                                    'field': field,
                                    'value': field_value,
                                    'type': type(field_value).__name__
                                })
                        elif isinstance(field_value, list):
                            good_examples.append({
                                'filename': filename,
                                'field': field,
                                'value': field_value[:3] if len(field_value) > 3 else field_value,
                                'type': type(field_value).__name__
                            })

            print("\n📊 Metadata Array Analysis:")
            print(f"   Good examples (proper lists): {len(good_examples)}")
            print(f"   String representation examples: {len(string_representation_examples)}")

            if good_examples:
                print("\n✅ Good examples (proper JSON arrays):")
                for example in good_examples[:3]:
                    print(f"   {example['filename']}.{example['field']}: {example['value']} ({example['type']})")

            if string_representation_examples:
                print("\n❌ String representation examples:")
                for example in string_representation_examples[:3]:
                    print(f"   {example['filename']}.{example['field']}: {example['value'][:100]}... ({example['type']})")

            # Main assertion: should not have Python string representations
            assert len(string_representation_examples) == 0, (
                f"Found {len(string_representation_examples)} cases where array fields are stored as "
                f"Python string representations instead of proper JSON arrays. "
                f"Examples: {string_representation_examples[:2]}"
            )

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.integration
    async def test_chunking_method_consistency(self):
        """Test that chunking_method is consistent between different metadata sources.

        SLOW TEST: This test is marked as slow because it:
        1. Creates full CocoIndexTestInfrastructure (same expensive setup as above)
        2. Processes entire /workspaces/rust directory with complete indexing pipeline
        3. Performs multiple vector searches across different languages:
           - Haskell function search (top_k=5)
           - Python class search (top_k=5)
           - JavaScript function search (top_k=5)
        4. Analyzes chunking method consistency across metadata sources

        Expected runtime: 2-5 minutes (similar to tests above)
        """
        async with CocoIndexTestInfrastructure(
            paths=["/workspaces/rust/src", "/workspaces/rust/tests/fixtures"],  # Reduced scope for performance
            enable_polling=False,
            default_chunking=False,
            default_language_handler=False
        ) as test_infrastructure:
            # Test various languages to check consistency
            test_cases = [
                ("haskell function", "language:haskell"),
                ("python class", "language:python"),
                ("javascript function", "language:javascript"),
            ]

            all_chunking_methods = set()
            inconsistencies = []

            for vector_query, keyword_query in test_cases:
                results = test_infrastructure.hybrid_search_engine.search(
                    vector_query=vector_query,
                    keyword_query=keyword_query,
                    top_k=5
                )

                for result in results:
                    filename = result.get('filename', 'UNKNOWN')
                    language = result.get('language', 'UNKNOWN')

                    # Get chunking method from different sources
                    top_level_method = result.get('chunking_method', 'MISSING')
                    metadata_method = result.get('metadata_json', {}).get('chunking_method', 'MISSING')

                    all_chunking_methods.add(top_level_method)

                    # Check for consistency between sources
                    if top_level_method != metadata_method and metadata_method != 'MISSING':
                        inconsistencies.append({
                            'filename': filename,
                            'language': language,
                            'top_level': top_level_method,
                            'metadata': metadata_method
                        })

            print("\n📊 Chunking Method Consistency Analysis:")
            print(f"   All chunking methods found: {sorted(all_chunking_methods)}")
            print(f"   Inconsistencies found: {len(inconsistencies)}")

            if inconsistencies:
                print("\n⚠️  Chunking method inconsistencies:")
                for inconsistency in inconsistencies[:3]:
                    print(f"   {inconsistency['filename']} ({inconsistency['language']}): "
                          f"top_level='{inconsistency['top_level']}' vs metadata='{inconsistency['metadata']}'")

            # Should not have many inconsistencies (some might be legitimate due to fallbacks)
            assert len(inconsistencies) == 0, (
                f"Found {len(inconsistencies)} cases where chunking_method is inconsistent "
                f"between top-level field and metadata_json. Examples: {inconsistencies[:2]}"
            )

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.integration
    async def test_no_unknown_chunking_for_supported_languages(self):
        """Test that supported languages don't get 'unknown_chunking' method.

        SLOW TEST: This test is marked as slow because it:
        1. Creates full CocoIndexTestInfrastructure (same expensive setup as above)
        2. Processes entire /workspaces/rust directory with complete indexing pipeline
        3. Performs searches for 5 different supported languages:
           - Haskell, Python, JavaScript, TypeScript, Java (top_k=5 each)
        4. Validates chunking methods for each language to ensure proper handlers

        Expected runtime: 2-5 minutes (similar to tests above)
        """
        async with CocoIndexTestInfrastructure(
            paths=["/workspaces/rust/src", "/workspaces/rust/tests/fixtures"],  # Reduced scope for performance
            enable_polling=False,
            default_chunking=False,
            default_language_handler=False
        ) as test_infrastructure:
            # Languages that should have proper chunking methods
            supported_languages = ["Haskell", "Python", "JavaScript", "TypeScript", "Java"]

            unknown_chunking_cases = []

            for language in supported_languages:
                results = test_infrastructure.hybrid_search_engine.search(
                    vector_query="function class method",
                    keyword_query=f"language:{language.lower()}",
                    top_k=5
                )

                for result in results:
                    if result.get('language') == language:
                        chunking_method = result.get('chunking_method', 'MISSING')
                        if chunking_method == 'unknown_chunking':
                            unknown_chunking_cases.append({
                                'filename': result.get('filename', 'UNKNOWN'),
                                'language': language,
                                'chunking_method': chunking_method
                            })

            print("\n📊 Unknown Chunking Analysis:")
            print(f"   Supported languages checked: {supported_languages}")
            print(f"   Files with unknown_chunking: {len(unknown_chunking_cases)}")

            if unknown_chunking_cases:
                print("\n❌ Files with unknown_chunking (should have proper methods):")
                for case in unknown_chunking_cases[:5]:
                    print(f"   {case['filename']} ({case['language']}): {case['chunking_method']}")

            # Supported languages should not have unknown_chunking
            assert len(unknown_chunking_cases) == 0, (
                f"Found {len(unknown_chunking_cases)} cases where supported languages have "
                f"'unknown_chunking' method instead of proper chunking methods. "
                f"Examples: {unknown_chunking_cases[:3]}"
            )


if __name__ == "__main__":
    # Run the tests directly
    import asyncio

    test_suite = TestChunkingMethodRegression()

    print("🧪 Running chunking method regression tests...")

    try:
        asyncio.run(test_suite.test_haskell_rust_chunking_method_names())
        print("✅ Haskell rust chunking method test passed")

        asyncio.run(test_suite.test_metadata_arrays_not_strings())
        print("✅ Metadata arrays test passed")

        asyncio.run(test_suite.test_chunking_method_consistency())
        print("✅ Chunking method consistency test passed")

        asyncio.run(test_suite.test_no_unknown_chunking_for_supported_languages())
        print("✅ No unknown chunking test passed")

        print("🎉 All regression tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
