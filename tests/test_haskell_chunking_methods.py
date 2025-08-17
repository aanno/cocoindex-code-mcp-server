#!/usr/bin/env python3

"""
Test Haskell chunking method names to ensure proper Rust integration.
This test verifies that Haskell files get the correct chunking method names
from the Rust code (like rust_haskell_ast_recursive) instead of generic names.
"""

import logging
from typing import Any, Dict

import pytest

from .common import CocoIndexTestInfrastructure

# Add src to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


LOGGER = logging.getLogger(__name__)


class TestHaskellChunkingMethods:
    """Test that Haskell files get proper chunking method names from Rust code."""

    @pytest.mark.asyncio
    async def test_haskell_chunking_methods_from_rust(self):
        """Test that Haskell files get proper Rust chunking method names."""
        infrastructure_config = {
            "paths": ["/workspaces/rust"],
            "enable_polling": False,
            "poll_interval": 30,
            "default_embedding": False,
            "default_chunking": False,
            "default_language_handler": False,
            "chunk_factor_percent": 100
        }

        async with CocoIndexTestInfrastructure(**infrastructure_config) as test_infrastructure:
            # Search for Haskell files to get their chunking methods
            results = test_infrastructure.hybrid_search_engine.search(
                vector_query="haskell function",
                keyword_query="language:haskell",
                top_k=20
            )

            # Should find some Haskell files
            assert len(results) > 0, "Should find Haskell files in the test corpus"

            # Track all chunking methods we see
            chunking_methods = set()
            haskell_results = []

            for result in results:
                if result.get('language') == 'Haskell':
                    chunking_method = result.get('chunking_method', 'MISSING')
                    chunking_methods.add(chunking_method)
                    haskell_results.append({
                        'filename': result.get('filename', 'UNKNOWN'),
                        'chunking_method': chunking_method,
                        'analysis_method': result.get('analysis_method', 'UNKNOWN'),
                        'metadata_json_chunking': self._extract_metadata_chunking_method(result)
                    })

            print(f"\n🔍 Found {len(haskell_results)} Haskell results")
            print(f"📊 Unique chunking methods: {sorted(chunking_methods)}")

            # Print detailed results for debugging
            for i, result in enumerate(haskell_results[:5]):  # Show first 5
                print(f"\n📄 Haskell Result {i + 1}:")
                print(f"   File: {result['filename']}")
                print(f"   Chunking Method: {result['chunking_method']}")
                print(f"   Analysis Method: {result['analysis_method']}")
                print(f"   Metadata JSON Chunking: {result['metadata_json_chunking']}")

            # Assertions for expected behavior
            assert len(haskell_results) > 0, "Should have found at least one Haskell file"

            # Check that we're NOT getting the old problematic values
            bad_methods = {'ast', 'unknown_chunking', 'ast_with_errors', 'regex_fallback'}
            found_bad_methods = chunking_methods.intersection(bad_methods)

            if found_bad_methods:
                print(f"\n❌ FOUND BAD CHUNKING METHODS: {found_bad_methods}")
                print("   Expected Rust-based methods like:")
                print("   - rust_haskell_ast_recursive")
                print("   - rust_haskell_error_recovery")
                print("   - rust_haskell_ast_with_errors")
                print("   - rust_haskell_regex_fallback_*")

                # Show which files have bad methods
                for result in haskell_results:
                    if result['chunking_method'] in bad_methods:
                        print(f"   Bad method '{result['chunking_method']}' in: {result['filename']}")

            # Check for expected Rust-based method names
            rust_method_prefixes = ['rust_haskell_', 'haskell_specialized_chunker']
            good_methods = set()
            for method in chunking_methods:
                if any(method.startswith(prefix) for prefix in rust_method_prefixes):
                    good_methods.add(method)

            print(f"\n✅ Found good Rust-based methods: {sorted(good_methods)}")

            # Main assertion: should have proper Rust-based chunking methods
            assert len(good_methods) > 0, (
                f"Expected to find Rust-based chunking methods starting with {rust_method_prefixes}, "
                f"but found: {sorted(chunking_methods)}"
            )

            # Additional assertion: should not have bad generic methods
            assert len(found_bad_methods) == 0, (
                f"Found problematic generic chunking methods: {found_bad_methods}. "
                f"These should be replaced with Rust-specific methods."
            )

    @pytest.mark.asyncio
    async def test_haskell_specialized_chunker_used(self):
        """Test that Haskell files use the specialized chunker, not generic AST chunking."""
        infrastructure_config = {
            "paths": ["/workspaces/rust"],
            "enable_polling": False,
            "default_chunking": False
        }

        async with CocoIndexTestInfrastructure(**infrastructure_config) as test_infrastructure:
            results = test_infrastructure.hybrid_search_engine.search(
                vector_query="haskell data type",
                keyword_query="language:haskell",
                top_k=10
            )

            haskell_files = [r for r in results if r.get('language') == 'Haskell']
            assert len(haskell_files) > 0, "Should find Haskell files"

            # Check analysis methods - should show Haskell-specific analysis
            analysis_methods = {r.get('analysis_method', 'UNKNOWN') for r in haskell_files}
            print(f"\n🔬 Analysis methods for Haskell files: {sorted(analysis_methods)}")

            # Should have haskell-specific analysis method
            haskell_analysis = [m for m in analysis_methods if 'haskell' in m.lower()]
            assert len(haskell_analysis) > 0, (
                f"Expected Haskell-specific analysis methods, but found: {sorted(analysis_methods)}"
            )

    def _extract_metadata_chunking_method(self, result: Dict[str, Any]) -> str:
        """Extract chunking method from metadata_json field."""
        try:
            import json
            metadata_json = result.get('metadata_json')
            if isinstance(metadata_json, dict):
                return metadata_json.get('chunking_method', 'NOT_FOUND')
            elif isinstance(metadata_json, str):
                metadata = json.loads(metadata_json)
                return metadata.get('chunking_method', 'NOT_FOUND')
            else:
                return 'NO_METADATA_JSON'
        except Exception as e:
            return f'ERROR: {e}'


if __name__ == "__main__":
    # Run the test directly
    test = TestHaskellChunkingMethods()
    infrastructure = CocoIndexTestInfrastructure()

    print("🧪 Running Haskell chunking method test...")
    try:
        test.test_haskell_chunking_methods_from_rust(infrastructure)
        print("✅ Test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
