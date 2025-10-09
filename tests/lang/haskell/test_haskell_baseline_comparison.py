#!/usr/bin/env python3

"""
Baseline comparison test for Haskell metadata extraction.
Compares our specialized HaskellASTVisitor implementation against CocoIndex defaults.
Similar to tests/lang/python/test_cocoindex_baseline_comparison.py
"""

from pathlib import Path
from typing import Dict, List, Union

import pytest

import_error = None
try:
    from cocoindex_code_mcp_server.ast_visitor import (
        analyze_code as generic_analyze_code,
    )
    from cocoindex_code_mcp_server.language_handlers.haskell_handler import (
        analyze_haskell_code,
    )
    HASKELL_VISITOR_AVAILABLE = True
except ImportError as e:
    HASKELL_VISITOR_AVAILABLE = False
    import_error = str(e)

try:
    COCOINDEX_AVAILABLE = True
except ImportError:
    COCOINDEX_AVAILABLE = False


@pytest.mark.skipif(not HASKELL_VISITOR_AVAILABLE, reason=f"Haskell visitor not available: {import_error}")
class TestHaskellBaselineComparison:
    """Compare our Haskell implementation vs CocoIndex baseline."""

    def setup_method(self):
        """Set up test fixtures."""
        # Load the test Haskell file
        self.fixture_path = Path(__file__).parent.parent.parent / 'fixtures' / 'lang_examples' / 'HaskellExample1.hs'

        if not self.fixture_path.exists():
            pytest.skip(f"Test fixture not found: {self.fixture_path}")

        with open(self.fixture_path, 'r') as f:
            self.haskell_code = f.read()

    def test_our_haskell_visitor_analysis(self) -> Dict[str, Union[str, int, List[str], bool, Dict[str, int]]]:
        """Test our specialized Haskell visitor implementation."""
        result = analyze_haskell_code(self.haskell_code, str(self.fixture_path))

        # Basic validation
        assert result['language'] == 'Haskell'
        assert result['analysis_method'] == 'haskell_chunk_visitor'
        assert result['line_count'] > 0
        assert result['char_count'] > 0

        # Extract key metrics for comparison
        metrics = {
            'analysis_method': result['analysis_method'],
            'functions_found': len(result.get('functions', [])),
            'function_names': sorted(result.get('functions', [])),
            'data_types_found': len(result.get('data_types', [])),
            'data_type_names': sorted(result.get('data_types', [])),
            'imports_found': len(result.get('imports', [])),
            'has_type_signatures': result.get('has_type_signatures', False),
            'complexity_score': result.get('complexity_score', 0),
            'node_stats': result.get('node_stats', {}),
            'function_details_count': len(result.get('function_details', [])),
            'data_type_details_count': len(result.get('data_type_details', [])),
        }

        print("\n=== Our Haskell Visitor Results ===")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        # Store for comparison
        self.our_metrics = metrics

        return metrics

    def test_generic_visitor_analysis(self) -> Dict[str, Union[str, int, List[str], bool, Dict[str, int]]]:
        """Test generic AST visitor with Haskell (should use our specialized visitor)."""
        result = generic_analyze_code(self.haskell_code, 'haskell', str(self.fixture_path))

        # Basic validation
        assert result['language'] == 'Haskell'
        assert 'tree_sitter' in result.get('analysis_method', '')

        # Extract key metrics for comparison
        metrics = {
            'analysis_method': result['analysis_method'],
            'functions_found': len(result.get('functions', [])),
            'function_names': sorted(result.get('functions', [])),
            'data_types_found': len(result.get('data_types', [])),
            'data_type_names': sorted(result.get('data_types', [])),
            'imports_found': len(result.get('imports', [])),
            'has_type_signatures': result.get('has_type_signatures', False),
            'complexity_score': result.get('complexity_score', 0),
            'node_stats': result.get('node_stats', {}),
            'function_details_count': len(result.get('function_details', [])),
            'data_type_details_count': len(result.get('data_type_details', [])),
        }

        print("\n=== Generic Visitor Results ===")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        # Store for comparison
        self.generic_metrics = metrics

        return metrics

    @pytest.mark.skipif(not COCOINDEX_AVAILABLE, reason="CocoIndex not available")
    def test_cocoindex_baseline_analysis(self) -> Dict[str, Union[str, int, List[str], bool, Dict[str, int]]]:
        """Test CocoIndex's built-in analysis for comparison."""
        try:
            # Use CocoIndex's basic text analysis
            # This is a simplified baseline since CocoIndex doesn't have Haskell-specific parsing
            lines = self.haskell_code.split('\n')

            # Basic pattern matching (CocoIndex-style fallback)
            functions = []
            data_types = []
            imports = []

            for line in lines:
                line = line.strip()
                if line.startswith('data '):
                    # Extract data type name
                    parts = line.split()
                    if len(parts) > 1:
                        data_types.append(parts[1])
                elif '::' in line and not line.startswith('--'):
                    # Extract function name from type signature
                    func_name = line.split('::')[0].strip()
                    if func_name and not func_name.startswith('-'):
                        functions.append(func_name)
                elif line.startswith('import '):
                    imports.append(line)

            # Remove duplicates and clean up
            functions = sorted(list(set(f for f in functions if f.isidentifier())))
            data_types = sorted(list(set(dt for dt in data_types if dt)))

            metrics: Dict[str, Union[str, int, List[str], bool, Dict[str, int]]] = {
                'analysis_method': 'cocoindex_text_fallback',
                'functions_found': len(functions),
                'function_names': functions,
                'data_types_found': len(data_types),
                'data_type_names': data_types,
                'imports_found': len(imports),
                'has_type_signatures': '::' in self.haskell_code,
                'complexity_score': 0,  # Not computed in baseline
                'node_stats': {},
                'function_details_count': 0,
                'data_type_details_count': 0,
            }

            print("\n=== CocoIndex Baseline Results ===")
            for key, value in metrics.items():
                print(f"{key}: {value}")

            self.baseline_metrics = metrics
            return metrics

        except Exception as e:
            pytest.skip(f"CocoIndex baseline analysis failed: {e}")

    def test_comparison_analysis(self):
        """Compare all analysis methods and report differences."""
        # Run all analyses first
        our_metrics = self.test_our_haskell_visitor_analysis()
        generic_metrics = self.test_generic_visitor_analysis()

        try:
            baseline_metrics = self.test_cocoindex_baseline_analysis()
        except BaseException:
            baseline_metrics = None

        print("\n" + "=" * 60)
        print("HASKELL ANALYSIS COMPARISON")
        print("=" * 60)

        # Compare our implementation vs generic visitor
        print("\n--- Our Specialized Visitor vs Generic Visitor ---")
        for key in ['functions_found', 'data_types_found', 'imports_found']:
            our_val = our_metrics.get(key, 0)
            generic_val = generic_metrics.get(key, 0)
            status = "✅ SAME" if our_val == generic_val else f"⚠️  DIFF ({our_val} vs {generic_val})"
            print(f"{key}: {status}")

        # Function name comparison
        our_func_names = our_metrics.get('function_names', [])
        generic_func_names = generic_metrics.get('function_names', [])
        our_funcs = set(our_func_names if isinstance(our_func_names, list) else [])
        generic_funcs = set(generic_func_names if isinstance(generic_func_names, list) else [])
        if our_funcs == generic_funcs:
            print(f"function_names: ✅ SAME ({len(our_funcs)} functions)")
        else:
            print("function_names: ⚠️  DIFF")
            print(f"  Our: {sorted(our_funcs)}")
            print(f"  Generic: {sorted(generic_funcs)}")
            print(f"  Only in ours: {sorted(our_funcs - generic_funcs)}")
            print(f"  Only in generic: {sorted(generic_funcs - our_funcs)}")

        # Data type comparison
        our_type_names = our_metrics.get('data_type_names', [])
        generic_type_names = generic_metrics.get('data_type_names', [])
        our_types = set(our_type_names if isinstance(our_type_names, list) else [])
        generic_types = set(generic_type_names if isinstance(generic_type_names, list) else [])
        if our_types == generic_types:
            print(f"data_type_names: ✅ SAME ({len(our_types)} types)")
        else:
            print("data_type_names: ⚠️  DIFF")
            print(f"  Our: {sorted(our_types)}")
            print(f"  Generic: {sorted(generic_types)}")

        # Compare with baseline if available
        if baseline_metrics:
            print("\n--- Our Implementation vs CocoIndex Baseline ---")
            our_func_names = our_metrics.get('function_names', [])
            baseline_func_names = baseline_metrics.get('function_names', [])
            our_funcs = set(our_func_names if isinstance(our_func_names, list) else [])
            baseline_funcs = set(baseline_func_names if isinstance(baseline_func_names, list) else [])

            print(f"Functions - Our: {len(our_funcs)}, Baseline: {len(baseline_funcs)}")
            if our_funcs != baseline_funcs:
                print(f"  Our extra: {sorted(our_funcs - baseline_funcs)}")
                print(f"  Baseline extra: {sorted(baseline_funcs - our_funcs)}")

            our_type_names = our_metrics.get('data_type_names', [])
            baseline_type_names = baseline_metrics.get('data_type_names', [])
            our_types = set(our_type_names if isinstance(our_type_names, list) else [])
            baseline_types = set(baseline_type_names if isinstance(baseline_type_names, list) else [])
            print(f"Data types - Our: {len(our_types)}, Baseline: {len(baseline_types)}")
            if our_types != baseline_types:
                print(f"  Our: {sorted(our_types)}")
                print(f"  Baseline: {sorted(baseline_types)}")

        # Quality assessment
        print("\n--- Quality Assessment ---")
        expected_functions = {'fibonacci', 'sumList', 'treeMap', 'compose', 'addTen', 'multiplyByTwo', 'main'}
        expected_types = {'Person', 'Tree'}

        our_func_names = our_metrics.get('function_names', [])
        our_type_names = our_metrics.get('data_type_names', [])
        our_funcs = set(our_func_names if isinstance(our_func_names, list) else [])
        our_types = set(our_type_names if isinstance(our_type_names, list) else [])

        func_recall = len(our_funcs & expected_functions) / max(len(expected_functions), 1)
        type_recall = len(our_types & expected_types) / max(len(expected_types), 1)

        print(f"Function recall: {func_recall:.2%} ({len(our_funcs & expected_functions)}/{len(expected_functions)})")
        print(f"Data type recall: {type_recall:.2%} ({len(our_types & expected_types)}/{len(expected_types)})")
        print(f"Functions found: {sorted(our_funcs)}")
        print(f"Types found: {sorted(our_types)}")
        print(f"Missing functions: {sorted(expected_functions - our_funcs)}")
        print(f"Missing types: {sorted(expected_types - our_types)}")

        # Assert basic quality thresholds
        assert func_recall >= 0.5, f"Function recall too low: {func_recall:.2%}"
        assert type_recall >= 0.5, f"Data type recall too low: {type_recall:.2%}"

        # Type-safe assertions
        has_type_signatures = our_metrics.get('has_type_signatures', False)
        complexity_score = our_metrics.get('complexity_score', 0)

        assert isinstance(has_type_signatures, bool) and has_type_signatures, "Should detect type signatures"
        assert isinstance(complexity_score, int) and complexity_score > 0, "Should calculate complexity"


if __name__ == "__main__":
    # Run with verbose output for debugging
    pytest.main([__file__, "-v", "-s", "--tb=short"])
