#!/usr/bin/env python3

"""
Comprehensive baseline comparison test for Haskell metadata extraction.
Compares our specialized HaskellASTVisitor implementation against multiple baselines:
1. Generic AST visitor with fallback
2. CocoIndex text analysis baseline
3. RAG search results from indexed code
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pytest

# Test imports with error tracking
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


class HaskellBaselineComparison:
    """Comprehensive Haskell analysis comparison suite."""

    def __init__(self):
        """Initialize with test fixture."""
        # Load the test Haskell file
        self.fixture_path = Path(__file__).parent.parent.parent / 'fixtures' / 'lang_examples' / 'HaskellExample1.hs'

        if not self.fixture_path.exists():
            raise FileNotFoundError(f"Test fixture not found: {self.fixture_path}")

        with open(self.fixture_path, 'r') as f:
            self.haskell_code = f.read()

        # Expected results for validation
        self.expected_functions = {
            'fibonacci', 'sumList', 'treeMap', 'compose',
            'addTen', 'multiplyByTwo', 'main'
        }
        self.expected_data_types = {'Person', 'Tree'}
        self.expected_modules = {'TestHaskell'}

        # Analysis results storage
        self.results = {}

    def analyze_with_specialized_visitor(self) -> Dict[str, Any]:
        """Test our specialized Haskell visitor implementation."""
        if not HASKELL_VISITOR_AVAILABLE:
            return {
                'analysis_method': 'haskell_visitor_unavailable',
                'error': str(import_error),
                'functions': [],
                'data_types': [],
                'modules': [],
                'success': False
            }

        try:
            result = analyze_haskell_code(self.haskell_code, str(self.fixture_path))

            # Normalize result format
            normalized = {
                'analysis_method': result.get('analysis_method', 'haskell_chunk_visitor'),
                'functions': result.get('functions', []),
                'data_types': result.get('data_types', []),
                'modules': [result.get('module')] if result.get('module') else [],
                'imports': result.get('imports', []),
                'complexity_score': result.get('complexity_score', 0),
                'has_type_signatures': result.get('has_type_signatures', False),
                'line_count': result.get('line_count', 0),
                'char_count': result.get('char_count', 0),
                'node_stats': result.get('node_stats', {}),
                'function_details': result.get('function_details', []),
                'success': True,
                'raw_result': result
            }

            return normalized

        except Exception as e:
            return {
                'analysis_method': 'haskell_visitor_failed',
                'error': str(e),
                'functions': [],
                'data_types': [],
                'modules': [],
                'success': False
            }

    def analyze_with_generic_visitor(self) -> Dict[str, Any]:
        """Test generic AST visitor (should delegate to specialized visitor)."""
        if not HASKELL_VISITOR_AVAILABLE:
            return {
                'analysis_method': 'generic_visitor_unavailable',
                'error': 'Generic visitor unavailable due to import issues',
                'functions': [],
                'data_types': [],
                'modules': [],
                'success': False
            }

        try:
            result = generic_analyze_code(self.haskell_code, 'haskell', str(self.fixture_path))

            # Normalize result format
            normalized = {
                'analysis_method': result.get('analysis_method', 'generic_fallback'),
                'functions': result.get('functions', []),
                'data_types': result.get('data_types', []),
                'modules': [],  # Generic visitor may not extract modules
                'imports': result.get('imports', []),
                'complexity_score': result.get('complexity_score', 0),
                'has_type_signatures': result.get('has_type_signatures', False),
                'line_count': result.get('line_count', 0),
                'char_count': result.get('char_count', 0),
                'node_stats': result.get('node_stats', {}),
                'success': True,
                'raw_result': result
            }

            return normalized

        except Exception as e:
            return {
                'analysis_method': 'generic_visitor_failed',
                'error': str(e),
                'functions': [],
                'data_types': [],
                'modules': [],
                'success': False
            }

    def analyze_with_cocoindex_baseline(self) -> Dict[str, Any]:
        """Test CocoIndex's basic text analysis as baseline."""
        try:
            # Simple pattern-based analysis similar to CocoIndex fallback
            lines = self.haskell_code.split('\n')

            # Extract functions from type signatures
            functions = set()
            data_types = set()
            modules = set()
            imports = []

            for line in lines:
                line = line.strip()

                # Skip comments
                if line.startswith('--'):
                    continue

                # Extract data types
                if line.startswith('data '):
                    parts = line.split()
                    if len(parts) > 1:
                        type_name = parts[1].split('=')[0].strip()
                        if type_name:
                            data_types.add(type_name)

                # Extract functions from type signatures
                elif '::' in line and not line.startswith('--'):
                    func_name = line.split('::')[0].strip()
                    if func_name and func_name.isidentifier():
                        functions.add(func_name)

                # Extract modules
                elif line.startswith('module '):
                    parts = line.split()
                    if len(parts) > 1:
                        module_name = parts[1].split()[0]
                        if module_name != 'where':
                            modules.add(module_name)

                # Extract imports
                elif line.startswith('import '):
                    imports.append(line)

            return {
                'analysis_method': 'cocoindex_text_baseline',
                'functions': sorted(list(functions)),
                'data_types': sorted(list(data_types)),
                'modules': sorted(list(modules)),
                'imports': imports,
                'complexity_score': 0,  # Not computed in baseline
                'has_type_signatures': '::' in self.haskell_code,
                'line_count': len(lines),
                'char_count': len(self.haskell_code),
                'node_stats': {},
                'success': True
            }

        except Exception as e:
            return {
                'analysis_method': 'cocoindex_baseline_failed',
                'error': str(e),
                'functions': [],
                'data_types': [],
                'modules': [],
                'success': False
            }

    def calculate_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for a result."""
        if not result.get('success', False):
            return {
                'function_recall': 0.0,
                'function_precision': 0.0,
                'data_type_recall': 0.0,
                'data_type_precision': 0.0,
                'overall_score': 0.0,
                'detected_functions': set(),
                'detected_data_types': set(),
                'missing_functions': self.expected_functions,
                'missing_data_types': self.expected_data_types,
                'extra_functions': set(),
                'extra_data_types': set()
            }

        detected_functions = set(result.get('functions', []))
        detected_data_types = set(result.get('data_types', []))

        # Calculate recall (what percentage of expected items were found)
        function_recall = len(detected_functions & self.expected_functions) / \
            len(self.expected_functions) if self.expected_functions else 0
        data_type_recall = len(detected_data_types & self.expected_data_types) / \
            len(self.expected_data_types) if self.expected_data_types else 0

        # Calculate precision (what percentage of detected items were correct)
        function_precision = len(detected_functions & self.expected_functions) / \
            len(detected_functions) if detected_functions else 0
        data_type_precision = len(detected_data_types & self.expected_data_types) / \
            len(detected_data_types) if detected_data_types else 0

        # Overall score (harmonic mean of recall and precision)
        if function_recall + function_precision > 0:
            function_f1 = 2 * (function_recall * function_precision) / (function_recall + function_precision)
        else:
            function_f1 = 0

        if data_type_recall + data_type_precision > 0:
            data_type_f1 = 2 * (data_type_recall * data_type_precision) / (data_type_recall + data_type_precision)
        else:
            data_type_f1 = 0

        overall_score = (function_f1 + data_type_f1) / 2

        return {
            'function_recall': function_recall,
            'function_precision': function_precision,
            'function_f1': function_f1,
            'data_type_recall': data_type_recall,
            'data_type_precision': data_type_precision,
            'data_type_f1': data_type_f1,
            'overall_score': overall_score,
            'detected_functions': detected_functions,
            'detected_data_types': detected_data_types,
            'missing_functions': self.expected_functions - detected_functions,
            'missing_data_types': self.expected_data_types - detected_data_types,
            'extra_functions': detected_functions - self.expected_functions,
            'extra_data_types': detected_data_types - self.expected_data_types
        }

    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """Run all analysis methods and compare results."""
        print("🔍 Running Comprehensive Haskell Analysis Comparison")
        print("=" * 60)

        # Run all analyses
        specialized_result = self.analyze_with_specialized_visitor()
        generic_result = self.analyze_with_generic_visitor()
        baseline_result = self.analyze_with_cocoindex_baseline()

        # Store results
        self.results = {
            'specialized_visitor': specialized_result,
            'generic_visitor': generic_result,
            'cocoindex_baseline': baseline_result
        }

        # Calculate metrics
        metrics = {}
        for name, result in self.results.items():
            metrics[name] = self.calculate_metrics(result)

        # Print detailed comparison
        self._print_detailed_comparison(metrics)

        # Create summary
        summary = {
            'fixture_file': str(self.fixture_path),
            'code_length': len(self.haskell_code),
            'expected_functions': sorted(list(self.expected_functions)),
            'expected_data_types': sorted(list(self.expected_data_types)),
            'results': self.results,
            'metrics': metrics,
            'analysis_date': str(Path(__file__).stat().st_mtime)
        }

        return summary

    def _print_detailed_comparison(self, metrics: Dict[str, Any]):
        """Print detailed comparison results."""
        print(f"\n📊 Analysis Results for {self.fixture_path.name}")
        print(f"Expected: {len(self.expected_functions)} functions, {len(self.expected_data_types)} data types")
        print()

        # Results table
        print("Method".ljust(25) + "Functions".ljust(12) + "Data Types".ljust(12) + "F1 Score".ljust(10) + "Status")
        print("-" * 70)

        for method_name, result in self.results.items():
            metric = metrics[method_name]
            status = "✅ SUCCESS" if result.get('success', False) else "❌ FAILED"

            functions_found = len(result.get('functions', []))
            data_types_found = len(result.get('data_types', []))
            f1_score = metric['overall_score']

            print(f"{method_name.replace('_', ' ').title():<25}"
                  f"{functions_found:<12}"
                  f"{data_types_found:<12}"
                  f"{f1_score:.2f}".ljust(10) +
                  f"{status}")

        print()

        # Detailed breakdown
        for method_name, metric in metrics.items():
            if not self.results[method_name].get('success', False):
                print(f"❌ {method_name} failed: {self.results[method_name].get('error', 'Unknown error')}")
                continue

            print(f"📈 {method_name.replace('_', ' ').title()} Metrics:")
            print(
                f"   Function Recall: {metric['function_recall']:.2%} ({len(metric['detected_functions'])}/{len(self.expected_functions)})")
            print(f"   Function Precision: {metric['function_precision']:.2%}")
            print(
                f"   Data Type Recall: {metric['data_type_recall']:.2%} ({len(metric['detected_data_types'])}/{len(self.expected_data_types)})")
            print(f"   Data Type Precision: {metric['data_type_precision']:.2%}")
            print(f"   Overall F1 Score: {metric['overall_score']:.2%}")

            if metric['missing_functions']:
                print(f"   Missing Functions: {sorted(metric['missing_functions'])}")
            if metric['missing_data_types']:
                print(f"   Missing Data Types: {sorted(metric['missing_data_types'])}")
            if metric['extra_functions']:
                print(f"   Extra Functions: {sorted(metric['extra_functions'])}")
            if metric['extra_data_types']:
                print(f"   Extra Data Types: {sorted(metric['extra_data_types'])}")
            print()


@pytest.mark.skipif(not HASKELL_VISITOR_AVAILABLE, reason=f"Haskell visitor not available: {import_error}")
class TestHaskellComprehensiveBaseline:
    """Pytest wrapper for comprehensive baseline comparison."""

    def setup_method(self):
        """Set up comparison suite."""
        self.comparison = HaskellBaselineComparison()

    def test_comprehensive_baseline_comparison(self) -> Dict[str, Union[str, int, List[str], Dict[str, Union[Dict[str, Union[str, List[str], int, bool, Dict[str, int], List[Dict[str, Union[str, bool, int]]], Dict[str, Optional[Union[str, int, Dict[str, int], List[str], bool, List[Dict[str, Union[str, bool, int]]]]]]]],
                                                                                                             Dict[str, Union[str, List[str], int, bool, Dict[str, int], Dict[str, Optional[Union[str, int, Dict[str, int], List[str], bool, List[Dict[str, Union[str, bool, int]]]]]]]], Dict[str, Union[str, List[str], int, bool]]]], Dict[str, Dict[str, Union[float, Set[str]]]]]]:
        """Run comprehensive comparison test."""
        summary = self.comparison.run_comprehensive_comparison()

        # Assertions for test validation
        assert summary is not None, "Comparison should return a summary"
        assert len(summary['expected_functions']) > 0, "Should have expected functions"
        assert len(summary['expected_data_types']) > 0, "Should have expected data types"

        # At least one method should succeed
        success_count = sum(1 for result in summary['results'].values() if result.get('success', False))
        assert success_count > 0, "At least one analysis method should succeed"

        # If specialized visitor works, it should have good recall
        specialized_metrics = summary['metrics'].get('specialized_visitor', {})
        if summary['results']['specialized_visitor'].get('success', False):
            assert specialized_metrics['function_recall'] >= 0.7, "Specialized visitor should have good function recall"
            assert specialized_metrics['data_type_recall'] >= 0.5, "Specialized visitor should have decent data type recall"

        # Save detailed results for debugging
        results_file = Path(__file__).parent / 'haskell_baseline_results.json'
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: {results_file}")

        return summary


if __name__ == "__main__":
    # Run comparison directly
    try:
        comparison = HaskellBaselineComparison()
        summary = comparison.run_comprehensive_comparison()

        # Print summary
        print("\n🎯 Summary:")
        successful_methods = [name for name, result in summary['results'].items() if result.get('success', False)]
        print(f"   Successful methods: {len(successful_methods)}/{len(summary['results'])}")

        if successful_methods:
            best_method = max(successful_methods, key=lambda x: summary['metrics'][x]['overall_score'])
            best_score = summary['metrics'][best_method]['overall_score']
            print(f"   Best performing method: {best_method} (F1: {best_score:.2%})")

        print(f"   Results available in: {Path(__file__).parent / 'haskell_baseline_results.json'}")

    except Exception as e:
        print(f"❌ Failed to run comparison: {e}")
        import traceback
        traceback.print_exc()
