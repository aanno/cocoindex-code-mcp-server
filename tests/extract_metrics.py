#!/usr/bin/env python3
"""Extract baseline test metrics."""

import json
import subprocess
import sys
from pathlib import Path

def run_baseline_test():
    """Run the baseline test and return metrics."""
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        'tests/lang/haskell/test_haskell_comprehensive_baseline.py',
        '-q'  # Quiet mode
    ], cwd=Path.cwd(), capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Test failed: {result.stderr}")
        return None
    
    # Read results file
    results_file = Path('tests/lang/haskell/haskell_baseline_results.json')
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None

def print_metrics(data):
    """Print formatted metrics."""
    for method in ['specialized_visitor', 'generic_visitor', 'cocoindex_baseline']:
        metrics = data['metrics'][method]
        print(f"\n{method.replace('_', ' ').title()}:")
        print(f"  Function Recall:    {metrics['function_recall']:.1%}")
        print(f"  Function Precision: {metrics['function_precision']:.1%}")
        print(f"  Function F1:        {metrics['function_f1']:.1%}")
        print(f"  Overall Score:      {metrics['overall_score']:.1%}")

def extract_specific_metric(data, method='specialized_visitor', metric='function_recall'):
    """Extract a specific metric value."""
    try:
        return data['metrics'][method][metric]
    except KeyError:
        return None

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Direct metric extraction mode
        if sys.argv[1] == '--extract':
            results_file = Path('tests/lang/haskell/haskell_baseline_results.json')
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                method = sys.argv[2] if len(sys.argv) > 2 else 'specialized_visitor'
                metric = sys.argv[3] if len(sys.argv) > 3 else 'function_recall'
                value = extract_specific_metric(data, method, metric)
                if value is not None:
                    print(f"{value:.3f}")
                else:
                    print(f"Metric {method}.{metric} not found")
                    sys.exit(1)
            else:
                print("Results file not found. Run the test first.")
                sys.exit(1)
        else:
            print(f"Usage: {sys.argv[0]} [--extract [method] [metric]]")
            print("Methods: specialized_visitor, generic_visitor, cocoindex_baseline")
            print("Metrics: function_recall, function_precision, function_f1, overall_score")
            sys.exit(1)
    else:
        # Full test run and display mode
        data = run_baseline_test()
        if data:
            print("📊 Baseline Test Metrics")
            print("=" * 40)
            print_metrics(data)
        else:
            print("Failed to extract metrics")
            sys.exit(1)