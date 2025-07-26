#!/usr/bin/env python3
"""Extract baseline test metrics for all languages."""

import json
import subprocess
import sys
from pathlib import Path

def run_baseline_test(test_type='all'):
    """Run baseline tests and return metrics."""
    if test_type == 'haskell':
        # Run original Haskell-specific test
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/lang/haskell/test_haskell_comprehensive_baseline.py',
            '-q'  # Quiet mode
        ], cwd=Path.cwd(), capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Haskell test failed: {result.stderr}")
            return None
        
        # Read results file
        results_file = Path('tests/lang/haskell/haskell_baseline_results.json')
        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)
    
    elif test_type == 'all':
        # Run comprehensive multi-language test
        result = subprocess.run([
            sys.executable, 'tests/all_languages_baseline.py'
        ], cwd=Path.cwd(), capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Multi-language test failed: {result.stderr}")
            return None
        
        # Read results file
        results_file = Path('tests/all_languages_baseline_results.json')
        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)
    
    return None

def print_metrics(data):
    """Print formatted metrics."""
    if 'metrics' in data:
        # Haskell-style format
        for method in ['specialized_visitor', 'generic_visitor', 'cocoindex_baseline']:
            if method in data['metrics']:
                metrics = data['metrics'][method]
                print(f"\n{method.replace('_', ' ').title()}:")
                print(f"  Function Recall:    {metrics['function_recall']:.1%}")
                print(f"  Function Precision: {metrics['function_precision']:.1%}")
                print(f"  Function F1:        {metrics['function_f1']:.1%}")
                print(f"  Overall Score:      {metrics['overall_score']:.1%}")
    elif 'results' in data:
        # Multi-language format
        print("\n📊 All Languages Summary:")
        print(f"{'Language':<12} {'Recall':<8} {'Precision':<10} {'F1 Score':<10} {'Status'}")
        print("-" * 60)
        
        for lang_name, result in data['results'].items():
            if result['success']:
                functions = result['functions']
                status = "✅"
                recall = f"{functions['recall']:.1%}"
                precision = f"{functions['precision']:.1%}"
                f1 = f"{functions['f1']:.1%}"
            else:
                status = "❌"
                recall = "N/A"
                precision = "N/A"
                f1 = "N/A"
            
            print(f"{lang_name:<12} {recall:<8} {precision:<10} {f1:<10} {status}")

def extract_specific_metric(data, language='haskell', method='specialized_visitor', metric='function_recall'):
    """Extract a specific metric value."""
    try:
        if 'metrics' in data:
            # Haskell-style format
            return data['metrics'][method][metric]
        elif 'results' in data:
            # Multi-language format
            if language in data['results']:
                result = data['results'][language]
                if result['success']:
                    if metric.startswith('function_'):
                        metric_name = metric.replace('function_', '')
                        return result['functions'][metric_name]
            return None
    except KeyError:
        return None

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Direct metric extraction mode
        if sys.argv[1] == '--extract':
            # Determine test type and results file
            test_type = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ['haskell', 'all'] else 'all'
            
            if test_type == 'haskell':
                results_file = Path('tests/lang/haskell/haskell_baseline_results.json')
            else:
                results_file = Path('tests/all_languages_baseline_results.json')
            
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                
                if test_type == 'haskell':
                    method = sys.argv[3] if len(sys.argv) > 3 else 'specialized_visitor'
                    metric = sys.argv[4] if len(sys.argv) > 4 else 'function_recall'
                    value = extract_specific_metric(data, 'haskell', method, metric)
                else:
                    language = sys.argv[3] if len(sys.argv) > 3 else 'c'
                    metric = sys.argv[4] if len(sys.argv) > 4 else 'function_recall'
                    value = extract_specific_metric(data, language, 'specialized_visitor', metric)
                
                if value is not None:
                    print(f"{value:.3f}")
                else:
                    print(f"Metric not found")
                    sys.exit(1)
            else:
                print(f"Results file {results_file} not found. Run the test first.")
                sys.exit(1)
        
        elif sys.argv[1] == '--help':
            print(f"Usage: {sys.argv[0]} [options]")
            print("Options:")
            print("  --extract [test_type] [language/method] [metric]")
            print("    test_type: 'haskell' or 'all' (default: all)")
            print("    For haskell: method = specialized_visitor|generic_visitor|cocoindex_baseline")
            print("    For all: language = c|rust|cpp|python|etc.")
            print("    metric: function_recall|function_precision|function_f1")
            print("  --test [test_type]")
            print("    Run specific test type: 'haskell' or 'all' (default: all)")
            sys.exit(0)
        
        elif sys.argv[1] == '--test':
            test_type = sys.argv[2] if len(sys.argv) > 2 else 'all'
            data = run_baseline_test(test_type)
            if data:
                print("📊 Baseline Test Metrics")
                print("=" * 40)
                print_metrics(data)
            else:
                print("Failed to extract metrics")
                sys.exit(1)
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            sys.exit(1)
    else:
        # Full test run and display mode (default: all languages)
        data = run_baseline_test('all')
        if data:
            print("📊 Baseline Test Metrics")
            print("=" * 40)
            print_metrics(data)
        else:
            print("Failed to extract metrics")
            sys.exit(1)
