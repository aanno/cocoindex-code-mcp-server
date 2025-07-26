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
        print("=" * 80)
        print("🌳 TREE-SITTER IMPLEMENTATION vs 📝 BASELINE COMPARISON")
        print("=" * 80)
        print(f"{'Language':<12} {'Implementation':<25} {'Baseline':<25} {'Improvement':<15}")
        print(f"{'':12} {'Recall Prec  F1':25} {'Recall Prec  F1':25} {'(F1 Δ)':15}")
        print("-" * 80)
        
        for lang_name, result in data['results'].items():
            if result['success']:
                # Tree-sitter metrics
                ts_functions = result['functions']
                ts_recall = f"{ts_functions['recall']:.1%}"
                ts_precision = f"{ts_functions['precision']:.1%}"
                ts_f1 = f"{ts_functions['f1']:.1%}"
                ts_summary = f"{ts_recall:5} {ts_precision:5} {ts_f1:5}"
                
                # Baseline metrics (if available)
                if 'baseline' in result and result['baseline'].get('success', False):
                    baseline_functions = result['baseline']['functions']
                    baseline_recall = f"{baseline_functions['recall']:.1%}"
                    baseline_precision = f"{baseline_functions['precision']:.1%}"
                    baseline_f1 = f"{baseline_functions['f1']:.1%}"
                    baseline_summary = f"{baseline_recall:5} {baseline_precision:5} {baseline_f1:5}"
                    
                    # Calculate improvement
                    improvement = ts_functions['f1'] - baseline_functions['f1']
                    if improvement > 0:
                        improvement_str = f"↗ +{improvement:.1%}"
                    elif improvement < 0:
                        improvement_str = f"↘ {improvement:.1%}"
                    else:
                        improvement_str = "→ Same"
                else:
                    baseline_summary = "Not available".ljust(25)
                    improvement_str = "N/A"
                
                print(f"{lang_name:<12} {ts_summary:25} {baseline_summary:25} {improvement_str:15}")
            else:
                print(f"{lang_name:<12} {'FAILED':25} {'N/A':25} {'N/A':15}")
        
        print("-" * 80)
        
        # Summary statistics
        successful_results = [r for r in data['results'].values() if r['success']]
        if successful_results:
            avg_ts_recall = sum(r['functions']['recall'] for r in successful_results) / len(successful_results)
            avg_ts_precision = sum(r['functions']['precision'] for r in successful_results) / len(successful_results)
            avg_ts_f1 = sum(r['functions']['f1'] for r in successful_results) / len(successful_results)
            
            baseline_results = [r for r in successful_results if 'baseline' in r and r['baseline'].get('success', False)]
            if baseline_results:
                avg_baseline_recall = sum(r['baseline']['functions']['recall'] for r in baseline_results) / len(baseline_results)
                avg_baseline_precision = sum(r['baseline']['functions']['precision'] for r in baseline_results) / len(baseline_results)
                avg_baseline_f1 = sum(r['baseline']['functions']['f1'] for r in baseline_results) / len(baseline_results)
                avg_improvement = avg_ts_f1 - avg_baseline_f1
                
                print(f"{'AVERAGE':<12} {avg_ts_recall:.1%} {avg_ts_precision:.1%} {avg_ts_f1:.1%}     {avg_baseline_recall:.1%} {avg_baseline_precision:.1%} {avg_baseline_f1:.1%}     {avg_improvement:+.1%}")
            else:
                print(f"{'AVERAGE':<12} {avg_ts_recall:.1%} {avg_ts_precision:.1%} {avg_ts_f1:.1%}     {'N/A':17}     {'N/A':15}")
        
        print("=" * 80)

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
