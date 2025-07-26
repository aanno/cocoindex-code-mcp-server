#!/usr/bin/env python3
"""
Comprehensive baseline test for all supported languages.
Tests both tree-sitter visitors and CocoIndex baseline analysis.
"""

import json
from pathlib import Path
from typing import Dict, Any, Set

# Package should be installed via maturin develop or pip install -e .

from cocoindex_code_mcp_server.ast_visitor import analyze_code

class LanguageBaseline:
    """Baseline test for a specific language."""
    
    def __init__(self, language: str, fixture_file: str, expected_functions: Set[str], 
                 expected_constructs: Dict[str, Set[str]] = None):
        self.language = language
        self.fixture_file = Path(f"tests/fixtures/{fixture_file}")
        self.expected_functions = expected_functions
        self.expected_constructs = expected_constructs or {}
        
    def run_test(self) -> Dict[str, Any]:
        """Run baseline test for this language."""
        if not self.fixture_file.exists():
            return {'success': False, 'error': f'Fixture file {self.fixture_file} not found'}
            
        with open(self.fixture_file) as f:
            code = f.read()
            
        # Run AST visitor analysis
        result = analyze_code(code, self.language, str(self.fixture_file))
        
        # Calculate metrics
        detected_functions = set(result.get('functions', []))
        
        # Function metrics
        true_positives = len(detected_functions & self.expected_functions)
        false_positives = len(detected_functions - self.expected_functions)
        false_negatives = len(self.expected_functions - detected_functions)
        
        precision = true_positives / len(detected_functions) if detected_functions else 0
        recall = true_positives / len(self.expected_functions) if self.expected_functions else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Construct-specific metrics
        construct_metrics = {}
        for construct_type, expected_items in self.expected_constructs.items():
            detected_items = set(result.get(construct_type, []))
            if expected_items:
                construct_recall = len(detected_items & expected_items) / len(expected_items)
                construct_precision = len(detected_items & expected_items) / len(detected_items) if detected_items else 0
                construct_metrics[construct_type] = {
                    'recall': construct_recall,
                    'precision': construct_precision,
                    'detected': detected_items,
                    'expected': expected_items,
                    'missing': expected_items - detected_items,
                    'extra': detected_items - expected_items
                }
        
        return {
            'success': result.get('success') is not False and 'analysis_method' in result,
            'language': self.language,
            'analysis_method': result.get('analysis_method'),
            'functions': {
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'detected': detected_functions,
                'expected': self.expected_functions,
                'missing': self.expected_functions - detected_functions,
                'extra': detected_functions - self.expected_functions
            },
            'constructs': construct_metrics,
            'raw_result': result
        }

class MultiLanguageBaseline:
    """Multi-language baseline test runner."""
    
    def __init__(self):
        self.languages = {
            'python': LanguageBaseline(
                'python', 'test_python.py',
                expected_functions={'fibonacci', 'is_prime'},
                expected_constructs={'classes': {'MathUtils'}}
            ),
            'haskell': LanguageBaseline(
                'haskell', 'test_haskell.hs', 
                expected_functions={'fibonacci', 'sumList', 'treeMap', 'compose', 'addTen', 'multiplyByTwo', 'main'},
                expected_constructs={'data_types': {'Person', 'Tree'}}
            ),
            'c': LanguageBaseline(
                'c', 'test_c.c',
                expected_functions={'add', 'print_point', 'create_point', 'get_default_color', 'main'},
                expected_constructs={'structs': {'Point'}, 'enums': {'Color'}}
            ),
            'rust': LanguageBaseline(
                'rust', 'test_rust.rs',
                expected_functions={'new', 'is_adult', 'fibonacci', 'main'},
                expected_constructs={'structs': {'Person'}}
            ),
            'cpp': LanguageBaseline(
                'cpp', 'test_c.c',  # Using C file for now as it's valid C++
                expected_functions={'add', 'print_point', 'create_point', 'get_default_color', 'main'},
                expected_constructs={'structs': {'Point'}}
            ),
            'kotlin': LanguageBaseline(
                'kotlin', 'test_kotlin.kt',
                expected_functions={'fibonacci', 'processResult', 'calculateSum', 'isAdult', 'greet', 'add', 'multiply', 'getHistory', 'main'},
                expected_constructs={'classes': {'Person', 'Result', 'Calculator'}}
            ),
            'java': LanguageBaseline(
                'java', 'test_java.java',
                expected_functions={'fibonacci', 'calculateSum', 'printList', 'isPrime', 'isAdult', 'greet', 'getName', 'getAge', 'getArea', 'getPerimeter', 'getColor', 'main'},
                expected_constructs={'classes': {'TestJava', 'Person', 'Shape', 'Rectangle'}}
            ),
            'typescript': LanguageBaseline(
                'typescript', 'test_typescript.ts',
                expected_functions={'fibonacci', 'greet', 'isAdult', 'getName', 'getAge', 'calculateSum', 'processUsers', 'main'},
                expected_constructs={'classes': {'Person'}}
            )
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run baseline tests for all languages."""
        results = {}
        summary = {
            'total_languages': len(self.languages),
            'successful_languages': 0,
            'failed_languages': 0,
            'average_function_recall': 0,
            'average_function_precision': 0,
            'languages_by_performance': []
        }
        
        for lang_name, baseline in self.languages.items():
            print(f"Testing {lang_name}...")
            result = baseline.run_test()
            results[lang_name] = result
            
            if result['success']:
                summary['successful_languages'] += 1
            else:
                summary['failed_languages'] += 1
                
            # Add to performance ranking
            f1_score = result.get('functions', {}).get('f1', 0)
            summary['languages_by_performance'].append({
                'language': lang_name,
                'f1_score': f1_score,
                'recall': result.get('functions', {}).get('recall', 0),
                'precision': result.get('functions', {}).get('precision', 0),
                'analysis_method': result.get('analysis_method', 'unknown')
            })
        
        # Sort by F1 score
        summary['languages_by_performance'].sort(key=lambda x: x['f1_score'], reverse=True)
        
        # Calculate averages for successful languages
        successful_results = [r for r in results.values() if r['success']]
        if successful_results:
            summary['average_function_recall'] = sum(r['functions']['recall'] for r in successful_results) / len(successful_results)
            summary['average_function_precision'] = sum(r['functions']['precision'] for r in successful_results) / len(successful_results)
        
        return {'results': results, 'summary': summary}
    
    def print_summary(self, data: Dict[str, Any]):
        """Print formatted summary of all language tests."""
        print("\n" + "="*80)
        print("🔍 MULTI-LANGUAGE BASELINE TEST RESULTS")
        print("="*80)
        
        summary = data['summary']
        print(f"📊 Overall Statistics:")
        print(f"  • Total languages tested: {summary['total_languages']}")
        print(f"  • Successful: {summary['successful_languages']}")
        print(f"  • Failed: {summary['failed_languages']}")
        print(f"  • Average function recall: {summary['average_function_recall']:.1%}")
        print(f"  • Average function precision: {summary['average_function_precision']:.1%}")
        
        print(f"\n🏆 Language Performance Ranking:")
        print(f"{'Language':<12} {'F1 Score':<10} {'Recall':<8} {'Precision':<10} {'Method':<20}")
        print("-" * 70)
        
        for lang_perf in summary['languages_by_performance']:
            status = "✅" if lang_perf['f1_score'] > 0 else "❌"
            print(f"{lang_perf['language']:<12} {lang_perf['f1_score']:.3f}     {lang_perf['recall']:.1%}   {lang_perf['precision']:.1%}      {lang_perf['analysis_method']:<20} {status}")
        
        print(f"\n📋 Detailed Results by Language:")
        for lang_name, result in data['results'].items():
            if result['success']:
                functions = result['functions']
                print(f"\n{lang_name.upper()}:")
                print(f"  Functions - Recall: {functions['recall']:.1%}, Precision: {functions['precision']:.1%}, F1: {functions['f1']:.1%}")
                print(f"  Found: {sorted(functions['detected'])}")
                print(f"  Expected: {sorted(functions['expected'])}")
                if functions['missing']:
                    print(f"  Missing: {sorted(functions['missing'])}")
                if functions['extra']:
                    print(f"  Extra: {sorted(functions['extra'])}")
            else:
                print(f"\n{lang_name.upper()}: ❌ FAILED - {result.get('error', 'Unknown error')}")

def main():
    """Main test runner."""
    baseline = MultiLanguageBaseline()
    results = baseline.run_all_tests()
    
    baseline.print_summary(results)
    
    # Save detailed results
    output_file = Path('tests/all_languages_baseline_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return results

if __name__ == '__main__':
    main()
