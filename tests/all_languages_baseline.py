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
        
    def analyze_with_cocoindex_baseline(self, code: str) -> Dict[str, Any]:
        """Simple pattern-based baseline analysis similar to CocoIndex text analysis."""
        try:
            lines = code.split('\n')
            functions = set()
            classes = set()
            
            for line in lines:
                line = line.strip()
                
                # Skip comments based on language
                if self.language == 'python' and line.startswith('#'):
                    continue
                elif self.language in ['c', 'cpp', 'java', 'javascript', 'typescript'] and line.startswith('//'):
                    continue
                elif self.language == 'haskell' and line.startswith('--'):
                    continue
                elif self.language == 'rust' and line.startswith('//'):
                    continue
                elif self.language == 'kotlin' and line.startswith('//'):
                    continue
                
                # Language-specific function detection
                if self.language == 'python':
                    if line.startswith('def ') and '(' in line:
                        func_name = line.split('def ')[1].split('(')[0].strip()
                        if func_name.isidentifier():
                            functions.add(func_name)
                    elif line.startswith('class '):
                        class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                        if class_name.isidentifier():
                            classes.add(class_name)
                            
                elif self.language == 'haskell':
                    if '::' in line and not line.startswith('--'):
                        func_name = line.split('::')[0].strip()
                        if func_name and func_name.replace("'", "").isidentifier():
                            functions.add(func_name)
                    elif line.startswith('data '):
                        parts = line.split()
                        if len(parts) > 1:
                            type_name = parts[1].split('=')[0].strip()
                            if type_name:
                                classes.add(type_name)
                                
                elif self.language in ['c', 'cpp']:
                    # Simple function detection: return_type function_name(
                    import re
                    func_pattern = r'\b(\w+)\s*\('
                    if '(' in line and ')' in line and not line.strip().startswith('//'):
                        # Look for function definitions
                        matches = re.findall(func_pattern, line)
                        for match in matches:
                            if match not in ['if', 'while', 'for', 'switch', 'printf', 'scanf']:
                                functions.add(match)
                    # Struct detection
                    if line.startswith('struct ') or 'struct ' in line:
                        struct_match = re.search(r'struct\s+(\w+)', line)
                        if struct_match:
                            classes.add(struct_match.group(1))
                            
                elif self.language == 'rust':
                    if line.startswith('fn ') and '(' in line:
                        func_name = line.split('fn ')[1].split('(')[0].strip()
                        if func_name.isidentifier():
                            functions.add(func_name)
                    elif line.startswith('struct ') or 'struct ' in line:
                        import re
                        struct_match = re.search(r'struct\s+(\w+)', line)
                        if struct_match:
                            classes.add(struct_match.group(1))
                            
                elif self.language in ['java', 'kotlin']:
                    # Method detection
                    import re
                    if 'fun ' in line and '(' in line:  # Kotlin
                        func_match = re.search(r'fun\s+(\w+)\s*\(', line)
                        if func_match:
                            functions.add(func_match.group(1))
                    elif ' ' in line and '(' in line and ')' in line:  # Java methods
                        # Look for method patterns: visibility return_type method_name(
                        method_pattern = r'\b(\w+)\s*\('
                        matches = re.findall(method_pattern, line)
                        for match in matches:
                            if match not in ['if', 'while', 'for', 'switch', 'System', 'new']:
                                functions.add(match)
                    # Class detection
                    if 'class ' in line:
                        class_match = re.search(r'class\s+(\w+)', line)
                        if class_match:
                            classes.add(class_match.group(1))
                    elif 'data class ' in line:  # Kotlin data class
                        class_match = re.search(r'data\s+class\s+(\w+)', line)
                        if class_match:
                            classes.add(class_match.group(1))
                            
                elif self.language in ['javascript', 'typescript']:
                    if line.startswith('function ') and '(' in line:
                        func_name = line.split('function ')[1].split('(')[0].strip()
                        if func_name.isidentifier():
                            functions.add(func_name)
                    elif 'class ' in line:
                        import re
                        class_match = re.search(r'class\s+(\w+)', line)
                        if class_match:
                            classes.add(class_match.group(1))
                    # Method detection inside classes
                    elif re.match(r'\s*\w+\s*\(.*\)\s*[:{]', line):
                        method_match = re.match(r'\s*(\w+)\s*\(', line)
                        if method_match and method_match.group(1) not in ['if', 'while', 'for', 'constructor']:
                            functions.add(method_match.group(1))
            
            return {
                'analysis_method': 'cocoindex_text_baseline',
                'functions': sorted(list(functions)),
                'classes': sorted(list(classes)),
                'success': True
            }
            
        except Exception as e:
            return {
                'analysis_method': 'cocoindex_baseline_failed',
                'error': str(e),
                'functions': [],
                'classes': [],
                'success': False
            }
        
    def run_test(self) -> Dict[str, Any]:
        """Run baseline test for this language comparing tree-sitter vs baseline."""
        if not self.fixture_file.exists():
            return {'success': False, 'error': f'Fixture file {self.fixture_file} not found'}
            
        with open(self.fixture_file) as f:
            code = f.read()
            
        # Run both analyses
        tree_sitter_result = analyze_code(code, self.language, str(self.fixture_file))
        baseline_result = self.analyze_with_cocoindex_baseline(code)
        
        # Calculate metrics for tree-sitter implementation
        ts_detected_functions = set(tree_sitter_result.get('functions', []))
        ts_true_positives = len(ts_detected_functions & self.expected_functions)
        ts_precision = ts_true_positives / len(ts_detected_functions) if ts_detected_functions else 0
        ts_recall = ts_true_positives / len(self.expected_functions) if self.expected_functions else 0
        ts_f1 = 2 * (ts_precision * ts_recall) / (ts_precision + ts_recall) if (ts_precision + ts_recall) > 0 else 0
        
        # Calculate metrics for baseline implementation
        baseline_detected_functions = set(baseline_result.get('functions', []))
        baseline_true_positives = len(baseline_detected_functions & self.expected_functions)
        baseline_precision = baseline_true_positives / len(baseline_detected_functions) if baseline_detected_functions else 0
        baseline_recall = baseline_true_positives / len(self.expected_functions) if self.expected_functions else 0
        baseline_f1 = 2 * (baseline_precision * baseline_recall) / (baseline_precision + baseline_recall) if (baseline_precision + baseline_recall) > 0 else 0
        
        # Construct-specific metrics for tree-sitter
        construct_metrics = {}
        for construct_type, expected_items in self.expected_constructs.items():
            detected_items = set(tree_sitter_result.get(construct_type, []))
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
            'success': tree_sitter_result.get('success') is not False and 'analysis_method' in tree_sitter_result,
            'language': self.language,
            'tree_sitter': {
                'analysis_method': tree_sitter_result.get('analysis_method'),
                'functions': {
                    'recall': ts_recall,
                    'precision': ts_precision,
                    'f1': ts_f1,
                    'detected': ts_detected_functions,
                    'expected': self.expected_functions,
                    'missing': self.expected_functions - ts_detected_functions,
                    'extra': ts_detected_functions - self.expected_functions
                },
                'constructs': construct_metrics,
                'raw_result': tree_sitter_result
            },
            'baseline': {
                'analysis_method': baseline_result.get('analysis_method'),
                'functions': {
                    'recall': baseline_recall,
                    'precision': baseline_precision,
                    'f1': baseline_f1,
                    'detected': baseline_detected_functions,
                    'expected': self.expected_functions,
                    'missing': self.expected_functions - baseline_detected_functions,
                    'extra': baseline_detected_functions - self.expected_functions
                },
                'success': baseline_result.get('success', False),
                'raw_result': baseline_result
            },
            # Legacy format for backward compatibility
            'analysis_method': tree_sitter_result.get('analysis_method'),
            'functions': {
                'recall': ts_recall,
                'precision': ts_precision,
                'f1': ts_f1,
                'detected': ts_detected_functions,
                'expected': self.expected_functions,
                'missing': self.expected_functions - ts_detected_functions,
                'extra': ts_detected_functions - self.expected_functions
            },
            'constructs': construct_metrics,
            'raw_result': tree_sitter_result
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
        
        print(f"\n🏆 Language Performance Ranking (Tree-Sitter Implementation):")
        print(f"{'Language':<12} {'F1 Score':<10} {'Recall':<8} {'Precision':<10} {'Method':<20}")
        print("-" * 70)
        
        for lang_perf in summary['languages_by_performance']:
            status = "✅" if lang_perf['f1_score'] > 0 else "❌"
            print(f"{lang_perf['language']:<12} {lang_perf['f1_score']:.3f}     {lang_perf['recall']:.1%}   {lang_perf['precision']:.1%}      {lang_perf['analysis_method']:<20} {status}")
        
        print(f"\n📋 Detailed Results by Language:")
        for lang_name, result in data['results'].items():
            if result['success']:
                # Tree-sitter results
                ts_functions = result['functions']
                print(f"\n{lang_name.upper()}:")
                print(f"  🌳 Tree-Sitter - Recall: {ts_functions['recall']:.1%}, Precision: {ts_functions['precision']:.1%}, F1: {ts_functions['f1']:.1%}")
                print(f"     Found: {sorted(ts_functions['detected'])}")
                
                # Baseline results (if available)
                if 'baseline' in result and result['baseline'].get('success', False):
                    baseline_functions = result['baseline']['functions']
                    print(f"  📝 Baseline - Recall: {baseline_functions['recall']:.1%}, Precision: {baseline_functions['precision']:.1%}, F1: {baseline_functions['f1']:.1%}")
                    print(f"     Found: {sorted(baseline_functions['detected'])}")
                    
                    # Comparison
                    ts_f1 = ts_functions['f1']
                    baseline_f1 = baseline_functions['f1']
                    improvement = ts_f1 - baseline_f1
                    if improvement > 0:
                        print(f"  ⬆️ Improvement: Tree-sitter is {improvement:.1%} better (F1 score)")
                    elif improvement < 0:
                        print(f"  ⬇️ Regression: Tree-sitter is {abs(improvement):.1%} worse (F1 score)")
                    else:
                        print(f"  ➡️ Same performance")
                else:
                    print(f"  📝 Baseline - Not available")
                
                print(f"  Expected: {sorted(ts_functions['expected'])}")
                if ts_functions['missing']:
                    print(f"  Missing: {sorted(ts_functions['missing'])}")
                if ts_functions['extra']:
                    print(f"  Extra: {sorted(ts_functions['extra'])}")
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
