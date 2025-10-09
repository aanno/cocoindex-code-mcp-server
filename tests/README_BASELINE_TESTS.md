# Baseline Tests Documentation

This document explains the baseline comparison tests for language analysis in CocoIndex Code MCP Server.

## Overview

Baseline tests compare our tree-sitter-based code analysis against simple text-based pattern matching. They provide quantitative metrics (precision, recall, F1 scores) to validate parser quality and track improvements/regressions across language implementations.

## Test Types

### Multi-Language Baseline (`tests/all_languages_baseline.py`)

Tests all supported languages (Python, Haskell, C, C++, Rust, Kotlin, Java, TypeScript) by comparing:
- **Tree-sitter analysis** - Our main implementation using tree-sitter parsers
- **Text baseline** - Simple regex-based pattern matching for reference

### Language-Specific Comprehensive Tests

Example: `tests/lang/haskell/test_haskell_comprehensive_baseline.py`

These tests compare three analysis methods for specific languages:
1. **Specialized Handler** - Custom language-specific implementation (e.g., Haskell handler)
2. **Generic Visitor** - Generic AST visitor (should delegate to specialized handler)
3. **Text Baseline** - Simple pattern matching baseline

## Metrics Calculation

### Precision and Recall

The baseline test calculates standard information retrieval metrics:

```python
# Recall: What percentage of expected items were found
function_recall = len(detected ∩ expected) / len(expected)

# Precision: What percentage of detected items were correct  
function_precision = len(detected ∩ expected) / len(detected)

# F1 Score: Harmonic mean of precision and recall
f1_score = 2 * (precision * recall) / (precision + recall)
```

### Example Calculation

For a hypothetical test:
- **Expected**: 7 items
- **Detected**: 12 items (7 correct + 5 false positives)

**Metrics:**
- **Recall**: 7/7 = 100% (found all expected items)
- **Precision**: 7/12 = 58.33% (7 correct out of 12 detected)
- **F1 Score**: 2 * (1.0 * 0.583) / (1.0 + 0.583) = 73.68%

## Running Baseline Tests

### All Languages Baseline

Run baseline tests for all supported languages at once:

```bash
# Run the all-languages baseline test
python tests/all_languages_baseline.py

# Results are automatically saved to JSON
cat tests/all_languages_baseline_results.json | jq '.'

# Extract specific language metrics
cat tests/all_languages_baseline_results.json | jq '.python'
cat tests/all_languages_baseline_results.json | jq '.haskell'

# Extract F1 scores for all languages
cat tests/all_languages_baseline_results.json | jq 'to_entries[] | {language: .key, f1: .value.tree_sitter.f1}'

# Compare tree-sitter vs baseline for a specific language
cat tests/all_languages_baseline_results.json | jq '.python | {tree_sitter: .tree_sitter.f1, baseline: .baseline.f1}'
```

The `all_languages_baseline.py` script tests Python, Haskell, C, C++, Rust, Kotlin, Java, and TypeScript in one run, providing a comprehensive view of parser quality across all supported languages.

### Language-Specific Test Run

For detailed testing of individual languages:

```bash
python -m pytest tests/lang/haskell/test_haskell_comprehensive_baseline.py -v
```

### Verbose Output with Metrics

```bash
python -m pytest tests/lang/haskell/test_haskell_comprehensive_baseline.py -v -s
```

### Debug Output (shows all chunks processed)

```bash
python -m pytest tests/lang/haskell/test_haskell_comprehensive_baseline.py -v -s --log-cli-level=DEBUG
```

## Extracting Metrics

### Method 1: JSON Results File

The test automatically saves detailed metrics to a JSON file:

```bash
# Run the test
python -m pytest tests/lang/haskell/test_haskell_comprehensive_baseline.py

# Extract specific metrics using jq
cat tests/lang/haskell/haskell_baseline_results.json | jq '.metrics.specialized_visitor.function_recall'
cat tests/lang/haskell/haskell_baseline_results.json | jq '.metrics.specialized_visitor.function_precision'
cat tests/lang/haskell/haskell_baseline_results.json | jq '.metrics.specialized_visitor.overall_score'

# Get all function-related metrics
cat tests/lang/haskell/haskell_baseline_results.json | jq '.metrics.specialized_visitor | {recall: .function_recall, precision: .function_precision, f1: .function_f1}'
```

### Method 2: Programmatic Access

You can import and run the test programmatically:

```python
import sys
# sys.path.append('tests/lang/haskell')
from test_haskell_comprehensive_baseline import TestHaskellComprehensiveBaseline

# Create test instance and run comparison
test = TestHaskellComprehensiveBaseline()
test.setup_method(None)
summary = test.test_comprehensive_baseline_comparison()

# Extract metrics
specialized_metrics = summary['metrics']['specialized_visitor']
print(f"Function Recall: {specialized_metrics['function_recall']:.2%}")
print(f"Function Precision: {specialized_metrics['function_precision']:.2%}")
print(f"Overall F1 Score: {specialized_metrics['overall_score']:.2%}")
```

### Method 3: Custom Script

Create a script to extract specific metrics:

```python
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

if __name__ == '__main__':
    data = run_baseline_test()
    if data:
        print_metrics(data)
    else:
        print("Failed to extract metrics")
        sys.exit(1)
```

## Understanding the Metrics

### Good vs Bad Results

**Good Results:**

- **High Recall (>90%)**: Finds most expected functions/types
- **High Precision (>80%)**: Few false positives
- **Balanced F1 Score (>85%)**: Good overall performance

**Concerning Results:**

- **Low Recall (<70%)**: Missing many expected items
- **Low Precision (<60%)**: Too many false positives
- **Large Gap**: Big difference between precision and recall

### Current Haskell Results (After Improvement)

| Method | Function Recall | Function Precision | F1 Score | Status |
|--------|-----------------|-------------------|----------|---------|
| Specialized Visitor | 100% | 58.3% | 73.7% | ✅ Good recall, acceptable precision |
| Generic Visitor | 100% | 58.3% | 73.7% | ✅ Same as specialized |
| CocoIndex Baseline | 100% | 100% | 100% | ✅ Perfect (ground truth) |

### Trade-offs

The current Haskell implementation shows a classic precision/recall trade-off:

- **100% Recall**: We catch all actual functions (including `addTen = (+) 10`)
- **58% Precision**: We also catch variable bindings (`person`, `numbers`, etc.)

This is often acceptable for code analysis where missing functions is worse than finding extra bindings.

## Adding New Language Tests

To create baseline tests for a new language:

1. **Create test fixture**: `tests/fixtures/test_<language>.<ext>`
2. **Define expected items**: List all functions, classes, types, etc.
3. **Create test class**: Inherit from a base comparison class
4. **Implement analysis methods**: Add language-specific parsers
5. **Set quality thresholds**: Define minimum acceptable metrics

## Monitoring and CI Integration

### Quality Gates

You can use these metrics as quality gates in CI:

```yaml
# .github/workflows/test.yml
- name: Run Baseline Tests
  run: |
    python -m pytest tests/lang/haskell/test_haskell_comprehensive_baseline.py
    
- name: Check Quality Metrics
  run: |
    RECALL=$(cat tests/lang/haskell/haskell_baseline_results.json | jq -r '.metrics.specialized_visitor.function_recall')
    if (( $(echo "$RECALL < 0.9" | bc -l) )); then
      echo "Function recall $RECALL below threshold 0.9"
      exit 1
    fi
```

### Tracking Over Time

Store metrics in a time series database or append to a CSV:

```bash
echo "$(date),$(cat tests/lang/haskell/haskell_baseline_results.json | jq -r '.metrics.specialized_visitor.function_recall')" >> metrics_history.csv
```

## Troubleshooting

### Test Failures

1. **Syntax Error**: Check test fixture for valid syntax
2. **Import Error**: Ensure all dependencies are installed
3. **Low Metrics**: Check if parser is correctly identifying language constructs

### Debug Commands

```bash
# See what chunks are being processed
python -m pytest tests/lang/haskell/test_haskell_comprehensive_baseline.py -s --log-cli-level=DEBUG | grep "Processed.*chunk"

# Check expected vs detected items
cat tests/lang/haskell/haskell_baseline_results.json | jq '.metrics.specialized_visitor | {detected: .detected_functions, missing: .missing_functions, extra: .extra_functions}'
```

## Language Support & Current Results

All tests are working correctly with the following F1 scores:

- **Python**: 100% (perfect)
- **Haskell**: 100% (perfect)
- **C**: 100% (perfect)
- **C++**: 100% (perfect)
- **Rust**: 100% (perfect)
- **Kotlin**: 100% (perfect)
- **TypeScript**: 94.1% (extra `constructor` detected)
- **Java**: 88.9% (constructors detected as functions)

### Implementation Notes

**Haskell** uses a custom Rust implementation (`rust/src/lib.rs`) via Maturin/PyO3 for performance, while other languages use Python tree-sitter bindings. All languages delegate to specialized handlers in `src/cocoindex_code_mcp_server/language_handlers/`.

## Best Practices

1. **Run tests after changes**: Always run baseline tests when modifying parsers
2. **Track metrics over time**: Monitor for regressions
3. **Balance precision/recall**: Consider your use case requirements
4. **Update fixtures carefully**: Changes affect all baseline measurements
5. **Document trade-offs**: Explain why certain precision/recall balances are acceptable
6. **Architecture awareness**: Understand that Haskell uses a fundamentally different implementation approach
