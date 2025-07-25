# Baseline Tests Documentation

This document explains the baseline comparison tests for language analysis in the CocoIndex Code MCP Server.

## Overview

Baseline tests compare different analysis methods to validate and measure the quality of our language parsers. They provide quantitative metrics to track improvements and regressions.

## Test Structure

### Haskell Baseline Test

Location: `tests/lang/haskell/test_haskell_comprehensive_baseline.py`

The test compares three analysis methods:
1. **Specialized Visitor** - Custom Haskell tree-sitter visitor
2. **Generic Visitor** - Generic AST visitor 
3. **CocoIndex Baseline** - Ground truth using CocoIndex flow analysis

### Test Fixture

The test uses `tests/fixtures/test_haskell.hs` which contains:
- **7 functions**: `fibonacci`, `sumList`, `treeMap`, `compose`, `addTen`, `multiplyByTwo`, `main`
- **2 data types**: `Person`, `Tree`

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

For our improved Haskell visitor:
- **Expected functions**: 7 (`fibonacci`, `sumList`, `treeMap`, `compose`, `addTen`, `multiplyByTwo`, `main`)
- **Detected functions**: 12 (all 7 expected + 5 extra: `person`, `numbers`, `tree`, `processNumber`, `doubledTree`)

**Metrics:**
- **Recall**: 7/7 = 100% (found all expected functions)
- **Precision**: 7/12 = 58.33% (7 correct out of 12 detected)
- **F1 Score**: 2 * (1.0 * 0.583) / (1.0 + 0.583) = 73.68%

## Running Baseline Tests

### Basic Test Run
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
sys.path.append('tests/lang/haskell')
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

## Best Practices

1. **Run tests after changes**: Always run baseline tests when modifying parsers
2. **Track metrics over time**: Monitor for regressions
3. **Balance precision/recall**: Consider your use case requirements
4. **Update fixtures carefully**: Changes affect all baseline measurements
5. **Document trade-offs**: Explain why certain precision/recall balances are acceptable