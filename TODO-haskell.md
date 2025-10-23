# Haskell Parser Issue - TODO

## Problem Summary

The Haskell AST parser fails on complex files due to tree-sitter parsing errors, resulting in 0 chunks extracted and empty functions/imports arrays despite valid Haskell code containing clear function definitions.

## Root Cause Analysis

**Architecture**: Custom Rust implementation with maturin bindings (`/workspaces/rust/rust/src/lib.rs`) using `tree-sitter-haskell`, unlike other language parsers.

**Core Issue**: The `haskell_example_1.hs` test file generates **47 ERROR nodes** in the AST parse tree, causing semantic chunk extraction to fail completely.

### Specific Failures

- ✅ **Simple code**: Works perfectly (7 chunks: `signature`, `function` types)
- ❌ **Complex file**: 0 chunks extracted, 405 nodes with 47 ERROR nodes
- ❌ **Module declaration**: Even `module haskell_example_1 where` creates ERROR nodes
- ✅ **Data section only**: 2 chunks extracted successfully when isolated

## Technical Details

### Current Behavior

1. `get_haskell_ast_chunks()` returns empty vector when ERROR nodes prevent extraction
2. Falls back to `get_haskell_ast_chunks_with_fallback()` → produces `regex_chunk` types
3. Handler expects `function`/`bind`/`signature` but receives `regex_chunk` → empty results

### Code Location

- **Chunk extraction**: `/workspaces/rust/rust/src/lib.rs:190-298` (`extract_chunks_recursive`)
- **Target node types**: Lines 201-210 (only processes `signature`, `function`, `bind`, etc.)
- **AST parsing**: Lines 376-393 (`get_haskell_ast_chunks`)

## Evidence

```
Testing simple code (155 chars, 8 lines)
  ✅ get_haskell_ast_chunks: 7 chunks
     Types: ['signature', 'function', 'function', 'signature', 'function']

Testing complex code (1571 chars, 58 lines)
  ✅ get_haskell_ast_chunks: 0 chunks ❌ No chunks returned!
  ✅ get_haskell_ast_chunks_with_fallback: 44 chunks
     Types: ['regex_chunk', 'regex_chunk', 'regex_chunk']
```

## Required Fixes

### Priority 1: Error Recovery in Rust Implementation

Modify `extract_chunks_recursive()` to handle ERROR nodes gracefully:

1. **Process children of ERROR nodes** recursively to find valid semantic chunks
2. **Implement fault-tolerant extraction** that doesn't skip entire subtrees with errors
3. **Better error logging** to identify specific tree-sitter parsing issues

### Priority 2: Module Declaration Parsing

Fix tree-sitter-haskell grammar issues:

1. Investigate why `module ModuleName where` creates ERROR nodes
2. Test with different module declaration formats
3. Ensure standard Haskell module syntax parses correctly

### Priority 3: Handler Compatibility

Update Python handler to process chunks even when some ERROR nodes exist:

1. Add fallback processing for partial AST results
2. Improve error reporting to identify specific parsing failures

## Testing

- ✅ Test case exists: `tests/test_haskell_kotlin_analysis_issues.py` (marked as expected failure)
- ✅ Debug infrastructure: Functions to analyze AST nodes and ERROR locations
- ✅ Incremental testing: Confirms simple → complex progression where errors occur

## Expected Outcome

After fix: `haskell_example_1.hs` should detect:

- **Functions**: fibonacci, sumList, treeMap, compose, addTen, multiplyByTwo, main (7 functions)
- **Data types**: Person, Tree (2 types)
- **Imports**: No imports in test file
- **Analysis method**: `haskell_chunk_visitor` (already correct)

## Impact

This affects all real-world Haskell files with module declarations, making the analyzer unusable for production Haskell codebases despite working perfectly on simple code snippets.
