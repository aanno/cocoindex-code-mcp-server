# Metadata Enhancement for Code Search

This document describes how to enhance metadata extraction and keyword search for different programming languages in CocoIndex.

## Overview

The hybrid search system supports rich metadata extraction that can be extended for any programming language. Currently, Python has full metadata support, and this guide shows how to add similar support for other languages.

## Current Implementation

### Python Metadata Extraction

The system extracts rich metadata for Python code including:

- **Functions/Methods**: Names, parameters, return types, decorators
- **Classes**: Names, inheritance, methods
- **Imports**: Module names and aliases
- **Code Patterns**: Type hints, async functions, complexity metrics
- **Structural Info**: Private methods, dunder methods, class variables

### Integration Points

1. **Analysis Module**: `src/python_code_analyzer.py` - AST-based analysis
2. **Search Integration**: `src/hybrid_search.py` - Runtime metadata extraction
3. **Display Format**: Enhanced readable and JSON output formats

## Adding Metadata for New Languages

### Step 1: Create Language Analyzer

Create a new analyzer module following the Python pattern:

```python
# src/{language}_analyzer.py

import ast  # or appropriate parser
from typing import Dict, Any, List

class {Language}CodeAnalyzer:
    """Analyzer for {Language} code chunks."""

    def analyze_code(self, code: str, filename: str = "") -> Dict[str, Any]:
        """Extract metadata from {Language} code."""
        try:
            # Parse code using language-specific parser
            # Extract functions, classes, imports, etc.
            return self._build_metadata(code, filename)
        except Exception as e:
            return self._build_fallback_metadata(code, filename)

    def _build_metadata(self, code: str, filename: str) -> Dict[str, Any]:
        """Build comprehensive metadata dictionary."""
        return {
            "language": "{Language}",
            "filename": filename,
            "functions": [...],
            "classes": [...],
            "imports": [...],
            # Add language-specific fields
        }
```

### Step 2: Integrate with Search Engine

Update `src/hybrid_search.py` to include the new language:

```python
from {language}_analyzer import analyze_{language}_code

# In _format_result method:
elif language == "{Language}":
    try:
        metadata = analyze_{language}_code(code, filename)
        result.update({
            "functions": metadata.get("functions", []),
            "classes": metadata.get("classes", []),
            # Add language-specific fields
        })
    except Exception as e:
        result.update({"analysis_error": str(e)})
```

### Step 3: Update Display Formats

Enhance the readable format in `format_results_readable()`:

```python
# Add language-specific metadata display
if result['language'] == '{Language}' and result.get('functions'):
    # Format {Language}-specific metadata
    metadata_parts.append(f"Functions: {', '.join(result['functions'])}")
```

## Language-Specific Considerations

### Tree-sitter Languages

For languages with tree-sitter support, use the tree-sitter Python bindings:

```python
import tree_sitter_python as tsp
from tree_sitter import Language, Parser

# Create parser
parser = Parser()
parser.set_language(Language(tsp.language(), "python"))

# Parse and extract metadata
tree = parser.parse(code.encode('utf8'))
```

### Regex-based Fallbacks

For languages without full parser support, implement regex-based extraction:

```python
def _extract_functions_regex(self, code: str) -> List[str]:
    """Extract function names using regex patterns."""
    patterns = {
        'rust': r'fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        'go': r'func\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        'javascript': r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
    }
    return re.findall(patterns.get(language, ''), code)
```

## Database Schema Evolution

### Current Approach: JSON Fields

The current implementation adds metadata as JSON fields at search time:

```python
result.update({
    "functions": [...],           # Array of function names
    "classes": [...],            # Array of class names
    "metadata_json": json.dumps(metadata)  # Full metadata as JSON
})
```

### Future Migration: Database Fields

To add metadata as proper database fields:

1. **Extend CocoIndex Flow**: Add metadata extraction functions to `cocoindex_config.py`
2. **Update Schema**: Add columns to the database table
3. **Migrate Data**: Run migration to populate new fields

Example migration approach:

```python
# In cocoindex_config.py
@cocoindex.op.function()
def extract_metadata_fields(code: str, language: str) -> Dict[str, Any]:
    """Extract metadata as separate fields."""
    if language == "Python":
        metadata = analyze_python_code(code)
        return {
            "function_names": json.dumps(metadata.get("functions", [])),
            "class_names": json.dumps(metadata.get("classes", [])),
            "import_modules": json.dumps(metadata.get("imports", [])),
            "complexity_score": metadata.get("complexity_score", 0),
            "has_type_hints": metadata.get("has_type_hints", False),
        }
    return {}

# Add to flow processing
with file["chunks"].row() as chunk:
    metadata_fields = chunk[["text", "language"]].transform(extract_metadata_fields)
    code_embeddings.collect(
        # ... existing fields ...
        function_names=metadata_fields["function_names"],
        class_names=metadata_fields["class_names"],
        # ... other metadata fields ...
    )
```

## Keyword Search Extensions

### Adding Metadata Search

Once metadata is in database fields, extend keyword search:

```python
# In keyword_search_parser.py
SEARCHABLE_FIELDS = [
    "filename", "language", "code", "source_name",
    "function_names", "class_names", "import_modules"  # New metadata fields
]

# Enable searches like:
# "function_names:contains(process_data)"
# "class_names:contains(DataProcessor)"
# "import_modules:contains(numpy)"
```

### Advanced Search Patterns

Support complex metadata queries:

```
language:Python and function_names:contains(async) and has_type_hints:true
class_names:exists() and complexity_score:>10
import_modules:contains(tensorflow) or import_modules:contains(pytorch)
```

## Performance Considerations

### Caching Metadata

For frequently accessed code chunks, consider caching metadata:

```python
import functools
from typing import Dict, Any

@functools.lru_cache(maxsize=1000)
def cached_analyze_code(code_hash: str, code: str, language: str) -> Dict[str, Any]:
    """Cache metadata analysis results."""
    if language == "Python":
        return analyze_python_code(code)
    return {}
```

### Incremental Updates

When code changes, only recompute metadata for affected chunks:

```python
def should_recompute_metadata(old_code: str, new_code: str) -> bool:
    """Determine if metadata needs recomputing."""
    # Simple check: if function/class definitions changed
    old_defs = extract_definitions(old_code)
    new_defs = extract_definitions(new_code)
    return old_defs != new_defs
```

## Testing Metadata Extraction

### Unit Tests

Create comprehensive tests for each language analyzer:

```python
def test_python_analyzer():
    code = '''
def hello(name: str) -> str:
    return f"Hello {name}"

class Greeter:
    def greet(self): pass
    '''

    metadata = analyze_python_code(code)
    assert "hello" in metadata["functions"]
    assert "Greeter" in metadata["classes"]
    assert metadata["has_type_hints"] is True
```

### Integration Tests

Test the full search pipeline with metadata:

```python
def test_metadata_search():
    results = search_engine.search(
        vector_query="greeting function",
        keyword_query="language:Python and has_type_hints:true"
    )

    assert len(results) > 0
    assert all(r["language"] == "Python" for r in results)
    assert all(r.get("has_type_hints") for r in results)
```

## Example: Haskell Metadata

Based on the existing Haskell support, here's how to add metadata:

```python
# src/haskell_analyzer.py
def analyze_haskell_code(code: str, filename: str = "") -> Dict[str, Any]:
    """Extract Haskell metadata."""
    return {
        "language": "Haskell",
        "filename": filename,
        "functions": extract_haskell_functions(code),
        "types": extract_type_definitions(code),
        "modules": extract_module_imports(code),
        "type_signatures": extract_type_signatures(code),
        "has_type_classes": "class " in code,
        "has_instances": "instance " in code,
    }
```

This approach provides a scalable foundation for adding rich metadata support to any programming language in the CocoIndex search system.
