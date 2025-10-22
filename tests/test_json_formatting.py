#!/usr/bin/env python3

"""
Test suite for JSON formatting functionality in hybrid search results.
Tests the human-readable formatting of code and metadata_json fields.
"""

import json
from typing import Any, Dict, List


def format_results_as_json(results: List[Dict[str, Any]], indent: int = 2) -> str:
    """Format search results as JSON string with human-readable code and metadata_json fields."""

    def format_single_result(result: Dict[str, Any], indent_level: int = 1) -> str:
        """Format a single result with custom handling for code and metadata_json fields."""
        lines = ["{"]

        for i, (key, value) in enumerate(result.items()):
            is_last = i == len(result) - 1
            comma = "" if is_last else ","
            indent_str = "  " * indent_level

            if key in ['code', 'metadata_json'] and isinstance(value, str):
                # For code and metadata_json, output the raw string without JSON escaping
                # Use triple quotes to preserve formatting
                formatted_value = f'"""{value}"""'
                lines.append(f'{indent_str}"{key}": {formatted_value}{comma}')
            else:
                # For other fields, use normal JSON formatting
                formatted_value = json.dumps(value, default=str)
                lines.append(f'{indent_str}"{key}": {formatted_value}{comma}')

        lines.append("}")
        return "\n".join(lines)

    # Format the entire results array
    if not results:
        return "[]"

    output_lines = ["["]
    for i, result in enumerate(results):
        is_last = i == len(results) - 1
        comma = "" if is_last else ","

        # Format single result with proper indentation
        formatted_result = format_single_result(result, indent_level=2)
        # Indent the entire result block
        indented_result = "\n".join(f"  {line}" for line in formatted_result.split("\n"))
        output_lines.append(f"{indented_result}{comma}")

    output_lines.append("]")
    return "\n".join(output_lines)


# Sample data similar to what you showed in your output
test_results = [
    {
        "filename": "cocoindex/python/cocoindex/targets.py",
        "language": "Python",
        "code": "@dataclass\nclass KuzuConnection:\n    \"\"\"Connection spec for Kuzu.\"\"\"\n\n    api_server_url: str\n\n\nclass Kuzu(op.TargetSpec):\n    \"\"\"Graph storage powered by Kuzu.\"\"\"\n\n    connection: AuthEntryReference[KuzuConnection]\n    mapping: Nodes | Relationships",
        "score": 0.2507555036385951,
        "start": {"line": 115, "column": 1, "offset": 2338},
        "end": {"line": 135, "column": 38, "offset": 2795},
        "source": "files",
        "score_type": "vector",
        "functions": [],
        "classes": ["KuzuConnection", "Kuzu", "KuzuDeclaration"],
        "imports": [],
        "complexity_score": 11,
        "has_type_hints": False,
        "has_async": False,
        "has_classes": True,
        "private_methods": [],
        "dunder_methods": [],
        "decorators": [],
        "metadata_json": "{\"language\": \"Python\", \"filename\": \"cocoindex/python/cocoindex/targets.py\", \"line_count\": 21, \"char_count\": 457, \"functions\": [], \"classes\": [\"KuzuConnection\", \"Kuzu\", \"KuzuDeclaration\"]}"
    },
    {
        "filename": "cocoindex/python/ops/functions/split_recursively.rs",
        "language": "Rust",
        "code": "    add_treesitter_language(\n        &mut map,\n        \"Go\",\n        [\".go\", \"golang\"],\n        tree_sitter_go::LANGUAGE,\n        [],\n    );\n    add_treesitter_language(\n        &mut map,\n        \"HTML\",\n        [\".html\", \".htm\"],\n        tree_sitter_html::LANGUAGE,\n        [],\n    );",
        "score": 0.23254502720728532,
        "start": {"line": 129, "column": 1, "offset": 3615},
        "end": {"line": 165, "column": 7, "offset": 4542},
        "source": "files",
        "score_type": "vector"
    }
]


class TestJSONFormatting:
    """Test class for JSON formatting functionality."""

    def test_code_field_formatting(self):
        """Test that code fields are formatted without JSON escaping."""
        test_data = [{
            "filename": "test.py",
            "code": "def hello():\n    print(\"Hello, World!\")\n    return True"
        }]

        result = format_results_as_json(test_data)

        # Should contain readable newlines, not escaped \n
        assert "def hello():" in result
        assert "print(\"Hello, World!\")" in result
        assert "return True" in result
        # Should NOT contain escaped newlines in the code content
        assert "\\n" not in result

    def test_metadata_json_field_formatting(self):
        """Test that metadata_json fields are formatted without JSON escaping."""
        test_data = [{
            "filename": "test.py",
            "metadata_json": "{\"functions\": [\"hello\"], \"classes\": []}"
        }]

        result = format_results_as_json(test_data)

        # Should contain readable JSON, not escaped quotes
        assert '"functions": ["hello"]' in result
        assert '"classes": []' in result
        # Should NOT contain escaped quotes in the metadata content
        assert '\\"' not in result

    def test_other_fields_normal_formatting(self):
        """Test that non-code/metadata_json fields use normal JSON formatting."""
        test_data = [{
            "filename": "test.py",
            "score": 0.95,
            "start": {"line": 1, "column": 1},
            "language": "Python"
        }]

        result = format_results_as_json(test_data)

        # These should be normally JSON formatted
        assert '"score": 0.95' in result
        assert '"language": "Python"' in result
        assert '"line": 1' in result

    def test_empty_results(self):
        """Test formatting of empty results."""
        result = format_results_as_json([])
        assert result == "[]"

    def test_mixed_content_formatting(self):
        """Test formatting with both code and regular fields."""
        test_data = [{
            "filename": "test.py",
            "code": "class Example:\n    def method(self):\n        pass",
            "metadata_json": "{\"classes\": [\"Example\"]}",
            "score": 0.8,
            "language": "Python"
        }]

        result = format_results_as_json(test_data)

        # Code should be readable
        assert "class Example:" in result
        assert "def method(self):" in result
        # Metadata JSON should be readable
        assert '"classes": ["Example"]' in result
        # Regular fields should be normal JSON
        assert '"score": 0.8' in result
        assert '"language": "Python"' in result


def run_manual_test():
    """Manual test runner for demonstration purposes."""
    print("🧪 Testing improved JSON formatting:")
    print("=" * 50)

    # Sample data similar to the original example
    test_results = [
        {
            "filename": "cocoindex/python/cocoindex/targets.py",
            "language": "Python",
            "code": "@dataclass\nclass KuzuConnection:\n    \"\"\"Connection spec for Kuzu.\"\"\"\n\n    api_server_url: str\n\n\nclass Kuzu(op.TargetSpec):\n    \"\"\"Graph storage powered by Kuzu.\"\"\"\n\n    connection: AuthEntryReference[KuzuConnection]\n    mapping: Nodes | Relationships",
            "score": 0.2507555036385951,
            "start": {"line": 115, "column": 1, "offset": 2338},
            "end": {"line": 135, "column": 38, "offset": 2795},
            "source": "files",
            "score_type": "vector",
            "functions": [],
            "classes": ["KuzuConnection", "Kuzu", "KuzuDeclaration"],
            "imports": [],
            "complexity_score": 11,
            "has_type_hints": False,
            "has_async": False,
            "has_classes": True,
            "private_methods": [],
            "dunder_methods": [],
            "decorators": [],
            "metadata_json": "{\"language\": \"Python\", \"filename\": \"cocoindex/python/cocoindex/targets.py\", \"line_count\": 21, \"char_count\": 457, \"functions\": [], \"classes\": [\"KuzuConnection\", \"Kuzu\", \"KuzuDeclaration\"]}"
        },
        {
            "filename": "cocoindex/src/ops/functions/split_recursively.rs",
            "language": "Rust",
            "code": "    add_treesitter_language(\n        &mut map,\n        \"Go\",\n        [\".go\", \"golang\"],\n        tree_sitter_go::LANGUAGE,\n        [],\n    );\n    add_treesitter_language(\n        &mut map,\n        \"HTML\",\n        [\".html\", \".htm\"],\n        tree_sitter_html::LANGUAGE,\n        [],\n    );",
            "score": 0.23254502720728532,
            "start": {"line": 129, "column": 1, "offset": 3615},
            "end": {"line": 165, "column": 7, "offset": 4542},
            "source": "files",
            "score_type": "vector"
        }
    ]

    print(format_results_as_json(test_results))


if __name__ == "__main__":
    # Check if pytest is available for proper testing
    try:
        print("✅ Use 'pytest test_json_formatting.py' to run the full test suite")
    except ImportError:
        print("📝 pytest not available, running manual test...")

    run_manual_test()
