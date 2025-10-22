#!/usr/bin/env python3
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 aanno <aanno@users.noreply.github.com>
#
# This file is part of cocoindex_code_mcp_server from
# https://github.com/aanno/cocoindex-code-mcp-server
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Python code analyzer for extracting rich metadata from code chunks.

Enhanced with tree-sitter AST analysis and multi-level fallback strategies.
"""

import ast
import json
import re
from typing import Any, Dict, List, Optional, Set, Union

from . import LOGGER

# Import the new tree-sitter based analyzer
try:
    from .tree_sitter_python_analyzer import create_python_analyzer

    TREE_SITTER_ANALYZER_AVAILABLE = True
except ImportError as e:
    LOGGER.warning("Tree-sitter Python analyzer not available: %s", e)
    TREE_SITTER_ANALYZER_AVAILABLE = False


class PythonCodeAnalyzer:
    """Analyzer for extracting metadata from Python code chunks."""

    def __init__(self) -> None:
        self.max_recursion_depth: int = 200  # Prevent infinite recursion
        self.visited_nodes: Set[int] = set()  # Cycle detection
        self.reset()

    def reset(self) -> None:
        """Reset the analyzer state."""
        self.functions: List[Dict[str, Any]] = []
        self.classes: List[Dict[str, Any]] = []
        self.imports: List[Dict[str, Any]] = []
        self.variables: List[Dict[str, Any]] = []
        self.decorators: Set[str] = set()
        self.complexity_score: float = 0
        self.visited_nodes.clear()  # Clear visited nodes on reset

    def analyze_code(self, code: str, filename: str = "") -> Dict[str, Any]:
        """
        Analyze Python code and extract rich metadata.

        Args:
            code: Python source code to analyze
            filename: Optional filename for context


        Returns:
            Dictionary containing extracted metadata
        """
        self.reset()

        # Add basic size limits to prevent analyzing enormous code chunks
        if len(code) > 100000:  # 100KB limit
            LOGGER.debug("Code chunk too large (%s chars), using fallback analysis", len(code))
            return self._build_fallback_metadata(code, filename)

        try:
            # Parse the code into an AST
            tree = ast.parse(code, filename=filename)

            # Visit all nodes to extract metadata
            self._visit_node(tree)

            # Calculate additional metrics
            self._calculate_metrics(code)

            return self._build_metadata(code, filename)

        except SyntaxError as e:
            # Don't log warnings for syntax errors - this is expected when chunking breaks code
            LOGGER.debug("Syntax error in Python code (using fallback): %s", e)
            return self._build_fallback_metadata(code, filename)
        except Exception as e:
            LOGGER.error("Error analyzing Python code: %s", e)
            return self._build_fallback_metadata(code, filename)

    def _visit_node(self, node: ast.AST, class_context: Optional[str] = None, depth: int = 0) -> None:
        """Recursively visit AST nodes to extract metadata with bounds checking."""
        # Prevent infinite recursion
        if depth > self.max_recursion_depth:
            LOGGER.debug("Max recursion depth %s reached, stopping AST traversal", self.max_recursion_depth)
            return

        # Cycle detection using node memory address
        node_id = id(node)
        if node_id in self.visited_nodes:
            LOGGER.debug("Cycle detected in AST traversal, skipping node")
            return
        self.visited_nodes.add(node_id)

        try:
            if isinstance(node, ast.FunctionDef):
                self._extract_function_info(node, class_context)
            elif isinstance(node, ast.AsyncFunctionDef):
                self._extract_function_info(node, class_context, is_async=True)
            elif isinstance(node, ast.ClassDef):
                self._extract_class_info(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._extract_import_info(node)
            elif isinstance(node, ast.Assign):
                self._extract_variable_info(node, class_context)

            # Recursively visit child nodes with depth tracking
            for child in ast.iter_child_nodes(node):
                if isinstance(node, ast.ClassDef):
                    # Pass class context to child nodes
                    self._visit_node(child, node.name, depth + 1)
                else:
                    self._visit_node(child, class_context, depth + 1)
        finally:
            # Remove from visited set when done (allow revisiting in different contexts)
            self.visited_nodes.discard(node_id)

    def _extract_function_info(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, class_context: Optional[str] = None, is_async: bool = False
    ) -> None:
        """Extract information about function definitions."""
        func_info: Dict[str, Any] = {
            "name": node.name,
            "type": "async_function" if is_async else "method" if class_context else "function",
            "class": class_context,
            "line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno),
            "column": getattr(node, "col_offset", 0) + 1,  # Convert to 1-based
            "end_column": getattr(node, "end_col_offset", 0) + 1,
            "lines_of_code": (node.lineno, getattr(node, "end_lineno", node.lineno)),
            "parameters": [],
            "return_type": None,
            "decorators": [],
            "docstring": ast.get_docstring(node),
            "is_private": node.name.startswith("_"),
            "is_dunder": node.name.startswith("__") and node.name.endswith("__"),
        }

        # Extract parameters
        for arg in node.args.args:
            param_info = {
                "name": arg.arg,
                "type_annotation": self._get_type_annotation(arg.annotation) if arg.annotation else None,
                "default": None,
            }
            func_info["parameters"].append(param_info)

        # Extract default values
        defaults = node.args.defaults
        if defaults:
            # Match defaults to parameters (defaults align to the end of the parameter list)
            param_count = len(func_info["parameters"])
            default_count = len(defaults)
            for i, default in enumerate(defaults):
                param_index = param_count - default_count + i
                if 0 <= param_index < param_count:
                    func_info["parameters"][param_index]["default"] = self._get_ast_value(default, 0)

        # Extract return type annotation
        if node.returns:
            func_info["return_type"] = self._get_type_annotation(node.returns)

        # Extract decorators
        for decorator in node.decorator_list:
            decorator_name = self._get_decorator_name(decorator, 0)
            func_info["decorators"].append(decorator_name)
            self.decorators.add(decorator_name)

        self.functions.append(func_info)

    def _extract_class_info(self, node: ast.ClassDef) -> None:
        """Extract information about class definitions."""
        class_info = {
            "name": node.name,
            "line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno),
            "column": getattr(node, "col_offset", 0) + 1,  # Convert to 1-based
            "end_column": getattr(node, "end_col_offset", 0) + 1,
            "lines_of_code": (node.lineno, getattr(node, "end_lineno", node.lineno)),
            "bases": [self._get_type_annotation(base) for base in node.bases],
            "decorators": [self._get_decorator_name(dec, 0) for dec in node.decorator_list],
            "docstring": ast.get_docstring(node),
            "methods": [],
            "is_private": node.name.startswith("_"),
        }

        # Add class decorators to global decorators list (like function decorators)
        for decorator in node.decorator_list:
            decorator_name = self._get_decorator_name(decorator, 0)
            self.decorators.add(decorator_name)

        # Extract methods (will be added by function extraction with class context)
        self.classes.append(class_info)

    def _extract_import_info(self, node: ast.AST) -> None:
        """Extract import information."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_info = {"module": alias.name, "alias": alias.asname, "type": "import", "line": node.lineno}
                self.imports.append(import_info)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                import_info = {
                    "module": module,
                    "name": alias.name,
                    "alias": alias.asname,
                    "type": "from_import",
                    "line": node.lineno,
                    "level": node.level,  # For relative imports
                }
                self.imports.append(import_info)

    def _extract_variable_info(self, node: ast.Assign, class_context: Optional[str] = None) -> None:
        """Extract variable assignment information."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_info = {
                    "name": target.id,
                    "class": class_context,
                    "line": node.lineno,
                    "type": "class_variable" if class_context else "variable",
                    "is_private": target.id.startswith("_"),
                }
                self.variables.append(var_info)

    def _get_type_annotation(self, annotation: ast.AST, depth: int = 0) -> str:
        """Convert AST type annotation to string with recursion protection."""
        # Prevent infinite recursion in type annotations
        if depth > 10:
            return "Complex"

        try:
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Constant):
                return repr(annotation.value)
            elif isinstance(annotation, ast.Attribute):
                return f"{self._get_type_annotation(annotation.value, depth + 1)}.{annotation.attr}"
            elif isinstance(annotation, ast.Subscript):
                value = self._get_type_annotation(annotation.value, depth + 1)
                slice_val = self._get_type_annotation(annotation.slice, depth + 1)
                return f"{value}[{slice_val}]"
            elif isinstance(annotation, ast.Tuple):
                elements = [self._get_type_annotation(elt, depth + 1) for elt in annotation.elts[:5]]  # Limit elements
                return f"({', '.join(elements)})"
            else:
                return ast.unparse(annotation)[:100]  # Limit length
        except Exception:
            return "Unknown"

    def _get_decorator_name(self, decorator: ast.AST, depth: int = 0) -> str:
        """Extract decorator name with recursion protection."""
        if depth > 5:  # Limit decorator complexity
            return "complex_decorator"

        try:
            if isinstance(decorator, ast.Name):
                return decorator.id
            elif isinstance(decorator, ast.Attribute):
                return f"{self._get_type_annotation(decorator.value, depth + 1)}.{decorator.attr}"
            elif isinstance(decorator, ast.Call):
                return self._get_decorator_name(decorator.func, depth + 1)
            else:
                return ast.unparse(decorator)[:50]  # Limit length
        except Exception:
            return "unknown_decorator"

    def _get_ast_value(self, node: ast.AST, depth: int = 0) -> Any:
        """Extract value from AST node with recursion protection."""
        if depth > 10:  # Prevent deep recursion in value extraction
            return "complex_value"

        try:
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.List):
                # Limit list size to prevent huge structures
                return [self._get_ast_value(elt, depth + 1) for elt in node.elts[:10]]
            elif isinstance(node, ast.Dict):
                # Limit dict size
                items = list(zip(node.keys, node.values))[:5]
                return {
                    self._get_ast_value(k, depth + 1) if k else "None": (
                        self._get_ast_value(v, depth + 1) if v else "None"
                    )
                    for k, v in items
                }
            else:
                return ast.unparse(node)[:100]  # Limit length
        except Exception:
            return "unknown_value"

    def _calculate_metrics(self, code: str) -> None:
        """Calculate code complexity and other metrics."""
        # Simple complexity metrics
        self.complexity_score = (
            len(self.functions) * 2
            + len(self.classes) * 3
            + code.count("if ")
            + code.count("for ")
            + code.count("while ")
            + code.count("try:")
            + code.count("except")
            + len([f for f in self.functions if f["decorators"]])
        )

    def _build_metadata(self, code: str, filename: str) -> Dict[str, Any]:
        """Build the final metadata dictionary."""
        import hashlib
        import json

        # Extract unique module names from imports
        imported_modules = list(set([imp["module"] for imp in self.imports if imp["module"] and imp["module"] != ""]))

        # Group functions by class
        class_methods: Dict[str, List[Dict[str, Any]]] = {}
        standalone_functions: List[Dict[str, Any]] = []

        for func in self.functions:
            if func["class"]:
                if func["class"] not in class_methods:
                    class_methods[func["class"]] = []
                class_methods[func["class"]].append(func)
            else:
                standalone_functions.append(func)

        # Calculate content hash for unique identification
        content_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]

        # Build node relationships
        node_relationships = self._build_node_relationships()

        # Enhanced metadata following RAG-pychunk recommendations
        metadata = {
            # Basic info (RAG-pychunk: file, node_type)
            "language": "Python",
            "filename": filename,
            "file": filename,  # RAG-pychunk compatibility
            "node_type": "MODULE",  # RAG-pychunk compatibility
            "line_count": len(code.split("\n")),
            "char_count": len(code),
            # Position information (RAG-pychunk: lines_of_code)
            "lines_of_code": (1, len(code.split("\n"))),
            "start_line": 1,
            "end_line": len(code.split("\n")),
            "start_column": 1,
            "end_column": len(code.split("\n")[-1]) if code.split("\n") else 1,
            # Unique identifier (RAG-pychunk: hash)
            "hash": content_hash,
            "content_hash": content_hash,
            # Node relationships (RAG-pychunk: node_relationships)
            "node_relationships": node_relationships,
            # Functions and methods
            "functions": [f["name"] for f in standalone_functions],
            "function_details": standalone_functions,
            "method_count": len([f for f in self.functions if f["class"]]),
            # Classes
            "classes": [c["name"] for c in self.classes],
            "class_details": self.classes,
            "class_methods": class_methods,
            # Imports
            "imports": imported_modules,
            "import_details": self.imports,
            # Variables
            "variables": [v["name"] for v in self.variables if not v["class"]],
            "class_variables": [v["name"] for v in self.variables if v["class"]],
            # Decorators
            "decorators": list(set(self.decorators)),
            # Complexity
            "complexity_score": self.complexity_score,
            "has_async": any(f["type"] == "async_function" for f in self.functions),
            "has_classes": len(self.classes) > 0,
            "has_decorators": len(self.decorators) > 0,
            # Code patterns
            "has_docstrings": any(f.get("docstring") for f in self.functions + self.classes),
            "has_type_hints": any(
                f.get("return_type") or any(p.get("type_annotation") for p in f.get("parameters", []))
                for f in self.functions
            ),
            "private_methods": [f["name"] for f in self.functions if f["is_private"]],
            "dunder_methods": [f["name"] for f in self.functions if f["is_dunder"]],
            # Promoted metadata fields for database columns
            "analysis_method": "python_code_analyzer",
            # don't set chunking method in analyzer
            # "chunking_method": "ast_tree_sitter",
            # "tree_sitter_chunking_error": False,
            "tree_sitter_analyze_error": False,
            "decorators_used": list(set(self.decorators)),
            # Additional metadata (RAG-pychunk: additional_metadata)
            "additional_metadata": {
                "analysis_method": "python_ast",
                "parser_version": "3.x",
                "extracted_at": self._get_timestamp(),
                "total_functions": len(self.functions),
                "total_classes": len(self.classes),
                "total_imports": len(self.imports),
                "code_patterns": {
                    "async_functions": len([f for f in self.functions if f["type"] == "async_function"]),
                    "private_elements": len([f for f in self.functions if f["is_private"]])
                    + len([c for c in self.classes if c["is_private"]]),
                    "documented_elements": len([f for f in self.functions if f.get("docstring")])
                    + len([c for c in self.classes if c.get("docstring")]),
                },
            },
        }

        # Add metadata_json field for compatibility
        metadata["metadata_json"] = json.dumps(metadata, default=str)

        return metadata

    def _build_node_relationships(self) -> Dict[str, Any]:
        """Build node relationships mapping parent-child and scope relationships."""
        relationships: Dict[str, Any] = {
            "parent": None,  # Module level has no parent
            "children": [],
            "scope": "module",
            "contains": {
                "functions": [f["name"] for f in self.functions if not f["class"]],
                "classes": [c["name"] for c in self.classes],
                "imports": [imp["module"] for imp in self.imports if imp["module"]],
                "variables": [v["name"] for v in self.variables if not v["class"]],
            },
            "class_hierarchies": {},
            "import_dependencies": [],
        }

        # Build class hierarchies
        for class_info in self.classes:
            class_name = class_info["name"]
            relationships["class_hierarchies"][class_name] = {
                "bases": class_info.get("bases", []),
                "methods": [f["name"] for f in self.functions if f.get("class") == class_name],
                "variables": [v["name"] for v in self.variables if v.get("class") == class_name],
            }

        # Build import dependencies
        for imp in self.imports:
            if imp["module"]:
                dep = {"module": imp["module"], "type": imp["type"]}
                if imp.get("name"):
                    dep["imports"] = [imp["name"]]
                relationships["import_dependencies"].append(dep)

        return relationships

    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _build_fallback_metadata(self, code: str, filename: str) -> Dict[str, Any]:
        """Build basic metadata when AST parsing fails."""
        # Use regex fallback for basic function detection
        function_names = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
        class_names = re.findall(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]", code)
        import_matches = re.findall(r"(?:from\s+(\S+)\s+)?import\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)

        # Extract imports more carefully
        imports = []
        for match in import_matches:
            if match[0]:  # from X import Y
                imports.append(match[0])
            else:  # import Y
                imports.append(match[1])

        # Basic complexity estimation
        complexity = (
            len(function_names)
            + len(class_names)
            + code.count("if ")
            + code.count("for ")
            + code.count("while ")
            + code.count("try:")
            + code.count("except")
        )

        # Check for patterns
        has_async = "async def" in code
        has_type_hints = "->" in code or ":" in code  # Basic heuristic
        has_classes = len(class_names) > 0

        fallback_metadata = {
            "language": "Python",
            "filename": filename,
            "line_count": len(code.split("\n")),
            "char_count": len(code),
            "functions": function_names,
            "classes": class_names,
            "imports": imports,
            "complexity_score": complexity,
            "has_type_hints": has_type_hints,
            "has_async": has_async,
            "has_classes": has_classes,
            "decorators_used": [],  # Could regex for @decorator if needed
            "analysis_method": "python_regex_fallback",
        }

        # Add metadata_json field for compatibility
        import json

        fallback_metadata["metadata_json"] = json.dumps(fallback_metadata, default=str)

        return fallback_metadata


def analyze_python_code(code: str, filename: str = "") -> Union[Dict[str, Any], None]:
    """
    Enhanced Python code analysis with tree-sitter support and fallback strategies.

    This function now uses the TreeSitterPythonAnalyzer which provides:
    - Tree-sitter AST analysis for better structure understanding
    - Python AST analysis for detailed semantic information
    - Multi-level fallback strategies for robustness
    - Enhanced metadata extraction with position information

    Args:
        code: Python source code
        filename: Optional filename for context

    Returns:
        Dictionary containing extracted metadata with enhanced information
    """
    if TREE_SITTER_ANALYZER_AVAILABLE:
        # Use the enhanced tree-sitter based analyzer
        analyzer = create_python_analyzer(prefer_tree_sitter=True)
        metadata = analyzer.analyze_code(code, filename)
        LOGGER.debug("Use the enhanced tree-sitter based analyzer for %s: %s", filename, metadata)

        # Ensure metadata_json field for compatibility with existing code
        if metadata and "metadata_json" not in metadata:
            metadata["metadata_json"] = json.dumps(metadata, default=str)

        return metadata
    else:
        # Fallback to the enhanced legacy analyzer (not tree-sitter, but still enhanced)
        LOGGER.info("Using enhanced Python AST analyzer (tree-sitter not available)")
        fallback_analyzer: Any = PythonCodeAnalyzer()
        return fallback_analyzer.analyze_code(code, filename)


if __name__ == "__main__":
    print("Python code analyzer module - use tests/test_python_analyzer_integration.py for testing")
