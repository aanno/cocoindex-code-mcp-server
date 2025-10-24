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
Tree-sitter based Python code analyzer.

Combines the generic AST visitor framework with Python-specific node handlers.
"""

import ast
from ast import AST, expr
from typing import Any, Dict, List, Optional, Set, Union

from ...ast_visitor import (
    ASTParserFactory,
    GenericMetadataVisitor,
    MultiLevelAnalyzer,
    TreeWalker,
)
from ...language_handlers import get_handler_for_language
from ...parser_util import update_defaults
from . import LOGGER


class TreeSitterPythonAnalyzer:
    """
    Enhanced Python code analyzer using tree-sitter with fallback to Python AST.
    Provides comprehensive metadata extraction with multiple analysis strategies.
    """

    def __init__(self, prefer_tree_sitter: bool = True) -> None:
        """
        Initialize the analyzer.

        Args:
            prefer_tree_sitter: Whether to prefer tree-sitter over Python AST
        """
        self.prefer_tree_sitter = prefer_tree_sitter
        self.multilevel_analyzer = MultiLevelAnalyzer()
        self.parser_factory = ASTParserFactory()

    def analyze_code(self, code: str, filename: str = "") -> Union[Dict[str, Any], None]:
        """
        Analyze Python code and extract comprehensive metadata.

        Args:
            code: Python source code to analyze
            filename: Optional filename for context

        Returns:
            Dictionary containing extracted metadata
        """
        if not code.strip():
            return self._empty_metadata(filename)

        # Try multiple analysis strategies
        metadata = self._try_enhanced_python_analysis(code, filename)

        if not metadata or metadata.get("analysis_method") == "basic_text":
            # Fallback to enhanced Python AST analyzer
            try:
                from ...lang.python.python_code_analyzer import PythonCodeAnalyzer

                LOGGER.debug("Using enhanced PythonCodeAnalyzer for detailed analysis (bad) for %s", filename)
                fallback_analyzer = PythonCodeAnalyzer()
                fallback_metadata = fallback_analyzer.analyze_code(code, filename)
                if fallback_metadata:
                    metadata = fallback_metadata
            except ImportError:
                # Last resort: use multi-level analyzer
                LOGGER.debug("Using MultiLevelAnalyzer as fallback for Python code (very bad) for %s", filename)
                fallback_metadata = self.multilevel_analyzer.analyze_code(code, "python", filename)
                if fallback_metadata:
                    metadata = fallback_metadata

        # Ensure we have all expected fields
        return self._normalize_metadata(metadata, code, filename)

    def _try_enhanced_python_analysis(self, code: str, filename: str) -> Optional[Dict[str, Any]]:
        """Try enhanced Python analysis with tree-sitter + Python AST."""

        if self.prefer_tree_sitter:
            # Strategy 1: Tree-sitter analysis
            ts_metadata = self._try_tree_sitter_analysis(code, filename)
            LOGGER.debug("Strategy 1: Tree-sitter analysis result for %s: %s", filename, ts_metadata)
            if ts_metadata and ts_metadata.get("analysis_method") == "tree_sitter":
                # Enhance with Python AST for better semantic analysis
                ast_metadata = self._try_python_ast_analysis(code, filename)
                if ast_metadata:
                    LOGGER.debug("Enhanced tree-sitter metadata with Python AST for %s: %s", filename, ast_metadata)
                    return self._merge_metadata(ts_metadata, ast_metadata)
                return ts_metadata

        # Strategy 2: Python AST analysis (primary or fallback)
        ast_metadata = self._try_python_ast_analysis(code, filename)
        if ast_metadata:
            if self.prefer_tree_sitter:
                # Try to enhance with tree-sitter position info
                ts_metadata = self._try_tree_sitter_analysis(code, filename)
                LOGGER.debug("Strategy 2: Python AST analysis result for %s: %s", filename, ast_metadata)
                if ts_metadata:
                    return self._merge_metadata(ast_metadata, ts_metadata)
            return ast_metadata

        return None

    def _try_tree_sitter_analysis(self, code: str, filename: str) -> Optional[Dict[str, Any]]:
        """Try tree-sitter based analysis."""
        try:
            # Use the parser factory to get a Python parser
            tree = self.parser_factory.parse_code(code, "python")
            if not tree:
                return None

            # Create a visitor with Python handler
            visitor = GenericMetadataVisitor("python")
            python_handler = get_handler_for_language("python")

            if python_handler:
                visitor.add_handler(python_handler)

            # Walk the tree
            walker = TreeWalker(code, tree)
            metadata = walker.walk(visitor)

            # Get handler summary if available
            if python_handler and hasattr(python_handler, "get_summary"):
                handler_summary = python_handler.get_summary()
                metadata.update(handler_summary)

            metadata["analysis_method"] = "tree_sitter"
            metadata["language"] = "python"

            return metadata

        except Exception as e:
            LOGGER.warning("Tree-sitter Python analysis failed: %s", e)
            return None

    def _try_python_ast_analysis(self, code: str, filename: str) -> Optional[Dict[str, Any]]:
        """Try enhanced Python AST analysis for detailed semantic information."""
        try:
            # Use the enhanced PythonCodeAnalyzer instead of the simple visitor
            from ...lang.python.python_code_analyzer import PythonCodeAnalyzer

            analyzer = PythonCodeAnalyzer()
            metadata = analyzer.analyze_code(code, filename)

            if metadata:
                # Ensure analysis method is set correctly
                metadata["analysis_method"] = "python_code_analyzer"
                return metadata
            else:
                return None

        except ImportError:
            # Fallback to simple AST visitor if import fails
            try:
                tree = ast.parse(code, filename=filename)

                # Create a Python AST visitor
                visitor = PythonASTVisitor()
                visitor.visit(tree)

                metadata = visitor.get_metadata()
                # metadata.update({
                update_defaults(
                    metadata,
                    {
                        "analysis_method": "python_ast",
                        "language": "python",
                        "filename": filename,
                        "line_count": len(code.split("\n")),
                        "char_count": len(code),
                        # Promoted metadata fields for database columns
                        # "analysis_method": "tree_sitter_python_analyzer",
                        # don't set chunking method in analyzer
                        # "chunking_method": "ast_tree_sitter",
                        # "tree_sitter_chunking_error": False,
                        "tree_sitter_analyze_error": False,
                        "decorators_used": metadata.get("decorators", []),
                    },
                )

                return metadata

            except Exception as e:
                LOGGER.warning("Python AST analysis failed: %s", e)
                return None
        except SyntaxError as e:
            LOGGER.warning("Python AST parsing failed (syntax error): %s", e)
            return None
        except Exception as e:
            LOGGER.warning("Python AST analysis failed: %s", e)
            return None

    def _merge_metadata(self, primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
        """Merge metadata from two analysis methods."""
        merged = primary.copy()

        # Merge lists and sets
        for key in ["functions", "classes", "imports", "variables", "decorators"]:
            if key in secondary:
                primary_items = set(merged.get(key, []))
                secondary_items = set(secondary.get(key, []))
                merged[key] = list(primary_items | secondary_items)

        # Take the higher complexity score
        if "complexity_score" in secondary:
            merged["complexity_score"] = max(merged.get("complexity_score", 0), secondary["complexity_score"])

        # Merge boolean flags (OR operation)
        for key in ["has_async", "has_classes", "has_decorators", "has_type_hints", "has_docstrings"]:
            if key in secondary:
                merged[key] = merged.get(key, False) or secondary.get(key, False)

        # Merge class_details with decorators from both sources
        if "class_details" in secondary:
            # Create a mapping of class names to details for easier merging
            primary_classes = {cls.get("name"): cls for cls in merged.get("class_details", [])}
            secondary_classes = {cls.get("name"): cls for cls in secondary["class_details"]}

            merged_class_details = []
            for class_name in set(primary_classes.keys()) | set(secondary_classes.keys()):
                primary_cls = primary_classes.get(class_name, {})
                secondary_cls = secondary_classes.get(class_name, {})

                # Start with the more detailed one (usually secondary from Python AST)
                if len(secondary_cls) > len(primary_cls):
                    merged_cls = secondary_cls.copy()
                    # Add any missing fields from primary
                    for k, v in primary_cls.items():
                        if k not in merged_cls:
                            merged_cls[k] = v
                else:
                    merged_cls = primary_cls.copy()
                    # Merge important fields from secondary (especially decorators)
                    for k, v in secondary_cls.items():
                        if k == "decorators" and v:  # Prefer non-empty decorators
                            merged_cls[k] = v
                        elif k not in merged_cls:
                            merged_cls[k] = v

                merged_class_details.append(merged_cls)

            merged["class_details"] = merged_class_details

        # Handle function_details and import_details as before
        for key in ["function_details", "import_details"]:
            if key in secondary and len(secondary[key]) > len(merged.get(key, [])):
                merged[key] = secondary[key]

        # Combine analysis methods
        primary_method = merged.get("analysis_method", "unknown")
        secondary_method = secondary.get("analysis_method", "unknown")
        merged["analysis_method"] = f"{primary_method}+{secondary_method}"

        return merged

    def _normalize_metadata(self, metadata: Union[Dict[str, Any], None], code: str, filename: str) -> Dict[str, Any]:
        """Ensure metadata has all expected fields without overriding enhanced metadata."""
        base_metadata = {
            "language": "Python",
            "filename": filename,
            "line_count": len(code.split("\n")),
            "char_count": len(code),
            "functions": [],
            "classes": [],
            "imports": [],
            "variables": [],
            "decorators": [],
            "complexity_score": 0,
            "has_async": False,
            "has_classes": False,
            "has_decorators": False,
            "has_type_hints": False,
            "has_docstrings": False,
            "private_methods": [],
            "dunder_methods": [],
            "function_details": [],
            "class_details": [],
            "import_details": [],
            "analysis_method": "unknown",
        }

        # Only add base fields that are missing, don't override enhanced ones
        if metadata:
            result = metadata.copy()
            for key, value in base_metadata.items():
                if key not in result:
                    result[key] = value
            return result
        else:
            return base_metadata

    def _empty_metadata(self, filename: str) -> Dict[str, Any]:
        """Return empty metadata structure."""
        return self._normalize_metadata({}, "", filename)


class PythonASTVisitor(ast.NodeVisitor):
    """
    Python AST visitor for extracting detailed semantic information.
    Complements tree-sitter analysis with Python-specific semantics.
    """

    def __init__(self) -> None:
        self.functions: List[Dict[str, Any]] = []
        self.classes: List[Dict[str, Any]] = []
        self.imports: List[Dict[str, Any]] = []
        self.variables: List[str] = []
        self.decorators: Set[str] = set()
        self.complexity_score: float = 0
        self.current_class: Optional[str] = None
        self.scope_stack: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        self._process_function(node, is_async=False)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions."""
        self._process_function(node, is_async=True)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        self._process_class(node)

        # Enter class scope
        old_class = self.current_class
        self.current_class = node.name
        self.scope_stack.append(f"class:{node.name}")

        self.generic_visit(node)

        # Exit class scope
        self.current_class = old_class
        self.scope_stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            self.imports.append(
                {"module": alias.name, "alias": alias.asname, "is_from_import": False, "line": node.lineno}
            )

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from...import statements."""
        module = node.module or ""
        for alias in node.names:
            self.imports.append(
                {
                    "module": module,
                    "name": alias.name,
                    "alias": alias.asname,
                    "is_from_import": True,
                    "line": node.lineno,
                }
            )

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment statements."""
        # Only track module-level variables
        if len(self.scope_stack) == 0:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.variables.append(target.id)

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit annotated assignment statements."""
        # Only track module-level variables
        if len(self.scope_stack) == 0 and isinstance(node.target, ast.Name):
            self.variables.append(node.target.id)

        self.generic_visit(node)

    # Control flow nodes for complexity calculation
    def visit_If(self, node: ast.If) -> None:
        self.complexity_score += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity_score += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity_score += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.complexity_score += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self.complexity_score += 1
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.complexity_score += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.complexity_score += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.complexity_score += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.complexity_score += 1
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.complexity_score += 1
        self.generic_visit(node)

    def _process_function(self, node, is_async: bool = False):
        """Process function definition node."""
        func_info = {
            "name": node.name,
            "line": node.lineno,
            "is_async": is_async,
            "is_private": node.name.startswith("_"),
            "is_dunder": node.name.startswith("__") and node.name.endswith("__"),
            "decorators": [self._get_decorator_name(dec) for dec in node.decorator_list],
            "parameters": self._extract_parameters(node),
            "return_type": self._get_annotation_name(node.returns) if node.returns else None,
            "docstring": ast.get_docstring(node),
            "class": self.current_class,
        }

        self.functions.append(func_info)

        # Track decorators
        for dec_name in func_info["decorators"]:
            self.decorators.add(dec_name)

        # Add to class methods if in class scope
        if self.current_class:
            for cls_info in self.classes:
                if cls_info["name"] == self.current_class:
                    cls_info.setdefault("methods", []).append(node.name)
                    break

    def _process_class(self, node):
        """Process class definition node."""
        class_info = {
            "name": node.name,
            "line": node.lineno,
            "is_private": node.name.startswith("_"),
            "bases": [self._get_annotation_name(base) for base in node.bases],
            "decorators": [self._get_decorator_name(dec) for dec in node.decorator_list],
            "docstring": ast.get_docstring(node),
            "methods": [],
        }

        self.classes.append(class_info)

        # Track decorators
        for dec_name in class_info["decorators"]:
            self.decorators.add(dec_name)

    def _extract_parameters(self, node) -> List[Dict[str, Any]]:
        """Extract function parameters."""
        parameters = []

        args = node.args

        # Regular arguments
        for i, arg in enumerate(args.args):
            param_info = {
                "name": arg.arg,
                "type_annotation": self._get_annotation_name(arg.annotation) if arg.annotation else None,
                "default": None,
            }

            # Check for default values
            defaults_start = len(args.args) - len(args.defaults)
            if i >= defaults_start:
                default_index = i - defaults_start
                param_info["default"] = self._get_default_value(args.defaults[default_index])

            parameters.append(param_info)

        # *args
        if args.vararg:
            parameters.append(
                {
                    "name": f"*{args.vararg.arg}",
                    "type_annotation": (
                        self._get_annotation_name(args.vararg.annotation) if args.vararg.annotation else None
                    ),
                    "default": None,
                }
            )

        # Keyword-only arguments
        for i, arg in enumerate(args.kwonlyargs):
            param_info = {
                "name": arg.arg,
                "type_annotation": self._get_annotation_name(arg.annotation) if arg.annotation else None,
                "default": None,
            }

            # Check for default values in kw_defaults
            if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
                param_info["default"] = self._get_default_value(args.kw_defaults[i])

            parameters.append(param_info)

        # **kwargs
        if args.kwarg:
            parameters.append(
                {
                    "name": f"**{args.kwarg.arg}",
                    "type_annotation": (
                        self._get_annotation_name(args.kwarg.annotation) if args.kwarg.annotation else None
                    ),
                    "default": None,
                }
            )

        return parameters

    def _get_decorator_name(self, decorator: expr) -> str:
        """Extract decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_annotation_name(decorator.value)}.{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        else:
            return ast.unparse(decorator)

    def _get_annotation_name(self, annotation: expr) -> Union[str, None]:
        """Extract type annotation name."""
        if annotation is None:
            return None

        try:
            return ast.unparse(annotation)
        except BaseException:
            return str(annotation)

    def _get_default_value(self, default: AST) -> str:
        """Extract default parameter value."""
        try:
            return ast.unparse(default)
        except BaseException:
            return str(default)

    def get_metadata(self) -> Dict[str, Any]:
        """Get extracted metadata."""
        # Extract unique modules from imports
        imported_modules = list(set([imp["module"] for imp in self.imports if imp["module"] and imp["module"] != ""]))

        return {
            "functions": [f["name"] for f in self.functions if not f["class"]],
            "classes": [c["name"] for c in self.classes],
            "imports": imported_modules,
            "variables": list(set(self.variables)),
            "decorators": list(self.decorators),
            "complexity_score": self.complexity_score,
            "has_async": any(f["is_async"] for f in self.functions),
            "has_classes": len(self.classes) > 0,
            "has_decorators": len(self.decorators) > 0,
            "has_type_hints": any(
                f["return_type"] or any(p.get("type_annotation") for p in f["parameters"]) for f in self.functions
            ),
            "has_docstrings": any(f.get("docstring") or f.get("has_docstring") for f in self.functions + self.classes),
            "private_methods": [f["name"] for f in self.functions if f["is_private"]],
            "dunder_methods": [f["name"] for f in self.functions if f["is_dunder"]],
            "function_details": self.functions,
            "class_details": self.classes,
            "import_details": self.imports,
        }


# Convenience function to create analyzer
def create_python_analyzer(prefer_tree_sitter: bool = True) -> TreeSitterPythonAnalyzer:
    """Create a new Python analyzer instance."""
    return TreeSitterPythonAnalyzer(prefer_tree_sitter=prefer_tree_sitter)


# Convenience function for direct analysis
def analyze_python_code(code: str, filename: str = "", prefer_tree_sitter: bool = True) -> Union[Dict[str, Any], None]:
    """Analyze Python code with the enhanced analyzer."""
    analyzer = create_python_analyzer(prefer_tree_sitter=prefer_tree_sitter)
    return analyzer.analyze_code(code, filename)
