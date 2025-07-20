#!/usr/bin/env python3

"""
Python code analyzer for extracting rich metadata from code chunks.
Enhanced with tree-sitter AST analysis and multi-level fallback strategies.
"""

import ast
import re
import json
from typing import List, Dict, Any, Optional, Set
import logging
from __init__ import LOGGER

# Import the new tree-sitter based analyzer
try:
    from tree_sitter_python_analyzer import TreeSitterPythonAnalyzer, create_python_analyzer
    TREE_SITTER_ANALYZER_AVAILABLE = True
except ImportError as e:
    LOGGER.warning(f"Tree-sitter Python analyzer not available: {e}")
    TREE_SITTER_ANALYZER_AVAILABLE = False

class PythonCodeAnalyzer:
    """Analyzer for extracting metadata from Python code chunks."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the analyzer state."""
        self.functions = []
        self.classes = []
        self.imports = []
        self.variables = []
        self.decorators = []
        self.complexity_score = 0
    
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
        
        try:
            # Parse the code into an AST
            tree = ast.parse(code, filename=filename)
            
            # Visit all nodes to extract metadata
            self._visit_node(tree)
            
            # Calculate additional metrics
            self._calculate_metrics(code)
            
            return self._build_metadata(code, filename)
            
        except SyntaxError as e:
            LOGGER.warning(f"Syntax error in Python code: {e}")
            return self._build_fallback_metadata(code, filename)
        except Exception as e:
            LOGGER.error(f"Error analyzing Python code: {e}")
            return self._build_fallback_metadata(code, filename)
    
    def _visit_node(self, node: ast.AST, class_context: str = None):
        """Recursively visit AST nodes to extract metadata."""
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
        
        # Recursively visit child nodes
        for child in ast.iter_child_nodes(node):
            if isinstance(node, ast.ClassDef):
                # Pass class context to child nodes
                self._visit_node(child, node.name)
            else:
                self._visit_node(child, class_context)
    
    def _extract_function_info(self, node: ast.FunctionDef, class_context: str = None, is_async: bool = False):
        """Extract information about function definitions."""
        func_info = {
            "name": node.name,
            "type": "async_function" if is_async else "method" if class_context else "function",
            "class": class_context,
            "line": node.lineno,
            "parameters": [],
            "return_type": None,
            "decorators": [],
            "docstring": ast.get_docstring(node),
            "is_private": node.name.startswith('_'),
            "is_dunder": node.name.startswith('__') and node.name.endswith('__'),
        }
        
        # Extract parameters
        for arg in node.args.args:
            param_info = {
                "name": arg.arg,
                "type_annotation": self._get_type_annotation(arg.annotation) if arg.annotation else None,
                "default": None
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
                    func_info["parameters"][param_index]["default"] = self._get_ast_value(default)
        
        # Extract return type annotation
        if node.returns:
            func_info["return_type"] = self._get_type_annotation(node.returns)
        
        # Extract decorators
        for decorator in node.decorator_list:
            decorator_name = self._get_decorator_name(decorator)
            func_info["decorators"].append(decorator_name)
            self.decorators.append(decorator_name)
        
        self.functions.append(func_info)
    
    def _extract_class_info(self, node: ast.ClassDef):
        """Extract information about class definitions."""
        class_info = {
            "name": node.name,
            "line": node.lineno,
            "bases": [self._get_type_annotation(base) for base in node.bases],
            "decorators": [self._get_decorator_name(dec) for dec in node.decorator_list],
            "docstring": ast.get_docstring(node),
            "methods": [],
            "is_private": node.name.startswith('_'),
        }
        
        # Extract methods (will be added by function extraction with class context)
        self.classes.append(class_info)
    
    def _extract_import_info(self, node: ast.AST):
        """Extract import information."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_info = {
                    "module": alias.name,
                    "alias": alias.asname,
                    "type": "import",
                    "line": node.lineno
                }
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
                    "level": node.level  # For relative imports
                }
                self.imports.append(import_info)
    
    def _extract_variable_info(self, node: ast.Assign, class_context: str = None):
        """Extract variable assignment information."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_info = {
                    "name": target.id,
                    "class": class_context,
                    "line": node.lineno,
                    "type": "class_variable" if class_context else "variable",
                    "is_private": target.id.startswith('_'),
                }
                self.variables.append(var_info)
    
    def _get_type_annotation(self, annotation: ast.AST) -> str:
        """Convert AST type annotation to string."""
        try:
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Constant):
                return repr(annotation.value)
            elif isinstance(annotation, ast.Attribute):
                return f"{self._get_type_annotation(annotation.value)}.{annotation.attr}"
            elif isinstance(annotation, ast.Subscript):
                value = self._get_type_annotation(annotation.value)
                slice_val = self._get_type_annotation(annotation.slice)
                return f"{value}[{slice_val}]"
            elif isinstance(annotation, ast.Tuple):
                elements = [self._get_type_annotation(elt) for elt in annotation.elts]
                return f"({', '.join(elements)})"
            else:
                return ast.unparse(annotation)
        except Exception:
            return "Unknown"
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Extract decorator name."""
        try:
            if isinstance(decorator, ast.Name):
                return decorator.id
            elif isinstance(decorator, ast.Attribute):
                return f"{self._get_type_annotation(decorator.value)}.{decorator.attr}"
            elif isinstance(decorator, ast.Call):
                return self._get_decorator_name(decorator.func)
            else:
                return ast.unparse(decorator)
        except Exception:
            return "unknown_decorator"
    
    def _get_ast_value(self, node: ast.AST) -> Any:
        """Extract value from AST node."""
        try:
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.List):
                return [self._get_ast_value(elt) for elt in node.elts]
            elif isinstance(node, ast.Dict):
                return {
                    self._get_ast_value(k): self._get_ast_value(v) 
                    for k, v in zip(node.keys, node.values)
                }
            else:
                return ast.unparse(node)
        except Exception:
            return "unknown_value"
    
    def _calculate_metrics(self, code: str):
        """Calculate code complexity and other metrics."""
        # Simple complexity metrics
        self.complexity_score = (
            len(self.functions) * 2 +
            len(self.classes) * 3 +
            code.count('if ') +
            code.count('for ') +
            code.count('while ') +
            code.count('try:') +
            code.count('except') +
            len([f for f in self.functions if f['decorators']])
        )
    
    def _build_metadata(self, code: str, filename: str) -> Dict[str, Any]:
        """Build the final metadata dictionary."""
        # Extract unique module names from imports
        imported_modules = list(set([
            imp['module'] for imp in self.imports 
            if imp['module'] and imp['module'] != ''
        ]))
        
        # Group functions by class
        class_methods = {}
        standalone_functions = []
        
        for func in self.functions:
            if func['class']:
                if func['class'] not in class_methods:
                    class_methods[func['class']] = []
                class_methods[func['class']].append(func)
            else:
                standalone_functions.append(func)
        
        return {
            # Basic info
            "language": "Python",
            "filename": filename,
            "line_count": len(code.split('\n')),
            "char_count": len(code),
            
            # Functions and methods
            "functions": [f['name'] for f in standalone_functions],
            "function_details": standalone_functions,
            "method_count": len([f for f in self.functions if f['class']]),
            
            # Classes
            "classes": [c['name'] for c in self.classes],
            "class_details": self.classes,
            "class_methods": class_methods,
            
            # Imports
            "imports": imported_modules,
            "import_details": self.imports,
            
            # Variables
            "variables": [v['name'] for v in self.variables if not v['class']],
            "class_variables": [v['name'] for v in self.variables if v['class']],
            
            # Decorators
            "decorators": list(set(self.decorators)),
            
            # Complexity
            "complexity_score": self.complexity_score,
            "has_async": any(f['type'] == 'async_function' for f in self.functions),
            "has_classes": len(self.classes) > 0,
            "has_decorators": len(self.decorators) > 0,
            
            # Code patterns
            "has_docstrings": any(f.get('docstring') for f in self.functions + self.classes),
            "has_type_hints": any(f.get('return_type') or any(p.get('type_annotation') for p in f.get('parameters', [])) for f in self.functions),
            "private_methods": [f['name'] for f in self.functions if f['is_private']],
            "dunder_methods": [f['name'] for f in self.functions if f['is_dunder']],
        }
    
    def _build_fallback_metadata(self, code: str, filename: str) -> Dict[str, Any]:
        """Build basic metadata when AST parsing fails."""
        # Use regex fallback for basic function detection
        function_names = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        class_names = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]', code)
        import_matches = re.findall(r'(?:from\s+(\S+)\s+)?import\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        
        return {
            "language": "Python",
            "filename": filename,
            "line_count": len(code.split('\n')),
            "char_count": len(code),
            "functions": function_names,
            "classes": class_names,
            "imports": [match[0] or match[1] for match in import_matches],
            "analysis_method": "regex_fallback",
            "complexity_score": len(function_names) + len(class_names),
        }


def analyze_python_code(code: str, filename: str = "") -> Dict[str, Any]:
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
        
        # Ensure metadata_json field for compatibility with existing code
        if 'metadata_json' not in metadata:
            metadata['metadata_json'] = json.dumps(metadata, default=str)
        
        return metadata
    else:
        # Fallback to the legacy analyzer
        LOGGER.info("Using legacy Python analyzer (tree-sitter not available)")
        analyzer = PythonCodeAnalyzer()
        return analyzer.analyze_code(code, filename)


if __name__ == "__main__":
    print("Python code analyzer module - use tests/test_python_analyzer_integration.py for testing")
