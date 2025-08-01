#!/usr/bin/env python3

"""
Generic AST visitor framework for tree-sitter based code analysis.
Provides language-agnostic interfaces for extracting metadata from source code.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
import tree_sitter
from tree_sitter import Language, Node, Parser, Tree

TREE_SITTER_AVAILABLE = True
LOGGER = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a position in source code."""
    line: int
    column: int
    byte_offset: int


@dataclass
class CodeSpan:
    """Represents a span of code with start and end positions."""
    start: Position
    end: Position
    text: str = ""


@dataclass
class NodeContext:
    """Context information for AST node processing."""
    node: Node
    parent: Optional[Node] = None
    depth: int = 0
    scope_stack: List[str] = field(default_factory=list)
    source_text: str = ""

    def get_node_text(self) -> str:
        """Get the text content of the current node."""
        if self.source_text and hasattr(self.node, 'start_byte') and hasattr(self.node, 'end_byte'):
            return self.source_text[self.node.start_byte:self.node.end_byte]
        return ""

    def get_position(self) -> Position:
        """Get the position of the current node."""
        if hasattr(self.node, 'start_point'):
            point = self.node.start_point
            return Position(
                line=point[0] + 1,  # Convert to 1-based
                column=point[1] + 1,
                byte_offset=getattr(self.node, 'start_byte', 0)
            )
        return Position(line=1, column=1, byte_offset=0)


class NodeHandler(Protocol):
    """Protocol for language-specific node handlers."""

    def can_handle(self, node_type: str) -> bool:
        """Check if this handler can process the given node type."""
        ...

    def extract_metadata(self, context: NodeContext) -> Dict[str, Any]:
        """Extract metadata from a node with context."""
        ...


class ASTVisitor(ABC):
    """Abstract base class for AST visitors."""

    def __init__(self, language: str = "unknown"):
        self.language = language
        self.metadata: Dict[str, str] = {}
        self.errors: List[str] = []

    @abstractmethod
    def visit_node(self, context: NodeContext) -> Optional[Dict[str, Any]]:
        """Visit a single AST node and extract metadata."""

    def on_enter_node(self, context: NodeContext) -> None:
        """Called when entering a node (pre-order)."""

    def on_exit_node(self, context: NodeContext) -> None:
        """Called when exiting a node (post-order)."""

    def get_metadata(self) -> Dict[str, Any]:
        """Get the accumulated metadata."""
        return self.metadata

    def get_errors(self) -> List[str]:
        """Get any errors encountered during processing."""
        return self.errors


class GenericMetadataVisitor(ASTVisitor):
    """Generic visitor that uses pluggable node handlers."""

    def __init__(self, language: str = "unknown"):
        super().__init__(language)
        self.handlers: List[NodeHandler] = []
        self.node_stats: Dict[str, int] = {}
        self.complexity_score: float = 0

    def add_handler(self, handler: NodeHandler) -> None:
        """Add a node handler to the visitor."""
        self.handlers.append(handler)

    def visit_node(self, context: NodeContext) -> Optional[Dict[str, Any]]:
        """Visit a node using registered handlers."""
        node_type = context.node.type if hasattr(context.node, 'type') else str(type(context.node))

        # Track node statistics
        self.node_stats[node_type] = self.node_stats.get(node_type, 0) + 1

        # Try each handler
        metadata = {}
        for handler in self.handlers:
            if handler.can_handle(node_type):
                try:
                    handler_metadata = handler.extract_metadata(context)
                    if handler_metadata:
                        metadata.update(handler_metadata)
                except Exception as e:
                    error_msg = f"Handler error for {node_type}: {e}"
                    self.errors.append(error_msg)
                    LOGGER.warning(error_msg)

        # Update complexity based on node type
        self._update_complexity(node_type)

        return metadata if metadata else None

    def _update_complexity(self, node_type: str) -> None:
        """Update complexity score based on node type."""
        # Universal complexity indicators
        complexity_weights = {
            # Control flow
            'if_statement': 1, 'if_expression': 1, 'conditional_expression': 1,
            'for_statement': 1, 'while_statement': 1, 'for_in_statement': 1,
            'switch_statement': 2, 'case_statement': 1,
            'try_statement': 1, 'catch_clause': 1, 'except_clause': 1,

            # Functions and methods
            'function_definition': 2, 'method_definition': 2, 'function_declaration': 2,
            'lambda': 1, 'arrow_function': 1,

            # Classes and structures
            'class_definition': 3, 'class_declaration': 3, 'struct_declaration': 3,

            # Advanced constructs
            'generator_expression': 2, 'list_comprehension': 1,
            'with_statement': 1, 'async_function_definition': 2,
        }

        weight = complexity_weights.get(node_type, 0)
        self.complexity_score += weight


class TreeWalker:
    """Language-agnostic tree-sitter AST traversal."""

    def __init__(self, source_text: str, tree: Tree):
        self.source_text = source_text
        self.tree = tree
        self.visit_order = "pre"  # "pre", "post", or "both"

    def walk(self, visitor: ASTVisitor, visit_order: str = "pre") -> Dict[str, Any]:
        """Walk the AST tree using the provided visitor."""
        self.visit_order = visit_order

        if not self.tree or not hasattr(self.tree, 'root_node'):
            LOGGER.warning("Invalid tree for walking")
            return {}

        root_node = self.tree.root_node
        self._walk_recursive(root_node, visitor, None, 0, [])

        # Add tree-level metadata
        metadata = visitor.get_metadata()

        # Get tree language as string, not Language object (which is not JSON serializable)
        tree_language = 'unknown'
        if hasattr(self.tree, 'language') and self.tree.language:
            tree_language = str(self.tree.language) if self.tree.language else 'unknown'

        metadata.update({
            'node_stats': getattr(visitor, 'node_stats', {}),
            'complexity_score': getattr(visitor, 'complexity_score', 0),
            'parse_errors': len(visitor.get_errors()),
            'tree_language': tree_language
        })

        return metadata

    def _walk_recursive(self, node: Node, visitor: ASTVisitor, parent: Optional[Node],
                        depth: int, scope_stack: List[str]) -> None:
        """Recursively walk AST nodes."""
        context = NodeContext(
            node=node,
            parent=parent,
            depth=depth,
            scope_stack=scope_stack.copy(),
            source_text=self.source_text
        )

        # Pre-order processing
        if self.visit_order in ["pre", "both"]:
            visitor.on_enter_node(context)
            visitor.visit_node(context)

        # Update scope for certain node types
        new_scope = self._get_scope_name(context)
        if new_scope:
            scope_stack.append(new_scope)

        # Recursively process children
        if hasattr(node, 'children'):
            for child in node.children:
                self._walk_recursive(child, visitor, node, depth + 1, scope_stack)

        # Post-order processing
        if self.visit_order in ["post", "both"]:
            visitor.on_exit_node(context)
            if self.visit_order == "post":
                visitor.visit_node(context)

        # Clean up scope
        if new_scope:
            scope_stack.pop()

    def _get_scope_name(self, context: NodeContext) -> Optional[str]:
        """Extract scope name from certain node types."""
        node_type = context.node.type if hasattr(context.node, 'type') else ""

        if node_type in ['function_definition', 'method_definition', 'function_declaration']:
            # Try to extract function name
            for child in context.node.children if hasattr(context.node, 'children') else []:
                if hasattr(child, 'type') and child.type in ['identifier', 'name']:
                    return context.source_text[child.start_byte:child.end_byte]

        elif node_type in ['class_definition', 'class_declaration']:
            # Try to extract class name
            for child in context.node.children if hasattr(context.node, 'children') else []:
                if hasattr(child, 'type') and child.type in ['identifier', 'name', 'type_identifier']:
                    return context.source_text[child.start_byte:child.end_byte]

        return None


class ASTParserFactory:
    """Factory for creating language-specific tree-sitter parsers."""

    # Language mapping from file extensions to tree-sitter languages
    LANGUAGE_MAP = {
        '.c': 'c',
        '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', '.h': 'cpp', '.hpp': 'cpp',
        '.cs': 'c_sharp',
        '.css': 'css', '.scss': 'css',
        '.go': 'go',
        '.hs': 'haskell', '.lhs': 'haskell',
        '.html': 'html', '.htm': 'html',
        '.java': 'java',
        '.js': 'javascript', '.mjs': 'javascript', '.cjs': 'javascript',
        '.json': 'json',
        '.kt': 'kotlin', '.kts': 'kotlin',
        '.md': 'markdown', '.mdx': 'markdown',
        '.php': 'php',
        '.py': 'python', # '.pyi': 'python',
        '.rb': 'ruby',
        '.rs': 'rust',
        '.scala': 'scala',
        '.sql': 'sql',
        '.swift': 'swift',
        '.ts': 'typescript', '.tsx': 'tsx',
        '.xml': 'xml',
        '.yaml': 'yaml', '.yml': 'yaml',
    }

    def __init__(self):
        self._parsers = {}  # Cache parsers
        self._languages = {}  # Cache languages

    def get_language_for_file(self, filename: str) -> Optional[str]:
        """Detect language from filename."""
        path = Path(filename)
        extension = path.suffix.lower()

        # Check extension mapping
        if extension in self.LANGUAGE_MAP:
            return self.LANGUAGE_MAP[extension]

        # Check special filenames
        basename = path.name.lower()
        if basename in ['makefile', 'dockerfile', 'jenkinsfile']:
            return basename

        return None

    def create_parser(self, language: str) -> Optional[Parser]:
        """Create a tree-sitter parser for the given language."""
        if not TREE_SITTER_AVAILABLE:
            LOGGER.warning("Tree-sitter not available")
            return None

        # Return cached parser if available
        if language in self._parsers:
            result = self._parsers[language]
            # LOGGER.debug(f"Using cached parser for {language}")
            return result

        try:
            # Create parser based on available tree-sitter language packages
            parser = None
            language_obj = None

            if language == 'python':
                try:
                    import tree_sitter_python
                    language_obj = tree_sitter.Language(tree_sitter_python.language())
                except ImportError:
                    LOGGER.warning("tree-sitter-python not available")
                    return None

            elif language == 'c_sharp':
                try:
                    import tree_sitter_c_sharp
                    language_obj = tree_sitter.Language(tree_sitter_c_sharp.language())
                except ImportError:
                    LOGGER.warning("tree-sitter-c-sharp not available")
                    return None

            elif language == 'java':
                try:
                    import tree_sitter_java
                    language_obj = tree_sitter.Language(tree_sitter_java.language())
                except ImportError:
                    LOGGER.warning("tree-sitter-java not available")
                    return None

            elif language in ['typescript', 'tsx']:
                try:
                    import tree_sitter_typescript
                    if language == 'tsx':
                        language_obj = tree_sitter.Language(tree_sitter_typescript.language_tsx())
                    else:
                        language_obj = tree_sitter.Language(tree_sitter_typescript.language_typescript())
                except ImportError:
                    LOGGER.warning("tree-sitter-typescript not available")
                    return None

            elif language == 'c':
                try:
                    import tree_sitter_c
                    language_obj = tree_sitter.Language(tree_sitter_c.language())
                except ImportError:
                    LOGGER.warning("tree-sitter-c not available")
                    return None

            elif language in ['cpp', 'cc', 'cxx']:
                try:
                    import tree_sitter_cpp
                    language_obj = tree_sitter.Language(tree_sitter_cpp.language())
                except ImportError:
                    LOGGER.warning("tree-sitter-cpp not available")
                    return None

            elif language == 'rust':
                try:
                    import tree_sitter_rust
                    language_obj = tree_sitter.Language(tree_sitter_rust.language())
                except ImportError:
                    LOGGER.warning("tree-sitter-rust not available")
                    return None

            elif language == 'kotlin':
                try:
                    import tree_sitter_kotlin
                    language_obj = tree_sitter.Language(tree_sitter_kotlin.language())
                except ImportError:
                    LOGGER.warning("tree-sitter-kotlin not available")
                    return None

            elif language == 'haskell':
                # Haskell uses a specialized visitor, no parser needed here
                LOGGER.debug("Haskell uses specialized visitor, not generic parser")
                return None

            else:
                LOGGER.debug(f"Tree-sitter language '{language}' not yet implemented")
                return None

            if language_obj:
                parser = Parser()
                parser.language = language_obj

                # Cache the parser
                self._parsers[language] = parser
                LOGGER.debug(f"Created tree-sitter parser for {language}")

            return parser

        except Exception as e:
            LOGGER.error(f"Failed to create parser for {language}: {e}")
            return None

    def parse_code(self, code: str, language: str) -> Optional[Tree]:
        """Parse code using tree-sitter."""
        parser = self.create_parser(language)
        if not parser:
            return None

        try:
            return parser.parse(bytes(code, 'utf-8'))
        except Exception as e:
            LOGGER.error(f"Failed to parse code with {language}: {e}")
            return None


class MultiLevelAnalyzer:
    """Multi-level code analyzer with fallback strategies."""

    def __init__(self) -> None:
        self.parser_factory = ASTParserFactory()

    def analyze_code(self, code: str, language: str = "unknown",
                     filename: str = "") -> Dict[str, Any]:
        """Analyze code using multiple strategies with fallback."""

        # Detect language if not provided
        if language == "unknown" and filename:
            detected = self.parser_factory.get_language_for_file(filename)
            if detected:
                language = detected

        metadata = {
            'language': language,
            'filename': filename,
            'line_count': len(code.split('\n')),
            'char_count': len(code),
            'analysis_method': 'unknown',
            'errors': []
        }

        # Strategy 1: Tree-sitter AST parsing
        if TREE_SITTER_AVAILABLE:
            tree_metadata = self._try_treesitter_analysis(code, language, filename)
            LOGGER.debug(f"Strategy 1: Tree-sitter analysis result for {filename}: {tree_metadata}")
            if tree_metadata:
                metadata.update(tree_metadata)
                metadata['analysis_method'] = 'tree_sitter'
                return metadata

        # Strategy 2: Language-specific AST (Python only for now)
        if language == 'python':
            python_metadata = self._try_python_ast_analysis(code, filename)
            LOGGER.debug(f"Strategy 2: Python AST analysis result for {filename}: {python_metadata}")
            if python_metadata:
                metadata.update(python_metadata)
                metadata['analysis_method'] = 'python_ast'
                return metadata

        # Strategy 3: Enhanced regex patterns
        regex_metadata = self._try_regex_analysis(code, language)
        if regex_metadata:
            LOGGER.debug(f"Strategy 3: Regex analysis result for {filename}: {regex_metadata}")
            metadata.update(regex_metadata)
            metadata['analysis_method'] = 'enhanced_regex'
            return metadata

        # Strategy 4: Basic text analysis
        basic_metadata = self._basic_text_analysis(code, language)
        LOGGER.debug(f"Strategy 4: Basic text analysis result for {filename}: {basic_metadata}")
        metadata.update(basic_metadata)
        metadata['analysis_method'] = 'basic_text'

        return metadata

    def _try_treesitter_analysis(self, code: str, language: str, filename: str = "") -> Optional[Dict[str, Any]]:
        """Try tree-sitter based analysis."""
        # Special handling for languages using dedicated visitors
        if language == 'haskell':
            try:
                from .language_handlers.haskell_visitor import analyze_haskell_code
                metadata = analyze_haskell_code(code, "")
                LOGGER.debug("Used specialized Haskell visitor")
                return metadata
            except ImportError:
                LOGGER.debug("Haskell visitor not available, falling back to generic")
            except Exception as e:
                LOGGER.warning(f"Haskell visitor failed: {e}")

        elif language == 'c':
            try:
                from .language_handlers.c_visitor import analyze_c_code
                metadata = analyze_c_code(code, filename)
                LOGGER.debug("Used specialized C visitor")
                return metadata
            except ImportError:
                LOGGER.debug("C visitor not available, falling back to generic")
            except Exception as e:
                LOGGER.warning(f"C visitor failed: {e}")

        elif language in ['cpp', 'cc', 'cxx']:
            try:
                from .language_handlers.cpp_visitor import analyze_cpp_code
                metadata = analyze_cpp_code(code, language, filename)
                LOGGER.debug("Used specialized C++ visitor")
                return metadata
            except ImportError:
                LOGGER.debug("C++ visitor not available, falling back to generic")
            except Exception as e:
                LOGGER.warning(f"C++ visitor failed: {e}")

        elif language == 'rust':
            try:
                from .language_handlers.rust_visitor import analyze_rust_code
                metadata = analyze_rust_code(code, filename)
                LOGGER.debug("Used specialized Rust visitor")
                return metadata
            except ImportError:
                LOGGER.debug("Rust visitor not available, falling back to generic")
            except Exception as e:
                LOGGER.warning(f"Rust visitor failed: {e}")

        elif language == 'kotlin':
            try:
                from .language_handlers.kotlin_visitor import analyze_kotlin_code
                metadata = analyze_kotlin_code(code, filename)
                LOGGER.debug("Used specialized Kotlin visitor")
                return metadata
            except ImportError:
                LOGGER.debug("Kotlin visitor not available, falling back to generic")
            except Exception as e:
                LOGGER.warning(f"Kotlin visitor failed: {e}")

        elif language == 'java':
            try:
                from .language_handlers.java_visitor import analyze_java_code
                metadata = analyze_java_code(code, filename)
                LOGGER.debug("Used specialized Java visitor")
                return metadata
            except ImportError:
                LOGGER.debug("Java visitor not available, falling back to generic")
            except Exception as e:
                LOGGER.warning(f"Java visitor failed: {e}")

        elif language in ['javascript', 'js']:
            try:
                from .language_handlers.javascript_visitor import (
                    analyze_javascript_code,
                )
                metadata = analyze_javascript_code(code, language, filename)
                LOGGER.debug("Used specialized JavaScript visitor")
                return metadata
            except ImportError:
                LOGGER.debug("JavaScript visitor not available, falling back to generic")
            except Exception as e:
                LOGGER.warning(f"JavaScript visitor failed: {e}")

        elif language in ['typescript', 'tsx']:
            try:
                from .language_handlers.typescript_visitor import (
                    analyze_typescript_code,
                )
                metadata = analyze_typescript_code(code, language, filename)
                LOGGER.debug("Used specialized TypeScript visitor")
                return metadata
            except ImportError:
                LOGGER.debug("TypeScript visitor not available, falling back to generic")
            except Exception as e:
                LOGGER.warning(f"TypeScript visitor failed: {e}")

        # Generic tree-sitter analysis for other languages
        try:
            tree = self.parser_factory.parse_code(code, language)
            if not tree:
                return None

            visitor = GenericMetadataVisitor(language)

            # Add language-specific handler if available
            handler = None
            try:
                from .language_handlers import get_handler_for_language
                handler = get_handler_for_language(language)
                if handler:
                    visitor.add_handler(handler)
                    LOGGER.debug(f"Added {language} handler to visitor")
            except ImportError:
                LOGGER.debug("Language handlers not available")

            walker = TreeWalker(code, tree)
            metadata = walker.walk(visitor)

            # Add language-specific summary if handler was used
            if handler:
                try:
                    language_summary = handler.get_summary()
                    metadata.update(language_summary)
                    metadata['analysis_method'] = f'tree_sitter+{language}_handler'
                except Exception as e:
                    LOGGER.warning(f"Error getting {language} handler summary: {e}")

            return metadata

        except Exception as e:
            LOGGER.warning(f"Tree-sitter analysis failed: {e}")
            return None

    def _try_python_ast_analysis(self, code: str, filename: str) -> Optional[Dict[str, Any]]:
        """Try Python AST analysis as fallback."""
        try:
            import ast
            tree = ast.parse(code, filename=filename)

            # This would use the existing Python analyzer logic
            # For now, return basic info
            return {
                'has_python_ast': True,
                'python_ast_nodes': len(list(ast.walk(tree)))
            }

        except Exception as e:
            LOGGER.warning(f"Python AST analysis failed: {e}")
            return None

    def _try_regex_analysis(self, code: str, language: str) -> Optional[Dict[str, Any]]:
        """Enhanced regex-based analysis."""
        try:
            # Universal patterns that work across languages
            patterns = {
                'functions': [
                    r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # Python
                    r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # JavaScript
                    r'fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # Rust
                    r'func\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # Go
                ],
                'classes': [
                    r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # Python, C++, Java
                    r'struct\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # Rust, C, Go
                    r'interface\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # TypeScript, Java, Go
                ],
                'imports': [
                    r'import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',  # Python, JavaScript
                    r'from\s+([a-zA-Z_][a-zA-Z0-9_\.]*)\s+import',  # Python
                    r'use\s+([a-zA-Z_][a-zA-Z0-9_:]*)',  # Rust
                    r'#include\s*[<"]([^>"]+)[>"]',  # C/C++
                ]
            }

            metadata: Dict[str, Any] = {}

            for category, pattern_list in patterns.items():
                matches = set()
                for pattern in pattern_list:
                    for match in re.finditer(pattern, code, re.MULTILINE):
                        if match.group(1):
                            matches.add(match.group(1))
                metadata[category] = list(matches)

            # Simple complexity estimation
            complexity_indicators = ['if', 'for', 'while', 'switch', 'case', 'try', 'catch']
            complexity = sum(len(re.findall(rf'\b{indicator}\b', code)) for indicator in complexity_indicators)
            metadata['complexity_score'] = complexity

            return metadata

        except Exception as e:
            LOGGER.warning(f"Regex analysis failed: {e}")
            return None

    def _basic_text_analysis(self, code: str, language: str) -> Dict[str, Any]:
        """Basic text-based analysis as last resort."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        return {
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
            'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
            'has_functions': 'def ' in code or 'function ' in code or 'fn ' in code,
            'has_classes': 'class ' in code or 'struct ' in code,
            'has_imports': 'import ' in code or '#include' in code or 'use ' in code,
        }


# Factory function for easy instantiation
def create_analyzer() -> MultiLevelAnalyzer:
    """Create a new multi-level code analyzer."""
    return MultiLevelAnalyzer()


# Convenience function for direct analysis
def analyze_code(code: str, language: str = "unknown", filename: str = "") -> Dict[str, Any]:
    """Analyze code with automatic fallback strategies."""
    analyzer = create_analyzer()
    return analyzer.analyze_code(code, language, filename)
