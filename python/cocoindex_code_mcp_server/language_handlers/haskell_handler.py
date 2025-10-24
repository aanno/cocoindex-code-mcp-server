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
Haskell-specific AST node handler for tree-sitter based analysis.

Handles Haskell-specific constructs like data types, type classes, modules, etc.

TODO: (tp)
This file does use the rust _haskell_tree_sitter implementation.
Hence I'm not sure if it is really in use (or helpful).
"""

import os
import re

# Import from parent directory
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cocoindex_code_mcp_server.ast_visitor import Position
from tree_sitter import Node

from ..ast_visitor import NodeContext
from . import LOGGER

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@dataclass
class HaskellFunction:
    """Represents a Haskell function with detailed metadata."""

    name: str
    line: int
    column: int
    type_signature: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    is_infix: bool = False
    is_private: bool = False
    complexity: int = 0


@dataclass
class HaskellDataType:
    """Represents a Haskell data type (data, newtype, type)."""

    name: str
    line: int
    column: int
    kind: str  # 'data', 'newtype', 'type'
    constructors: List[str] = field(default_factory=list)
    type_parameters: List[str] = field(default_factory=list)
    deriving_clauses: List[str] = field(default_factory=list)


@dataclass
class HaskellTypeClass:
    """Represents a Haskell type class."""

    name: str
    line: int
    column: int
    constraints: List[str] = field(default_factory=list)
    type_variables: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)


@dataclass
class HaskellInstance:
    """Represents a Haskell type class instance."""

    class_name: str
    type_name: str
    line: int
    column: int
    constraints: List[str] = field(default_factory=list)


@dataclass
class HaskellImport:
    """Represents a Haskell import statement."""

    module: str
    qualified: bool = False
    alias: Optional[str] = None
    imports: List[str] = field(default_factory=list)  # specific imports
    hiding: List[str] = field(default_factory=list)  # hidden imports
    line: int = 0


@dataclass
class HaskellModule:
    """Represents a Haskell module declaration."""

    name: str
    exports: List[str] = field(default_factory=list)
    line: int = 0


class HaskellNodeHandler:
    """Haskell-specific tree-sitter node handler."""

    # Haskell-specific node types mapping
    # These are the actual node types returned by _haskell_tree_sitter
    HASKELL_NODE_TYPES = {
        # Module and imports
        "module": "module_declaration",
        "import": "import_statement",
        "qualified_import": "qualified_import",
        # Functions and bindings
        "function": "function_definition",
        "bind": "function_definition",  # Handle bind chunks as potential function definitions
        "signature": "type_signature",
        "pattern": "pattern_binding",
        "variable": "variable_binding",
        # Data types (may need adjustment based on actual _haskell_tree_sitter output)
        "data": "data_declaration",
        "data_type": "data_declaration",
        "newtype": "newtype_declaration",
        "type": "type_declaration",
        "type_synonym": "type_declaration",
        "gadt": "gadt_declaration",
        # Type classes and instances
        "class": "class_declaration",
        "instance": "instance_declaration",
        # Expressions
        "application": "function_application",
        "lambda": "lambda_expression",
        "case": "case_expression",
        "if": "conditional_expression",
        "let": "let_expression",
        "where": "where_clause",
        "do": "do_expression",
        # Literals
        "integer": "integer_literal",
        "float": "float_literal",
        "char": "char_literal",
        "string": "string_literal",
        # Comments
        "comment": "comment",
        "pragma": "language_pragma",
    }

    def __init__(self) -> None:
        self.functions: List[HaskellFunction] = []
        self.data_types: List[HaskellDataType] = []
        self.type_classes: List[HaskellTypeClass] = []
        self.instances: List[HaskellInstance] = []
        self.imports: List[HaskellImport] = []
        self.module: Optional[HaskellModule] = None
        self.type_signatures: Dict[str, str] = {}  # function_name -> type_signature
        self.complexity_score = 0
        self.current_scope: List[str] = []

    def can_handle(self, node_type: str) -> bool:
        """Check if this handler can process the given node type."""
        return node_type in self.HASKELL_NODE_TYPES

    def extract_metadata(self, context: NodeContext) -> Dict[str, Any]:
        """Extract metadata from a Haskell chunk or AST node."""
        # Get node type from chunk or standard node
        if hasattr(context.node, "node_type") and callable(context.node.node_type):
            node_type = context.node.node_type()  # haskell_tree_sitter chunk
        elif hasattr(context.node, "type"):
            node_type = context.node.type  # standard tree-sitter node
        else:
            node_type = str(type(context.node))

        if not self.can_handle(node_type):
            return {}

        # For chunks, use specialized chunk handlers
        if hasattr(context.node, "text") and callable(context.node.text):
            return self._handle_chunk(context, node_type)

        # For standard tree-sitter nodes, use the original handlers
        handler_method = f"_handle_{self.HASKELL_NODE_TYPES[node_type]}"

        if hasattr(self, handler_method):
            try:
                result = getattr(self, handler_method)(context) or {}
                LOGGER.debug("Handled %s with result: %s", node_type, result)
                return result
            except Exception as e:
                LOGGER.warning("Error handling %s: %s", node_type, e)
                return {}

        return {}

    def _handle_chunk(self, context: NodeContext, node_type: str) -> Dict[str, Any]:
        """Handle _haskell_tree_sitter chunks directly."""
        chunk = context.node
        # Handle chunk.text() returning bytes or None
        raw_text = chunk.text() if hasattr(chunk, "text") and callable(chunk.text) else None
        if raw_text is None:
            chunk_text = ""
        elif isinstance(raw_text, bytes):
            chunk_text = raw_text.decode("utf-8", errors="ignore")
        else:
            chunk_text = str(raw_text)
        position = context.get_position()

        if node_type == "data_type":
            return self._handle_data_type_chunk(chunk, chunk_text, position)
        elif node_type == "function":
            return self._handle_function_chunk(chunk, chunk_text, position)
        elif node_type == "bind":
            # Handle bind chunks as potential function definitions
            return self._handle_function_chunk(chunk, chunk_text, position)
        elif node_type == "signature":
            return self._handle_signature_chunk(chunk, chunk_text, position)
        elif node_type == "import":
            return self._handle_import_chunk(chunk, chunk_text, position)
        elif node_type == "module":
            return self._handle_module_chunk(chunk, chunk_text, position)
        else:
            LOGGER.debug("No specific chunk handler for %s", node_type)
            return {}

    def _handle_data_type_chunk(self, chunk, chunk_text: str, position) -> Dict[str, Any]:
        """Handle data type declaration chunks."""
        # Parse data type from text
        data_type_info = self._parse_data_type_text(chunk_text)
        if not data_type_info:
            return {}

        data_type = HaskellDataType(
            name=data_type_info["name"],
            line=position.line,
            column=position.column,
            kind=data_type_info["kind"],
            constructors=data_type_info.get("constructors", []),
            type_parameters=data_type_info.get("type_parameters", []),
            deriving_clauses=data_type_info.get("deriving", []),
        )

        self.data_types.append(data_type)

        return {
            f"{data_type.kind}_declaration_found": {
                "name": data_type.name,
                "kind": data_type.kind,
                "constructor_count": len(data_type.constructors),
                "type_parameter_count": len(data_type.type_parameters),
                "has_deriving": len(data_type.deriving_clauses) > 0,
            }
        }

    def _handle_function_chunk(self, chunk: Node, chunk_text: str, position: Position) -> Dict[str, Any]:
        """Handle function definition chunks."""
        func_info = self._parse_function_text(chunk_text)
        if not func_info:
            return {}

        function = HaskellFunction(
            name=func_info["name"],
            line=position.line,
            column=position.column,
            type_signature=self.type_signatures.get(func_info["name"]),
            parameters=func_info.get("parameters", []),
            is_infix=func_info.get("is_infix", False),
            is_private=func_info["name"].startswith("_"),
            complexity=self._calculate_text_complexity(chunk_text),
        )

        self.functions.append(function)
        self.complexity_score += function.complexity

        return {
            "function_found": {
                "name": function.name,
                "has_type_signature": function.type_signature is not None,
                "parameter_count": len(function.parameters),
                "is_infix": function.is_infix,
                "is_private": function.is_private,
                "complexity": function.complexity,
            }
        }

    def _handle_signature_chunk(self, chunk: Node, chunk_text: str, position: Position) -> Dict[str, Any]:
        """Handle type signature chunks."""
        sig_info = self._parse_type_signature_text(chunk_text)
        if not sig_info:
            return {}

        # Store for later association with function definitions
        for name in sig_info["names"]:
            self.type_signatures[name] = sig_info["type"]

        return {
            "type_signature_found": {
                "names": sig_info["names"],
                "type": sig_info["type"],
                "is_polymorphic": self._is_polymorphic_type(sig_info["type"]),
                "has_constraints": "=>" in sig_info["type"],
            }
        }

    def _handle_import_chunk(self, chunk: Node, chunk_text: str, position: Position) -> Dict[str, Any]:
        """Handle import statement chunks."""
        import_info = self._parse_import_text(chunk_text)
        if not import_info:
            return {}

        haskell_import = HaskellImport(
            module=import_info["module"],
            qualified=import_info.get("qualified", False),
            alias=import_info.get("alias"),
            imports=import_info.get("imports", []),
            hiding=import_info.get("hiding", []),
            line=position.line,
        )

        self.imports.append(haskell_import)

        return {
            "import_found": {
                "module": haskell_import.module,
                "qualified": haskell_import.qualified,
                "has_alias": haskell_import.alias is not None,
                "import_count": len(haskell_import.imports),
                "has_hiding": len(haskell_import.hiding) > 0,
            }
        }

    def _handle_module_chunk(self, chunk: Node, chunk_text: str, position: Position) -> Dict[str, Any]:
        """Handle module declaration chunks."""
        module_info = self._parse_module_text(chunk_text)
        if not module_info:
            return {}

        self.module = HaskellModule(
            name=module_info["name"], exports=module_info.get("exports", []), line=position.line
        )

        return {
            "module_found": {
                "name": self.module.name,
                "has_exports": len(self.module.exports) > 0,
                "export_count": len(self.module.exports),
            }
        }

    def _handle_module_declaration(self, context: NodeContext) -> Dict[str, Any]:
        """Handle module declarations."""
        node = context.node
        module_info = self._extract_module_info(node, context.source_text)

        if not module_info:
            return {}

        position = context.get_position()

        self.module = HaskellModule(
            name=module_info["name"], exports=module_info.get("exports", []), line=position.line
        )

        return {
            "module_found": {
                "name": self.module.name,
                "has_exports": len(self.module.exports) > 0,
                "export_count": len(self.module.exports),
            }
        }

    def _handle_import_statement(self, context: NodeContext) -> Dict[str, Any]:
        """Handle import statements."""
        return self._extract_import_info(context, qualified=False)

    def _handle_qualified_import(self, context: NodeContext) -> Dict[str, Any]:
        """Handle qualified import statements."""
        return self._extract_import_info(context, qualified=True)

    def _extract_import_info(self, context: NodeContext, qualified: bool) -> Dict[str, Any]:
        """Extract import information."""
        # Since haskell_tree_sitter gives us text chunks, extract from text directly
        import_text = context.get_node_text().strip()
        position = context.get_position()

        # Parse import text using regex patterns
        import_info = self._parse_import_text(import_text)
        if not import_info:
            return {}

        haskell_import = HaskellImport(
            module=import_info["module"],
            qualified=qualified or import_info.get("qualified", False),
            alias=import_info.get("alias"),
            imports=import_info.get("imports", []),
            hiding=import_info.get("hiding", []),
            line=position.line,
        )

        self.imports.append(haskell_import)

        return {
            "import_found": {
                "module": haskell_import.module,
                "qualified": haskell_import.qualified,
                "has_alias": haskell_import.alias is not None,
                "import_count": len(haskell_import.imports),
                "has_hiding": len(haskell_import.hiding) > 0,
            }
        }

    def _handle_type_signature(self, context: NodeContext) -> Dict[str, Any]:
        """Handle type signatures."""
        # Extract from text directly since haskell_tree_sitter gives us chunks
        signature_text = context.get_node_text().strip()
        sig_info = self._parse_type_signature_text(signature_text)

        if not sig_info:
            return {}

        # Store for later association with function definitions
        for name in sig_info["names"]:
            self.type_signatures[name] = sig_info["type"]

        return {
            "type_signature_found": {
                "names": sig_info["names"],
                "type": sig_info["type"],
                "is_polymorphic": self._is_polymorphic_type(sig_info["type"]),
                "has_constraints": "=>" in sig_info["type"],
            }
        }

    def _handle_function_definition(self, context: NodeContext) -> Dict[str, Any]:
        """Handle function definitions."""
        node = context.node
        func_info = self._extract_function_info(node, context.source_text)

        if not func_info:
            return {}

        position = context.get_position()

        function = HaskellFunction(
            name=func_info["name"],
            line=position.line,
            column=position.column,
            type_signature=self.type_signatures.get(func_info["name"]),
            parameters=func_info.get("parameters", []),
            is_infix=func_info.get("is_infix", False),
            is_private=func_info["name"].startswith("_"),
            complexity=self._calculate_function_complexity(node, context.source_text),
        )

        self.functions.append(function)
        self.complexity_score += function.complexity

        return {
            "function_found": {
                "name": function.name,
                "has_type_signature": function.type_signature is not None,
                "parameter_count": len(function.parameters),
                "is_infix": function.is_infix,
                "is_private": function.is_private,
                "complexity": function.complexity,
            }
        }

    # def _handle_data_declaration(self, context: NodeContext) -> Dict[str, Any]:
    #     """Handle data type declarations."""
    #     return self._handle_type_declaration(context, 'data')

    # def _handle_newtype_declaration(self, context: NodeContext) -> Dict[str, Any]:
    #     """Handle newtype declarations."""
    #     return self._handle_type_declaration(context, 'newtype')

    # def _handle_type_declaration(self, context: NodeContext) -> Dict[str, Any]:
    #     """Handle type synonym declarations."""
    #     return self._handle_type_declaration(context, 'type')

    def _handle_data_declaration(self, context: NodeContext) -> Dict[str, Any]:
        return self._handle_type_declaration(context, "data")

    def _handle_newtype_declaration(self, context: NodeContext) -> Dict[str, Any]:
        return self._handle_type_declaration(context, "newtype")

    def _handle_typedef_declaration(self, context: NodeContext) -> Dict[str, Any]:
        return self._handle_type_declaration(context, "type")

    def _handle_type_declaration(self, context: NodeContext, kind: str) -> Dict[str, Any]:
        """Handle various type declarations."""
        node = context.node
        type_info = self._extract_type_declaration(node, context.source_text, kind)

        if not type_info:
            return {}

        position = context.get_position()

        data_type = HaskellDataType(
            name=type_info["name"],
            line=position.line,
            column=position.column,
            kind=kind,
            constructors=type_info.get("constructors", []),
            type_parameters=type_info.get("type_parameters", []),
            deriving_clauses=type_info.get("deriving", []),
        )

        self.data_types.append(data_type)

        return {
            f"{kind}_declaration_found": {
                "name": data_type.name,
                "kind": kind,
                "constructor_count": len(data_type.constructors),
                "type_parameter_count": len(data_type.type_parameters),
                "has_deriving": len(data_type.deriving_clauses) > 0,
            }
        }

    def _handle_class_declaration(self, context: NodeContext) -> Dict[str, Any]:
        """Handle type class declarations."""
        node = context.node
        class_info = self._extract_class_declaration(node, context.source_text)

        if not class_info:
            return {}

        position = context.get_position()

        type_class = HaskellTypeClass(
            name=class_info["name"],
            line=position.line,
            column=position.column,
            constraints=class_info.get("constraints", []),
            type_variables=class_info.get("type_variables", []),
            methods=class_info.get("methods", []),
        )

        self.type_classes.append(type_class)

        return {
            "class_declaration_found": {
                "name": type_class.name,
                "constraint_count": len(type_class.constraints),
                "type_variable_count": len(type_class.type_variables),
                "method_count": len(type_class.methods),
            }
        }

    def _handle_instance_declaration(self, context: NodeContext) -> Dict[str, Any]:
        """Handle instance declarations."""
        node = context.node
        instance_info = self._extract_instance_declaration(node, context.source_text)

        if not instance_info:
            return {}

        position = context.get_position()

        instance = HaskellInstance(
            class_name=instance_info["class_name"],
            type_name=instance_info["type_name"],
            line=position.line,
            column=position.column,
            constraints=instance_info.get("constraints", []),
        )

        self.instances.append(instance)

        return {
            "instance_declaration_found": {
                "class_name": instance.class_name,
                "type_name": instance.type_name,
                "has_constraints": len(instance.constraints) > 0,
            }
        }

    def _handle_case_expression(self, context: NodeContext) -> Dict[str, Any]:
        """Handle case expressions."""
        self.complexity_score += 2  # Case expressions add complexity
        return {"case_complexity": 2}

    def _handle_conditional_expression(self, context: NodeContext) -> Dict[str, Any]:
        """Handle if-then-else expressions."""
        self.complexity_score += 1
        return {"conditional_complexity": 1}

    def _handle_lambda_expression(self, context: NodeContext) -> Dict[str, Any]:
        """Handle lambda expressions."""
        self.complexity_score += 1
        return {"lambda_complexity": 1}

    def _handle_do_expression(self, context: NodeContext) -> Dict[str, Any]:
        """Handle do expressions."""
        self.complexity_score += 1
        return {"do_complexity": 1}

    # Helper methods for extracting specific information

    def _extract_module_info(self, node, source_text: str) -> Optional[Dict[str, Any]]:
        """Extract module name and exports."""
        if not hasattr(node, "children"):
            return None

        module_info = {"name": "", "exports": []}

        for child in node.children:
            if hasattr(child, "type"):
                if child.type == "module_id":
                    module_info["name"] = source_text[child.start_byte: child.end_byte]
                elif child.type == "exports":
                    module_info["exports"] = self._extract_export_list(child, source_text)

        return module_info if module_info["name"] else None

    def _extract_export_list(self, exports_node: object, source_text: str) -> List[str]:
        """Extract list of exported items."""
        exports: List[str] = []

        if not hasattr(exports_node, "children"):
            return exports

        for child in exports_node.children:
            if hasattr(child, "type") and child.type in ["variable", "constructor", "type"]:
                export_name = source_text[child.start_byte: child.end_byte]
                exports.append(export_name)

        return exports

    def _parse_import_statement(self, node: object, source_text: str) -> Optional[Dict[str, Any]]:
        """Parse import statement components."""
        if not hasattr(node, "children"):
            return None

        import_info = {"module": "", "qualified": False, "alias": None, "imports": [], "hiding": []}

        for child in node.children:
            if hasattr(child, "type"):
                if child.type == "module_id":
                    import_info["module"] = source_text[child.start_byte: child.end_byte]
                elif child.type == "qualified":
                    import_info["qualified"] = True
                elif child.type == "as":
                    # Next sibling should be the alias
                    import_info["alias"] = self._get_next_identifier(child, source_text)
                elif child.type == "imports":
                    import_info["imports"] = self._extract_import_list(child, source_text)
                elif child.type == "hiding":
                    import_info["hiding"] = self._extract_import_list(child, source_text)

        return import_info if import_info["module"] else None

    def _extract_import_list(self, import_list_node: object, source_text: str) -> List[str]:
        """Extract list of imported/hidden items."""
        items: List[str] = []

        if not hasattr(import_list_node, "children"):
            return items

        for child in import_list_node.children:
            if hasattr(child, "type") and child.type in ["variable", "constructor", "type"]:
                item_name = source_text[child.start_byte: child.end_byte]
                items.append(item_name)

        return items

    def _extract_type_signature(self, node, source_text: str) -> Optional[Dict[str, Any]]:
        """Extract type signature information."""
        if not hasattr(node, "children"):
            return None

        sig_info: Dict[str, Any] = {"names": [], "type": ""}

        # Type signatures have pattern: names :: type
        collecting_names = True

        for child in node.children:
            if hasattr(child, "type"):
                if child.type == "::":
                    collecting_names = False
                elif collecting_names and child.type in ["variable", "identifier"]:
                    name = source_text[child.start_byte: child.end_byte]
                    sig_info["names"].append(name)
                elif not collecting_names:
                    # Everything after :: is the type
                    if not sig_info["type"]:
                        sig_info["type"] = source_text[child.start_byte: node.end_byte].strip()

        return sig_info if sig_info["names"] and sig_info["type"] else None

    def _extract_function_info(self, node, source_text: str) -> Optional[Dict[str, Any]]:
        """Extract function definition information."""
        if not hasattr(node, "children"):
            return None

        func_info = {"name": "", "parameters": [], "is_infix": False}

        for child in node.children:
            if hasattr(child, "type"):
                if child.type in ["variable", "identifier"]:
                    if not func_info["name"]:
                        func_info["name"] = source_text[child.start_byte: child.end_byte]
                elif child.type == "patterns":
                    func_info["parameters"] = self._extract_pattern_list(child, source_text)
                elif child.type == "infix":
                    func_info["is_infix"] = True

        return func_info if func_info["name"] else None

    def _extract_pattern_list(self, patterns_node: object, source_text: str) -> List[str]:
        """Extract function parameter patterns."""
        patterns: List[str] = []

        if not hasattr(patterns_node, "children"):
            return patterns

        for child in patterns_node.children:
            if hasattr(child, "type"):
                pattern_text = source_text[child.start_byte: child.end_byte]
                patterns.append(pattern_text)

        return patterns

    def _extract_type_declaration(self, node: Node, source_text: str, kind: str) -> Optional[Dict[str, Any]]:
        """Extract type declaration information."""
        if not hasattr(node, "children"):
            return None

        type_info: Dict[str, Any] = {"name": "", "type_parameters": [], "constructors": [], "deriving": []}

        for child in node.children:
            if hasattr(child, "type"):
                if child.type == "type_constructor":
                    type_info["name"] = source_text[child.start_byte: child.end_byte]
                elif child.type == "type_variable":
                    param_name = source_text[child.start_byte: child.end_byte]
                    type_info["type_parameters"].append(param_name)
                elif child.type == "constructors":
                    type_info["constructors"] = self._extract_constructor_list(child, source_text)
                elif child.type == "deriving":
                    type_info["deriving"] = self._extract_deriving_clause(child, source_text)

        return type_info if type_info["name"] else None

    def _extract_constructor_list(self, constructors_node: object, source_text: str) -> List[str]:
        """Extract data constructor names."""
        constructors: List[str] = []

        if not hasattr(constructors_node, "children"):
            return constructors

        for child in constructors_node.children:
            if hasattr(child, "type") and child.type == "constructor":
                constructor_name = source_text[child.start_byte: child.end_byte]
                constructors.append(constructor_name)

        return constructors

    def _extract_deriving_clause(self, deriving_node: object, source_text: str) -> List[str]:
        """Extract deriving clause type classes."""
        deriving_classes: List[str] = []

        if not hasattr(deriving_node, "children"):
            return deriving_classes

        for child in deriving_node.children:
            if hasattr(child, "type") and child.type == "type_constructor":
                class_name = source_text[child.start_byte: child.end_byte]
                deriving_classes.append(class_name)

        return deriving_classes

    def _extract_class_declaration(self, node: Node, source_text: str) -> Optional[Dict[str, Any]]:
        """Extract type class declaration information."""
        if not hasattr(node, "children"):
            return None

        class_info: Dict[str, Any] = {"name": "", "constraints": [], "type_variables": [], "methods": []}

        for child in node.children:
            if hasattr(child, "type"):
                if child.type == "type_constructor":
                    class_info["name"] = source_text[child.start_byte: child.end_byte]
                elif child.type == "type_variable":
                    var_name = source_text[child.start_byte: child.end_byte]
                    class_info["type_variables"].append(var_name)
                elif child.type == "constraints":
                    class_info["constraints"] = self._extract_constraint_list(child, source_text)
                elif child.type == "where":
                    class_info["methods"] = self._extract_class_methods(child, source_text)

        return class_info if class_info["name"] else None

    def _extract_instance_declaration(self, node, source_text: str) -> Optional[Dict[str, Any]]:
        """Extract instance declaration information."""
        if not hasattr(node, "children"):
            return None

        instance_info = {"class_name": "", "type_name": "", "constraints": []}

        for child in node.children:
            if hasattr(child, "type"):
                if child.type == "type_constructor":
                    if not instance_info["class_name"]:
                        instance_info["class_name"] = source_text[child.start_byte: child.end_byte]
                    else:
                        instance_info["type_name"] = source_text[child.start_byte: child.end_byte]
                elif child.type == "constraints":
                    instance_info["constraints"] = self._extract_constraint_list(child, source_text)

        return instance_info if instance_info["class_name"] and instance_info["type_name"] else None

    def _extract_constraint_list(self, constraints_node: object, source_text: str) -> List[str]:
        """Extract constraint list from type class/instance declarations."""
        constraints: List[str] = []

        if not hasattr(constraints_node, "children"):
            return constraints

        for child in constraints_node.children:
            if hasattr(child, "type"):
                constraint_text = source_text[child.start_byte: child.end_byte]
                constraints.append(constraint_text)

        return constraints

    def _extract_class_methods(self, where_node: object, source_text: str) -> List[str]:
        """Extract method names from type class where clause."""
        methods: List[str] = []

        if not hasattr(where_node, "children"):
            return methods

        for child in where_node.children:
            if hasattr(child, "type") and child.type == "signature":
                # Extract method names from type signatures
                sig_info = self._extract_type_signature(child, source_text)
                if sig_info:
                    methods.extend(sig_info["names"])

        return methods

    def _parse_import_text(self, import_text: str) -> Optional[Dict[str, Any]]:
        """Parse import statement from text using regex."""
        import_info = {"module": "", "qualified": False, "alias": None, "imports": [], "hiding": []}

        # Basic import pattern: import [qualified] Module [as Alias] [(imports)] [hiding (items)]
        import_pattern = r"^import\s+(?:(qualified)\s+)?([A-Z][a-zA-Z0-9_.]*)"
        match = re.match(import_pattern, import_text.strip())

        if not match:
            return None

        import_info["qualified"] = match.group(1) is not None
        import_info["module"] = match.group(2)

        # Look for 'as' alias
        as_pattern = r"\s+as\s+([A-Z][a-zA-Z0-9_]*)"
        as_match = re.search(as_pattern, import_text)
        if as_match:
            import_info["alias"] = as_match.group(1)

        # Look for specific imports
        imports_pattern = r"\(([^)]+)\)"
        imports_match = re.search(imports_pattern, import_text)
        if imports_match:
            imports_str = imports_match.group(1)
            if "hiding" in import_text:
                import_info["hiding"] = [item.strip() for item in imports_str.split(",")]
            else:
                import_info["imports"] = [item.strip() for item in imports_str.split(",")]

        return import_info

    def _parse_type_signature_text(self, signature_text: str) -> Optional[Dict[str, Any]]:
        """Parse type signature from text using regex."""
        # Pattern: name1, name2, ... :: Type
        sig_pattern = r"^([a-zA-Z_][a-zA-Z0-9_,\s]*?)\s*::\s*(.+)$"
        match = re.match(sig_pattern, signature_text.strip())

        if not match:
            return None

        names_str = match.group(1)
        type_str = match.group(2).strip()

        # Parse names (comma-separated)
        names = [name.strip() for name in names_str.split(",") if name.strip()]

        return {"names": names, "type": type_str}

    def _get_next_identifier(self, node: object, source_text: str) -> Optional[str]:
        """Get the next identifier sibling after the given node."""
        if not hasattr(node, "parent") or not hasattr(node.parent, "children"):
            return None

        found_current = False
        for child in node.parent.children:
            if found_current and hasattr(child, "type") and child.type in ["identifier", "module_id"]:
                return source_text[child.start_byte: child.end_byte]
            if child == node:
                found_current = True

        return None

    def _is_polymorphic_type(self, type_signature: str) -> bool:
        """Check if a type signature is polymorphic (contains type variables)."""
        # Simple heuristic: contains lowercase type variables
        type_vars = re.findall(r"\b[a-z][a-zA-Z0-9_]*\b", type_signature)
        # Filter out common keywords
        keywords = {"where", "let", "in", "case", "of", "if", "then", "else", "do"}
        return any(var not in keywords for var in type_vars)

    def _calculate_function_complexity(self, node: Node, source_text: str) -> int:
        """Calculate cyclomatic complexity for a Haskell function."""
        complexity = 1  # Base complexity

        # Get function text
        function_text = source_text[node.start_byte: node.end_byte] if hasattr(node, "start_byte") else ""

        # Count complexity indicators
        complexity += function_text.count("case ")
        complexity += function_text.count("if ")
        complexity += function_text.count("|")  # Guards
        complexity += function_text.count("where")
        complexity += function_text.count("let ")
        complexity += len(re.findall(r"\\.*->", function_text))  # Lambda expressions

        return complexity

    def _parse_data_type_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse data type declaration from text."""
        # Patterns for different data type declarations
        data_patterns = [
            r"^data\s+([A-Z][a-zA-Z0-9_\']*)\s*([a-z\s]*?)\s*=\s*(.+?)(?:\s+deriving\s*\(([^)]+)\))?$",
            r"^newtype\s+([A-Z][a-zA-Z0-9_\']*)\s*([a-z\s]*?)\s*=\s*(.+?)(?:\s+deriving\s*\(([^)]+)\))?$",
            r"^type\s+([A-Z][a-zA-Z0-9_\']*)\s*([a-z\s]*?)\s*=\s*(.+)$",
        ]

        for pattern in data_patterns:
            match = re.match(pattern, text.strip(), re.MULTILINE | re.DOTALL)
            if match:
                kind = (
                    "data" if pattern.startswith("^data") else ("newtype" if pattern.startswith("^newtype") else "type")
                )
                name = match.group(1)
                type_params = [p.strip() for p in match.group(2).split() if p.strip()] if match.group(2) else []

                # Parse constructors (simplified)
                constructors = []
                if match.group(3):
                    constructor_text = match.group(3).strip()
                    # Simple constructor parsing (just get the names)
                    constructors = [c.strip().split()[0] for c in constructor_text.split("|") if c.strip()]

                # Parse deriving clause
                deriving = []
                if len(match.groups()) > 3 and match.group(4):
                    deriving = [d.strip() for d in match.group(4).split(",")]

                return {
                    "kind": kind,
                    "name": name,
                    "type_parameters": type_params,
                    "constructors": constructors,
                    "deriving": deriving,
                }

        return None

    def _parse_function_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse function definition from text."""
        # Pattern for function definition: name params = body
        func_pattern = r"^([a-zA-Z_][a-zA-Z0-9_\']*)\s*([^=]*?)\s*=\s*(.+)$"
        match = re.match(func_pattern, text.strip(), re.MULTILINE | re.DOTALL)

        if match:
            name = match.group(1)
            params_text = match.group(2).strip()

            # Parse parameters (simplified)
            parameters: List[str] = []
            if params_text:
                # Simple parameter parsing - just split on whitespace
                parameters = params_text.split()

            return {"name": name, "parameters": parameters, "is_infix": name.startswith("(") and name.endswith(")")}

        return None

    def _parse_module_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse module declaration from text."""
        # Pattern: module ModuleName (exports) where
        module_pattern = r"^module\s+([A-Z][a-zA-Z0-9_.]*)(?:\s*\(([^)]*)\))?\s+where"
        match = re.match(module_pattern, text.strip())

        if match:
            name = match.group(1)
            exports = []

            if match.group(2):
                exports_text = match.group(2)
                exports = [e.strip() for e in exports_text.split(",") if e.strip()]

            return {"name": name, "exports": exports}

        return None

    def _calculate_text_complexity(self, text: str) -> int:
        """Calculate complexity from text content."""
        complexity = 1  # Base complexity

        # Count complexity indicators
        complexity += text.count("case ")
        complexity += text.count("if ")
        complexity += text.count("|")  # Guards
        complexity += text.count("where")
        complexity += text.count("let ")
        complexity += len(re.findall(r"\\.*->", text))  # Lambda expressions

        return complexity

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all extracted metadata."""
        return {
            "module": self.module.name if self.module else None,
            "functions": [func.name for func in self.functions],
            "data_types": [dt.name for dt in self.data_types],
            "type_classes": [tc.name for tc in self.type_classes],
            "instances": [f"{inst.class_name} {inst.type_name}" for inst in self.instances],
            "imports": [imp.module for imp in self.imports],
            "complexity_score": self.complexity_score,
            # Detailed statistics
            "has_module_declaration": self.module is not None,
            "has_exports": self.module is not None and len(self.module.exports) > 0,
            "has_type_signatures": len(self.type_signatures) > 0,
            "has_data_types": len(self.data_types) > 0,
            "has_type_classes": len(self.type_classes) > 0,
            "has_instances": len(self.instances) > 0,
            "qualified_imports": len([imp for imp in self.imports if imp.qualified]),
            # Detailed breakdown
            "function_details": [
                {
                    "name": func.name,
                    "line": func.line,
                    "has_type_signature": func.type_signature is not None,
                    "parameter_count": len(func.parameters),
                    "is_infix": func.is_infix,
                    "is_private": func.is_private,
                    "complexity": func.complexity,
                }
                for func in self.functions
            ],
            "data_type_details": [
                {
                    "name": dt.name,
                    "line": dt.line,
                    "kind": dt.kind,
                    "constructor_count": len(dt.constructors),
                    "type_parameter_count": len(dt.type_parameters),
                    "has_deriving": len(dt.deriving_clauses) > 0,
                }
                for dt in self.data_types
            ],
            "import_details": [
                {
                    "module": imp.module,
                    "qualified": imp.qualified,
                    "alias": imp.alias,
                    "imports": imp.imports,
                    "hiding": imp.hiding,
                    "line": imp.line,
                }
                for imp in self.imports
            ],
        }


def analyze_haskell_code(content: str, filename: str = "") -> Dict[str, Any]:
    """
    Analyze Haskell code using the current Rust-based implementation.

    This function provides compatibility with the old haskell_visitor.analyze_haskell_code
    interface while using the modern Rust-based tree-sitter implementation.

    Args:
        content: Haskell source code to analyze
        filename: Optional filename for context

    Returns:
        Dictionary containing analysis results with keys:
        - success: bool indicating if analysis succeeded
        - analysis_method: string describing the method used
        - functions: list of function names
        - imports: list of import module names
        - classes: list of type class names (empty for Haskell as it uses 'type_classes')
        - data_classes: list of data type names
        - data_types: list of all data type names
        - type_classes: list of type class names
        - complexity_score: int complexity metric
        - (and other metadata fields)
    """
    try:
        # Import haskell_tree_sitter here to avoid import errors if not available
        import cocoindex_code_mcp_server._haskell_tree_sitter as hts

        # Use Rust-based chunking to get AST chunks
        chunks = hts.get_haskell_ast_chunks_with_fallback(content)

        if len(chunks) == 0:
            LOGGER.warning("No chunks produced for Haskell file %s", filename)
            return {
                "success": False,
                "analysis_method": "haskell_chunk_visitor",
                "functions": [],
                "imports": [],
                "classes": [],
                "data_classes": [],
                "data_types": [],
                "type_classes": [],
                "complexity_score": 0,
                "error": "No AST chunks produced",
            }

        # Create handler and process chunks
        handler = HaskellNodeHandler()

        for chunk in chunks:
            # Create a minimal Node-like object for the chunk
            class ChunkNode:
                """Adapter to make chunk compatible with NodeContext."""

                def __init__(self, chunk):
                    self._chunk = chunk
                    self.type = chunk.node_type()
                    self.start_byte = chunk.start_byte()
                    self.end_byte = chunk.end_byte()

                def node_type(self):
                    return self._chunk.node_type()

                def text(self):
                    return self._chunk.text()

            # Create proper NodeContext with all required fields
            chunk_node = ChunkNode(chunk)
            context = NodeContext(
                node=chunk_node,  # type: ignore[arg-type]
                parent=None,
                depth=0,
                scope_stack=[],
                source_text=content,
            )

            handler.extract_metadata(context)

        # Get summary
        summary = handler.get_summary()

        # Determine chunking method and extract module info from chunks
        chunking_method = "rust_haskell_ast"
        has_errors = False
        module_names = set()
        for chunk in chunks:
            chunk_metadata = chunk.metadata() if hasattr(chunk, "metadata") and callable(chunk.metadata) else {}
            if "chunking_method" in chunk_metadata:
                chunking_method = chunk_metadata["chunking_method"]
            if chunk_metadata.get("has_error") == "true" or chunk_metadata.get("tree_sitter_chunking_error") == "true":
                has_errors = True
            # Extract module names from metadata
            if "modules" in chunk_metadata:
                import json

                try:
                    modules_list = json.loads(chunk_metadata["modules"])
                    module_names.update(modules_list)
                except BaseException:
                    pass

        # Determine the main module (first in alphabetical order, excluding imported modules)
        main_module = None
        if module_names:
            # The main module is typically the first module that's not an import
            # For now, just use the first module alphabetically
            main_module = sorted(module_names)[0] if module_names else None

        # Add compatibility fields for test expectations
        summary.update(
            {
                "success": True,
                "language": "Haskell",  # Display language name
                "filename": filename,
                "line_count": len(content.split("\n")),
                "char_count": len(content),
                "analysis_method": "haskell_chunk_visitor",
                "chunking_method": chunking_method,
                "module": main_module,
                "modules": list(module_names) if module_names else [],
                "has_module_declaration": main_module is not None,
                "classes": summary.get("type_classes", []),  # Compatibility: map type_classes to classes
                "data_classes": summary.get("data_types", []),  # Compatibility: map data_types to data_classes
                "error_count": 0,  # Would need to count actual errors from chunks
                "parse_errors": 0,
                "coverage_complete": not has_errors,
                "should_fallback": False,
            }
        )

        return summary

    except Exception as e:
        LOGGER.error("Haskell analysis failed for %s: %s", filename, e)
        return {
            "success": False,
            "language": "Haskell",
            "filename": filename,
            "line_count": len(content.split("\n")) if content else 0,
            "char_count": len(content) if content else 0,
            "analysis_method": "haskell_chunk_visitor",
            "functions": [],
            "imports": [],
            "classes": [],
            "data_classes": [],
            "data_types": [],
            "type_classes": [],
            "complexity_score": 0,
            "error": str(e),
        }
