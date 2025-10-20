#!/usr/bin/env python3
"""Debug AST parsing for new languages."""

import sys

from cocoindex_code_mcp_server.ast_visitor import (
    ASTParserFactory,
    GenericMetadataVisitor,
    TreeWalker,
)

sys.path.append("src")


def debug_language(code: str, language: str, filename: str):
    print(f"\n=== Debugging {language.upper()} ===")
    print(f"Code:\n{code}")

    factory = ASTParserFactory()
    parser = factory.create_parser(language)

    if not parser:
        print(f"❌ No parser for {language}")
        return

    tree = factory.parse_code(code, language)
    if not tree:
        print(f"❌ Failed to parse {language} code")
        return

    print("✅ Parsed successfully with tree-sitter")

    # Debug what nodes we find
    visitor = GenericMetadataVisitor(language)
    walker = TreeWalker(code, tree)

    def debug_visit(node, path=""):
        node_type = node.type if hasattr(node, "type") else str(type(node))
        text = node.text.decode("utf-8") if hasattr(node, "text") else ""
        if len(text) > 50:
            text = text[:50] + "..."
        print(f"  Node: {node_type} | Text: {repr(text)}")

    # Walk tree manually to see structure
    def walk_tree(node, depth=0):
        if depth > 3:  # Limit depth
            return
        indent = "  " * depth
        node_type = node.type if hasattr(node, "type") else str(type(node))
        text = node.text.decode("utf-8") if hasattr(node, "text") else ""
        if len(text) > 30:
            text = text[:30] + "..."
        print(f"{indent}{node_type}: {repr(text)}")

        if hasattr(node, "children"):
            for child in node.children:
                walk_tree(child, depth + 1)

    print("Tree structure:")
    walk_tree(tree.root_node)

    # Try visitor
    metadata = walker.walk(visitor)
    print(f"Visitor result: {metadata}")


if __name__ == "__main__":
    # Test C
    c_code = """int add(int a, int b) {
    return a + b;
}"""
    debug_language(c_code, "c", "test.c")

    # Test Rust
    rust_code = """fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}"""
    debug_language(rust_code, "rust", "test.rs")
