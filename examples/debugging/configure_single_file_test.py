#!/usr/bin/env python3

"""
Configure CocoIndex to test a single file.
"""

import sys

from cocoindex_code_mcp_server.cocoindex_config import update_flow_config

sys.path.insert(0, "src")


# Configure for single file test
update_flow_config(
    paths=["python/cocoindex_code_mcp_server/language_handlers/cpp_visitor.py"],
    use_default_chunking=False,  # Use AST chunking
    use_default_language_handler=False,  # Use proper language handler
)

print("✅ Configured CocoIndex for single file test")
print("📁 File: python/cocoindex_code_mcp_server/language_handlers/cpp_visitor.py")
print("🔧 AST chunking: enabled")
print("🔧 Proper language handler: enabled")
