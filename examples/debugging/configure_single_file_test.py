#!/usr/bin/env python3

"""
Configure CocoIndex to test a single file.
"""

import sys
sys.path.insert(0, 'src')

from cocoindex_code_mcp_server.cocoindex_config import update_flow_config

# Configure for single file test
update_flow_config(
    paths=['src/cocoindex_code_mcp_server/language_handlers/cpp_visitor.py'], 
    use_default_chunking=False,  # Use AST chunking
    use_default_language_handler=False  # Use proper language handler
)

print("✅ Configured CocoIndex for single file test")
print("📁 File: src/cocoindex_code_mcp_server/language_handlers/cpp_visitor.py")
print("🔧 AST chunking: enabled")
print("🔧 Proper language handler: enabled")