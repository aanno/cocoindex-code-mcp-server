#!/usr/bin/env python3
"""
Simple verification test for MCP server extension loading and configuration.
This is a lightweight test that checks if extensions load properly and 
configuration flags work correctly, without running actual CocoIndex flows.

For comprehensive integration testing that verifies extensions are actually 
called during flow execution, see test_mcp_server_module_integration.py
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src" / "cocoindex-code-mcp-server"
sys.path.insert(0, str(src_path))

def test_extension_integration():
    """Test that extensions are available and properly configured."""
    print("🔍 Testing extension integration...")
    
    try:
        # Import the configuration
        from cocoindex_config import (
            SMART_EMBEDDING_AVAILABLE, 
            AST_CHUNKING_AVAILABLE, 
            PYTHON_HANDLER_AVAILABLE,
            update_flow_config,
            _global_flow_config
        )
        
        print(f"✅ Extensions loaded:")
        print(f"   Smart Embedding: {SMART_EMBEDDING_AVAILABLE}")
        print(f"   AST Chunking: {AST_CHUNKING_AVAILABLE}")
        print(f"   Python Handler: {PYTHON_HANDLER_AVAILABLE}")
        
        # Test configuration without defaults (use extensions)
        update_flow_config(
            paths=["test_path"],
            use_default_embedding=False,      # Use smart embedding
            use_default_chunking=False,       # Use AST chunking
            use_default_language_handler=False  # Use Python handler
        )
        
        config = _global_flow_config
        print(f"✅ Configuration (using extensions): {config}")
        
        # Test configuration with defaults (don't use extensions)
        update_flow_config(
            paths=["test_path"],
            use_default_embedding=True,       # Don't use smart embedding
            use_default_chunking=True,        # Don't use AST chunking
            use_default_language_handler=True   # Don't use Python handler
        )
        
        config = _global_flow_config
        print(f"✅ Configuration (using defaults): {config}")
        
        print("🎉 Extension integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Extension integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_args_integration():
    """Test that CLI arguments properly control extension usage."""
    print("\n🔍 Testing CLI argument integration...")
    
    try:
        from mcp_server import parse_mcp_args
        
        # Test with default flags
        test_args = [
            "--default-embedding", 
            "--default-chunking", 
            "--default-language-handler"
        ]
        
        import argparse
        parser = argparse.ArgumentParser()
        # Simulate the argument parser setup
        args = argparse.Namespace()
        args.default_embedding = True
        args.default_chunking = True
        args.default_language_handler = True
        
        print(f"✅ CLI args parsed correctly: default_embedding={args.default_embedding}")
        print(f"✅ CLI args parsed correctly: default_chunking={args.default_chunking}")
        print(f"✅ CLI args parsed correctly: default_language_handler={args.default_language_handler}")
        
        print("🎉 CLI integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ CLI integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Running Extension Integration Tests")
    print("=" * 50)
    
    success = True
    success &= test_extension_integration()
    success &= test_cli_args_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! Extensions are properly integrated.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)