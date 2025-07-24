#!/usr/bin/env python3

"""
Start script for the CocoIndex RAG MCP Server.
This script forwards all arguments to the main MCP server.
"""

import os
import sys
from main_mcp_server import main as server_main
import asyncio

def main():
    """Start the MCP server with forwarded arguments."""
    print("🚀 Starting CocoIndex RAG MCP Server...", file=sys.stderr)
    
    # Check environment variables
    required_env = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing_env = [env for env in required_env if not os.getenv(env)]
    
    if missing_env:
        print(f"⚠️  Warning: Missing environment variables: {missing_env}", file=sys.stderr)
        print("Using defaults (this may cause connection errors):", file=sys.stderr)
        print("  DB_HOST=localhost, DB_NAME=cocoindex, DB_USER=postgres, DB_PASSWORD=password", file=sys.stderr)
        print(file=sys.stderr)
    
    # Forward all command line arguments to the main server
    # This preserves the original sys.argv for argument parsing
    try:
        asyncio.run(server_main())
    except KeyboardInterrupt:
        print("\n👋 MCP server stopped.", file=sys.stderr)
    except Exception as e:
        print(f"❌ Server error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
