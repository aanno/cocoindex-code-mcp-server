#!/bin/bash -x

python -m cocoindex_code_mcp_server.main_mcp_server --rescan --default-embedding --port 3033 /workspaces/rust/
