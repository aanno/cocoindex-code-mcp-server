#!/bin/bash -x

# needed for coverage over threads and processes
export COVERAGE_PROCESS_START=/workspaces/rust/.coveragerc
coverage run --source=cocoindex-code-mcp-server,tests,src/cocoindex-code-mcp-server \
  src/cocoindex-code-mcp-server/main_mcp_server.py --coverage /worspaces/rust "$@"
