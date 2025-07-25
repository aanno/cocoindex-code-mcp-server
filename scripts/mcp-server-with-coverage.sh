#!/bin/bash -x

# needed for coverage over threads and processes
export COVERAGE_PROCESS_START=/workspaces/rust/.coveragerc
coverage run --source=cocoindex_code_mcp_server,tests,src/cocoindex_code_mcp_server \
  src/cocoindex_code_mcp_server/main_mcp_server.py --coverage /worspaces/rust "$@"
