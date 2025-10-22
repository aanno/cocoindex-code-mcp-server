#!/bin/bash -x

# needed for coverage over threads and processes
export COVERAGE_PROCESS_START=/workspaces/rust/.coveragerc
coverage run --source=cocoindex_code_mcp_server,tests,python/cocoindex_code_mcp_server \
  -m cocoindex_code_mcp_server.main_mcp_server --port 3033 /workspaces/rust/ "$@"
# --coverage has to be reimplemented
