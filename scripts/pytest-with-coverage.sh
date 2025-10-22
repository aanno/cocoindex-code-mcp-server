#!/bin/bash -x

# needed for coverage over threads and processes
export COVERAGE_PROCESS_START=/workspaces/rust/.coveragerc
coverage run --data-file=.coverage-test --source=cocoindex_code_mcp_server,tests,python/cocoindex_code_mcp_server \
  -m pytest -c pytest.ini "$@"
