#!/bin/bash -x

# needed for coverage over threads and processes
export COVERAGE_PROCESS_START=/workspaces/rust/.coveragerc
coverage run --data-file=.coverage-test --source=cocoindex-code-mcp-server,tests,src/cocoindex-code-mcp-server \
  -m pytest -c pytest.ini "$@"
