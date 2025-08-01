#!/bin/bash -x

mypy --config-file pyproject.toml --check-untyped-defs src/cocoindex_code_mcp_server
