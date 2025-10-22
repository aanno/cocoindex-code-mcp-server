#!/bin/bash -x

mypy --config-file pyproject.toml --check-untyped-defs python/cocoindex_code_mcp_server
