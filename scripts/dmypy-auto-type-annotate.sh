#!/bin/bash -x

dmypy start -- --config-file pyproject.toml --check-untyped-defs
dmypy check src/cocoindex_code_mcp_server/main_mcp_server.py

#
# auto-type-annotate --application-directories .:src src/cocoindex_code_mcp_server/*/*/*.py
# mypy --install-types --non-interactive
