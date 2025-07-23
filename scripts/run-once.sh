#!/bin/bash -x

cd /workspaces/rust
maturin dev
pip install -e ".[mcp-server,test]"
