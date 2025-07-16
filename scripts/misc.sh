#!/bin/bash

# missing vs code extensions:
# claude-code
# vs code mcp server (JuehangQin.vscode-mcp-server)

pnpm install -g @anthropic-ai/claude-code

pnpm list --global --json --long claude-code
# /home/vscode/.local/share/pnpm/global/5
ln -s /home/vscode/.local/share/pnpm/global/5/node_modules/\@anthropic-ai/claude-code/cli.js .volta/bin/claude

pip install --upgrade pip
pip install pre-commit maturin psycopg "psycopg[pool]" pgvector "sentence-transformers" 
# ensure that this is _not_ saved in .local

sudo chown -R vscode:vscode /home/vscode/.cargo/

# In main.py, only main and the argument parsing should reside. We have to refactor it in multiple files and name them appropriate... 
