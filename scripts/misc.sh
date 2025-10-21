#!/bin/bash

# missing vs code extensions:
# claude-code
# vs code mcp server (JuehangQin.vscode-mcp-server)

# important in bashrc
. ~/.venv/bin/activate
# path must contain
# /home/vscode/.local/share/pnpm
# $ pnpm install -g @anthropic-ai/claude-code
#  ERROR  The configured global bin directory "/home/vscode/.local/share/pnpm" is not in PATH

pnpm install -g @anthropic-ai/claude-code

sudo chown -R vscode:vscode /home/vscode/.cargo/
# /home/vscode/.local/share/pnpm/global/5
ln -s /home/vscode/.local/share/pnpm/global/5/node_modules/\@anthropic-ai/claude-code/cli.js ~/.volta/bin/claude

pip install --upgrade pip
pip install pre-commit maturin psycopg "psycopg[pool]" pgvector "sentence-transformers"
# ensure that this is _not_ saved in .local

sudo chown -R vscode:vscode /home/vscode/.cargo/

# In main_interactive_query.py, only main and the argument parsing should reside. We have to refactor it in multiple files and name them appropriate...

# Why we use tree-sitter-haskell in version 0.21 instead of 0.23.1 ?

# Add least the need the following extra support:
#
# * SQL
# * shell/bash
# * Kotlin
# * dart
# * Markdown
# * asciidoc

# Is there a way to add ASTChunk as dependency (instead of using the check-out submodule)?
