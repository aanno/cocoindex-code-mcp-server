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

# In main.py, only main and the argument parsing should reside. We have to refactor it in multiple files and name them appropriate... 

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


# At src/hybrid_search.py run_interactive_hybrid_search the line 316:\\
#   \
#   vector_query = input("Vector query (semantic search): ").strip()\
#   \
#   has the problem to not handle multi line input.\
#   This interactive input should be refactored by using the prompt_toolkit 
#   library allowing users to enter text with newlines (\n, \r\n) and mark 
#   completion with keys Ctrl+Enter. Along this lines:

#   from prompt_toolkit import PromptSession
# from prompt_toolkit.key_binding import KeyBindings

# bindings = KeyBindings()
# @bindings.add('c-enter')  # Ctrl+Enter
# def _(event):
#     event.app.exit(result=event.app.current_buffer.text)

# session = PromptSession(key_bindings=bindings, multiline=True)

# print("Enter your text (finish with Ctrl-Enter):")
# text = session.prompt()
# print("You entered:\n", text)

