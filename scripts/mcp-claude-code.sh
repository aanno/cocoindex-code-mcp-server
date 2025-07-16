#!/bin/bash

# must run in dev container
pnpm dlx supergateway --stdio '/home/vscode/.volta/bin/claude mcp serve' --port 3011 --baseUrl http://localhost:3011 --ssePath /sse --messagePath /message
