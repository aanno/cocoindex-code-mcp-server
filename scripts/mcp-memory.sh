#!/bin/bash

# must run _outside_ dev container
pnpm dlx supergateway --stdio 'uv --directory /stratis/home/tpasch/dev/scm/github/mcp-memory-service run memory' --port 3012 --baseUrl http://localhost:3012 --ssePath /sse --messagePath /message
