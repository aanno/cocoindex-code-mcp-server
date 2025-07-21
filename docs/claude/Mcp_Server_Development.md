# MCP Server Development Gotchas

This document captures critical gotchas and lessons learned during the development of the CocoIndex RAG MCP Server. These insights will help future developers avoid common pitfalls and understand the subtle requirements of MCP protocol implementation.

## 🚨 Critical Gotchas

### 1. **Tool Serialization: Null Fields Break Tool Discovery**

**Problem**: Claude Desktop cannot see tools even when MCP protocol communication works perfectly.

**Root Cause**: Including `null` fields in tool definitions confuses Claude Desktop's tool discovery mechanism.

**Symptoms**:
- MCP server shows as "connected" in Claude Desktop
- Direct HTTP tests return tools correctly
- `tools/list` method works via curl/supergateway
- But Claude Desktop shows no available tools

**Bad Implementation**:
```python
# This includes null fields and breaks Claude Desktop
return [types.Tool(name="my_tool", description="...", inputSchema={})]
# Results in: {"name": "my_tool", "title": null, "description": "...", "outputSchema": null, ...}
```

**Correct Implementation**:
```python
# Use exclude_none=True to remove null fields
tools = await handle_list_tools()
return {
    "result": {"tools": [tool.model_dump(mode='json', exclude_none=True) for tool in tools]}
}
```

**Why This Matters**: The MCP specification shows minimal tool examples without optional null fields. Claude Desktop expects this clean format and fails to parse tools with unexpected null values.

### 2. **HTTP vs Stdio Transport Confusion**

**Problem**: Mixing up HTTP and stdio transport requirements leads to connection failures.

**Key Differences**:

| Aspect | Stdio Transport | HTTP Transport |
|--------|----------------|----------------|
| Claude Desktop Integration | Via MCP client library | Via supergateway proxy |
| Stdout Requirements | Must be 100% clean JSON-RPC | Normal application output OK |
| Debugging | Very difficult | Easy with curl/browser |
| Error Messages | Cryptic protocol errors | Clear HTTP status codes |
| Development | Complex setup | Simple server/client |

**Gotcha**: Don't use stdio transport for development - always start with HTTP transport and only switch to stdio when integrating with Claude Desktop.

### 3. **Supergateway Configuration Pitfalls**

**Problem**: Wrong supergateway flags cause silent connection failures.

**Transport Type Matching**:
```bash
# WRONG: Using SSE flag with StreamableHTTP server
pnpm dlx supergateway --sse "http://localhost:3033/mcp"

# CORRECT: Using streamableHttp flag with StreamableHTTP server  
pnpm dlx supergateway --streamableHttp "http://localhost:3033/mcp"
```

**Debugging**: Always test supergateway connection separately before blaming your MCP server:
```bash
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}' | \
pnpm dlx supergateway --streamableHttp "http://localhost:3033/mcp" --logLevel debug
```

### 4. **Claude Desktop Configuration File Location**

**Gotcha**: Don't edit `.mcp.json` - it's not used by Claude Desktop!

**Wrong File**: `/workspaces/rust/.mcp.json` (ignored by Claude Desktop)
**Correct File**: `~/.config/Claude/claude_desktop_config.json`

**Config Format**:
```json
{
  "mcpServers": {
    "your-server-name": {
      "command": "pnpm",
      "args": ["dlx", "supergateway", "--streamableHttp", "http://localhost:3033/mcp"]
    }
  }
}
```

### 5. **Async Context Management in MCP Handlers**

**Problem**: Improper async handling causes resource leaks or blocking operations.

**Bad Pattern**:
```python
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    # DON'T: Synchronous database operations
    result = sync_database_query(arguments["query"])
    return [types.TextContent(type="text", text=result)]
```

**Good Pattern**:
```python
@server.call_tool() 
async def handle_call_tool(name: str, arguments: dict):
    # DO: Proper async operations with connection pooling
    async with connection_pool.connection() as conn:
        result = await async_database_query(conn, arguments["query"])
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
```

### 6. **Error Handling and Client Experience**

**Problem**: Poor error handling creates confusing user experience.

**Bad Error Response**:
```python
# DON'T: Let exceptions bubble up as MCP protocol errors
async def handle_call_tool(name: str, arguments: dict):
    result = some_operation_that_might_fail()  # Raises exception
    return [types.TextContent(type="text", text=result)]
```

**Good Error Response**:
```python
# DO: Catch exceptions and return user-friendly error messages
async def handle_call_tool(name: str, arguments: dict):
    try:
        result = some_operation_that_might_fail()
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        return [types.TextContent(
            type="text", 
            text=f"Error executing tool '{name}': {str(e)}"
        )]
```

### 7. **JSON Schema Validation Gotchas**

**Problem**: Invalid JSON schemas cause tool calls to fail silently.

**Common Mistakes**:
```python
# DON'T: Invalid schema structure
inputSchema = {
    "properties": {  # Missing "type": "object"
        "query": {"type": "string"}
    }
}

# DON'T: Wrong required format
inputSchema = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": "query"  # Should be ["query"]
}
```

**Correct Schema**:
```python
inputSchema = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query text"
        }
    },
    "required": ["query"]
}
```

### 8. **Database Connection Management**

**Problem**: Improper connection handling causes resource exhaustion.

**Anti-Pattern**:
```python
# DON'T: Create new connections for each request
async def search_database(query: str):
    conn = await asyncpg.connect(DATABASE_URL)
    result = await conn.fetch("SELECT * FROM table WHERE text = $1", query)
    await conn.close()
    return result
```

**Best Practice**:
```python
# DO: Use connection pooling
connection_pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=10)

async def search_database(query: str):
    async with connection_pool.connection() as conn:
        result = await conn.fetch("SELECT * FROM table WHERE text = $1", query)
        return result
```

### 9. **Development vs Production Configuration**

**Problem**: Hardcoded development settings cause production failures.

**Gotchas**:
- Database URLs with localhost
- Fixed port numbers
- Debug logging enabled
- Missing environment variable handling

**Solution**: Always use environment variables with sensible defaults:
```python
import os

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
MCP_PORT = int(os.getenv("MCP_PORT", "3033"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
```

### 10. **Testing Strategy Pitfalls**

**Problem**: Testing only happy paths misses integration issues.

**What to Test**:
1. **Protocol Compliance**: Test actual JSON-RPC message exchange
2. **Tool Schema Validation**: Ensure schemas work with real MCP clients
3. **Error Conditions**: Test database failures, invalid inputs, timeouts
4. **Integration**: Test with actual supergateway and Claude Desktop
5. **Performance**: Test with realistic data sizes and concurrent requests

**Testing Tools**:
```python
# Create test utilities like these:
def test_mcp_protocol_compliance():
    """Test actual MCP protocol messages"""
    
def test_claude_desktop_simulation():
    """Simulate Claude Desktop interaction"""
    
def test_tool_schema_validation():
    """Validate tool schemas work with MCP clients"""
```

## 🔍 Debugging Techniques

### 1. **MCP Protocol Debugging**

**Test Server Directly**:
```bash
curl -X POST http://localhost:3033/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```

**Test Supergateway Connection**:
```bash
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}' | \
timeout 5s pnpm dlx supergateway --streamableHttp "http://localhost:3033/mcp" --logLevel debug
```

### 2. **Tool Format Validation**

Create a validation script to check your tool format:
```python
def validate_tool_format(tool_response):
    """Check if tool response matches Claude Desktop expectations"""
    tools = tool_response.get("result", {}).get("tools", [])
    for tool in tools:
        # Check for null fields that confuse Claude Desktop
        null_fields = [k for k, v in tool.items() if v is None]
        if null_fields:
            print(f"WARNING: Tool '{tool['name']}' has null fields: {null_fields}")
        
        # Validate required fields
        required = ["name", "description", "inputSchema"]
        missing = [f for f in required if f not in tool]
        if missing:
            print(f"ERROR: Tool '{tool['name']}' missing fields: {missing}")
```

### 3. **Connection Tracing**

Add logging to trace the connection flow:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# In your MCP handlers
logger = logging.getLogger(__name__)

@server.list_tools()
async def handle_list_tools():
    logger.debug("Tools list requested")
    tools = [...]
    logger.debug(f"Returning {len(tools)} tools")
    return tools
```

## 📚 Best Practices

### 1. **Start Simple**
- Begin with HTTP transport, not stdio
- Test with curl before integrating with Claude Desktop
- Start with one simple tool, then expand

### 2. **Follow the Specification**
- Use minimal tool definitions without optional null fields
- Include proper JSON schemas for all parameters
- Handle errors gracefully with user-friendly messages

### 3. **Test Integration Early**
- Test with supergateway before Claude Desktop integration
- Create simulation scripts for debugging
- Validate tool format matches expectations

### 4. **Monitor and Log**
- Log all MCP protocol interactions
- Monitor database connection usage
- Track tool usage patterns

### 5. **Version Compatibility**
- Pin MCP library versions in requirements
- Test with multiple MCP client versions
- Document compatibility requirements

## 🎯 Key Success Factors

1. **Clean Tool Serialization**: Always use `exclude_none=True`
2. **Proper Transport Configuration**: Match supergateway flags to server type
3. **Comprehensive Testing**: Test protocol, integration, and edge cases
4. **Error Handling**: Provide clear, actionable error messages
5. **Documentation**: Document gotchas and debugging procedures

Remember: MCP is a protocol specification, not just a Python library. Understanding the protocol requirements is crucial for successful implementation.