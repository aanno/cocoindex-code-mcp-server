# Current State - CocoIndex Code MCP Server Integration

## Problem Status: CRITICAL - CocoIndex Evaluation Still Hanging

### Current Issue
- `cocoindex evaluate` command is still not terminating properly
- Despite fixing infinite recursion issues in Python AST analyzer, the evaluation continues to hang
- Need to implement and test `--default-*` flags to isolate whether the issue is in our custom extensions

### Last Actions Taken
1. **Fixed infinite recursion in AST traversal** (lines 75-112 in python_code_analyzer.py):
   - Added max recursion depth limit (200 levels)
   - Implemented cycle detection using visited_nodes set
   - Added depth tracking to prevent stack overflow
   - Limited code chunk size to 100KB
   - Added recursion protection to _get_type_annotation, _get_decorator_name, _get_ast_value methods

2. **Improved error handling** in metadata extraction:
   - Changed syntax error logging from WARNING to DEBUG level
   - Enhanced fallback metadata with all required fields
   - Made individual field extraction functions more robust

3. **Discovered missing flag implementation**:
   - Found that `--default-embedding`, `--default-chunking`, `--default-language-handler` flags are defined in MCP server but NOT being passed to `update_flow_config()`
   - These flags are critical for testing whether our custom extensions are causing the hang

### Next Critical Actions Needed
1. **URGENT**: Fix the MCP server to pass default flags to `update_flow_config()` (line ~1419 in mcp_server.py)
2. **Test with all default flags**: Run `cocoindex evaluate` with `--default-embedding --default-chunking --default-language-handler` to isolate the issue
3. **If defaults work**: Systematically re-enable extensions one by one to identify the problematic component
4. **If defaults still hang**: The issue is in base CocoIndex or our flow configuration, not our extensions

### Files Modified in Last Session
- `/workspaces/rust/src/cocoindex-code-mcp-server/lang/python/python_code_analyzer.py` - Added recursion limits and cycle detection
- `/workspaces/rust/src/cocoindex-code-mcp-server/cocoindex_config.py` - Improved error handling in metadata extraction
- `/workspaces/rust/src/cocoindex-code-mcp-server/arg_parser.py` - Added missing default flag definitions

### Critical Code Locations
- **MCP server flag handling**: Line 1419 in mcp_server.py - `update_flow_config()` call missing default flags
- **Flow configuration**: Lines 620-632 in cocoindex_config.py - `update_flow_config()` function
- **AST analyzer protection**: Lines 75-112 in python_code_analyzer.py - Recursion protection

### Evaluation Directories
- Latest attempt: `eval_CodeEmbedding_250723_113505` (aborted, hanging)
- Previous attempts moved to `old/` folder

### TODO List Status
All major fixes completed except:
- [ ] **CRITICAL**: Implement proper flag passing from MCP server to flow config
- [ ] Test evaluation with default flags to isolate issue source
- [ ] Create unit tests for python_code_analyzer.py (mentioned as needed)

### Key Insight
The user correctly identified that we need to test with `--default-*` flags to determine if the issue is in our custom extensions or in the base CocoIndex functionality. This is the most logical next step for debugging.