# Extension Integration Tests

This directory contains tests for MCP server extension integration.

## Test Files

### `test_extension_integration.py`

**Purpose**: Simple verification test for extension loading and configuration

- ✅ Quick sanity check that extensions load properly
- ✅ Verifies CLI flags control extension usage
- ✅ Tests configuration without running actual CocoIndex flows
- ⚡ Fast execution, no external dependencies

**Use when**: You want to quickly verify that extensions are available and configuration works

### `test_main_mcp_server_module_integration.py`

**Purpose**: Comprehensive integration test for actual extension usage

- 🔍 Creates test corpus and runs real CocoIndex flows
- 🕵️ Uses pytest-mock to spy on extension function calls
- ✅ Verifies extensions are actually called during flow execution
- 🐌 Slower execution, requires cocoindex runtime

**Use when**: You want to verify that extensions are actually being called during real usage

## Running Tests

```bash
# Quick verification test
python -m pytest tests/test_extension_integration.py -v

# Comprehensive integration test
python -m pytest tests/test_main_mcp_server_module_integration.py -v

# All extension tests
python -m pytest tests/ -k "extension" -v
```

## Test Strategy

1. **Extension Loading**: Both tests verify extensions load properly
2. **Configuration**: Both tests verify CLI flags control extension usage
3. **Integration**: Only the comprehensive test verifies actual function calls
4. **Corpus Testing**: Only the comprehensive test uses real file processing

## CLI Flags Tested

- `--default-embedding` → bypasses smart_code_embedding.py
- `--default-chunking` → bypasses ast_chunking.py
- `--default-language-handler` → bypasses python_handler.py

When flags are NOT set, extensions are used (default behavior).
