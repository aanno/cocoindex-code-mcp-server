# Backlog TODOs

* yml is not supported because of `eval_CodeEmbedding_*`
  from `cocoindex evaluate` will trash the scanning (recursion problem?)
* logic for PYTHON_HANDLER_AVAILABLE is not implemented
* AST Visitor in rust
* Use AST Visitor everywhere
* Reactivate skip tests
* Implement haskell support in parity with python
  (improve chunking, check embedding, implement own metadata extractor)
* Graph support (for GraphRAG)
* src/cocoindex-code-mcp-server/lang/python/python_code_analyzer.py is bad
  and needs more tests and fixing
* unify command line argument parsing (in arg_parser.py)
* use cocoindex API (instead of `cocoindex evaluate` and `cocoindex update`)
  (new main for this?)
* test for metadata extraction (for table what is supported where)
* MCP server resource problem (see skip test in tests/mcp_server/test_mcp_integration_http.py)
* Convert other MCP server integration test to tests/mcp_server/test_mcp_integration_http.py technology

## smart embedding

* Smart embedding failed for language Python, falling back to basic model: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.
* device = 'cpu' (search for that)
