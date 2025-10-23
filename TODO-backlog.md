# Backlog TODOs

* yml is not supported because of `eval_CodeEmbedding_*`
  from `cocoindex evaluate` will trash the scanning (recursion problem?)
* AST Visitor in rust
* Use AST Visitor everywhere
* Reactivate skipped tests
* Improve haskell support to parity with python
  (improve chunking, check embedding, implement own metadata extractor)
  - variables should not be detected as functions
* Graph support (for GraphRAG)
* python/cocoindex_code_mcp_server/lang/python/python_code_analyzer.py is bad
  and needs more tests and fixing
* use cocoindex API (instead of `cocoindex evaluate` and `cocoindex update`)
  (new main for this?)
* test for metadata extraction (for table what is supported where)
* MCP server resource problem (see skip test in tests/mcp_server/test_mcp_integration_http.py)
* Convert other MCP server integration test to tests/mcp_server/test_mcp_integration_http.py technology
* pgvector on podman: use a volume for the data
* bug report against python mcp library for the resource bug
* use click for all arguments parsing
* unify command line argument parsing (in arg_parser_old.py)
* add hypothesis property testing: <https://hypothesis.readthedocs.io/en/latest/>
