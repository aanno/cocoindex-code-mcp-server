# RAG / cocoindex-rag MCP — Findings & Known Issues

Tested 2026-04-14 against the indexed Java Spring Boot project and `cocoindex-code-mcp-server`.

---

## What is indexed

| Language | Source paths | Chunking method |
|---|---|---|
| Java | `admin-service/`, `data-mgmt-service/`, … | `astchunk_library` (tree-sitter AST) |
| Python | `cocoindex-code-mcp-server/**/*.py` | `ast_tree_sitter` |
| Rust | `cocoindex-code-mcp-server/rust/src/lib.rs` | `ast_fallback_unavailable` (raw windows) |

---

## Bugs

### BUG-1: Rust AST chunking falls back to raw line windows

**Symptom:** All Rust results carry `chunking_method: ast_fallback_unavailable`.
The AST chunker cannot split Rust code, so chunks are raw 180-line windows — far too large and without semantic boundaries.

**Impact:** Vector search quality for Rust is poor; chunks contain multiple unrelated functions.
The Rust AST visitor (`rust_ast_visitor`) successfully *analyses* the code (structs, impls, functions are extracted), but the *chunker* doesn't split it.

**Root cause to investigate:** `ASTChunkOperation` in `cocoindex_config.py` / `astchunk` library — Rust is either not registered or the tree-sitter grammar for Rust is missing from the astchunk chunker path.

---

### BUG-2: Low vector relevance scores for Java (GraphCodeBERT)

Important for queries, but will not be fixed soon.

**Symptom:** Java vector search returns scores in the 0.07–0.10 range, effectively random.

**Root cause:** Java is embedded with `microsoft/graphcodebert-base` (see `LANGUAGE_MODEL_GROUPS` in `cocoindex_config.py`). GraphCodeBERT was pretrained exclusively on English-language open-source code. This project's Java codebase uses German domain identifiers throughout — `Datensatz`, `Datensegment`, `MandantenBzr`, `Quelldatensegment`, `Erhebungsteilnehmer`, etc. These tokens are out-of-distribution for GraphCodeBERT; it maps them into a region of the embedding space that does not correlate with English query terms.

**Workarounds (no re-index needed):**

- Use `search-hybrid` with `keyword_weight: 0.6–0.7` instead of the default 0.3 — keyword search is unaffected by language and works well.
- Query using actual domain terms (`datensatz lock release`) rather than English descriptions (`record lock release`).

**Fix (requires re-index):** Switch Java from GraphCodeBERT to the fallback model `sentence-transformers/all-mpnet-base-v2`. This general-purpose model was trained on a much broader corpus including non-English text and handles mixed-language identifiers better. Change `LANGUAGE_MODEL_GROUPS["graphcodebert"]["languages"]` to exclude `"java"`, so Java falls through to the `"fallback"` group. Then re-run `cocoindex update`. See also BUG-3 for a better long-term option.

**Long-term fix:** Replace or augment GraphCodeBERT for Java with a multilingual model, e.g. `intfloat/multilingual-e5-base` or `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`. These handle German identifiers natively. However, this requires adding a new model group, re-indexing, and ensuring vector dimensions are consistent (all current models output 768D — `multilingual-e5-base` is also 768D, so it is a drop-in).

### `search-vector` and `search-hybrid` enforce `language` param

This is NOT a bug, but a design choice. That way, we could easily use different embedding models for different languages (e.g. GraphCodeBERT for Java/Python, UniXcoder for Rust/TypeScript, fallback for everything else) without risking invalid cross-model vector comparisons.

**Symptom:** Calling `search-vector` or `search-hybrid` without `language` or `embedding_model` returns:

```
Either 'language' or 'embedding_model' parameter is required for search.
```

**Reason:** The index stores embeddings from multiple models (GraphCodeBERT for Java/Python/etc., UniXcoder for Rust/TypeScript/etc., fallback for others). Comparing vectors across models is mathematically invalid, so the server enforces model isolation. This is *correct behaviour*, not a bug per se — but it prevents cross-language semantic queries without specifying a model explicitly.

**Workaround:** Use `search-keyword` for language-agnostic searches, or specify `embedding_model` explicitly (e.g. `embedding_model: "microsoft/graphcodebert-base"`) to search across all languages embedded with that model.

## Minor observations

- **Vector score ceiling is low even for perfect matches (~0.10).** This is a cosmetic/calibration issue — cosine similarity values are not being rescaled to a [0, 1] display range. Relative ranking is still meaningful; absolute scores are not.
- **Java imports field is always empty** in chunk metadata (`imports: []`). The `java_ast_visitor` analysis method extracts functions, classes, packages, and annotations but skips import declarations in the chunk slice output. Full-file import detection works (visible in raw `metadata_json.imports`), but it is not propagated to the flat schema columns.
- **Python function name extraction occasionally wrong:** The Python analyzer sometimes reports internal parse artefacts as function names (e.g. `"rbo"`, `"irm:"`, `"    \"\""` instead of the real names). Appears to be a tree-sitter edge case with decorated functions where the decorator line is parsed ambiguously.
