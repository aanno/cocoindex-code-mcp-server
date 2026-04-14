# Fix Plan — BUG-1: Rust AST Chunking Falls Back to Raw Line Windows

**Date:** 2026-04-14
**Affected file:** `python/cocoindex_code_mcp_server/ast_chunking.py`
**Symptom:** All Rust results carry `chunking_method: ast_fallback_unavailable`.
**Regression test:** `tests/test_rust_ast_chunking.py` — 5 tests currently FAIL (intentional).

---

## Root cause (confirmed)

`LANGUAGE_MAP` in `ast_chunking.py` has a comment: _"C++, C, Kotlin, Rust are not supported by ASTChunk library"_.
`_is_supported_language("Rust")` therefore returns `False`.
`_fallback_chunking()` has no Rust branch, so Rust falls straight to
`_simple_text_chunking(code, language, "ast_fallback_unavailable")` — raw 180-line windows.

`tree-sitter-rust` Python bindings **are already installed** (`pyproject.toml` line 74:
`"tree-sitter-rust>=0.21.2,<0.23.0"`), so the grammar is available but unused.

---

## Options

### Option 1 — Pure-Python tree-sitter-rust in `ast_chunking.py`

Add a `_rust_tree_sitter_chunking()` method to `ASTChunkExecutor` (Python side only).
Use the existing `tree-sitter` + `tree-sitter-rust` Python bindings directly:

```python
import tree_sitter_rust
from tree_sitter import Language, Parser

RUST_LANGUAGE = Language(tree_sitter_rust.language())
RUST_CHUNK_NODE_TYPES = {
    "function_item", "struct_item", "impl_item",
    "enum_item", "trait_item", "mod_item",
}

def _rust_tree_sitter_chunking(self, code: str) -> List[ChunkRow]:
    parser = Parser(RUST_LANGUAGE)
    tree = parser.parse(code.encode())
    # walk top-level nodes and yield chunks
    ...
```

Wire it into `_fallback_chunking()` with an `elif language == "Rust":` branch.

**Constraint:** Our `pyproject.toml` pins `tree-sitter>=0.23.0,<0.24.0`.
The Python API for `Language(tree_sitter_rust.language())` was introduced in tree-sitter 0.22;
it works with 0.23.x. ✓

### Option A — Pin astchunk to unmerged PR #6 fork

**PR:** [yilinjz/astchunk#6](https://github.com/yilinjz/astchunk/pulls) by `xiaoquisme`
_"update tree sitter language pack to support 40+ languages"_ (open as of 2026-04-14).

The PR replaces all individual tree-sitter grammar packages with `tree-sitter-language-pack>=0.6.0`,
expands `LANGUAGE_MAP` to 40+ languages **including `"rust": "rust"`**, and adds `get_supported_languages()`.

To use it, `pyproject.toml` would change to:
```toml
"astchunk @ git+https://github.com/xiaoquisme/astchunk.git@<branch>"
"tree-sitter-language-pack>=0.6.0"
```

**Key incompatibility:** PR #6 requires `tree-sitter>=0.25.0`, but our project currently pins
`tree-sitter>=0.23.0,<0.24.0` (required for `tree-sitter-haskell 0.23.x` and other individual
grammar packages). Adopting this PR would require bumping the entire tree-sitter dependency stack
**and** rebuilding the `_haskell_tree_sitter` Rust extension (Cargo.toml pins `tree-sitter = "0.23"`).
That is a large, risky change for a one-language fix.

### Option B — Rust extension in `rust/src/lib.rs` via PyO3 + maturin

The project already has a Rust+PyO3 extension (`_haskell_tree_sitter` module) built with maturin.
It uses `tree-sitter-haskell` via `tree_sitter_haskell::LANGUAGE` and implements recursive
semantic chunking (`extract_semantic_chunks_with_recursive_splitting`), error-aware splitting,
chunk merging, and rich `ChunkingResult` / `HaskellChunk` output types.

Adding Rust language support means:
1. Add `tree-sitter-rust = "0.23"` to `rust/Cargo.toml`.
2. Implement `get_rust_ast_chunks(source: &str) -> PyResult<Vec<HaskellChunk>>` in `lib.rs`,
   mirroring `get_haskell_ast_chunks()` but with `tree_sitter_rust::LANGUAGE` and Rust node types:
   `function_item`, `struct_item`, `impl_item`, `enum_item`, `trait_item`, `mod_item`.
3. Export the new function from the `#[pymodule]` block.
4. Run `maturin develop` to rebuild.
5. Add an `elif language == "Rust":` branch in `_fallback_chunking()` (`ast_chunking.py`)
   that calls `hts.get_rust_ast_chunks(code)`.

The `HaskellChunk` struct is generic (fields: `text`, `start_byte`, `end_byte`,
`start_line`, `end_line`, `node_type`, `metadata`) and reusable for Rust.

The chunking method string would be `"rust_ts_ast"` (or `"rust_ts_ast_with_errors"`
if errors are encountered, following the Haskell naming convention).

---

## Trade-offs comparison

| | Option 1 | Option A | Option B |
|---|---|---|---|
| **Rust added** | ✓ (Python tree-sitter) | ✓ (via astchunk PR #6) | ✓ (Rust + PyO3) |
| **Build step needed** | No | No | Yes (`maturin develop`) |
| **Dependency risk** | Low (bindings already installed) | High (unmerged PR, tree-sitter version conflict) | Low (add one Cargo dep) |
| **tree-sitter upgrade needed** | No | Yes (0.23 → 0.25, full stack) | No (stays at 0.23) |
| **Chunk quality** | Medium — custom walker written in Python | High — uses astchunk's tested splitter | High — full recursive splitting, error handling, chunk merging |
| **Consistency with Haskell** | Partial | Different code path | Full — same architecture |
| **Naming convention** | Custom | `astchunk_library` (same as Python/Java) | `rust_ts_ast` (mirrors `rust_haskell_ast`) |
| **Future language additions** | Per-language effort | Free (50+ langs available) | Per-language effort |
| **Merge complexity** | Small | Large (breaks Haskell build) | Medium |
| **Recommended?** | Fallback if B fails | Not recommended now | **Yes** |

**Recommendation: Option B.**
It reuses the existing, battle-tested Rust chunking infrastructure, requires no dangerous
dependency bumps, and gives better chunk quality than a hand-rolled Python walker.
Option A is attractive long-term but blocked by the `tree-sitter 0.23 → 0.25` upgrade
(which also requires rebuilding the `_haskell_tree_sitter` extension). Defer to a separate ticket.

---

## Detailed implementation steps (Option B)

### Step 1 — `rust/Cargo.toml`

Add one dependency:
```toml
tree-sitter-rust = "0.23"
```

### Step 2 — `rust/src/lib.rs`

#### 2a. Add use declaration
```rust
// at top, after tree_sitter_haskell import
// (tree_sitter_rust provides the LANGUAGE constant)
```

#### 2b. Add Rust-specific chunk node types constant
```rust
/// Top-level Rust AST node types that form natural chunk boundaries
const RUST_CHUNK_NODE_TYPES: &[&str] = &[
    "function_item",
    "struct_item",
    "impl_item",
    "enum_item",
    "trait_item",
    "mod_item",
];
```

#### 2c. Implement `extract_rust_semantic_chunks`

A simplified version of `extract_semantic_chunks_with_recursive_splitting` adapted
for Rust node types. The key difference: Rust `impl_item` blocks can be very large
(contain many methods), so they need recursive splitting on `function_item` children
when they exceed `max_chunk_size`.

Chunking method names:
- `"rust_ts_ast"` — clean parse
- `"rust_ts_ast_with_errors"` — parse succeeded but error nodes present
- `"rust_ts_regex_fallback"` — error count ≥ `ERROR_FALLBACK_THRESHOLD`

#### 2d. Implement `get_rust_ast_chunks` pyfunction
```rust
#[pyfunction]
fn get_rust_ast_chunks(source: &str) -> PyResult<Vec<HaskellChunk>> {
    let mut parser = Parser::new();
    let language = tree_sitter_rust::LANGUAGE.into();
    parser.set_language(&language).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to set Rust language: {}", e)
        )
    })?;
    match parser.parse(source, None) {
        Some(tree) => {
            let default_params = ChunkingParams {
                chunk_size: 1800, min_chunk_size: 200,
                chunk_overlap: 0, max_chunk_size: 1800,
            };
            let result = extract_rust_semantic_chunks(&tree, source, &default_params);
            Ok(result.chunks)
        }
        None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Failed to parse Rust source"
        )),
    }
}
```

#### 2e. Export from `#[pymodule]`
```rust
m.add_function(wrap_pyfunction!(get_rust_ast_chunks, m)?)?;
```

### Step 3 — `python/cocoindex_code_mcp_server/ast_chunking.py`

In `_fallback_chunking`, add before the final `_simple_text_chunking` call:
```python
elif language == "Rust":
    try:
        import cocoindex_code_mcp_server._haskell_tree_sitter as hts
        chunks_raw = hts.get_rust_ast_chunks(code)
        result_chunks: List[ChunkRow] = []
        for i, chunk in enumerate(chunks_raw):
            metadata = chunk.metadata()
            result_chunks.append(ChunkRow({
                "content": chunk.text(),
                "location": f"line:{chunk.start_line()}#{i}",
                "start": chunk.start_line(),
                "end": chunk.end_line(),
                "chunking_method": metadata.get("chunking_method", "rust_ts_ast"),
            }))
        LOGGER.info("Rust AST chunking created %s chunks", len(result_chunks))
        return result_chunks
    except Exception as e:
        LOGGER.error("Rust tree-sitter chunking failed: %s", e)
```

### Step 4 — Build

```bash
cd rust && maturin develop
```

### Step 5 — Verify tests

```bash
python3 -m pytest tests/test_rust_ast_chunking.py -v
```

Expected outcome: the 5 tests in `TestRustASTChunkingBug` now **pass**,
`TestRustChunkingFallbackCurrentBehavior::test_current_behavior_rust_gets_fallback` now **fails**
(delete that class after the fix is confirmed).

---

## Files changed summary

| File | Change |
|---|---|
| `rust/Cargo.toml` | Add `tree-sitter-rust = "0.23"` |
| `rust/src/lib.rs` | Add `RUST_CHUNK_NODE_TYPES`, `extract_rust_semantic_chunks`, `get_rust_ast_chunks`, export |
| `python/cocoindex_code_mcp_server/ast_chunking.py` | Add `elif language == "Rust":` branch in `_fallback_chunking` |
| `tests/test_rust_ast_chunking.py` | Delete `TestRustChunkingFallbackCurrentBehavior` class after fix |

---

## Deferred work

- **Option A follow-up ticket:** Upgrade `tree-sitter` stack to 0.25 and adopt astchunk PR #6
  to get C, C++, Go, Kotlin, Swift, etc. as a single batch. This is a larger change that also
  requires rebuilding the `_haskell_tree_sitter` extension for the new tree-sitter ABI.
- **`_haskell_tree_sitter` rename:** The module name and `HaskellChunk` / `HaskellParser`
  types are Haskell-specific in name only. A follow-up could rename them to `CodeChunk`,
  `CodeParser`, `_code_tree_sitter` to better reflect multi-language use.
