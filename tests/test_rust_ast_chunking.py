#!/usr/bin/env python3
"""
Regression tests for BUG-1: Rust AST chunking falls back to ast_fallback_unavailable.

Root cause: LANGUAGE_MAP in ast_chunking.py does not include "Rust", so
_is_supported_language("Rust") returns False, and _fallback_chunking() has no
Rust-specific branch — Rust falls through to _simple_text_chunking with
chunking_method="ast_fallback_unavailable".

These tests demonstrate the bug (currently failing) and will pass once fixed.
The fix should add Rust tree-sitter based chunking (tree-sitter-rust is already
a dependency in pyproject.toml).
"""

from pathlib import Path

import pytest

from cocoindex_code_mcp_server.ast_chunking import (
    LANGUAGE_MAP,
    ASTChunkExecutor,
    ASTChunkSpec,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "lang_examples"
RUST_FIXTURE = FIXTURES_DIR / "rust_example_1.rs"


def make_executor() -> ASTChunkExecutor:
    """Create an ASTChunkExecutor instance suitable for direct unit testing.

    We bypass the @op.executor_class() registration machinery using __new__,
    then set the two instance attributes the executor actually needs.
    """
    executor = ASTChunkExecutor.__new__(ASTChunkExecutor)
    executor.spec = ASTChunkSpec(max_chunk_size=1800)
    executor._builders = {}
    return executor


class TestRustASTChunkingBug:
    """BUG-1: Rust falls back to raw line-window chunking instead of AST chunking."""

    # ------------------------------------------------------------------ #
    # Root-cause checks                                                    #
    # ------------------------------------------------------------------ #

    def test_rust_not_in_language_map(self):
        """LANGUAGE_MAP must contain 'Rust' for AST chunking to work.

        Currently FAILS: Rust is explicitly absent from LANGUAGE_MAP.
        """
        assert "Rust" in LANGUAGE_MAP, (
            "Rust is not in LANGUAGE_MAP — this is the root cause of BUG-1. "
            "Add a tree-sitter based entry so Rust does not fall back to simple text chunking."
        )

    def test_is_supported_language_rust(self):
        """ASTChunkExecutor._is_supported_language('Rust') must return True.

        Currently FAILS because LANGUAGE_MAP has no Rust entry.
        """
        executor = make_executor()
        assert executor._is_supported_language("Rust"), (
            "_is_supported_language('Rust') returned False — Rust is not supported "
            "by the ASTChunk library path. A custom tree-sitter branch is needed."
        )

    # ------------------------------------------------------------------ #
    # Symptom checks (what the user observes in search results)            #
    # ------------------------------------------------------------------ #

    def test_rust_fixture_exists(self):
        """Sanity check: the fixture file used by the other tests is present."""
        assert RUST_FIXTURE.exists(), f"Missing fixture: {RUST_FIXTURE}"

    def test_rust_chunking_does_not_produce_fallback_method(self):
        """Chunking Rust code must NOT produce chunking_method='ast_fallback_unavailable'.

        Currently FAILS: all Rust chunks carry 'ast_fallback_unavailable'
        because the language is unsupported and falls through to
        _simple_text_chunking().
        """
        rust_code = RUST_FIXTURE.read_text()
        executor = make_executor()
        chunks = executor(rust_code, language="Rust")

        assert len(chunks) > 0, "Executor returned no chunks for Rust code"

        bad = [c for c in chunks if c.chunking_method == "ast_fallback_unavailable"]
        assert len(bad) == 0, (
            f"{len(bad)}/{len(chunks)} Rust chunk(s) have chunking_method='ast_fallback_unavailable'. "
            "Rust should use a tree-sitter AST-based chunking method, not the raw line-window fallback."
        )

    def test_rust_chunking_produces_semantic_chunks(self):
        """Rust chunks must reflect semantic boundaries (functions/structs/impls), not raw windows.

        Currently FAILS: raw line windows from _simple_text_chunking() don't
        split at function or impl boundaries.

        The fixture file rust_example_1.rs has these top-level items:
          - struct Person (lines 8-11)
          - impl Person  (lines 13-23, containing 2 methods)
          - fn fibonacci (lines 25-31)
          - fn main      (lines 33-48)

        With proper AST chunking each item (or method body) should be its own chunk.
        With the current fallback a single 180-line window swallows everything.
        """
        rust_code = RUST_FIXTURE.read_text()
        executor = make_executor()
        chunks = executor(rust_code, language="Rust")

        # The fixture has 4 top-level items; expect at least that many chunks
        assert len(chunks) >= 4, (
            f"Expected at least 4 semantic chunks for rust_example_1.rs "
            f"(struct + impl + 2 fns), got {len(chunks)}. "
            "This suggests raw line-window chunking is still in use."
        )

    def test_rust_chunk_method_name_indicates_ast(self):
        """chunking_method for Rust chunks must contain 'ast' or 'tree_sitter'.

        Currently FAILS because the value is 'ast_fallback_unavailable'.
        After the fix the method name should be something like 'ast_tree_sitter'
        or 'rust_tree_sitter_ast'.
        """
        rust_code = RUST_FIXTURE.read_text()
        executor = make_executor()
        chunks = executor(rust_code, language="Rust")

        assert len(chunks) > 0, "Executor returned no chunks"

        for chunk in chunks:
            method = chunk.chunking_method
            assert "fallback" not in method, (
                f"Chunk at {chunk.location!r} has chunking_method={method!r} "
                "which contains 'fallback' — this is not a proper AST method."
            )
            assert "ast" in method or "tree_sitter" in method, (
                f"Chunk at {chunk.location!r} has chunking_method={method!r} "
                "which does not indicate AST/tree-sitter based chunking."
            )


class TestRustChunkingFallbackCurrentBehavior:
    """Document the CURRENT (broken) behavior so the bug is unambiguous.

    These tests deliberately assert the broken state and will need to be
    deleted (or inverted) after the fix is applied.
    """

    def test_current_behavior_rust_gets_fallback(self):
        """Confirm that Rust currently produces ast_fallback_unavailable chunks.

        This test PASSES right now (it documents the bug).
        Delete this test once BUG-1 is fixed.
        """
        rust_code = RUST_FIXTURE.read_text()
        executor = make_executor()
        chunks = executor(rust_code, language="Rust")

        assert len(chunks) > 0, "No chunks produced"
        methods = {c.chunking_method for c in chunks}
        assert methods == {"ast_fallback_unavailable"}, (
            f"Expected exactly {{'ast_fallback_unavailable'}} but got {methods}. "
            "If this test starts failing, BUG-1 may be fixed — delete this class."
        )
