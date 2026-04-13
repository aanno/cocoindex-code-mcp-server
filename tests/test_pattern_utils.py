"""
Tests for pattern_utils: gitignore-style pattern conversion and file filtering.

Uses tests/fixtures/lang_examples as a real directory tree for filesystem tests.
"""

from pathlib import Path

import pytest

from cocoindex_code_mcp_server.pattern_utils import (
    PathFilter,
    collect_gitignore_patterns,
    gitignore_pattern_to_globset,
    parse_gitignore_file,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "lang_examples"


# ---------------------------------------------------------------------------
# gitignore_pattern_to_globset — unit tests
# ---------------------------------------------------------------------------

class TestGitignorePatternToGlobset:
    """Unit tests for single-pattern conversion (new list[str] API)."""

    # Extension patterns: non-anchored, get **/ prefix
    def test_plain_extension_pattern_gets_double_star(self):
        assert gitignore_pattern_to_globset("*.py") == ["**/*.py"]

    # Already-anchored **/ patterns: left as-is
    def test_double_star_pattern_unchanged(self):
        assert gitignore_pattern_to_globset("**/node_modules") == ["**/node_modules"]

    # Non-anchored names: get **/ prefix
    def test_plain_name_gets_double_star(self):
        assert gitignore_pattern_to_globset("target") == ["**/target"]

    # Directory pattern: get **/ prefix + /** children variant
    def test_trailing_slash_produces_two_patterns(self):
        assert gitignore_pattern_to_globset("target/") == ["**/target", "**/target/**"]

    # Root-anchored: no **/ prefix
    def test_leading_slash_anchors_to_root(self):
        assert gitignore_pattern_to_globset("/dist") == ["dist"]

    def test_leading_and_trailing_slash_root_dir(self):
        assert gitignore_pattern_to_globset("/build/") == ["build", "build/**"]

    # Nested (contains / but not root-anchored): not further modified
    def test_nested_pattern_with_trailing_slash(self):
        result = gitignore_pattern_to_globset("src/build/")
        assert "src/build" in result or "**/src/build" in result

    # Empty / skipped patterns
    def test_blank_line_returns_empty_list(self):
        assert gitignore_pattern_to_globset("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert gitignore_pattern_to_globset("   ") == []

    def test_comment_returns_empty_list(self):
        assert gitignore_pattern_to_globset("# this is a comment") == []

    def test_negation_returns_empty_list(self):
        assert gitignore_pattern_to_globset("!important.lock") == []

    def test_negation_with_path_returns_empty_list(self):
        assert gitignore_pattern_to_globset("!src/keep.py") == []

    # Already-anchored **/ patterns with trailing slash
    def test_complex_glob_unchanged(self):
        assert gitignore_pattern_to_globset("**/__pycache__") == ["**/__pycache__"]

    def test_double_star_extension(self):
        assert gitignore_pattern_to_globset("**/*.pyc") == ["**/*.pyc"]

    # Non-anchored names with special characters
    def test_question_mark_gets_double_star(self):
        assert gitignore_pattern_to_globset("file?.py") == ["**/file?.py"]

    def test_character_class_gets_double_star(self):
        assert gitignore_pattern_to_globset("[Mm]akefile") == ["**/[Mm]akefile"]


# ---------------------------------------------------------------------------
# parse_gitignore_file — tests using a real temp .gitignore
# ---------------------------------------------------------------------------

class TestParseGitignoreFile:
    """Tests for reading a .gitignore file and converting its patterns."""

    @pytest.fixture()
    def gitignore(self, tmp_path):
        """Write a .gitignore in tmp_path and return (path, root)."""
        gi = tmp_path / ".gitignore"
        gi.write_text(
            "# Build outputs\n"
            "target/\n"
            "/dist\n"
            "*.lock\n"
            "node_modules\n"
            "!important.lock\n"
            "\n"
            "**/__pycache__\n",
            encoding="utf-8",
        )
        return gi, tmp_path

    def test_returns_list(self, gitignore):
        gi, root = gitignore
        result = parse_gitignore_file(gi, root)
        assert isinstance(result, list)

    def test_comment_and_blank_excluded(self, gitignore):
        gi, root = gitignore
        result = parse_gitignore_file(gi, root)
        assert "# Build outputs" not in result
        assert "" not in result

    def test_trailing_slash_produces_recursive_patterns(self, gitignore):
        gi, root = gitignore
        result = parse_gitignore_file(gi, root)
        # target/ → **/target and **/target/**
        assert "**/target" in result
        assert "**/target/**" in result

    def test_leading_slash_anchored_to_root(self, gitignore):
        gi, root = gitignore
        result = parse_gitignore_file(gi, root)
        assert "dist" in result

    def test_plain_extension_included(self, gitignore):
        gi, root = gitignore
        result = parse_gitignore_file(gi, root)
        assert "**/*.lock" in result

    def test_plain_name_included(self, gitignore):
        gi, root = gitignore
        result = parse_gitignore_file(gi, root)
        assert "**/node_modules" in result

    def test_negation_excluded(self, gitignore):
        gi, root = gitignore
        result = parse_gitignore_file(gi, root)
        # negation pattern itself must not appear
        assert "!important.lock" not in result
        assert "important.lock" not in result

    def test_double_star_included(self, gitignore):
        gi, root = gitignore
        result = parse_gitignore_file(gi, root)
        assert "**/__pycache__" in result

    def test_nested_gitignore_patterns_anchored(self, tmp_path):
        """A .gitignore inside a subdirectory should be prefixed with that subdir."""
        sub = tmp_path / "src"
        sub.mkdir()
        gi = sub / ".gitignore"
        gi.write_text("*.gen\nbuild/\n", encoding="utf-8")
        result = parse_gitignore_file(gi, tmp_path)
        # *.gen → **/src/*.gen (nested, gets **/ prefix from nesting)
        assert any("src" in p and ".gen" in p for p in result), f"No src/*.gen pattern in {result}"
        assert any("src" in p and "build" in p for p in result), f"No src/build pattern in {result}"

    def test_missing_file_returns_empty(self, tmp_path):
        result = parse_gitignore_file(tmp_path / "nonexistent.gitignore", tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# collect_gitignore_patterns — filesystem tests using lang_examples
# ---------------------------------------------------------------------------

class TestCollectGitignorePatterns:
    """Tests for walking a directory tree and collecting .gitignore patterns."""

    def test_no_gitignore_returns_empty(self):
        """lang_examples has no .gitignore so result must be empty."""
        result = collect_gitignore_patterns([str(FIXTURES_DIR)])
        assert result == []

    def test_nonexistent_root_returns_empty(self):
        result = collect_gitignore_patterns(["/nonexistent/path/xyz"])
        assert result == []

    def test_empty_root_list_returns_empty(self):
        result = collect_gitignore_patterns([])
        assert result == []

    def test_single_gitignore_patterns_collected(self, tmp_path):
        """One .gitignore at root — all its patterns should appear."""
        gi = tmp_path / ".gitignore"
        gi.write_text("*.log\nbuild/\n", encoding="utf-8")
        result = collect_gitignore_patterns([str(tmp_path)])
        assert "**/*.log" in result
        assert "**/build" in result
        assert "**/build/**" in result

    def test_multiple_gitignores_merged(self, tmp_path):
        """Patterns from .gitignores in different subdirs are all returned."""
        (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
        sub = tmp_path / "lib"
        sub.mkdir()
        (sub / ".gitignore").write_text("*.o\n", encoding="utf-8")
        result = collect_gitignore_patterns([str(tmp_path)])
        assert "**/*.log" in result
        assert any("lib" in p and ".o" in p for p in result), f"No lib/*.o pattern in {result}"

    def test_nested_gitignore_anchored_to_root(self, tmp_path):
        """Pattern in a nested .gitignore is prefixed with the subdirectory path."""
        sub = tmp_path / "src" / "gen"
        sub.mkdir(parents=True)
        (sub / ".gitignore").write_text("*.pb.go\n", encoding="utf-8")
        result = collect_gitignore_patterns([str(tmp_path)])
        assert any("src/gen" in p and ".pb.go" in p for p in result), (
            f"No src/gen/*.pb.go pattern in {result}"
        )

    def test_multiple_root_paths(self, tmp_path):
        """Patterns from distinct root directories are all collected."""
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        (root_a / ".gitignore").write_text("*.log\n", encoding="utf-8")
        (root_b / ".gitignore").write_text("dist/\n", encoding="utf-8")
        result = collect_gitignore_patterns([str(root_a), str(root_b)])
        assert "**/*.log" in result
        assert "**/dist" in result

    def test_negation_patterns_not_in_result(self, tmp_path):
        """Negation lines in .gitignore must not appear in the output."""
        (tmp_path / ".gitignore").write_text("*.log\n!important.log\n", encoding="utf-8")
        result = collect_gitignore_patterns([str(tmp_path)])
        assert "**/*.log" in result
        assert "!important.log" not in result
        assert "important.log" not in result

    def test_comment_lines_not_in_result(self, tmp_path):
        (tmp_path / ".gitignore").write_text("# ignore build\nbuild/\n", encoding="utf-8")
        result = collect_gitignore_patterns([str(tmp_path)])
        assert "**/build" in result
        assert "# ignore build" not in result


# ---------------------------------------------------------------------------
# Integration: pattern_utils wired into SOURCE_CONFIG / update_flow_config
# ---------------------------------------------------------------------------

class TestUpdateFlowConfigPatterns:
    """Verify that update_flow_config correctly threads patterns into _global_flow_config."""

    def test_extra_excluded_none_by_default(self):
        from cocoindex_code_mcp_server.cocoindex_config import (
            _global_flow_config,
            update_flow_config,
        )
        update_flow_config()
        assert _global_flow_config["extra_excluded_patterns"] is None

    def test_extra_included_none_by_default(self):
        from cocoindex_code_mcp_server.cocoindex_config import (
            _global_flow_config,
            update_flow_config,
        )
        update_flow_config()
        assert _global_flow_config["extra_included_patterns"] is None

    def test_extra_excluded_stored(self):
        from cocoindex_code_mcp_server.cocoindex_config import (
            _global_flow_config,
            update_flow_config,
        )
        update_flow_config(extra_excluded_patterns=["*.lock", "**/vendor/**"])
        assert _global_flow_config["extra_excluded_patterns"] == ["*.lock", "**/vendor/**"]

    def test_extra_included_stored(self):
        from cocoindex_code_mcp_server.cocoindex_config import (
            _global_flow_config,
            update_flow_config,
        )
        update_flow_config(extra_included_patterns=["*.nix", "*.dhall"])
        assert _global_flow_config["extra_included_patterns"] == ["*.nix", "*.dhall"]

    def test_extra_included_empty_list_replaces_defaults(self):
        """Passing an empty list is different from None — it replaces the include list."""
        from cocoindex_code_mcp_server.cocoindex_config import (
            _global_flow_config,
            update_flow_config,
        )
        update_flow_config(extra_included_patterns=[])
        assert _global_flow_config["extra_included_patterns"] == []

    def teardown_method(self, _method):
        """Reset config after each test so state doesn't leak."""
        from cocoindex_code_mcp_server.cocoindex_config import update_flow_config
        update_flow_config()


# ---------------------------------------------------------------------------
# Integration: gitignore_pattern_to_globset used via CLI normalization path
# ---------------------------------------------------------------------------

class TestCliPatternNormalization:
    """Test the normalization logic that main() applies before update_flow_config()."""

    def _normalize_excludes(self, patterns):
        result = []
        for p in patterns:
            result.extend(gitignore_pattern_to_globset(p))
        return result or None

    def _normalize_includes(self, patterns):
        if not patterns:
            return None
        result = []
        for p in patterns:
            result.extend(gitignore_pattern_to_globset(p))
        return result

    def test_exclude_trailing_slash_normalized(self):
        result = self._normalize_excludes(["target/"])
        assert result == ["**/target", "**/target/**"]

    def test_exclude_multiple_patterns(self):
        result = self._normalize_excludes(["*.lock", "**/tests/**"])
        assert result == ["**/*.lock", "**/tests/**"]

    def test_exclude_negation_silently_dropped(self):
        result = self._normalize_excludes(["!keep.log"])
        assert result is None  # nothing left after dropping negation

    def test_include_replaces_when_given(self):
        result = self._normalize_includes(["*.nix", "*.dhall"])
        assert result == ["**/*.nix", "**/*.dhall"]

    def test_include_none_when_empty_tuple(self):
        result = self._normalize_includes(())
        assert result is None

    def test_include_leading_slash_stripped(self):
        result = self._normalize_includes(["/src/*.py"])
        # /src/*.py is root-anchored (has leading /); contains / so NOT further prefixed
        assert result is not None
        assert any("src" in p and ".py" in p for p in result), f"Unexpected: {result}"


# ---------------------------------------------------------------------------
# .gitignore integration: collect_gitignore_patterns wired via main() logic
# ---------------------------------------------------------------------------

class TestGitignoreIntegration:
    """Test .gitignore auto-exclusion as wired in main() — without invoking Click."""

    def _run_gitignore_step(self, final_paths, no_gitignore=False):
        """Replicate the gitignore block from main() for testing."""
        normalized_excludes: list[str] = []
        if not no_gitignore and final_paths:
            gitignore_excludes = collect_gitignore_patterns(final_paths)
            if gitignore_excludes:
                normalized_excludes = gitignore_excludes + normalized_excludes
        return normalized_excludes

    def test_no_gitignore_in_lang_examples_gives_empty(self):
        """lang_examples has no .gitignore — nothing added."""
        result = self._run_gitignore_step([str(FIXTURES_DIR)])
        assert result == []

    def test_no_gitignore_flag_skips_collection(self, tmp_path):
        """--no-gitignore suppresses collection even when .gitignore exists."""
        (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
        result = self._run_gitignore_step([str(tmp_path)], no_gitignore=True)
        assert result == []

    def test_gitignore_patterns_collected_by_default(self, tmp_path):
        """Without --no-gitignore, patterns from .gitignore appear in excludes."""
        (tmp_path / ".gitignore").write_text("*.log\nbuild/\n", encoding="utf-8")
        result = self._run_gitignore_step([str(tmp_path)])
        assert "**/*.log" in result
        assert "**/build" in result

    def test_gitignore_patterns_prepended_before_cli_excludes(self, tmp_path):
        """gitignore patterns come before explicit --exclude patterns."""
        (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
        # Replicate: gitignore_excludes + normalized_excludes (cli)
        gitignore_excludes = collect_gitignore_patterns([str(tmp_path)])
        cli_excludes = ["**/vendor/**"]
        combined = gitignore_excludes + cli_excludes
        assert any("*.log" in p for p in combined)
        assert combined.index(next(p for p in combined if "*.log" in p)) < combined.index("**/vendor/**")

    def test_gitignore_patterns_from_multiple_roots(self, tmp_path):
        """Patterns are collected from all root paths."""
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        (root_a / ".gitignore").write_text("*.log\n", encoding="utf-8")
        (root_b / ".gitignore").write_text("dist/\n", encoding="utf-8")
        result = self._run_gitignore_step([str(root_a), str(root_b)])
        assert "**/*.log" in result
        assert "**/dist" in result

    def test_gitignore_nested_patterns_anchored(self, tmp_path):
        """Patterns from nested .gitignore are prefixed with their subdir."""
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / ".gitignore").write_text("*.gen\n", encoding="utf-8")
        result = self._run_gitignore_step([str(tmp_path)])
        assert any("src" in p and ".gen" in p for p in result), f"No src/*.gen pattern in {result}"

    def test_gitignore_negation_not_in_excludes(self, tmp_path):
        """Negation lines must not end up as exclude patterns."""
        (tmp_path / ".gitignore").write_text("*.log\n!keep.log\n", encoding="utf-8")
        result = self._run_gitignore_step([str(tmp_path)])
        assert "**/*.log" in result
        assert "!keep.log" not in result
        assert "keep.log" not in result

    def test_no_paths_gives_empty(self):
        """When final_paths is None/empty, nothing is collected."""
        assert self._run_gitignore_step(None) == []  # type: ignore[arg-type]
        assert self._run_gitignore_step([]) == []

    def test_gitignore_combined_with_cli_excludes_via_update_flow_config(self, tmp_path):
        """End-to-end: gitignore + CLI excludes both end up in _global_flow_config."""
        from cocoindex_code_mcp_server.cocoindex_config import (
            _global_flow_config,
            update_flow_config,
        )
        (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")

        gitignore_excludes = collect_gitignore_patterns([str(tmp_path)])
        cli_excludes = ["**/vendor/**"]
        all_excludes = gitignore_excludes + cli_excludes

        update_flow_config(extra_excluded_patterns=all_excludes)
        stored = _global_flow_config["extra_excluded_patterns"]
        assert "**/*.log" in stored  # type: ignore[operator]
        assert "**/vendor/**" in stored  # type: ignore[operator]

    def teardown_method(self, _method):
        from cocoindex_code_mcp_server.cocoindex_config import update_flow_config
        update_flow_config()


# ---------------------------------------------------------------------------
# Regression: gitignore_pattern_to_globset must return list[str] and produce
# patterns that cover files INSIDE nested directories (e.g. admin-service/target/).
#
# Bug: the old implementation returned str | None and stripped trailing /
# without adding **/ prefix, so "target/" → "target" which only matches the
# directory node itself, not files beneath it such as
# "admin-service/target/classes/db/sql/datensatz.sql".
# ---------------------------------------------------------------------------

class TestGitignorePatternToGlobsetNewApi:
    """Regression tests for the list[str] API and correct **/ handling."""

    # --- return-type contract -----------------------------------------------

    def test_return_type_is_list_for_name(self):
        result = gitignore_pattern_to_globset("target")
        assert isinstance(result, list), f"Expected list, got {type(result)}: {result!r}"

    def test_return_type_is_list_for_dir_pattern(self):
        result = gitignore_pattern_to_globset("target/")
        assert isinstance(result, list), f"Expected list, got {type(result)}: {result!r}"

    def test_blank_returns_empty_list(self):
        assert gitignore_pattern_to_globset("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert gitignore_pattern_to_globset("   ") == []

    def test_comment_returns_empty_list(self):
        assert gitignore_pattern_to_globset("# comment") == []

    def test_negation_returns_empty_list(self):
        assert gitignore_pattern_to_globset("!keep.log") == []

    # --- non-anchored patterns must gain **/ prefix -------------------------

    def test_plain_name_gets_double_star_prefix(self):
        """'target' should become ['**/target'] so it matches at any depth."""
        patterns = gitignore_pattern_to_globset("target")
        assert isinstance(patterns, list)
        assert "**/target" in patterns, f"Expected '**/target' in {patterns}"

    def test_dir_pattern_includes_recursive_children(self):
        """'target/' should include '**/target/**' to cover files inside it."""
        patterns = gitignore_pattern_to_globset("target/")
        assert isinstance(patterns, list)
        assert "**/target/**" in patterns, (
            f"Expected '**/target/**' in {patterns} — "
            "files under admin-service/target/ won't be excluded"
        )

    def test_dir_pattern_also_matches_the_dir_itself(self):
        """'target/' should include '**/target' to match the directory node."""
        patterns = gitignore_pattern_to_globset("target/")
        assert isinstance(patterns, list)
        assert "**/target" in patterns, f"Expected '**/target' in {patterns}"

    # --- root-anchored patterns must NOT gain **/ prefix --------------------

    def test_root_anchored_no_double_star(self):
        """/dist is root-anchored and must not become **/dist."""
        patterns = gitignore_pattern_to_globset("/dist")
        assert isinstance(patterns, list)
        assert not any(p.startswith("**/") for p in patterns), (
            f"/dist is root-anchored; unexpected **/ prefix in {patterns}"
        )

    def test_root_anchored_plain_value_present(self):
        patterns = gitignore_pattern_to_globset("/dist")
        assert "dist" in patterns, f"Expected 'dist' in {patterns}"

    def test_root_anchored_dir_includes_children(self):
        """/dist/ should produce ['dist', 'dist/**']."""
        patterns = gitignore_pattern_to_globset("/dist/")
        assert isinstance(patterns, list)
        assert "dist" in patterns
        assert "dist/**" in patterns

    # --- patterns that already start with **/ must not be duplicated --------

    def test_already_double_star_not_duplicated(self):
        patterns = gitignore_pattern_to_globset("**/node_modules")
        assert isinstance(patterns, list)
        assert "**/**/node_modules" not in patterns, (
            f"Duplicated **/ prefix detected: {patterns}"
        )

    def test_already_double_star_dir_no_duplicate_prefix(self):
        patterns = gitignore_pattern_to_globset("**/__pycache__/")
        assert isinstance(patterns, list)
        assert "**/**/__pycache__" not in patterns

    # --- key regression: nested target/ -------------------------------------------------------

    def test_target_slash_matches_deep_file_via_pathspec(self):
        """Regression: 'target/' patterns must match admin-service/target/classes/foo.sql."""
        import pathspec as ps

        patterns = gitignore_pattern_to_globset("target/")
        assert patterns, "Expected non-empty pattern list for 'target/'"

        spec = ps.PathSpec.from_lines("gitignore", patterns)
        problematic_path = "admin-service/target/classes/db/sql/datensatz.sql"
        assert spec.match_file(problematic_path), (
            f"'target/' converted to {patterns} but does NOT match '{problematic_path}'. "
            "Files inside nested target/ directories will not be excluded."
        )

    def test_explicit_exclude_target_glob_matches_deep_file(self):
        """Regression: CLI --exclude 'target/**' must also cover nested target/ dirs."""
        import pathspec as ps

        patterns = gitignore_pattern_to_globset("target/**")
        assert patterns, "Expected non-empty pattern list for 'target/**'"

        spec = ps.PathSpec.from_lines("gitignore", patterns)
        problematic_path = "admin-service/target/classes/db/sql/datensatz.sql"
        assert spec.match_file(problematic_path), (
            f"'target/**' converted to {patterns} but does NOT match '{problematic_path}'. "
            "Explicit --exclude 'target/**' won't cover nested directories."
        )

    # --- parse_gitignore_file consistency -----------------------------------

    def test_parse_gitignore_file_target_slash(self, tmp_path):
        """A .gitignore with 'target/' should yield patterns matching nested files."""
        import pathspec as ps

        gi = tmp_path / ".gitignore"
        gi.write_text("target/\n", encoding="utf-8")

        patterns = parse_gitignore_file(gi, tmp_path)
        assert patterns, "Expected at least one pattern from 'target/' in .gitignore"

        spec = ps.PathSpec.from_lines("gitignore", patterns)
        problematic_path = "admin-service/target/classes/db/sql/datensatz.sql"
        assert spec.match_file(problematic_path), (
            f"parse_gitignore_file produced {patterns} from 'target/' but "
            f"it does NOT match '{problematic_path}'"
        )


# ---------------------------------------------------------------------------
# PathFilter — unit tests
# ---------------------------------------------------------------------------

class TestPathFilter:
    """Tests for PathFilter.is_included() and filter_results()."""

    # --- is_included: basic include/exclude logic ---------------------------

    def test_matching_include_passes(self):
        pf = PathFilter(["**/*.py"], [])
        assert pf.is_included("src/main.py")

    def test_non_matching_include_blocked(self):
        pf = PathFilter(["**/*.py"], [])
        assert not pf.is_included("src/main.rs")

    def test_matching_exclude_blocks(self):
        pf = PathFilter(["**/*.sql"], ["**/target/**"])
        assert not pf.is_included("admin-service/target/classes/db/sql/schema.sql")

    def test_include_passes_exclude_not_triggered(self):
        pf = PathFilter(["**/*.sql"], ["**/target/**"])
        assert pf.is_included("src/db/schema.sql")

    def test_empty_exclude_list_passes_anything_in_include(self):
        pf = PathFilter(["**/*.py", "**/*.rs"], [])
        assert pf.is_included("a/b/c/lib.rs")

    def test_empty_include_list_blocks_everything(self):
        pf = PathFilter([], [])
        assert not pf.is_included("src/main.py")

    # --- is_included: directory exclusion at any depth ----------------------

    def test_nested_target_excluded(self):
        pf = PathFilter(["**/*.java"], ["**/target", "**/target/**"])
        assert not pf.is_included("admin-service/target/classes/Foo.class")

    def test_root_level_target_excluded(self):
        pf = PathFilter(["**/*.class"], ["**/target/**"])
        assert not pf.is_included("target/classes/Foo.class")

    def test_file_outside_target_not_excluded(self):
        pf = PathFilter(["**/*.java"], ["**/target", "**/target/**"])
        assert pf.is_included("src/main/java/Foo.java")

    # --- filter_results: object form ----------------------------------------

    def test_filter_results_objects(self):
        class FakeResult:
            def __init__(self, filename):
                self.filename = filename

        pf = PathFilter(["**/*.py"], ["**/target/**"])
        results = [
            FakeResult("src/main.py"),
            FakeResult("target/generated/code.py"),
            FakeResult("lib/util.py"),
        ]
        kept = pf.filter_results(results)
        assert len(kept) == 2
        assert kept[0].filename == "src/main.py"
        assert kept[1].filename == "lib/util.py"

    # --- filter_results: dict form ------------------------------------------

    def test_filter_results_dicts(self):
        pf = PathFilter(["**/*.rs"], ["**/target/**"])
        results = [
            {"filename": "src/lib.rs", "score": 0.9},
            {"filename": "target/debug/build.rs", "score": 0.8},
            {"filename": "src/main.rs", "score": 0.7},
        ]
        kept = pf.filter_results(results)
        assert len(kept) == 2
        assert kept[0]["filename"] == "src/lib.rs"
        assert kept[1]["filename"] == "src/main.rs"

    def test_filter_results_empty_input(self):
        pf = PathFilter(["**/*.py"], [])
        assert pf.filter_results([]) == []

    def test_filter_results_all_excluded(self):
        pf = PathFilter(["**/*.sql"], ["**/target/**"])
        results = [{"filename": "target/sql/a.sql"}, {"filename": "target/sql/b.sql"}]
        assert pf.filter_results(results) == []

    def test_filter_results_none_excluded(self):
        pf = PathFilter(["**/*.py"], [])
        results = [{"filename": "a.py"}, {"filename": "src/b.py"}]
        assert len(pf.filter_results(results)) == 2

    # --- integration with SOURCE_CONFIG patterns ----------------------------

    def test_builtin_includes_match_common_files(self):
        """SOURCE_CONFIG included_patterns (now **/*.py form) must match real paths."""
        from cocoindex_code_mcp_server.mappers import SOURCE_CONFIG
        pf = PathFilter(SOURCE_CONFIG["included_patterns"], [])
        assert pf.is_included("src/main.py")
        assert pf.is_included("lib/util.rs")
        assert pf.is_included("pom.xml")
        assert pf.is_included("deep/nested/module/Foo.java")

    def test_builtin_excludes_block_target_contents(self):
        """SOURCE_CONFIG excluded_patterns must block files inside target/."""
        from cocoindex_code_mcp_server.mappers import SOURCE_CONFIG
        pf = PathFilter(SOURCE_CONFIG["included_patterns"], SOURCE_CONFIG["excluded_patterns"])
        assert not pf.is_included("admin-service/target/classes/db/sql/schema.sql")
        assert not pf.is_included("target/release/binary")

    def test_builtin_allows_src_files(self):
        """src/ files must pass through both include and exclude filters."""
        from cocoindex_code_mcp_server.mappers import SOURCE_CONFIG
        pf = PathFilter(SOURCE_CONFIG["included_patterns"], SOURCE_CONFIG["excluded_patterns"])
        assert pf.is_included("src/main/java/com/example/App.java")
        assert pf.is_included("src/lib.rs")


# ---------------------------------------------------------------------------
# use_builtin_excludes wiring
# ---------------------------------------------------------------------------

class TestUseBuiltinExcludes:
    """Verify update_flow_config stores use_builtin_excludes correctly."""

    def test_default_is_true(self):
        from cocoindex_code_mcp_server.cocoindex_config import (
            _global_flow_config,
            update_flow_config,
        )
        update_flow_config()
        assert _global_flow_config["use_builtin_excludes"] is True

    def test_set_false(self):
        from cocoindex_code_mcp_server.cocoindex_config import (
            _global_flow_config,
            update_flow_config,
        )
        update_flow_config(use_builtin_excludes=False)
        assert _global_flow_config["use_builtin_excludes"] is False

    def teardown_method(self, _method):
        from cocoindex_code_mcp_server.cocoindex_config import update_flow_config
        update_flow_config()


# ---------------------------------------------------------------------------
# SOURCE_CONFIG pattern correctness
# ---------------------------------------------------------------------------

class TestSourceConfigPatterns:
    """Verify that the updated SOURCE_CONFIG patterns have the right form."""

    def test_all_include_patterns_have_double_star_prefix(self):
        from cocoindex_code_mcp_server.mappers import SOURCE_CONFIG
        for pat in SOURCE_CONFIG["included_patterns"]:
            assert pat.startswith("**/"), (
                f"Include pattern '{pat}' missing '**/' prefix — "
                "bare patterns are ambiguous in globset"
            )

    def test_no_bare_target_in_excludes(self):
        from cocoindex_code_mcp_server.mappers import SOURCE_CONFIG
        assert "target" not in SOURCE_CONFIG["excluded_patterns"], (
            "Bare 'target' still present — should be '**/target'"
        )

    def test_target_exclude_has_recursive_form(self):
        from cocoindex_code_mcp_server.mappers import SOURCE_CONFIG
        excl = SOURCE_CONFIG["excluded_patterns"]
        assert "**/target" in excl
        assert "**/target/**" in excl
