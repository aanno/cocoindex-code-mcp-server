"""
Tests for pattern_utils: gitignore-style pattern conversion and file filtering.

Uses tests/fixtures/lang_examples as a real directory tree for filesystem tests.
"""

from pathlib import Path

import pytest

from cocoindex_code_mcp_server.pattern_utils import (
    collect_gitignore_patterns,
    gitignore_pattern_to_globset,
    parse_gitignore_file,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "lang_examples"


# ---------------------------------------------------------------------------
# gitignore_pattern_to_globset — unit tests
# ---------------------------------------------------------------------------

class TestGitignorePatternToGlobset:
    """Unit tests for single-pattern conversion."""

    def test_plain_extension_pattern_unchanged(self):
        assert gitignore_pattern_to_globset("*.py") == "*.py"

    def test_double_star_pattern_unchanged(self):
        assert gitignore_pattern_to_globset("**/node_modules") == "**/node_modules"

    def test_plain_name_unchanged(self):
        assert gitignore_pattern_to_globset("target") == "target"

    def test_trailing_slash_stripped(self):
        assert gitignore_pattern_to_globset("target/") == "target"

    def test_leading_slash_stripped(self):
        assert gitignore_pattern_to_globset("/dist") == "dist"

    def test_leading_and_trailing_slash_stripped(self):
        assert gitignore_pattern_to_globset("/build/") == "build"

    def test_nested_pattern_unchanged(self):
        assert gitignore_pattern_to_globset("src/build/") == "src/build"

    def test_blank_line_returns_none(self):
        assert gitignore_pattern_to_globset("") is None

    def test_whitespace_only_returns_none(self):
        assert gitignore_pattern_to_globset("   ") is None

    def test_comment_returns_none(self):
        assert gitignore_pattern_to_globset("# this is a comment") is None

    def test_negation_returns_none(self):
        assert gitignore_pattern_to_globset("!important.lock") is None

    def test_negation_with_path_returns_none(self):
        assert gitignore_pattern_to_globset("!src/keep.py") is None

    def test_complex_glob_unchanged(self):
        assert gitignore_pattern_to_globset("**/__pycache__") == "**/__pycache__"

    def test_double_star_extension(self):
        assert gitignore_pattern_to_globset("**/*.pyc") == "**/*.pyc"

    def test_question_mark_unchanged(self):
        assert gitignore_pattern_to_globset("file?.py") == "file?.py"

    def test_character_class_unchanged(self):
        assert gitignore_pattern_to_globset("[Mm]akefile") == "[Mm]akefile"


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

    def test_trailing_slash_stripped(self, gitignore):
        gi, root = gitignore
        result = parse_gitignore_file(gi, root)
        assert "target" in result

    def test_leading_slash_stripped(self, gitignore):
        gi, root = gitignore
        result = parse_gitignore_file(gi, root)
        assert "dist" in result

    def test_plain_extension_included(self, gitignore):
        gi, root = gitignore
        result = parse_gitignore_file(gi, root)
        assert "*.lock" in result

    def test_plain_name_included(self, gitignore):
        gi, root = gitignore
        result = parse_gitignore_file(gi, root)
        assert "node_modules" in result

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
        assert "src/*.gen" in result
        assert "src/build" in result

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
        assert "*.log" in result
        assert "build" in result

    def test_multiple_gitignores_merged(self, tmp_path):
        """Patterns from .gitignores in different subdirs are all returned."""
        (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
        sub = tmp_path / "lib"
        sub.mkdir()
        (sub / ".gitignore").write_text("*.o\n", encoding="utf-8")
        result = collect_gitignore_patterns([str(tmp_path)])
        assert "*.log" in result
        assert "lib/*.o" in result

    def test_nested_gitignore_anchored_to_root(self, tmp_path):
        """Pattern in a nested .gitignore is prefixed with the subdirectory path."""
        sub = tmp_path / "src" / "gen"
        sub.mkdir(parents=True)
        (sub / ".gitignore").write_text("*.pb.go\n", encoding="utf-8")
        result = collect_gitignore_patterns([str(tmp_path)])
        assert "src/gen/*.pb.go" in result

    def test_multiple_root_paths(self, tmp_path):
        """Patterns from distinct root directories are all collected."""
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        (root_a / ".gitignore").write_text("*.log\n", encoding="utf-8")
        (root_b / ".gitignore").write_text("dist/\n", encoding="utf-8")
        result = collect_gitignore_patterns([str(root_a), str(root_b)])
        assert "*.log" in result
        assert "dist" in result

    def test_negation_patterns_not_in_result(self, tmp_path):
        """Negation lines in .gitignore must not appear in the output."""
        (tmp_path / ".gitignore").write_text("*.log\n!important.log\n", encoding="utf-8")
        result = collect_gitignore_patterns([str(tmp_path)])
        assert "*.log" in result
        assert "!important.log" not in result
        assert "important.log" not in result

    def test_comment_lines_not_in_result(self, tmp_path):
        (tmp_path / ".gitignore").write_text("# ignore build\nbuild/\n", encoding="utf-8")
        result = collect_gitignore_patterns([str(tmp_path)])
        assert "build" in result
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
            converted = gitignore_pattern_to_globset(p)
            if converted:
                result.append(converted)
        return result or None

    def _normalize_includes(self, patterns):
        if not patterns:
            return None
        result = []
        for p in patterns:
            converted = gitignore_pattern_to_globset(p)
            if converted:
                result.append(converted)
        return result

    def test_exclude_trailing_slash_normalized(self):
        result = self._normalize_excludes(["target/"])
        assert result == ["target"]

    def test_exclude_multiple_patterns(self):
        result = self._normalize_excludes(["*.lock", "**/tests/**"])
        assert result == ["*.lock", "**/tests/**"]

    def test_exclude_negation_silently_dropped(self):
        result = self._normalize_excludes(["!keep.log"])
        assert result is None  # nothing left after dropping negation

    def test_include_replaces_when_given(self):
        result = self._normalize_includes(["*.nix", "*.dhall"])
        assert result == ["*.nix", "*.dhall"]

    def test_include_none_when_empty_tuple(self):
        result = self._normalize_includes(())
        assert result is None

    def test_include_leading_slash_stripped(self):
        result = self._normalize_includes(["/src/*.py"])
        assert result == ["src/*.py"]


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
        assert "*.log" in result
        assert "build" in result

    def test_gitignore_patterns_prepended_before_cli_excludes(self, tmp_path):
        """gitignore patterns come before explicit --exclude patterns."""
        (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
        # Replicate: gitignore_excludes + normalized_excludes (cli)
        gitignore_excludes = collect_gitignore_patterns([str(tmp_path)])
        cli_excludes = ["**/vendor/**"]
        combined = gitignore_excludes + cli_excludes
        assert combined.index("*.log") < combined.index("**/vendor/**")

    def test_gitignore_patterns_from_multiple_roots(self, tmp_path):
        """Patterns are collected from all root paths."""
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        (root_a / ".gitignore").write_text("*.log\n", encoding="utf-8")
        (root_b / ".gitignore").write_text("dist/\n", encoding="utf-8")
        result = self._run_gitignore_step([str(root_a), str(root_b)])
        assert "*.log" in result
        assert "dist" in result

    def test_gitignore_nested_patterns_anchored(self, tmp_path):
        """Patterns from nested .gitignore are prefixed with their subdir."""
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / ".gitignore").write_text("*.gen\n", encoding="utf-8")
        result = self._run_gitignore_step([str(tmp_path)])
        assert "src/*.gen" in result

    def test_gitignore_negation_not_in_excludes(self, tmp_path):
        """Negation lines must not end up as exclude patterns."""
        (tmp_path / ".gitignore").write_text("*.log\n!keep.log\n", encoding="utf-8")
        result = self._run_gitignore_step([str(tmp_path)])
        assert "*.log" in result
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
        assert "*.log" in stored  # type: ignore[operator]
        assert "**/vendor/**" in stored  # type: ignore[operator]

    def teardown_method(self, _method):
        from cocoindex_code_mcp_server.cocoindex_config import update_flow_config
        update_flow_config()
