"""
Utilities for translating gitignore-style patterns to cocoindex (globset) patterns.

CocoIndex's LocalFile source uses Rust's globset crate for pattern matching.
Gitignore patterns are parsed via pathspec, then converted to globset equivalents.

Conversion rules:

  Blank lines / comments → [] (skipped)
  !negation              → [] (globset has no negation; logged as warning)

  Root-anchored (/foo or /foo/):
    /foo   → ["foo"]            (exact path from root)
    /foo/  → ["foo", "foo/**"]  (directory: also match its contents)

  Already-recursive (**/ prefix):
    **/foo   → ["**/foo"]
    **/foo/  → ["**/foo", "**/foo/**"]

  Non-anchored (everything else):
    foo    → ["**/foo"]           (match at any depth)
    foo/   → ["**/foo", "**/foo/**"]  (directory at any depth + its contents)
    *.ext  → ["**/*.ext"]         (file extension at any depth)

The **/ prefix is critical: without it, a pattern like "target" only matches a
path whose *last component* is literally "target", not files inside a nested
target/ directory such as "admin-service/target/classes/Foo.class".
"""

import logging
from pathlib import Path

import pathspec

LOGGER = logging.getLogger(__name__)


def gitignore_pattern_to_globset(pattern: str) -> list[str]:
    """Convert a single gitignore-style pattern to a list of globset patterns.

    Returns an empty list if the pattern should be skipped (negation, blank, comment).
    May return two patterns for directory patterns (the dir itself + its contents).
    """
    stripped = pattern.strip()

    # Skip blank lines and comments
    if not stripped or stripped.startswith("#"):
        return []

    # Negation patterns are unsupported in globset
    if stripped.startswith("!"):
        LOGGER.warning("Pattern '%s' uses gitignore negation — not supported, skipped", stripped)
        return []

    # Note whether this is a directory pattern (trailing slash)
    is_dir = stripped.endswith("/")
    if is_dir:
        stripped = stripped[:-1]
    if not stripped:
        return []

    # Determine the prefix to use:
    #   Root-anchored (/foo)  → strip leading /, no **/ prefix
    #   Already **/ prefixed  → no extra prefix needed
    #   Everything else       → prepend **/ so it matches at any depth
    if stripped.startswith("/"):
        base = stripped[1:]
        prefix = ""
    elif stripped.startswith("**/"):
        base = stripped
        prefix = ""
    else:
        base = stripped
        prefix = "**/"

    if not base:
        return []

    result = [f"{prefix}{base}"]
    if is_dir:
        result.append(f"{prefix}{base}/**")
    return result


def parse_gitignore_file(gitignore_path: Path, root_path: Path) -> list[str]:
    """Read a .gitignore file and return globset patterns anchored to root_path.

    Each valid converted pattern is logged at DEBUG level.
    """
    try:
        lines = gitignore_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        LOGGER.warning("Could not read %s: %s", gitignore_path, e)
        return []

    # Use pathspec to enumerate non-blank, non-comment raw patterns.
    # We pass lines through pathspec to normalise encoding/whitespace, then
    # retrieve the original pattern string via the (untyped) .pattern attribute.
    spec = pathspec.PathSpec.from_lines("gitignore", lines)
    raw_patterns: list[str] = [
        pat for p in spec.patterns if (pat := str(getattr(p, "pattern", "") or ""))
    ]

    rel_dir = gitignore_path.parent.relative_to(root_path)

    results: list[str] = []
    for raw in raw_patterns:
        globset_patterns = gitignore_pattern_to_globset(raw)
        if not globset_patterns:
            # negation / blank already logged inside gitignore_pattern_to_globset
            continue

        for globset in globset_patterns:
            # Anchor patterns from nested .gitignore files relative to the root.
            # A pattern in src/.gitignore applies under src/, so prefix with that dir.
            if rel_dir != Path("."):
                globset = f"{rel_dir}/{globset}"

            LOGGER.debug("  .gitignore (%s) -> excluded: %s", gitignore_path, globset)
            results.append(globset)

    return results


class PathFilter:
    """Post-query filter enforcing current include/exclude patterns against result paths.

    Applied at query time so that DB entries from a previous broader scan are
    transparently excluded when the current session uses narrower patterns.
    Filtered-out paths are logged at INFO level.
    """

    def __init__(self, include_patterns: list[str], exclude_patterns: list[str]) -> None:
        self._include = pathspec.PathSpec.from_lines("gitignore", include_patterns)
        self._exclude = pathspec.PathSpec.from_lines("gitignore", exclude_patterns)

    def is_included(self, relative_path: str) -> bool:
        """Return True if the path passes the include/exclude filter."""
        if not self._include.match_file(relative_path):
            LOGGER.info("Post-filter excluded (no include match): %s", relative_path)
            return False
        if self._exclude.match_file(relative_path):
            LOGGER.info("Post-filter excluded (exclude match): %s", relative_path)
            return False
        return True

    def filter_results(self, results: list, filename_attr: str = "filename") -> list:
        """Return only results whose path passes is_included()."""
        out = []
        for r in results:
            path: str = r[filename_attr] if isinstance(r, dict) else getattr(r, filename_attr, "")
            if path and self.is_included(path):
                out.append(r)
        return out


def collect_gitignore_patterns(root_paths: list[str]) -> list[str]:
    """Walk each root path, find all .gitignore files, return merged globset patterns.

    Logs a summary per .gitignore file found.
    """
    all_patterns: list[str] = []

    for root_str in root_paths:
        root = Path(root_str).resolve()
        if not root.is_dir():
            continue

        gitignore_files = sorted(root.rglob(".gitignore"))
        if not gitignore_files:
            LOGGER.debug("No .gitignore files found under %s", root)
            continue

        for gi_path in gitignore_files:
            patterns = parse_gitignore_file(gi_path, root)
            LOGGER.info(
                "📄 .gitignore: %s  (%d pattern%s)",
                gi_path,
                len(patterns),
                "s" if len(patterns) != 1 else "",
            )
            for p in patterns:
                LOGGER.info("    -> excluded: %s", p)
            all_patterns.extend(patterns)

    return all_patterns
