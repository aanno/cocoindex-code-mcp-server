"""
Utilities for translating gitignore-style patterns to cocoindex (globset) patterns.

CocoIndex's LocalFile source uses Rust's globset crate for pattern matching.
Gitignore patterns are parsed via pathspec, then converted to globset equivalents.

Conversion rules:
  - Leading /  removed  (gitignore root-anchor has no globset equivalent)
  - Trailing / removed  (directory matching is handled by cocoindex itself)
  - !negation  skipped  (globset does not support negation; logged as warning)
  - Everything else passes through unchanged (both syntaxes agree on * ** ? [...])
"""

import logging
from pathlib import Path

import pathspec

LOGGER = logging.getLogger(__name__)


def gitignore_pattern_to_globset(pattern: str) -> str | None:
    """Convert a single gitignore-style pattern to a globset pattern.

    Returns None if the pattern should be skipped (negation, blank, comment).
    """
    stripped = pattern.strip()

    # Skip blank lines and comments
    if not stripped or stripped.startswith("#"):
        return None

    # Negation patterns are unsupported in globset
    if stripped.startswith("!"):
        LOGGER.warning("Pattern '%s' uses gitignore negation — not supported, skipped", stripped)
        return None

    # Remove leading slash (root-anchor): /dist → dist
    if stripped.startswith("/"):
        stripped = stripped[1:]

    # Remove trailing slash (directory marker): target/ → target
    if stripped.endswith("/"):
        stripped = stripped[:-1:]

    return stripped if stripped else None


def parse_gitignore_file(gitignore_path: Path, root_path: Path) -> list[str]:
    """Read a .gitignore file and return globset patterns anchored to root_path.

    Each valid converted pattern is logged at DEBUG level.
    """
    try:
        lines = gitignore_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        LOGGER.warning("Could not read %s: %s", gitignore_path, e)
        return []

    # Use pathspec to enumerate non-blank, non-comment raw patterns
    spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
    raw_patterns = [p.pattern for p in spec.patterns if p.pattern]

    rel_dir = gitignore_path.parent.relative_to(root_path)

    results: list[str] = []
    for raw in raw_patterns:
        globset = gitignore_pattern_to_globset(raw)
        if globset is None:
            # negation / blank already logged inside gitignore_pattern_to_globset
            continue

        # Anchor patterns from nested .gitignore files relative to the root.
        # A pattern in src/.gitignore applies under src/, so prefix with that dir.
        if rel_dir != Path("."):
            globset = f"{rel_dir}/{globset}"

        LOGGER.debug("  .gitignore (%s) -> excluded: %s", gitignore_path, globset)
        results.append(globset)

    return results


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
