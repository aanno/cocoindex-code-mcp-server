#!/usr/bin/env python3
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 aanno <aanno@users.noreply.github.com>
#
# This file is part of cocoindex_code_mcp_server from
# https://github.com/aanno/cocoindex-code-mcp-server
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Command-line argument parsing for the code embedding pipeline.
"""

import argparse

from cocoindex_code_mcp_server import LOGGER


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Code embedding pipeline with Haskell tree-sitter support and live updates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cocoindex_code_mcp_server.main_interactive_query.py                           # Use default path (cocoindex)
  python -m cocoindex_code_mcp_server.main_interactive_query.py /path/to/code             # Index single directory
  python -m cocoindex_code_mcp_server.main_interactive_query.py /path/to/code1 /path/to/code2  # Index multiple directories
  python -m cocoindex_code_mcp_server.main_interactive_query.py --paths /path/to/code     # Explicit paths argument

  # Live update mode
  python -m cocoindex_code_mcp_server.main_interactive_query.py --live                    # Live updates with event monitoring
  python -m cocoindex_code_mcp_server.main_interactive_query.py --live --poll 10         # Live updates with 10s polling
  python -m cocoindex_code_mcp_server.main_interactive_query.py --live --poll 60 /path/to/code  # Custom path with polling
        """,
    )

    parser.add_argument("paths", nargs="*", help="Code directory paths to index (default: cocoindex)")

    parser.add_argument("--paths", dest="explicit_paths", nargs="+", help="Alternative way to specify paths")

    parser.add_argument("--live", action="store_true", help="Enable live update mode with continuous monitoring")

    parser.add_argument(
        "--poll",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Enable file polling with specified interval in seconds (default: event-based monitoring)",
    )

    # Default mode flags to disable custom extensions
    parser.add_argument(
        "--default-embedding",
        action="store_true",
        help="Use CocoIndex default embedding instead of smart code embedding extension",
    )

    parser.add_argument(
        "--default-chunking",
        action="store_true",
        help="Use CocoIndex default SplitRecursively instead of AST chunking extension",
    )

    parser.add_argument(
        "--default-language-handler",
        action="store_true",
        help="Skip Python-specific language handlers and use basic metadata extraction",
    )

    return parser.parse_args()


def determine_paths(args):
    """Determine which paths to use based on parsed arguments."""
    paths = None
    if args.explicit_paths:
        paths = args.explicit_paths
    elif args.paths:
        paths = args.paths

    return paths


def display_configuration(args, paths):
    """Display the configuration based on parsed arguments."""
    if paths:
        if len(paths) == 1:
            LOGGER.info("📁 Indexing path: %s", paths[0])
        else:
            LOGGER.info("📁 Indexing %s paths:", len(paths))
            for i, path in enumerate(paths, 1):
                LOGGER.info("  %s. %s", i, path)
    else:
        LOGGER.info("📁 Using default path: cocoindex")

    # Display mode
    if args.live:
        LOGGER.info("🔴 Mode: Live updates")
        if args.poll > 0:
            LOGGER.info("⏰ Polling: %s seconds", args.poll)
        else:
            LOGGER.info("⚡ Monitoring: Event-based")
    else:
        LOGGER.info("🟢 Mode: One-time indexing")

    LOGGER.info("")  # Empty line for readability
