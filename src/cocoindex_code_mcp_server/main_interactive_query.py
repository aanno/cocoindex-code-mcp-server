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
Main entry point for the code embedding pipeline with Haskell tree-sitter support.
"""

from dotenv import load_dotenv

import cocoindex

# Import our modular components
from .arg_parser_old import determine_paths, display_configuration, parse_args
from .cocoindex_config import run_flow_update, update_flow_config
from .query_interactive import run_interactive_query_mode


def main():
    """Main entry point for the application."""
    load_dotenv()
    cocoindex.init()

    # Parse command line arguments
    args = parse_args()

    # Determine paths to use
    paths = determine_paths(args)

    # Display configuration
    display_configuration(args, paths)

    # Update flow configuration
    update_flow_config(paths=paths, enable_polling=args.poll > 0, poll_interval=args.poll)

    # Run the flow update
    run_flow_update(live_update=args.live, poll_interval=args.poll)

    # If not in live mode, run interactive query mode
    if not args.live:
        run_interactive_query_mode()


if __name__ == "__main__":
    main()
