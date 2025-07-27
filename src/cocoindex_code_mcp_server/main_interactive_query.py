#!/usr/bin/env python3

"""
Main entry point for the code embedding pipeline with Haskell tree-sitter support.
"""

from dotenv import load_dotenv

import cocoindex

# Import our modular components
from .arg_parser import determine_paths, display_configuration, parse_args
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
    update_flow_config(
        paths=paths,
        enable_polling=args.poll > 0,
        poll_interval=args.poll
    )

    # Run the flow update
    run_flow_update(
        live_update=args.live,
        poll_interval=args.poll
    )

    # If not in live mode, run interactive query mode
    if not args.live:
        run_interactive_query_mode()


if __name__ == "__main__":
    main()
