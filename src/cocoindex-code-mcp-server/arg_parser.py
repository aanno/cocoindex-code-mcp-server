#!/usr/bin/env python3

"""
Command-line argument parsing for the code embedding pipeline.
"""

import argparse

from __init__ import LOGGER


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Code embedding pipeline with Haskell tree-sitter support and live updates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                           # Use default path (cocoindex)
  python src/main.py /path/to/code             # Index single directory
  python src/main.py /path/to/code1 /path/to/code2  # Index multiple directories
  python src/main.py --paths /path/to/code     # Explicit paths argument
  
  # Live update mode
  python src/main.py --live                    # Live updates with event monitoring
  python src/main.py --live --poll 10         # Live updates with 10s polling
  python src/main.py --live --poll 60 /path/to/code  # Custom path with polling
        """
    )
    
    parser.add_argument(
        "paths", 
        nargs="*", 
        help="Code directory paths to index (default: cocoindex)"
    )
    
    parser.add_argument(
        "--paths",
        dest="explicit_paths",
        nargs="+",
        help="Alternative way to specify paths"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live update mode with continuous monitoring"
    )
    
    parser.add_argument(
        "--poll",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Enable file polling with specified interval in seconds (default: event-based monitoring)"
    )
    
    # Default mode flags to disable custom extensions
    parser.add_argument(
        "--default-embedding",
        action="store_true",
        help="Use CocoIndex default embedding instead of smart code embedding extension"
    )
    
    parser.add_argument(
        "--default-chunking", 
        action="store_true",
        help="Use CocoIndex default SplitRecursively instead of AST chunking extension"
    )
    
    parser.add_argument(
        "--default-language-handler",
        action="store_true", 
        help="Skip Python-specific language handlers and use basic metadata extraction"
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
            LOGGER.info(f"📁 Indexing path: {paths[0]}")
        else:
            LOGGER.info(f"📁 Indexing {len(paths)} paths:")
            for i, path in enumerate(paths, 1):
                LOGGER.info(f"  {i}. {path}")
    else:
        LOGGER.info("📁 Using default path: cocoindex")
    
    # Display mode
    if args.live:
        LOGGER.info("🔴 Mode: Live updates")
        if args.poll > 0:
            LOGGER.info(f"⏰ Polling: {args.poll} seconds")
        else:
            LOGGER.info("⚡ Monitoring: Event-based")
    else:
        LOGGER.info("🟢 Mode: One-time indexing")
    
    LOGGER.info("")  # Empty line for readability
