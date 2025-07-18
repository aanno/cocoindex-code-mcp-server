#!/usr/bin/env python3

"""
Command-line argument parsing for the code embedding pipeline.
"""

import argparse


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
            print(f"📁 Indexing path: {paths[0]}")
        else:
            print(f"📁 Indexing {len(paths)} paths:")
            for i, path in enumerate(paths, 1):
                print(f"  {i}. {path}")
    else:
        print("📁 Using default path: cocoindex")
    
    # Display mode
    if args.live:
        print("🔴 Mode: Live updates")
        if args.poll > 0:
            print(f"⏰ Polling: {args.poll} seconds")
        else:
            print("⚡ Monitoring: Event-based")
    else:
        print("🟢 Mode: One-time indexing")
    
    print()  # Empty line for readability