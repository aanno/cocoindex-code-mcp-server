#!/usr/bin/env python3

"""
Alternative main entry point for interactive hybrid search queries.
Combines vector similarity search with keyword metadata filtering.
"""

import argparse
from dotenv import load_dotenv
import cocoindex

# Import our modular components
from .cocoindex_config import update_flow_config, run_flow_update
from .db.pgvector.hybrid_search import run_interactive_hybrid_search


def parse_hybrid_search_args():
    """Parse command line arguments for hybrid search mode."""
    parser = argparse.ArgumentParser(
        description="Code embedding hybrid search with vector similarity and keyword filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main_hybrid_search.py                     # Use default path (cocoindex)
  python src/main_hybrid_search.py /path/to/code       # Index single directory
  python src/main_hybrid_search.py /path/to/code1 /path/to/code2  # Index multiple directories
  python src/main_hybrid_search.py --paths /path/to/code   # Explicit paths argument
  
  # Live update is enabled by default
  python src/main_hybrid_search.py --no-live           # Disable live updates
  python src/main_hybrid_search.py --poll 30           # Custom polling interval (default: 60s)

Hybrid Search Queries:
  The tool will prompt for two types of queries:
  
  1. Vector Query (semantic search):
     - "find authentication functions"
     - "error handling patterns"
     - "database connection setup"
  
  2. Keyword Query (metadata filtering):
     - language:python
     - language:python and filename:main_interactive_query.py
     - (language:python or language:rust) and exists(embedding)
     - filename:"test file.py" and language:python
     - exists(embedding) and (language:rust or language:go)
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
        "--no-live",
        action="store_true",
        help="Disable live update mode (live updates are enabled by default)"
    )
    
    parser.add_argument(
        "--poll",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Polling interval in seconds for live updates (default: 60)"
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


def display_hybrid_search_configuration(args, paths):
    """Display the configuration for hybrid search mode."""
    print("🔍 Hybrid Search Mode")
    print("=" * 50)
    
    # Display paths
    if paths:
        if len(paths) == 1:
            print(f"📁 Indexing path: {paths[0]}")
        else:
            print(f"📁 Indexing {len(paths)} paths:")
            for i, path in enumerate(paths, 1):
                print(f"  {i}. {path}")
    else:
        print("📁 Using default path: cocoindex")
    
    # Display mode configuration
    live_enabled = not args.no_live
    if live_enabled:
        print("🔴 Mode: Live updates ENABLED")
        print(f"⏰ Polling interval: {args.poll} seconds")
    else:
        print("🟡 Mode: Live updates DISABLED")
    
    print()
    print("🎯 Search Features:")
    print("  • Vector similarity search (semantic)")
    print("  • Keyword metadata filtering")
    print("  • Combined hybrid search with AND logic")
    print("  • JSON output for complex results")
    print()


def main():
    """Main entry point for hybrid search application."""
    load_dotenv()
    cocoindex.init()
    
    # Parse command line arguments
    args = parse_hybrid_search_args()
    
    # Determine paths to use
    paths = determine_paths(args)
    
    # Display configuration
    display_hybrid_search_configuration(args, paths)
    
    # Configure live updates (enabled by default)
    live_enabled = not args.no_live
    
    # Update flow configuration
    update_flow_config(
        paths=paths,
        enable_polling=live_enabled and args.poll > 0,
        poll_interval=args.poll
    )
    
    # Run the flow update
    if live_enabled:
        print("🚀 Starting indexing with live updates...")
        print("📊 This will build the initial index and then monitor for changes.")
        print("⚡ You can start searching while the system monitors for updates in the background.")
        print()
        
        # Start live update in background and then run interactive search
        try:
            # Setup and initial update
            from .cocoindex_config import code_embedding_flow
            flow = code_embedding_flow
            flow.setup()
            
            print("🔨 Building initial index...")
            stats = flow.update()
            print(f"✅ Initial index built: {stats}")
            print()
            
            # Start live updater in background
            print("👁️  Starting live monitoring...")
            live_options = cocoindex.FlowLiveUpdaterOptions(
                live_mode=True,
                print_stats=False  # Reduce noise during interactive search
            )
            
            with cocoindex.FlowLiveUpdater(flow, live_options) as updater:
                print("✅ Live monitoring active in background.")
                print("🔍 Starting interactive hybrid search...")
                print()
                
                # Run interactive search
                run_interactive_hybrid_search()
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")
            
    else:
        print("🔨 Building index (one-time)...")
        from .cocoindex_config import code_embedding_flow
        stats = code_embedding_flow.update()
        print(f"✅ Index built: {stats}")
        print()
        
        # Run interactive search without live updates
        run_interactive_hybrid_search()


if __name__ == "__main__":
    main()
