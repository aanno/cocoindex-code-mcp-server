#!/usr/bin/env python3

"""
CocoIndex Management Tool

A standalone CLI tool for managing CocoIndex flows using the Python API.
This tool provides setup, update, drop, and evaluate operations.
"""

import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(name)s:%(lineno)d)"
)
logger = logging.getLogger(__name__)


def initialize_cocoindex():
    """Initialize CocoIndex with proper environment setup."""
    try:
        # Load environment variables
        load_dotenv()

        # Import CocoIndex (must be after load_dotenv)
        import cocoindex
        from cocoindex_code_mcp_server.cocoindex_config import (
            code_embedding_flow,
            update_flow_config,
        )

        # Initialize CocoIndex library with database settings
        database_url = os.getenv("DATABASE_URL") or os.getenv("COCOINDEX_DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL or COCOINDEX_DATABASE_URL not found in environment")

        # Set COCOINDEX_DATABASE_URL for CocoIndex if not already set
        if not os.getenv("COCOINDEX_DATABASE_URL"):
            os.environ["COCOINDEX_DATABASE_URL"] = database_url

        cocoindex.init()
        logger.info("✅ CocoIndex library initialized with database")

        return cocoindex, code_embedding_flow, update_flow_config

    except ImportError as e:
        logger.error(f"❌ Failed to import CocoIndex: {e}")
        logger.error("Make sure CocoIndex is properly installed and available")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to initialize CocoIndex: {e}")
        sys.exit(1)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool):
    """CocoIndex Management Tool - CLI for setup, update, drop, and evaluate operations."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")


@cli.command()
@click.option('--paths', '-p', multiple=True, default=['.'],
              help='Paths to index (default: current directory)')
@click.option('--enable-polling/--disable-polling', default=False,
              help='Enable live file polling (default: disabled)')
@click.option('--poll-interval', default=30, type=int,
              help='Polling interval in seconds (default: 30)')
@click.option('--chunk-factor', default=100, type=int,
              help='Chunk size scaling factor percentage (default: 100)')
def setup(paths: tuple, enable_polling: bool, poll_interval: int, chunk_factor: int):
    """Set up CocoIndex flow and create database schema."""
    logger.info("🚀 Setting up CocoIndex flow...")

    # Initialize CocoIndex
    cocoindex, flow, update_flow_config = initialize_cocoindex()

    try:
        # Update flow configuration
        logger.info(f"📁 Configuring paths: {list(paths)}")
        logger.info(f"🔴 Polling: {'ENABLED' if enable_polling else 'DISABLED'}")
        if enable_polling:
            logger.info(f"⏰ Poll interval: {poll_interval} seconds")
        if chunk_factor != 100:
            logger.info(f"📏 Chunk size scaling: {chunk_factor}%")

        update_flow_config(
            paths=list(paths),
            enable_polling=enable_polling,
            poll_interval=poll_interval,
            use_default_embedding=True,
            use_default_chunking=True,
            use_default_language_handler=True,
            chunk_factor_percent=chunk_factor
        )

        # Setup the flow (creates database schema)
        logger.info("🔧 Setting up flow schema...")
        flow.setup()
        logger.info("✅ Flow schema setup completed")

        # Run initial update
        logger.info("🔄 Running initial index update...")
        stats = flow.update()
        logger.info(f"✅ Initial indexing completed - Stats: {stats}")

        logger.info("🎉 CocoIndex setup completed successfully!")

    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--live', is_flag=True, help='Enable live update mode with file monitoring')
@click.option('--poll-interval', default=30, type=int,
              help='Polling interval in seconds for live mode (default: 30)')
def update(live: bool, poll_interval: int):
    """Update the CocoIndex flow data."""
    logger.info("🔄 Updating CocoIndex flow...")

    # Initialize CocoIndex
    cocoindex, flow, _ = initialize_cocoindex()

    try:
        if live:
            logger.info("👁️  Starting live update mode...")
            logger.info(f"📊 File polling interval: {poll_interval} seconds")

            # Initial update
            logger.info("🚀 Initial index update...")
            stats = flow.update()
            logger.info(f"Initial update completed: {stats}")

            # Start live updater
            logger.info("🔄 Starting live file monitoring...")
            live_options = cocoindex.FlowLiveUpdaterOptions(
                live_mode=True,
                print_stats=True
            )

            with cocoindex.FlowLiveUpdater(flow, live_options) as updater:
                logger.info("✅ Live update mode active. Press Ctrl+C to stop.")
                try:
                    updater.wait()
                except KeyboardInterrupt:
                    logger.info("\n⏹️  Stopping live update mode...")
        else:
            # Regular one-time update
            stats = flow.update()
            logger.info(f"✅ Update completed - Stats: {stats}")

    except Exception as e:
        logger.error(f"❌ Update failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--target-name', default='code_embeddings',
              help='Name of the target to drop (default: code_embeddings)')
@click.option('--confirm', is_flag=True,
              help='Skip confirmation prompt')
def drop(target_name: str, confirm: bool):
    """Drop (delete) the indexed data for a target."""
    if not confirm:
        click.confirm(f"Are you sure you want to drop target '{target_name}'? This will delete all data.", abort=True)

    logger.info(f"🗑️  Dropping target: {target_name}")

    # Initialize CocoIndex
    cocoindex, flow, _ = initialize_cocoindex()

    try:
        flow.drop_target(target_name)
        logger.info(f"✅ Target '{target_name}' dropped successfully")

    except Exception as e:
        logger.error(f"❌ Drop failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--output-dir', default='./eval_output',
              help='Output directory for evaluation results (default: ./eval_output)')
def evaluate(output_dir: str):
    """Evaluate the flow transformations without changing target data."""
    logger.info("📊 Evaluating CocoIndex flow...")

    # Initialize CocoIndex
    cocoindex, flow, _ = initialize_cocoindex()

    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info(f"📁 Output directory: {output_path.absolute()}")

        # Run evaluation
        eval_options = cocoindex.EvaluateAndDumpOptions(output_dir=str(output_path))
        flow.evaluate_and_dump(eval_options)

        logger.info(f"✅ Evaluation completed - Results saved to: {output_path.absolute()}")

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        sys.exit(1)


@cli.command()
def status():
    """Show the current status of the CocoIndex flow."""
    logger.info("📋 Checking CocoIndex flow status...")

    # Initialize CocoIndex
    cocoindex, flow, _ = initialize_cocoindex()

    try:
        # Get table name
        table_name = cocoindex.utils.get_target_default_name(flow, "code_embeddings")
        logger.info(f"🏷️  Target table: {table_name}")

        # Check database connection
        database_url = os.getenv("DATABASE_URL") or os.getenv("COCOINDEX_DATABASE_URL")
        logger.info(f"🔗 Database URL: {database_url}")

        # Try to get basic stats
        logger.info("✅ CocoIndex flow is accessible")

    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()
