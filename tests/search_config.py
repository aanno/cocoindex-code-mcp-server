#!/usr/bin/env python3

"""
Search Test Configuration

Provides configuration class for search tests with defaults matching main_mcp_server.py options.
"""

import logging
from typing import Any, Dict, List, Optional


class SearchTestConfig:
    """Configuration class for search tests with defaults matching main_mcp_server.py options."""

    def __init__(
        self,
        paths: Optional[List[str]] = None,
        no_live: bool = True,  # --no-live default: True (disable live updates for tests)
        chunk_factor_percent: int = 100,
        default_embedding: bool = False,  # USE SMART EMBEDDINGS (language-specific models)
        default_chunking: bool = False,
        default_language_handler: bool = False,
        log_level: str = "DEBUG",  # --log-level default: DEBUG for tests
        poll_interval: int = 30,
    ):
        # Set paths default to /workspaces/rust as requested
        self.paths = paths or ["/workspaces/rust"]
        self.no_live = no_live
        self.chunk_factor_percent = chunk_factor_percent
        self.default_embedding = default_embedding
        self.default_chunking = default_chunking
        self.default_language_handler = default_language_handler
        self.log_level = log_level
        self.poll_interval = poll_interval

        # Configure logging immediately
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,  # Override existing configuration
        )

    @property
    def enable_polling(self) -> bool:
        """Convert --no-live flag to enable_polling boolean."""
        return not self.no_live

    def to_infrastructure_kwargs(self) -> Dict[str, Any]:
        """Convert config to CocoIndexTestInfrastructure kwargs."""
        return {
            "paths": self.paths,
            "default_embedding": self.default_embedding,
            "default_chunking": self.default_chunking,
            "default_language_handler": self.default_language_handler,
            "chunk_factor_percent": self.chunk_factor_percent,
            "enable_polling": self.enable_polling,
            "poll_interval": self.poll_interval,
        }

    def log_configuration(self, logger) -> None:
        """Log the current configuration for debugging."""
        logger.info("🔧 Search Test Configuration:")
        logger.info("  📁 Paths: %s", self.paths)
        logger.info("  🔴 Live updates: %s", "DISABLED" if self.no_live else "ENABLED")
        logger.info("  🎯 Default embedding: %s", "ENABLED" if self.default_embedding else "DISABLED")
        logger.info("  🎯 Default chunking: %s", "ENABLED" if self.default_chunking else "DISABLED")
        logger.info("  🎯 Default language handler: %s", "ENABLED" if self.default_language_handler else "DISABLED")
        logger.info("  📏 Chunk factor: %s%%", self.chunk_factor_percent)
        logger.info("  📊 Log level: %s", self.log_level)
        if not self.no_live:
            logger.info("  ⏰ Poll interval: %ss", self.poll_interval)
