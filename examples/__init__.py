#!/usr/bin/env python3
#
# Test package for haskell-tree-sitter project

import logging
import os
from logging.handlers import RotatingFileHandler

# Get WORKSPACE environment variable, fallback to current directory if not set
workspace_dir = os.environ.get("WORKSPACE", ".")

log_file_path = os.path.join(workspace_dir, "cocoindex_code_mcp_server-examples.log")

# Create a rotating file handler
rotating_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=2 * 1024 * 1024,  # 2 MB
    backupCount=3,
)
rotating_handler.setLevel(logging.DEBUG)

# Formatter for the file logs (can be same or different)
file_formatter = logging.Formatter(
    "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
rotating_handler.setFormatter(file_formatter)

# Set up console handler separately
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
console.setFormatter(console_formatter)

# Get root logger and configure it
LOGGER = logging.getLogger()  # root logger
LOGGER.setLevel(logging.DEBUG)  # or whatever level you want

# Remove all existing handlers
LOGGER.handlers.clear()

# Add the handlers
LOGGER.addHandler(rotating_handler)
LOGGER.addHandler(console)
