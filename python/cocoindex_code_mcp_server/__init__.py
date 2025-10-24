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

# Main package for haskell-tree-sitter project

import logging
import os
from logging.handlers import RotatingFileHandler

# This leads to circular dependencies (tp)
# from . import _haskell_tree_sitter as hts

# Get WORKSPACE environment variable, fallback to current directory if not set
workspace_dir = os.environ.get("WORKSPACE", ".")

log_file_path = os.path.join(workspace_dir, "cocoindex_code_mcp_server.log")

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
rotating_handler.setLevel(logging.DEBUG)

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
