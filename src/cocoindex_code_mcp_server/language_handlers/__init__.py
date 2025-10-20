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
Language-specific handlers for AST node processing.

Each handler implements the NodeHandler protocol for a specific programming language.
"""

import logging
from typing import Any, Dict, Optional, Type

LOGGER = logging.getLogger(__name__)

# Import language-specific handlers
try:
    from .python_handler import PythonNodeHandler

    PYTHON_HANDLER_AVAILABLE = True
except ImportError:
    PYTHON_HANDLER_AVAILABLE = False

try:
    from .haskell_handler import HaskellNodeHandler

    HASKELL_HANDLER_AVAILABLE = True
except ImportError:
    HASKELL_HANDLER_AVAILABLE = False

# Registry of available handlers

AVAILABLE_HANDLERS: Dict[str, Type[Any]] = {}

if PYTHON_HANDLER_AVAILABLE:
    AVAILABLE_HANDLERS["python"] = PythonNodeHandler

if HASKELL_HANDLER_AVAILABLE:
    AVAILABLE_HANDLERS["haskell"] = HaskellNodeHandler


def get_handler_for_language(language: str) -> Optional[Any]:
    """Get the appropriate handler for a programming language."""
    language_key = language.lower()

    if language_key in AVAILABLE_HANDLERS:
        return AVAILABLE_HANDLERS[language_key]()

    return None


def list_supported_languages() -> list[str]:
    """List all supported languages with dedicated handlers."""
    return list(AVAILABLE_HANDLERS.keys())


__all__ = ["get_handler_for_language", "list_supported_languages", "AVAILABLE_HANDLERS"]

if PYTHON_HANDLER_AVAILABLE:
    __all__.append("PythonNodeHandler")

if HASKELL_HANDLER_AVAILABLE:
    __all__.append("HaskellNodeHandler")
