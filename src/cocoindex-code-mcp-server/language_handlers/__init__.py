#!/usr/bin/env python3

"""
Language-specific handlers for AST node processing.
Each handler implements the NodeHandler protocol for a specific programming language.
"""

from typing import Dict, Any, Optional
import logging

LOGGER = logging.getLogger(__name__)

# Import language-specific handlers
try:
    from .python_handler import PythonNodeHandler
    PYTHON_HANDLER_AVAILABLE = True
except ImportError:
    PYTHON_HANDLER_AVAILABLE = False

# Registry of available handlers
AVAILABLE_HANDLERS = {}

if PYTHON_HANDLER_AVAILABLE:
    AVAILABLE_HANDLERS['python'] = PythonNodeHandler


def get_handler_for_language(language: str) -> Optional[Any]:
    """Get the appropriate handler for a programming language."""
    language_key = language.lower()
    
    if language_key in AVAILABLE_HANDLERS:
        return AVAILABLE_HANDLERS[language_key]()
    
    return None


def list_supported_languages() -> list[str]:
    """List all supported languages with dedicated handlers."""
    return list(AVAILABLE_HANDLERS.keys())


__all__ = [
    'get_handler_for_language',
    'list_supported_languages',
    'AVAILABLE_HANDLERS'
]

if PYTHON_HANDLER_AVAILABLE:
    __all__.append('PythonNodeHandler')
