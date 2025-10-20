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
JSON schemas used for MCP server endpoints for argument and result definitions.
"""

HYBRID_SEARCH_INPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "vector_query": {"type": "string", "description": "Text to embed and search for semantic similarity"},
        "keyword_query": {
            "type": "string",
            "description": "Keyword search query for metadata filtering. Syntax: field:value, exists(field), value_contains(field, 'text'), AND/OR operators",
        },
        "language": {
            "type": "string",
            "description": "Programming language to filter results by (e.g., 'Python', 'Rust'). Required for smart embeddings - ensures query uses correct embedding model for the language.",
        },
        "embedding_model": {
            "type": "string",
            "description": "Specific embedding model to use (e.g., 'microsoft/graphcodebert-base'). Alternative to language parameter. Required for smart embeddings.",
        },
        "top_k": {"type": "integer", "description": "Number of results to return", "default": 10},
        "vector_weight": {"type": "number", "description": "Weight for vector similarity score (0-1)", "default": 0.7},
        "keyword_weight": {"type": "number", "description": "Weight for keyword match score (0-1)", "default": 0.3},
    },
    "required": ["vector_query", "keyword_query"],
}

HYBRID_SEARCH_OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "query": {
            "type": "object",
            "properties": {
                "vector_query": {"type": "string"},
                "keyword_query": {"type": "string"},
                "top_k": {"type": "integer"},
                "vector_weight": {"type": "number"},
                "keyword_weight": {"type": "number"},
            },
            "required": [
                "vector_query",
            ],
            "additionalProperties": False,
        },
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "language": {"type": "string"},
                    "code": {"type": "string"},
                    "score": {"type": "number"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                    "source": {"type": "string"},
                    "score_type": {"type": "string"},
                    "location": {"type": "string"},
                    "source_name": {"type": "string"},
                    "functions": {"type": "array", "items": {"type": "string"}},
                    "classes": {"type": "array", "items": {"type": "string"}},
                    "imports": {"type": "array", "items": {"type": "string"}},
                    "complexity_score": {"type": "number"},
                    "has_type_hints": {"type": "boolean"},
                    "has_async": {"type": "boolean"},
                    "has_classes": {"type": "boolean"},
                    "metadata_json": {
                        "type": "object",
                        "properties": {
                            "language": {"type": "string"},
                            "filename": {"type": "string"},
                            "line_count": {"type": "integer"},
                            "char_count": {"type": "integer"},
                            "functions": {"type": "array", "items": {"type": "string"}},
                            "classes": {"type": "array", "items": {"type": "string"}},
                            "imports": {"type": "array", "items": {"type": "string"}},
                            "variables": {"type": "array", "items": {"type": "string"}},
                            "decorators": {"type": "array", "items": {"type": "string"}},
                            "complexity_score": {"type": "number"},
                            "has_async": {"type": "boolean"},
                            "has_classes": {"type": "boolean"},
                            "has_decorators": {"type": "boolean"},
                            "has_type_hints": {"type": "boolean"},
                            "has_docstrings": {"type": "boolean"},
                            "private_methods": {"type": "array", "items": {"type": "string"}},
                            "dunder_methods": {"type": "array", "items": {"type": "string"}},
                            "function_details": {"type": "array", "items": {}},
                            "class_details": {"type": "array", "items": {}},
                            "import_details": {"type": "array", "items": {}},
                            "analysis_method": {"type": "string"},
                            "metadata_json": {"type": "string"},
                        },
                        "required": [
                            "language",
                            "filename",
                            "line_count",
                            "char_count",
                            "analysis_method",
                            "metadata_json",
                        ],
                        "additionalProperties": True,
                    },
                },
                "required": [
                    "filename",
                    "language",
                    "code",
                    "score",
                    "start",
                    "end",
                    "source",
                    "score_type",
                    "location",
                    "source_name",
                    "metadata_json",
                ],
                "additionalProperties": True,
            },
        },
    },
    "required": ["query", "results"],
    "additionalProperties": False,
}


VECTOR_SEARCH_INPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Text to embed and search for semantic similarity"},
        "language": {
            "type": "string",
            "description": "Programming language to filter results by (e.g., 'Python', 'Rust'). Required for smart embeddings - ensures query uses correct embedding model for the language.",
        },
        "embedding_model": {
            "type": "string",
            "description": "Specific embedding model to use (e.g., 'microsoft/graphcodebert-base'). Alternative to language parameter. Required for smart embeddings.",
        },
        "top_k": {"type": "integer", "description": "Number of results to return", "default": 10},
    },
    "required": ["query"],
}

KEYWORD_SEARCH_INPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Keyword search query with AND/OR operators and parentheses grouping",
        },
        "top_k": {"type": "integer", "description": "Number of results to return", "default": 10},
    },
    "required": ["query"],
}

CODE_ANALYZE_INPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Code content to analyze"},
        "file_path": {"type": "string", "description": "File path for context"},
        "language": {"type": "string", "description": "Programming language (auto-detected if not provided)"},
    },
    "required": ["code", "file_path"],
}

CODE_EMBEDDINGS_INPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {"text": {"type": "string", "description": "Text to generate embeddings for"}},
    "required": ["text"],
}

EMPTY_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {},
    "required": [],
}
