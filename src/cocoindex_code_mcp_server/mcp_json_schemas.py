#!/usr/bin/env python3

"""
JSON schemas used for MCP server endpoints for argument and result definitions.
"""

HYBRID_SEARCH_INPUT_SCHEMA = {
                "type": "object",
                "properties": {
                    "vector_query": {
                        "type": "string",
                        "description": "Text to embed and search for semantic similarity"
                    },
                    "keyword_query": {
                        "type": "string",
                        "description": "Keyword search query for metadata filtering. Syntax: field:value, exists(field), value_contains(field, 'text'), AND/OR operators"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10
                    },
                    "vector_weight": {
                        "type": "number",
                        "description": "Weight for vector similarity score (0-1)",
                        "default": 0.7
                    },
                    "keyword_weight": {
                        "type": "number",
                        "description": "Weight for keyword match score (0-1)",
                        "default": 0.3
                    }
                },
                "required": ["vector_query", "keyword_query"]
            }

VECTOR_SEARCH_INPUT_SCHEMA = {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to embed and search for semantic similarity"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }

KEYWORD_SEARCH_INPUT_SCHEMA = {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword search query with AND/OR operators and parentheses grouping"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }

CODE_ANALYZE_INPUT_SCHEMA = {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code content to analyze"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "File path for context"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (auto-detected if not provided)"
                    }
                },
                "required": ["code", "file_path"]
            }

CODE_EMBEDDINGS_INPUT_SCHEMA = {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to generate embeddings for"
                    }
                },
                "required": ["text"]
            }

EMPTY_JSON_SCHEMA = {
                "type": "object",
                "properties": {},
                "required": []
            }
