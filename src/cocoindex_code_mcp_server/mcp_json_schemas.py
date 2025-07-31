#!/usr/bin/env python3

"""
JSON schemas used for MCP server endpoints for argument and result definitions.
"""

HYBRID_SEARCH_INPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
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
                "keyword_weight": {"type": "number"}
            },
            "required": [
                "vector_query",
                "keyword_query",
                "top_k",
                "vector_weight",
                "keyword_weight"
            ],
            "additionalProperties": False
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
                    "functions": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "classes": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "imports": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
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
                            "functions": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "classes": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "imports": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "variables": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "decorators": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "complexity_score": {"type": "number"},
                            "has_async": {"type": "boolean"},
                            "has_classes": {"type": "boolean"},
                            "has_decorators": {"type": "boolean"},
                            "has_type_hints": {"type": "boolean"},
                            "has_docstrings": {"type": "boolean"},
                            "private_methods": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "dunder_methods": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "function_details": {
                                "type": "array",
                                "items": {}
                            },
                            "class_details": {
                                "type": "array",
                                "items": {}
                            },
                            "import_details": {
                                "type": "array",
                                "items": {}
                            },
                            "analysis_method": {"type": "string"},
                            "metadata_json": {"type": "string"}
                        },
                        "required": [
                            "language",
                            "filename",
                            "line_count",
                            "char_count",
                            "functions",
                            "classes",
                            "imports",
                            "variables",
                            "decorators",
                            "complexity_score",
                            "has_async",
                            "has_classes",
                            "has_decorators",
                            "has_type_hints",
                            "has_docstrings",
                            "private_methods",
                            "dunder_methods",
                            "function_details",
                            "class_details",
                            "import_details",
                            "analysis_method",
                            "metadata_json"
                        ],
                        "additionalProperties": True
                    }
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
                    "functions",
                    "classes",
                    "imports",
                    "complexity_score",
                    "has_type_hints",
                    "has_async",
                    "has_classes",
                    "metadata_json"
                ],
                "additionalProperties": True
            }
        }
    },
    "required": ["query", "results"],
    "additionalProperties": False
}


VECTOR_SEARCH_INPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
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
    "$schema": "http://json-schema.org/draft-07/schema#",
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
    "$schema": "http://json-schema.org/draft-07/schema#",
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
    "$schema": "http://json-schema.org/draft-07/schema#",
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
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {},
    "required": []
}
