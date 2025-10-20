#!/usr/bin/env python3

"""
Field mapping utilities for handling backend-specific payload formats.

This module provides mapping between standardized ChunkMetadata schema
and backend-specific storage formats (PostgreSQL JSONB vs Qdrant payload).
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

from .schemas import (
    ChunkMetadata,
    FilterOperator,
    QueryFilter,
    SearchResult,
    SearchResultType,
)

###################################################
# SINGLE SOURCE OF TRUTH: All database columns and field mappings
#
# Important:
# If you change something here, the collect code `code_embeddings.collect(..)`
# in src/cocoindex_code_mcp_server/cocoindex_config.py
# has to be adapted as well!
#
# TODO:
# Might be a limitation of cocoindex, we should recheck this.
###################################################
CONST_FIELD_MAPPINGS = {
    # Core search result fields
    "filename": "filename",
    "language": "language",
    "location": "location",
    "code": "code",
    "start": "start",
    "end": "end",
    "source_name": "source_name",
    "embedding": "embedding",

    # Metadata fields (promoted from metadata_json)
    "functions": "functions",
    "classes": "classes",
    "imports": "imports",
    "complexity_score": "complexity_score",
    "has_type_hints": "has_type_hints",
    "has_async": "has_async",
    "has_classes": "has_classes",
    "metadata_json": "metadata_json",
    "analysis_method": "analysis_method",
    "chunking_method": "chunking_method",
    "tree_sitter_analyze_error": "tree_sitter_analyze_error",
    "tree_sitter_chunking_error": "tree_sitter_chunking_error",
    "decorators_used": "decorators_used",
    "dunder_methods": "dunder_methods",
    "function_details": "function_details",
    "docstring": "docstring",

    # Additional metadata fields
    "success": "success",
    "parse_errors": "parse_errors",
    "char_count": "char_count",

    # Language-specific fields
    "nodes_with_errors": "nodes_with_errors",  # Haskell
    "data_types": "data_types",  # Haskell
    "instances": "instances",  # Haskell
    "type_classes": "type_classes",  # Haskell
    "modules": "modules",  # Haskell, Rust
    "has_module": "has_module",  # Haskell
    "data_type_details": "data_type_details",  # Haskell
    "structs": "structs",  # Rust
    "traits": "traits",  # Rust
    "impls": "impls",  # Rust
    "exports": "exports",  # TypeScript/JavaScript
    "types": "types",  # TypeScript
    "enums": "enums",  # TypeScript
    "namespaces": "namespaces",  # TypeScript/JavaScript/C++

    # python
    "private_methods": "private_methods",
    "variables": "variables",
    "decorators": "decorators",
    "class_details": "class_details",
}

# Derived configurations from single source of truth
CONST_CORE_FIELDS = {"filename", "language", "location", "code", "start", "end", "source_name", "embedding"}
CONST_METADATA_FIELDS = {k for k in CONST_FIELD_MAPPINGS.keys() if k not in CONST_CORE_FIELDS}
CONST_SELECTABLE_FIELDS = {k for k in CONST_FIELD_MAPPINGS.keys() if k != "embedding"}  # embedding handled separately

# Fields stored in JSONB vs individual columns
CONST_JSONB_FIELDS = {"metadata_json"}
CONST_INDIVIDUAL_COLUMNS = CONST_FIELD_MAPPINGS.keys() - CONST_JSONB_FIELDS


# SINGLE SOURCE OF TRUTH: Language mappings from file extensions to normalized language names
CONST_LANGUAGE_MAPPINGS = {
    # Core programming languages
    ".c": "C", ".h": "C",
    ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".hpp": "C++",
    ".cs": "C#",
    ".css": "CSS", ".scss": "CSS",
    ".f": "Fortran", ".f90": "Fortran", ".f95": "Fortran", ".f03": "Fortran",
    ".go": "Go",
    ".html": "HTML", ".htm": "HTML",
    ".java": "Java",
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".json": "JSON",
    ".kt": "Kotlin", ".kts": "Kotlin",  # FIX: Use title case "Kotlin"
    ".md": "Markdown", ".mdx": "Markdown",
    ".pas": "Pascal", ".dpr": "Pascal",
    ".php": "PHP",
    ".py": "Python",  # ".pyi": "Python",
    ".r": "R", ".R": "R",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".scala": "Scala",
    ".sql": "SQL", ".ddl": "SQL", ".dml": "SQL",
    ".swift": "Swift",
    ".toml": "TOML",
    ".tsx": "TSX",
    ".ts": "TypeScript",
    ".xml": "XML",
    ".hs": "Haskell", ".lhs": "Haskell",
    ".yaml": "YAML", ".yml": "YAML",
}

# Language name normalization for internal processing (tree-sitter, handlers, etc.)
# Maps normalized display names to internal processing names
CONST_LANGUAGE_INTERNAL_NAMES = {
    "C": "c",
    "C++": "cpp",
    "C#": "c_sharp",
    "CSS": "css",
    "Fortran": "fortran",
    "Go": "go",
    "HTML": "html",
    "Java": "java",
    "JavaScript": "javascript",
    "JSON": "json",
    "Kotlin": "kotlin",  # FIX: Internal processing uses lowercase
    "Markdown": "markdown",
    "Pascal": "pascal",
    "PHP": "php",
    "Python": "python",
    "R": "r",
    "Ruby": "ruby",
    "Rust": "rust",
    "Scala": "scala",
    "SQL": "sql",
    "Swift": "swift",
    "TOML": "toml",
    "TSX": "tsx",
    "TypeScript": "typescript",
    "XML": "xml",
    "Haskell": "haskell",
    "YAML": "yaml",
}

# TODO: Maybe this is related to CONST_LANGUAGE_MAPPINGS
SOURCE_CONFIG = {
    "path": "<forgotten to set this>",
    "included_patterns": [
        # Python
        "*.py", "*.pyi", "*.pyx", "*.pxd",
        # Rust
        "*.rs", "*.toml",
        # Java/Kotlin/JVM
        "*.java", "*.kt", "*.kts", "*.scala", "*.clj", "*.cljs",
        # JavaScript/TypeScript
        "*.js", "*.jsx", "*.ts", "*.tsx", "*.mjs", "*.cjs",
        # Go
        "*.go", "go.mod", "go.sum",
        # Haskell
        "*.hs", "*.lhs", "*.cabal",
        # "*.yaml", "*.yml", "stack.yaml",
        # C/C++
        "*.c", "*.cc", "*.cpp", "*.cxx", "*.h", "*.hh", "*.hpp", "*.hxx",
        # C#/.NET
        "*.cs", "*.fs", "*.fsx", "*.vb", "*.csproj", "*.fsproj", "*.vbproj", "*.sln",
        # Build systems
        "Makefile", "makefile", "*.mk", "*.cmake", "CMakeLists.txt",
        "build.gradle", "build.gradle.kts", "settings.gradle", "gradle.properties",
        "pom.xml", "build.xml", "*.ant",
        # Shell/Scripts
        "*.sh", "*.bash", "*.zsh", "*.fish", "*.ps1", "*.bat", "*.cmd",
        # Web
        "*.html", "*.htm", "*.css", "*.scss", "*.sass", "*.less",
        "*.vue", "*.svelte", "*.astro", "*.php", "*.rb",
        # Swift/Objective-C
        "*.swift", "*.m", "*.mm", "*.pbxproj", "*.xcconfig",
        # Documentation/Config
        "*.md", "*.mdx", "*.rst", "*.txt", "*.json", "*.xml",
        "*.ini", "*.cfg", "*.conf", "*.properties", "*.env",
        # Database
        "*.sql", "*.ddl", "*.dml", "*.migration",
        # Other languages
        "*.lua", "*.pl", "*.pm", "*.r", "*.R", "*.jl", "*.dart",
        "*.ex", "*.exs", "*.erl", "*.hrl", "*.elm", "*.nim",
        "*.zig", "*.odin", "*.v", "*.gleam", "*.crystal",
        # Docker/Container
        "Dockerfile", "*.dockerfile",
        # "docker-compose.yml", "docker-compose.yaml",
        # CI/CD
        "Jenkinsfile", "*.jenkinsfile",
        # "*.gitlab-ci.yml", ".github/workflows/*.yml", ".github/workflows/*.yaml",
        # "azure-pipelines.yml",
        # Package managers
        "package.json", "package-lock.json", "yarn.lock", "requirements.txt",
        "Pipfile", "poetry.lock", "pyproject.toml", "setup.py", "setup.cfg",
        "Gemfile", "Gemfile.lock", "composer.json", "composer.lock",
        # IDEs/Editors
        "*.editorconfig", "*.gitignore", "*.gitattributes",
    ],
    "excluded_patterns": [
        "**/.*", "target", "**/node_modules", "**/build", "**/dist",
        "**/__pycache__", "**/bin", "**/obj", "**/out", "**/venv",
        "**/env", "**/.gradle", "**/.idea", "**/.vscode",
        "**/target/debug", "**/target/release", "**/*.class",
        "**/*.jar", "**/*.war", "**/*.ear", "**/*.pyc", "**/*.pyo",
        # cocoindex evaluate
        "**/eval_CodeEmbedding_*",
        # compiled and cached
        "**/*.o", "**/*.obj", "**/*.exe", "**/*.dll",
        # scm
        "**/.git", "**/.svn", "**/.hg",
        # misc
        "**/.DS_Store", "**/Thumbs.db", "**/*.tmp",
        # python
        "**/.venv",
    ]
}


# Helper functions for language mapping
def get_language_from_extension(filename: str) -> str:
    """Get normalized language name from file extension."""
    import os
    basename = os.path.basename(filename)

    # Handle special files without extensions
    if basename.lower() in ["makefile", "dockerfile", "jenkinsfile"]:
        return basename.lower().title()

    # Handle special patterns
    if basename.lower().startswith("cmakelists"):
        return "CMake"
    if basename.lower().startswith("build.gradle"):
        return "Gradle"
    if basename.lower().startswith("pom.xml"):
        return "Maven"
    if "docker-compose" in basename.lower():
        return "Dockerfile"
    if basename.startswith("go."):
        return "Go"
    if basename.lower() in ["stack.yaml", "cabal.project"]:
        return "Haskell"

    # Get extension and map to language
    ext = os.path.splitext(filename)[1].lower()
    return CONST_LANGUAGE_MAPPINGS.get(ext, ext[1:] if ext.startswith('.') else "Unknown")


def get_internal_language_name(display_name: str) -> str:
    """Get internal processing name from display language name."""
    return CONST_LANGUAGE_INTERNAL_NAMES.get(display_name, display_name.lower())


def get_display_language_name(internal_name: str) -> str:
    """Get display language name from internal processing name."""
    for display, internal in CONST_LANGUAGE_INTERNAL_NAMES.items():
        if internal == internal_name:
            return display
    return internal_name.title()  # Fallback: capitalize first letter


# Fields that should be indexed in Qdrant for fast filtering
CONST_INDEXED_FIELDS = {
    "filename", "language", "source_name", "complexity_score",
    "has_type_hints", "has_async", "has_classes"
}


T = TypeVar('T')


class FieldMapper(ABC, Generic[T]):
    """Abstract base class for backend-specific field mapping."""

    @abstractmethod
    def to_backend_format(self, metadata: ChunkMetadata) -> T:
        """Convert ChunkMetadata to backend-specific format."""

    @abstractmethod
    def from_backend_format(self, backend_data: T) -> ChunkMetadata:
        """Convert backend-specific format to ChunkMetadata."""

    @abstractmethod
    def map_query_filter(self, query_filter: QueryFilter) -> Any:
        """Convert QueryFilter to backend-specific filter format."""


class PostgresFieldMapper(FieldMapper[Dict[str, Any]]):
    """
    Field mapper for PostgreSQL backend with JSONB metadata storage.

    PostgreSQL stores metadata in JSONB columns and individual fields
    as separate columns for performance.
    """

    FIELD_MAPPINGS = dict(CONST_FIELD_MAPPINGS)
    JSONB_FIELDS = set(CONST_JSONB_FIELDS)
    INDIVIDUAL_COLUMNS = set(CONST_INDIVIDUAL_COLUMNS)

    def to_backend_format(self, metadata: ChunkMetadata) -> Dict[str, Any]:
        """
        Convert ChunkMetadata to PostgreSQL row format.

        Returns dictionary ready for INSERT/UPDATE operations.
        """
        row_data = {}

        for schema_field, pg_column in self.FIELD_MAPPINGS.items():
            if schema_field in metadata:
                value = metadata[schema_field]  # type: ignore

                # Special handling for JSONB fields
                if schema_field in self.JSONB_FIELDS and isinstance(value, dict):
                    row_data[pg_column] = json.dumps(value)
                else:
                    row_data[pg_column] = value

        return row_data

    def from_backend_format(self, pg_row: Dict[str, Any]) -> ChunkMetadata:
        """
        Convert PostgreSQL row to ChunkMetadata.

        Args:
            pg_row: Dictionary representing a PostgreSQL row

        Returns:
            ChunkMetadata with all available fields
        """
        metadata: ChunkMetadata = {}

        for schema_field, pg_column in self.FIELD_MAPPINGS.items():
            if pg_column in pg_row and pg_row[pg_column] is not None:
                value = pg_row[pg_column]

                # Parse JSONB fields
                if schema_field == "metadata_json" and isinstance(value, str):
                    try:
                        metadata[schema_field] = json.loads(value)  # type: ignore
                    except json.JSONDecodeError:
                        metadata[schema_field] = {}  # type: ignore
                else:
                    metadata[schema_field] = value  # type: ignore

        return metadata

    def map_query_filter(self, query_filter: QueryFilter) -> str:
        """
        Convert QueryFilter to PostgreSQL WHERE clause fragment.

        Args:
            query_filter: Filter to convert

        Returns:
            SQL WHERE clause fragment with parameter placeholder

        Raises:
            ValueError: If field or operator is not supported
        """
        field = query_filter.field
        operator = query_filter.operator

        # Map field name to PostgreSQL column
        if field not in self.FIELD_MAPPINGS:
            raise ValueError(f"Unknown field '{field}' for PostgreSQL backend")

        pg_column = self.FIELD_MAPPINGS[field]

        # Handle different operators
        if operator == FilterOperator.EQUALS:
            return f"{pg_column} = %s"
        elif operator == FilterOperator.NOT_EQUALS:
            return f"{pg_column} != %s"
        elif operator == FilterOperator.GREATER_THAN:
            return f"{pg_column} > %s"
        elif operator == FilterOperator.GREATER_EQUAL:
            return f"{pg_column} >= %s"
        elif operator == FilterOperator.LESS_THAN:
            return f"{pg_column} < %s"
        elif operator == FilterOperator.LESS_EQUAL:
            return f"{pg_column} <= %s"
        elif operator == FilterOperator.LIKE:
            return f"{pg_column} LIKE %s"
        elif operator == FilterOperator.ILIKE:
            return f"{pg_column} ILIKE %s"
        elif operator == FilterOperator.IN:
            # Handle IN operator with multiple placeholders
            if not isinstance(query_filter.value, (list, tuple)):
                raise ValueError("IN operator requires list/tuple value")
            placeholders = ', '.join(['%s'] * len(query_filter.value))
            return f"{pg_column} IN ({placeholders})"
        elif operator == FilterOperator.NOT_IN:
            if not isinstance(query_filter.value, (list, tuple)):
                raise ValueError("NOT IN operator requires list/tuple value")
            placeholders = ', '.join(['%s'] * len(query_filter.value))
            return f"{pg_column} NOT IN ({placeholders})"
        elif operator == FilterOperator.IS_NULL:
            return f"{pg_column} IS NULL"
        elif operator == FilterOperator.IS_NOT_NULL:
            return f"{pg_column} IS NOT NULL"
        elif operator == FilterOperator.CONTAINS:
            # JSONB contains operation
            if pg_column == "metadata_json":
                return f"{pg_column} @> %s::jsonb"
            else:
                # For array fields, use ANY
                return f"%s = ANY({pg_column})"
        else:
            raise ValueError(f"Unsupported operator '{operator}' for PostgreSQL backend")

    def build_insert_query(self, table_name: str, metadata: ChunkMetadata) -> tuple[str, List[Any]]:
        """
        Build INSERT query for PostgreSQL.

        Args:
            table_name: Target table name
            metadata: Metadata to insert

        Returns:
            tuple: (sql_query, parameters)
        """
        row_data = self.to_backend_format(metadata)

        columns = list(row_data.keys())
        placeholders = ['%s'] * len(columns)
        values = list(row_data.values())

        query = f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT (filename, location) 
            DO UPDATE SET {', '.join(f'{col} = EXCLUDED.{col}' for col in columns)}
        """

        return query, values


class QdrantFieldMapper(FieldMapper[Dict[str, Any]]):
    """
    Field mapper for Qdrant backend with payload-based metadata storage.

    Qdrant stores all metadata in the payload object, with some fields
    potentially indexed for filtering performance.
    """

    INDEXED_FIELDS = set(CONST_INDEXED_FIELDS)

    def to_backend_format(self, metadata: ChunkMetadata) -> Dict[str, Any]:
        """
        Convert ChunkMetadata to Qdrant payload format.

        All fields go into the payload, with vectors handled separately.
        """
        payload = {}

        # Copy all metadata fields to payload
        for key, value in metadata.items():
            if key != "embedding":  # Embedding handled separately in Qdrant
                payload[key] = value

        return payload

    def from_backend_format(self, qdrant_point: Dict[str, Any]) -> ChunkMetadata:
        """
        Convert Qdrant point to ChunkMetadata.

        Args:
            qdrant_point: Qdrant point data with payload

        Returns:
            ChunkMetadata extracted from payload
        """
        payload = qdrant_point.get("payload", {})

        # Qdrant payload can directly map to ChunkMetadata
        metadata: ChunkMetadata = {}

        # Copy all payload fields
        for key, value in payload.items():
            if key in ChunkMetadata.__annotations__:
                metadata[key] = value  # type: ignore

        return metadata

    def map_query_filter(self, query_filter: QueryFilter) -> Dict[str, Any]:
        """
        Convert QueryFilter to Qdrant filter format.

        Args:
            query_filter: Filter to convert

        Returns:
            Qdrant filter dictionary

        Raises:
            ValueError: If operator is not supported
        """
        field = query_filter.field
        operator = query_filter.operator
        value = query_filter.value

        # Qdrant filter format
        if operator == FilterOperator.EQUALS:
            return {"key": field, "match": {"value": value}}
        elif operator == FilterOperator.NOT_EQUALS:
            return {"key": field, "match": {"value": value}, "must_not": True}
        elif operator == FilterOperator.GREATER_THAN:
            return {"key": field, "range": {"gt": value}}
        elif operator == FilterOperator.GREATER_EQUAL:
            return {"key": field, "range": {"gte": value}}
        elif operator == FilterOperator.LESS_THAN:
            return {"key": field, "range": {"lt": value}}
        elif operator == FilterOperator.LESS_EQUAL:
            return {"key": field, "range": {"lte": value}}
        elif operator == FilterOperator.IN:
            return {"key": field, "match": {"any": value}}
        elif operator == FilterOperator.NOT_IN:
            return {"key": field, "match": {"any": value}, "must_not": True}
        elif operator == FilterOperator.IS_NULL:
            return {"is_empty": {"key": field}}
        elif operator == FilterOperator.IS_NOT_NULL:
            return {"is_empty": {"key": field}, "must_not": True}
        elif operator == FilterOperator.CONTAINS:
            # For text fields, use text match
            return {"key": field, "match": {"text": str(value)}}
        else:
            raise ValueError(f"Unsupported operator '{operator}' for Qdrant backend")

    def build_search_filters(self, filters: List[QueryFilter], logic: str = "AND") -> Dict[str, Any]:
        """
        Build Qdrant search filters from multiple QueryFilters.

        Args:
            filters: List of filters to combine
            logic: "AND" or "OR" logic

        Returns:
            Qdrant filter structure
        """
        if not filters:
            return {}

        qdrant_filters = [self.map_query_filter(f) for f in filters]

        if len(qdrant_filters) == 1:
            return qdrant_filters[0]

        if logic.upper() == "AND":
            return {"must": qdrant_filters}
        else:  # OR
            return {"should": qdrant_filters}


class ResultMapper:
    """Utility for mapping search results between backends and standardized format."""

    @staticmethod
    def from_postgres_result(
        pg_row: Dict[str, Any],
        score: float,
        score_type: SearchResultType = SearchResultType.HYBRID_COMBINED
    ) -> SearchResult:
        """Convert PostgreSQL row to SearchResult."""
        mapper = PostgresFieldMapper()
        metadata = mapper.from_backend_format(pg_row)

        result = SearchResult(
            filename=metadata.get("filename", ""),
            language=metadata.get("language", ""),
            code=metadata.get("code", ""),
            location=metadata.get("location", ""),
            start=metadata.get("start", 0),
            end=metadata.get("end", 0),
            score=score,
            score_type=score_type,
            source=metadata.get("source_name", ""),
            metadata=metadata
        )

        # Dynamically add all metadata fields as attributes to the SearchResult object
        # This allows the MCP server to access them via hasattr() and getattr()
        for key, value in metadata.items():
            # Skip core fields that are already set as SearchResult attributes
            if key not in {"filename", "language", "code", "location", "start", "end", "source_name", "embedding"}:
                setattr(result, key, value)

        return result

    @staticmethod
    def from_qdrant_result(
        qdrant_point: Dict[str, Any],
        score: float,
        score_type: SearchResultType = SearchResultType.VECTOR_SIMILARITY
    ) -> SearchResult:
        """Convert Qdrant search result to SearchResult."""
        mapper = QdrantFieldMapper()
        metadata = mapper.from_backend_format(qdrant_point)

        return SearchResult(
            filename=metadata.get("filename", ""),
            language=metadata.get("language", ""),
            code=metadata.get("code", ""),
            location=metadata.get("location", ""),
            start=metadata.get("start", 0),
            end=metadata.get("end", 0),
            score=score,
            score_type=score_type,
            source=metadata.get("source_name", ""),
            metadata=metadata
        )


# =============================================================================
# Factory for creating mappers
# =============================================================================

class MapperFactory:
    """Factory for creating appropriate field mappers."""

    @staticmethod
    def create_mapper(backend_type: str) -> FieldMapper:
        """Create appropriate mapper for backend type."""
        if backend_type.lower() == "postgres":
            return PostgresFieldMapper()
        elif backend_type.lower() == "qdrant":
            return QdrantFieldMapper()
        elif backend_type.lower() == "mock":
            # For testing purposes, return PostgresFieldMapper as default
            return PostgresFieldMapper()
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")


# =============================================================================
# Export for convenience
# =============================================================================

__all__ = [
    "FieldMapper",
    "PostgresFieldMapper",
    "QdrantFieldMapper",
    "ResultMapper",
    "MapperFactory"
]
