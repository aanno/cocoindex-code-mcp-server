#!/usr/bin/env python3

"""
CocoIndex configuration and flow definitions.
"""

import json

# Temporarily disabled due to cocoindex compatibility
# from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Union, Any, cast

import numpy as np
from numpy.typing import NDArray

import cocoindex
import logging
# from cocoindex_code_mcp_server import LOGGER

# from sentence_transformers import SentenceTransformer  # Use cocoindex.functions.SentenceTransformerEmbed instead
from .ast_chunking import ASTChunkOperation, Chunk
from .lang.haskell.haskell_ast_chunker import get_haskell_language_spec
from .lang.python.python_code_analyzer import analyze_python_code
from .smart_code_embedding import LanguageModelSelector
from .parser_util import update_defaults

LOGGER = logging.getLogger(__name__)  # root logger

# Models will be instantiated directly (HuggingFace handles caching)

DEFAULT_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STACKTRACE = False

# Import our custom extensions
try:
    from .smart_code_embedding import LanguageModelSelector
    SMART_EMBEDDING_AVAILABLE = True
    LOGGER.info("Smart code embedding enabled and loaded successfully")
except ImportError as e:
    SMART_EMBEDDING_AVAILABLE = False
    LOGGER.warning(f"Smart code embedding not available: {e}")

try:
    AST_CHUNKING_AVAILABLE = True
    # TODO: for the moment
    # AST_CHUNKING_AVAILABLE = False
    LOGGER.info("AST chunking extension loaded")
except ImportError as e:
    AST_CHUNKING_AVAILABLE = False
    LOGGER.warning(f"AST chunking not available: {e}")

try:
    PYTHON_HANDLER_AVAILABLE = True
    # TODO: for the moment
    # PYTHON_HANDLER_AVAILABLE = False
    LOGGER.info("Python language handler extension loaded")
except ImportError as e:
    PYTHON_HANDLER_AVAILABLE = False
    LOGGER.warning(f"Python language handler not available: {e}")


@dataclass
class ChunkingParams:
    """Parameters for chunking code."""
    chunk_size: int
    min_chunk_size: int
    chunk_overlap: int
    max_chunk_size: int = 0  # For recursive splitting (will be set to chunk_size * 2 if 0)


@dataclass
class CodeMetadata:
    """Metadata extracted from code chunks."""
    metadata_json: str
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity_score: int
    has_type_hints: bool
    has_async: bool
    has_classes: bool
    decorators_used: List[str]
    analysis_method: str


# Language mappings moved to mappers.py as single source of truth

# Language-specific chunking parameters
CHUNKING_PARAMS = {
    # Larger chunks for documentation and config files
    "Markdown": ChunkingParams(chunk_size=2000, min_chunk_size=500, chunk_overlap=200, max_chunk_size=3000),
    # "YAML": ChunkingParams(chunk_size=800, min_chunk_size=200, chunk_overlap=100, max_chunk_size=1200),
    "JSON": ChunkingParams(chunk_size=1500, min_chunk_size=300, chunk_overlap=200, max_chunk_size=2200),
    "XML": ChunkingParams(chunk_size=1500, min_chunk_size=300, chunk_overlap=200, max_chunk_size=2200),
    "TOML": ChunkingParams(chunk_size=800, min_chunk_size=200, chunk_overlap=100, max_chunk_size=1200),

    # Smaller chunks for dense code
    "C": ChunkingParams(chunk_size=800, min_chunk_size=200, chunk_overlap=150, max_chunk_size=1200),
    "C++": ChunkingParams(chunk_size=800, min_chunk_size=200, chunk_overlap=150, max_chunk_size=1200),
    "Rust": ChunkingParams(chunk_size=1000, min_chunk_size=250, chunk_overlap=200, max_chunk_size=1500),
    "Go": ChunkingParams(chunk_size=1000, min_chunk_size=250, chunk_overlap=200, max_chunk_size=1500),
    "Java": ChunkingParams(chunk_size=1200, min_chunk_size=300, chunk_overlap=250, max_chunk_size=1800),
    "C#": ChunkingParams(chunk_size=1200, min_chunk_size=300, chunk_overlap=250, max_chunk_size=1800),
    "Scala": ChunkingParams(chunk_size=1000, min_chunk_size=250, chunk_overlap=200, max_chunk_size=1500),

    # Medium chunks for scripting languages
    "Python": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=250, max_chunk_size=1500),
    "JavaScript": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=250, max_chunk_size=1500),
    "TypeScript": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=250, max_chunk_size=1500),
    "TSX": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=250, max_chunk_size=1500),
    "Ruby": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=250, max_chunk_size=1500),
    "PHP": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=250, max_chunk_size=1500),

    # Web and styling
    "HTML": ChunkingParams(chunk_size=1500, min_chunk_size=400, chunk_overlap=200, max_chunk_size=2200),
    "CSS": ChunkingParams(chunk_size=1000, min_chunk_size=250, chunk_overlap=150, max_chunk_size=1500),

    # Data and scientific
    "SQL": ChunkingParams(chunk_size=1200, min_chunk_size=300, chunk_overlap=200, max_chunk_size=1800),
    "R": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=200, max_chunk_size=1500),
    "Fortran": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=200, max_chunk_size=1500),

    # Others
    "Pascal": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=200, max_chunk_size=1500),
    "Swift": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=200, max_chunk_size=1500),
    "Haskell": ChunkingParams(chunk_size=1200, min_chunk_size=300, chunk_overlap=200, max_chunk_size=2500),

    # Default fallback
    "_DEFAULT": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=200, max_chunk_size=2000),
}

# Effective chunking parameters (potentially scaled)
import copy
EFFECTIVE_CHUNKING_PARAMS = copy.deepcopy(CHUNKING_PARAMS)


# Custom language configurations for files not supported by tree-sitter
CUSTOM_LANGUAGES = [
    # Build systems
    cocoindex.functions.CustomLanguageSpec(
        language_name="Makefile",
        aliases=[".makefile"],
        separators_regex=[r"\n\n+", r"\n\w+:", r"\n"]
    ),
    cocoindex.functions.CustomLanguageSpec(
        language_name="CMake",
        aliases=[".cmake"],
        separators_regex=[r"\n\n+", r"\n\w+\(", r"\n"]
    ),
    cocoindex.functions.CustomLanguageSpec(
        language_name="Dockerfile",
        aliases=[".dockerfile"],
        separators_regex=[r"\n\n+",
                          r"\n(FROM|RUN|COPY|ADD|EXPOSE|ENV|CMD|ENTRYPOINT)", r"\n"]
    ),
    cocoindex.functions.CustomLanguageSpec(
        language_name="Gradle",
        aliases=[".gradle"],
        separators_regex=[r"\n\n+", r"\n\w+\s*\{", r"\n"]
    ),
    cocoindex.functions.CustomLanguageSpec(
        language_name="Maven",
        aliases=[".maven"],
        separators_regex=[r"</\w+>\s*<\w+>", r"\n\n+", r"\n"]
    ),
    # Shell scripts
    cocoindex.functions.CustomLanguageSpec(
        language_name="Shell",
        aliases=[".sh", ".bash"],
        separators_regex=[r"\n\n+", r"\nfunction\s+\w+", r"\n\w+\(\)", r"\n"]
    ),
    # Configuration files
    cocoindex.functions.CustomLanguageSpec(
        language_name="Config",
        aliases=[".ini", ".cfg", ".conf"],
        separators_regex=[r"\n\n+", r"\n\[.*\]", r"\n"]
    ),
    # Haskell - using enhanced AST-aware separators
    get_haskell_language_spec(),
    # Kotlin
    cocoindex.functions.CustomLanguageSpec(
        language_name="Kotlin",
        aliases=["kt", ".kt", "kts", ".kts"],
        separators_regex=[r"\n\n+", r"\nfun\s+", r"\nclass\s+", r"\nobject\s+",
                          r"\ninterface\s+", r"\n"]
    ),
]


@cocoindex.op.function()
def extract_language(filename: str) -> str:
    """Extract the language from a filename for tree-sitter processing."""
    from .mappers import get_language_from_extension
    return get_language_from_extension(filename)


@cocoindex.op.function()
def get_chunking_params(language: str) -> ChunkingParams:
    """Get language-specific chunking parameters."""
    params = EFFECTIVE_CHUNKING_PARAMS.get(language, EFFECTIVE_CHUNKING_PARAMS["_DEFAULT"])
    
    # Ensure max_chunk_size is properly set
    if params.max_chunk_size <= 0:
        params = ChunkingParams(
            chunk_size=params.chunk_size,
            min_chunk_size=params.min_chunk_size,
            chunk_overlap=params.chunk_overlap,
            max_chunk_size=params.chunk_size * 2
        )
    
    return params


@cocoindex.op.function()
def create_default_metadata(content: str) -> str:
    """Create default metadata structure for default language handler."""
    default_metadata = {
        "functions": [],
        "classes": [],
        "imports": [],
        "complexity_score": 0,
        "has_type_hints": False,
        "has_async": False,
        "has_classes": False,
        "decorators_used": [],
        "analysis_method": "default_basic",
    }
    return json.dumps(default_metadata)


@cocoindex.op.function()
def extract_code_metadata(text: str, language: str, filename: str = "") -> str:
    """Extract rich metadata from code chunks based on language and return as JSON string."""
    # Check if we should use default language handler
    use_default_handler = _global_flow_config.get('use_default_language_handler', False)
    
    # DEBUG: Log configuration for specific files
    if filename and 'cpp_visitor.py' in filename:
        LOGGER.info(f"🔍 DEBUGGING extract_code_metadata for {filename}")
        LOGGER.info(f"   language: {language}")
        LOGGER.info(f"   use_default_handler: {use_default_handler}")
        LOGGER.info(f"   PYTHON_HANDLER_AVAILABLE: {PYTHON_HANDLER_AVAILABLE}")
        LOGGER.info(f"   _global_flow_config: {_global_flow_config}")

    try:
        if language == "Python" and PYTHON_HANDLER_AVAILABLE and not use_default_handler:
            # Use our advanced Python handler through the tree-sitter analyzer
            try:
                from .lang.python.tree_sitter_python_analyzer import (
                    TreeSitterPythonAnalyzer,
                )
                LOGGER.debug("Using TreeSitterPythonAnalyzer with integrated PythonNodeHandler")
                analyzer = TreeSitterPythonAnalyzer(prefer_tree_sitter=True)
                metadata = analyzer.analyze_code(text, filename)
            except Exception as e:
                LOGGER.debug(f"TreeSitterPythonAnalyzer failed, falling back to basic analysis: {e}")
                metadata = analyze_python_code(text, filename)
        elif language == "Python":
            metadata = analyze_python_code(text, filename)
        else:
            # For non-Python languages, use specialized analyzers
            metadata = None
            try:
                # Normalize language string for consistent matching
                lang_lower = language.lower() if language else ""
                
                if lang_lower == "rust":
                    from .language_handlers.rust_visitor import analyze_rust_code
                    metadata = analyze_rust_code(text, filename)
                elif lang_lower == "java": 
                    from .language_handlers.java_visitor import analyze_java_code
                    metadata = analyze_java_code(text, filename)
                elif lang_lower in ["javascript", "js"]:
                    from .language_handlers.javascript_visitor import analyze_javascript_code
                    metadata = analyze_javascript_code(text, "javascript", filename)
                elif lang_lower in ["typescript", "ts"]:
                    from .language_handlers.typescript_visitor import analyze_typescript_code
                    metadata = analyze_typescript_code(text, "typescript", filename)
                elif lang_lower in ["cpp", "c++", "cxx"]:
                    from .language_handlers.cpp_visitor import analyze_cpp_code
                    metadata = analyze_cpp_code(text, "cpp", filename)
                elif lang_lower == "c":
                    from .language_handlers.c_visitor import analyze_c_code
                    metadata = analyze_c_code(text, filename)
                elif lang_lower in ["kotlin", "kt"]:
                    from .language_handlers.kotlin_visitor import analyze_kotlin_code
                    metadata = analyze_kotlin_code(text, filename)
                elif lang_lower in ["haskell", "hs"]:
                    from .language_handlers.haskell_visitor import analyze_haskell_code
                    metadata = analyze_haskell_code(text, filename)
                else:
                    LOGGER.debug(f"No specialized analyzer for language: {language}")
                    
            except ImportError as e:
                LOGGER.warning(f"Failed to import analyzer for {language}: {e}")
            except Exception as e:
                LOGGER.warning(f"Analysis failed for {language}: {e}")
                
            # Fallback to basic metadata if analysis failed or no analyzer available
            if metadata is None:
                metadata = {}
            if not metadata.get('success', True):  # Fixed: Apply fallback when analysis FAILED
                update_defaults(metadata, {
                    "language": language,
                    "analysis_method": "no_success_analyze",
                    "functions": [],
                    "classes": [],
                    "imports": [],
                    "complexity_score": 0,
                    "has_type_hints": False,
                    "has_async": False,
                    "has_classes": False,
                    "decorators_used": [],
                    # Promoted metadata fields for database columns
                    "chunking_method": "no_success_chunking",
                    "tree_sitter_chunking_error": True,  # True because we failed to use tree-sitter
                    "tree_sitter_analyze_error": True,   # True because we failed to analyze properly
                })

        # Return ALL fields from metadata (generalized approach)
        if metadata is not None:
            # Start with all metadata fields and ensure critical ones have defaults
            result = dict(metadata)  # Copy all fields
            
            # Ensure essential fields have proper defaults if missing
            update_defaults(result, {
                "functions": [],
                "classes": [],
                "imports": [],
                "complexity_score": 0,
                "has_type_hints": False,
                "has_async": False,
                "has_classes": False,
                "decorators_used": [],
                "analysis_method": "unknown_analysis",
                "chunking_method": "unknown_chunking",
                "tree_sitter_chunking_error": False,
                "tree_sitter_analyze_error": False,
                "dunder_methods": [],
            })
        else:
            result = {}
        return json.dumps(result)

    except Exception as e:
        # Fallback to empty metadata if everything fails
        if filename and 'cpp_visitor.py' in filename:
            LOGGER.error(f"❌ EXCEPTION in extract_code_metadata for {filename}: {e}")
            import traceback
            LOGGER.error(f"   Traceback: {traceback.format_exc()}")
        else:
            LOGGER.debug(f"Metadata extraction failed for {filename}, using empty metadata: {e}")
        fallback_result = {
            "functions": [],
            "classes": [],
            "imports": [],
            "complexity_score": 0,
            "has_type_hints": False,
            "has_async": False,
            "has_classes": False,
            "decorators_used": [],
            "analysis_method": "error_fallback",
            # Promoted metadata fields for database columns
            "chunking_method": "error_fallback",
            "tree_sitter_chunking_error": True,  # True because we had an error
            "tree_sitter_analyze_error": True,   # True because we had an error
        }
        return json.dumps(fallback_result)


# =============================================================================
# Generalized Metadata Extraction Functions
# =============================================================================

@cocoindex.op.function()
def extract_string_field(metadata_json: str, field_name: str = "field", default_value: str = "") -> str:
    """
    Generic string field extractor from metadata JSON.
    
    Args:
        metadata_json: JSON string or dict containing metadata
        field_name: Name of the field to extract
        default_value: Default value if field is missing or extraction fails
        
    Returns:
        String value of the field
    """
    try:
        if not metadata_json:
            return default_value
        # Parse JSON string to dict
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        return str(metadata_dict.get(field_name, default_value))
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for {field_name}: {e}")
        return default_value


@cocoindex.op.function()
def extract_list_as_string_field(metadata_json: str, field_name: str = "field", default_value: str = "[]") -> str:
    """
    Generic list field extractor from metadata JSON, returned as string representation.
    
    Args:
        metadata_json: JSON string or dict containing metadata
        field_name: Name of the field to extract
        default_value: Default value if field is missing or extraction fails
        
    Returns:
        String representation of the list field
    """
    try:
        if not metadata_json:
            return default_value
        # Parse JSON string to dict
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        field_value = metadata_dict.get(field_name, [])
        # Ensure it's a list and convert to string representation
        if isinstance(field_value, list):
            return str(field_value)
        else:
            return str([field_value]) if field_value else default_value
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for {field_name}: {e}")
        return default_value


@cocoindex.op.function()
def extract_bool_field(metadata_json: str, field_name: str = "field", default_value: bool = False) -> bool:
    """
    Generic boolean field extractor from metadata JSON.
    
    Args:
        metadata_json: JSON string or dict containing metadata
        field_name: Name of the field to extract
        default_value: Default value if field is missing or extraction fails
        
    Returns:
        Boolean value of the field
    """
    try:
        if not metadata_json:
            return default_value
        # Parse JSON string to dict
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        return bool(metadata_dict.get(field_name, default_value))
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for {field_name}: {e}")
        return default_value


@cocoindex.op.function()
def extract_int_field(metadata_json: str, field_name: str = "field", default_value: int = 0) -> int:
    """
    Generic integer field extractor from metadata JSON.
    
    Args:
        metadata_json: JSON string or dict containing metadata
        field_name: Name of the field to extract
        default_value: Default value if field is missing or extraction fails
        
    Returns:
        Integer value of the field
    """
    try:
        if not metadata_json:
            return default_value
        # Parse JSON string to dict
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        field_value = metadata_dict.get(field_name, default_value)
        return int(field_value) if isinstance(field_value, (int, float, str)) and str(field_value).replace('.', '').replace('-', '').isdigit() else default_value
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for {field_name}: {e}")
        return default_value


# =============================================================================
# Specific Field Extraction Functions (using generalized extractors)
# =============================================================================

@cocoindex.op.function()
def extract_functions_field(metadata_json: str) -> str:
    """Extract functions field from metadata JSON."""
    return extract_list_as_string_field(metadata_json, "functions", "[]")


@cocoindex.op.function()
def extract_docstring_field(metadata_json: str) -> str:
    """Extract docstring field from metadata JSON."""
    return extract_string_field(metadata_json, "docstring", "")


@cocoindex.op.function()
def extract_classes_field(metadata_json: str) -> str:
    """Extract classes field from metadata JSON."""
    return extract_list_as_string_field(metadata_json, "classes", "[]")


@cocoindex.op.function()
def extract_imports_field(metadata_json: str) -> str:
    """Extract imports field from metadata JSON."""
    return extract_list_as_string_field(metadata_json, "imports", "[]")


@cocoindex.op.function()
def extract_complexity_score_field(metadata_json: str) -> int:
    """Extract complexity_score field from metadata JSON."""
    return extract_int_field(metadata_json, "complexity_score", 0)


@cocoindex.op.function()
def extract_has_type_hints_field(metadata_json: str) -> bool:
    """Extract has_type_hints field from metadata JSON."""
    return extract_bool_field(metadata_json, "has_type_hints", False)


@cocoindex.op.function()
def extract_has_async_field(metadata_json: str) -> bool:
    """Extract has_async field from metadata JSON."""
    return extract_bool_field(metadata_json, "has_async", False)


@cocoindex.op.function()
def extract_analysis_method_field(metadata_json: str) -> str:
    """Extract analysis_method field from metadata JSON."""
    return extract_string_field(metadata_json, "analysis_method", "unknown")


@cocoindex.op.function()
def extract_chunking_method_field(metadata_json: str) -> str:
    """Extract chunking_method field from metadata JSON."""
    return extract_string_field(metadata_json, "chunking_method", "unknown")


@cocoindex.op.function()
def extract_tree_sitter_chunking_error_field(metadata_json: str) -> bool:
    """Extract tree_sitter_chunking_error field from metadata JSON."""
    return extract_bool_field(metadata_json, "tree_sitter_chunking_error", False)


@cocoindex.op.function()
def extract_tree_sitter_analyze_error_field(metadata_json: str) -> bool:
    """Extract tree_sitter_analyze_error field from metadata JSON."""
    return extract_bool_field(metadata_json, "tree_sitter_analyze_error", False)


@cocoindex.op.function()
def extract_decorators_used_field(metadata_json: str) -> str:
    """Extract decorators_used field from metadata JSON."""
    return extract_list_as_string_field(metadata_json, "decorators_used", "[]")


@cocoindex.op.function()
def extract_dunder_methods_field(metadata_json: str) -> str:
    """Extract dunder_methods field from metadata JSON."""
    return extract_list_as_string_field(metadata_json, "dunder_methods", "[]")


@cocoindex.op.function()
def extract_success_field(metadata_json: str) -> bool:
    """Extract success field from metadata JSON."""
    try:
        if not metadata_json:
            return True
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        return bool(metadata_dict.get("success", True))
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for success: {e}")
        return True


@cocoindex.op.function()
def extract_parse_errors_field(metadata_json: str) -> int:
    """Extract parse_errors field from metadata JSON."""
    try:
        if not metadata_json:
            return 0
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        return int(metadata_dict.get("parse_errors", 0))
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for parse_errors: {e}")
        return 0


@cocoindex.op.function()
def extract_char_count_field(metadata_json: str) -> int:
    """Extract char_count field from metadata JSON."""
    try:
        if not metadata_json:
            return 0
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        return int(metadata_dict.get("char_count", 0))
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for char_count: {e}")
        return 0


# Haskell-specific fields
@cocoindex.op.function()
def extract_nodes_with_errors_field(metadata_json: str) -> str:
    """Extract nodes_with_errors field from metadata JSON (Haskell only)."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        nodes_with_errors = metadata_dict.get("nodes_with_errors", [])
        if isinstance(nodes_with_errors, list):
            return str(nodes_with_errors)
        else:
            return str([nodes_with_errors]) if nodes_with_errors else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for nodes_with_errors: {e}")
        return "[]"


@cocoindex.op.function()
def extract_data_types_field(metadata_json: str) -> str:
    """Extract data_types field from metadata JSON (Haskell only)."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        data_types = metadata_dict.get("data_types", [])
        if isinstance(data_types, list):
            return str(data_types)
        else:
            return str([data_types]) if data_types else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for data_types: {e}")
        return "[]"


@cocoindex.op.function()
def extract_instances_field(metadata_json: str) -> str:
    """Extract instances field from metadata JSON (Haskell only)."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        instances = metadata_dict.get("instances", [])
        if isinstance(instances, list):
            return str(instances)
        else:
            return str([instances]) if instances else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for instances: {e}")
        return "[]"


@cocoindex.op.function()
def extract_type_classes_field(metadata_json: str) -> str:
    """Extract type_classes field from metadata JSON (Haskell only)."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        type_classes = metadata_dict.get("type_classes", [])
        if isinstance(type_classes, list):
            return str(type_classes)
        else:
            return str([type_classes]) if type_classes else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for type_classes: {e}")
        return "[]"


@cocoindex.op.function()
def extract_modules_field(metadata_json: str) -> str:
    """Extract modules field from metadata JSON (Haskell, Rust)."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        modules = metadata_dict.get("modules", [])
        if isinstance(modules, list):
            return str(modules)
        else:
            return str([modules]) if modules else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for modules: {e}")
        return "[]"


@cocoindex.op.function()
def extract_has_module_field(metadata_json: str) -> bool:
    """Extract has_module field from metadata JSON (Haskell only)."""
    try:
        if not metadata_json:
            return False
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        return bool(metadata_dict.get("has_module", False))
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for has_module: {e}")
        return False


@cocoindex.op.function()
def extract_function_details_field(metadata_json: str) -> str:
    """Extract function_details field from metadata JSON (Haskell only)."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        function_details = metadata_dict.get("function_details", [])
        if isinstance(function_details, list):
            return str(function_details)
        else:
            return str([function_details]) if function_details else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for function_details: {e}")
        return "[]"


@cocoindex.op.function()
def extract_data_type_details_field(metadata_json: str) -> str:
    """Extract data_type_details field from metadata JSON (Haskell only)."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        data_type_details = metadata_dict.get("data_type_details", [])
        if isinstance(data_type_details, list):
            return str(data_type_details)
        else:
            return str([data_type_details]) if data_type_details else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for data_type_details: {e}")
        return "[]"


# Rust-specific fields
@cocoindex.op.function()
def extract_structs_field(metadata_json: str) -> str:
    """Extract structs field from metadata JSON (Rust only)."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        structs = metadata_dict.get("structs", [])
        if isinstance(structs, list):
            return str(structs)
        else:
            return str([structs]) if structs else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for structs: {e}")
        return "[]"


@cocoindex.op.function()
def extract_traits_field(metadata_json: str) -> str:
    """Extract traits field from metadata JSON (Rust only)."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        traits = metadata_dict.get("traits", [])
        if isinstance(traits, list):
            return str(traits)
        else:
            return str([traits]) if traits else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for traits: {e}")
        return "[]"


@cocoindex.op.function()
def extract_impls_field(metadata_json: str) -> str:
    """Extract impls field from metadata JSON (Rust only)."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        impls = metadata_dict.get("impls", [])
        if isinstance(impls, list):
            return str(impls)
        else:
            return str([impls]) if impls else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for impls: {e}")
        return "[]"


# TypeScript/JavaScript-specific fields
@cocoindex.op.function()
def extract_exports_field(metadata_json: str) -> str:
    """Extract exports field from metadata JSON (TypeScript, JavaScript only)."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        exports = metadata_dict.get("exports", [])
        if isinstance(exports, list):
            return str(exports)
        else:
            return str([exports]) if exports else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for exports: {e}")
        return "[]"


@cocoindex.op.function()
def extract_types_field(metadata_json: str) -> str:
    """Extract types field from metadata JSON (TypeScript only)."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        types = metadata_dict.get("types", [])
        if isinstance(types, list):
            return str(types)
        else:
            return str([types]) if types else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for types: {e}")
        return "[]"


@cocoindex.op.function()
def extract_enums_field(metadata_json: str) -> str:
    """Extract enums field from metadata JSON (TypeScript only)."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        enums = metadata_dict.get("enums", [])
        if isinstance(enums, list):
            return str(enums)
        else:
            return str([enums]) if enums else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for enums: {e}")
        return "[]"


@cocoindex.op.function()
def extract_namespaces_field(metadata_json: str) -> str:
    """Extract namespaces field from metadata JSON (TypeScript, JavaScript, C++ only)."""
    try:
        if not metadata_json:
            return "[]"
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        namespaces = metadata_dict.get("namespaces", [])
        if isinstance(namespaces, list):
            return str(namespaces)
        else:
            return str([namespaces]) if namespaces else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for namespaces: {e}")
        return "[]"


@cocoindex.op.function()
def ensure_unique_chunk_locations(chunks) -> List[Chunk]:
    """
    Post-process chunks to ensure location fields are unique within the file.
    This prevents PostgreSQL 'ON CONFLICT DO UPDATE' duplicate key errors.
    Keeps original chunk format for CocoIndex compatibility.
    """
    if not chunks:
        return []

    # Convert chunks to list if needed
    chunk_list = list(chunks) if hasattr(chunks, '__iter__') else [chunks]

    seen_locations = set()
    unique_chunks = []

    for i, chunk in enumerate(chunk_list):
        # Extract values from chunk (dict or dataclass) and convert to Chunk
        if hasattr(chunk, 'location'):
            # Already a Chunk dataclass
            base_loc = chunk.location
            text = chunk.content
            start = chunk.start
            end = chunk.end
            metadata = chunk.metadata
        elif isinstance(chunk, dict):
            # Dictionary format from SplitRecursively - convert to Chunk
            base_loc = chunk.get("location", f"chunk_{i}")
            text = chunk.get("content", chunk.get("text", ""))
            start = chunk.get("start", 0)
            end = chunk.get("end", 0)
            metadata = chunk.get("metadata", {})
        else:
            # Fallback for unexpected types
            base_loc = f"chunk_{i}"
            text = str(chunk) if chunk else ""
            start = 0
            end = 0
            metadata = {}

        # Make location unique
        unique_loc = base_loc
        suffix = 0
        while unique_loc in seen_locations:
            suffix += 1
            unique_loc = f"{base_loc}#{suffix}"

        seen_locations.add(unique_loc)

        # Always create Chunk dataclass with unique location, preserving metadata
        unique_chunk = Chunk(
            content=text,
            metadata=metadata,  # CRITICAL FIX: Preserve metadata instead of discarding it
            location=unique_loc,
            start=start,
            end=end
        )
        unique_chunks.append(unique_chunk)

    return unique_chunks


@cocoindex.op.function()
def convert_dataslice_to_string(content) -> str:
    """Convert CocoIndex DataSlice content to string."""
    try:
        result = str(content) if content else ""
        LOGGER.info(f"🔍 DataSlice conversion: input type={type(content)}, output_len={len(result)}")
        if len(result) == 0:
            LOGGER.error(f"❌ DataSlice conversion produced empty string! Input: {repr(content)}")
        return result
    except Exception as e:
        LOGGER.error(f"Failed to convert content to string: {e}")
        return ""

@cocoindex.op.function()
def extract_has_classes_field(metadata_json: str) -> bool:
    """Extract has_classes field from metadata JSON."""
    try:
        if not metadata_json:
            return False
        # Parse JSON string to dict
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        return bool(metadata_dict.get("has_classes", False))
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for has_classes: {e}")
        return False


@cocoindex.op.function()
def promote_metadata_fields(metadata_json: str) -> Dict[str, Any]:
    """
    Promote ALL fields from metadata_json to top-level fields with appropriate type conversion.
    This generalized approach automatically promotes any field found in metadata_json,
    making the system flexible and future-proof without needing to update field lists.
    """
    try:
        if not metadata_json:
            return {}
        
        metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        if not isinstance(metadata_dict, dict):
            return {}
        
        promoted = {}
        
        # Define type conversions for known fields
        field_conversions = {
            # String fields
            'analysis_method': lambda x: str(x) if x is not None else "unknown",
            'chunking_method': lambda x: str(x) if x is not None else "unknown",
            'language': lambda x: str(x) if x is not None else "unknown",
            'filename': lambda x: str(x) if x is not None else "",
            'docstring': lambda x: str(x) if x is not None else "",
            
            # Boolean fields (handle string "true"/"false" values)
            'tree_sitter_chunking_error': lambda x: x.lower() == "true" if isinstance(x, str) else bool(x) if x is not None else False,
            'tree_sitter_analyze_error': lambda x: x.lower() == "true" if isinstance(x, str) else bool(x) if x is not None else False,
            'has_type_hints': lambda x: bool(x) if x is not None else False,
            'has_async': lambda x: bool(x) if x is not None else False,
            'has_classes': lambda x: bool(x) if x is not None else False,
            'success': lambda x: bool(x) if x is not None else False,
            'tree_sitter_analyze_error': lambda x: bool(x) if x is not None else False,
            'tree_sitter_chunking_error': lambda x: bool(x) if x is not None else False,
            'has_docstrings': lambda x: bool(x) if x is not None else False,
            
            # Integer fields
            'complexity_score': lambda x: int(x) if x is not None and str(x).isdigit() else 0,
            'line_count': lambda x: int(x) if x is not None and str(x).isdigit() else 0,
            'char_count': lambda x: int(x) if x is not None and str(x).isdigit() else 0,
            'parse_errors': lambda x: int(x) if x is not None and str(x).isdigit() else 0,
            
            # List fields
            'functions': lambda x: list(x) if isinstance(x, (list, tuple)) else [],
            'classes': lambda x: list(x) if isinstance(x, (list, tuple)) else [],
            'imports': lambda x: list(x) if isinstance(x, (list, tuple)) else [],
            'decorators_used': lambda x: list(x) if isinstance(x, (list, tuple)) else [],
            'errors': lambda x: list(x) if isinstance(x, (list, tuple)) else [],
            'dunder_methods': lambda x: list(x) if isinstance(x, (list, tuple)) else [],
            'private_methods': lambda x: list(x) if isinstance(x, (list, tuple)) else [],
            'variables': lambda x: list(x) if isinstance(x, (list, tuple)) else [],
            'de': lambda x: list(x) if isinstance(x, (list, tuple)) else [],
            'variables': lambda x: list(x) if isinstance(x, (list, tuple)) else [],
            'decorators': lambda x: list(x) if isinstance(x, (list, tuple)) else [],
        }
        
        # Apply conversions for known fields
        for field, converter in field_conversions.items():
            if field in metadata_dict:
                try:
                    promoted[field] = converter(metadata_dict[field])
                except Exception as e:
                    LOGGER.debug(f"Failed to convert field {field}: {e}")
                    # Set safe defaults based on field type
                    if field in ['analysis_method', 'chunking_method', 'language', 'filename']:
                        promoted[field] = "unknown" if field != 'filename' else ""
                    elif field in ['tree_sitter_chunking_error', 'tree_sitter_analyze_error', 'has_type_hints', 'has_async', 'has_classes', 'success']:
                        promoted[field] = False
                    elif field in ['complexity_score', 'line_count', 'char_count', 'parse_errors']:
                        promoted[field] = 0
                    elif field in ['functions', 'classes', 'imports', 'decorators_used', 'errors']:
                        promoted[field] = []
        
        # For any remaining fields not in our conversion map, pass them through with basic type safety
        for field, value in metadata_dict.items():
            if field not in promoted and field not in ['metadata_json']:  # Avoid infinite recursion
                if isinstance(value, (str, int, float, bool, list, dict)):
                    promoted[field] = value
                else:
                    promoted[field] = str(value)  # Convert unknown types to string
        
        return promoted
        
    except Exception as e:
        LOGGER.debug(f"Failed to promote metadata fields: {e}")
        return {}


@cocoindex.transform_flow()
def code_to_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[NDArray[np.float32]]:
    """
    Default embedding using a SentenceTransformer model with caching.
    """

    return text.transform(
        # Embed text using SentenceTransformer model with meta tensor handling.
        cocoindex.functions.SentenceTransformerEmbed(
            model=DEFAULT_TRANSFORMER_MODEL
        )
    )

# Removed helper function that was causing DataScope context issues


@cocoindex.op.function()
def select_embedding_model_for_language(language: str) -> str:
    """
    Select appropriate embedding model based on programming language.
    """
    if not SMART_EMBEDDING_AVAILABLE:
        LOGGER.debug(f"Smart embedding not available for {language}, using default")
        return DEFAULT_TRANSFORMER_MODEL

    # Use the smart embedding selector with actual language value
    selector = LanguageModelSelector()
    selected_model = selector.select_model(language=language.lower())

    LOGGER.debug(f"Selected embedding model: {selected_model} for language: {language}")
    return selected_model


@cocoindex.transform_flow()
def graphcodebert_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[NDArray[np.float32]]:
    """GraphCodeBERT embedding for Python, Java, JavaScript, PHP, Ruby, Go, C, C++."""
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="microsoft/graphcodebert-base"
        )
    )


@cocoindex.transform_flow()
def unixcoder_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[NDArray[np.float32]]:
    """UniXcode embedding for Rust, TypeScript, C#, Kotlin, Scala, Swift, Dart."""
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="microsoft/unixcoder-base"
        )
    )


@cocoindex.transform_flow()
def fallback_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[NDArray[np.float32]]:
    """Fallback embedding for languages not supported by specialized models."""
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model=DEFAULT_TRANSFORMER_MODEL
        )
    )


# Language group to embedding model mapping for smart embedding
LANGUAGE_MODEL_GROUPS = {
    # GraphCodeBERT - optimized for these specific languages
    'graphcodebert': {
        'model': 'microsoft/graphcodebert-base',
        'languages': {'python', 'java', 'javascript', 'php', 'ruby', 'go', 'c', 'c++'}
    },
    # UniXcode - optimized for these languages
    'unixcoder': {
        'model': 'microsoft/unixcoder-base',
        'languages': {'rust', 'typescript', 'tsx', 'c#', 'kotlin', 'scala', 'swift', 'dart'}
    },
    # Fallback for all other languages
    'fallback': {
        'model': DEFAULT_TRANSFORMER_MODEL,
        'languages': set()  # Will catch everything else
    }
}


@cocoindex.op.function()
def get_embedding_model_group(language: str) -> str:
    """Get the appropriate embedding model group for a language."""
    lang_lower = language.lower()

    for group_name, group_info in LANGUAGE_MODEL_GROUPS.items():
        if group_name == 'fallback':
            continue  # Handle fallback last
        if lang_lower in group_info['languages']:
            return group_name

    # Default to fallback for unrecognized languages
    return 'fallback'


@cocoindex.op.function()
def get_cocoindex_split_recursively_chunking_method(content: str) -> str:
    """Return chunking method for default SplitRecursively chunking."""
    return "cocoindex_split_recursively"


@cocoindex.op.function()
def get_ast_tree_sitter_chunking_method(content: str) -> str:
    """Return chunking method for AST tree-sitter chunking."""
    return "ast_tree_sitter"


@cocoindex.op.function()
def get_ast_fallback_chunking_method(content: str) -> str:
    """Return chunking method for AST fallback chunking."""
    return "ast_fallback_unavailable"


@cocoindex.op.function()
def get_file_chunking_method(chunking_method_used: str) -> str:
    """Return the chunking method used for this file."""
    return chunking_method_used


@cocoindex.op.function()
def get_chunking_method_from_metadata(metadata_json: str) -> str:
    """Extract chunking method directly from metadata without preference logic."""
    return extract_string_field(metadata_json, "chunking_method", "unknown_chunking")



# Global configuration for flow parameters
_global_flow_config = {
    'paths': ["."],  # Use current directory for testing
    'enable_polling': False,
    'poll_interval': 30,
    'use_smart_embedding': True,  # Enable smart language-aware embedding
}


@cocoindex.flow_def(name="CodeEmbedding")
def code_embedding_flow(
    flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
) -> None:
    """
    Define an improved flow that embeds files with syntax-aware chunking.
    Reads configuration from global _global_flow_config.
    """
    # Get configuration from global settings
    paths = _global_flow_config.get('paths', ["cocoindex"])
    enable_polling = _global_flow_config.get('enable_polling', False)
    poll_interval = _global_flow_config.get('poll_interval', 30)

    # Add multiple sources - CocoIndex supports this natively!
    all_files_sources = []

    # Cast paths to list to satisfy mypy
    paths_list = list(paths) if hasattr(paths, '__iter__') else ["cocoindex"]
    for i, path in enumerate(paths_list):
        source_name = f"files_{i}" if len(paths_list) > 1 else "files"
        LOGGER.info(f"Adding source: {path} as '{source_name}'")

        # Configure LocalFile source with optional polling
        source_config = {
            "path": path,
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

        # Note: Polling configuration is handled by CocoIndex live updater, not LocalFile
        if enable_polling:
            LOGGER.info(f"  Polling enabled: {poll_interval}s interval (handled by live updater)")

        data_scope[source_name] = flow_builder.add_source(
            cocoindex.sources.LocalFile(**source_config)
        )
        all_files_sources.append(source_name)

    # Create a single collector for all sources
    code_embeddings = data_scope.add_collector()

    # Process each source with the same logic
    for source_name in all_files_sources:
        with data_scope[source_name].row() as file:
            file["language"] = file["filename"].transform(extract_language)
            file["chunking_params"] = file["language"].transform(get_chunking_params)

            # Choose chunking method based on configuration
            use_default_chunking = _global_flow_config.get('use_default_chunking', False)

            if use_default_chunking or not AST_CHUNKING_AVAILABLE:
                if not use_default_chunking and not AST_CHUNKING_AVAILABLE:
                    LOGGER.info("AST chunking not available, using default recursive splitting")
                else:
                    LOGGER.info("Using default recursive splitting (--default-chunking flag set)")
                raw_chunks = file["content"].transform(
                    cocoindex.functions.SplitRecursively(custom_languages=CUSTOM_LANGUAGES),
                    language=file["language"],
                    chunk_size=file["chunking_params"]["chunk_size"],
                    min_chunk_size=file["chunking_params"]["min_chunk_size"],
                    chunk_overlap=file["chunking_params"]["chunk_overlap"],
                )
                # Ensure unique locations for default chunking
                file["chunks"] = raw_chunks.transform(ensure_unique_chunk_locations)
                # Set chunking method for this file
                file["chunking_method_used"] = file["content"].transform(get_cocoindex_split_recursively_chunking_method)
            else:
                LOGGER.info("Using AST chunking extension")
                if ASTChunkOperation is not None:
                    raw_chunks = file["content"].transform(
                        ASTChunkOperation,
                        language=file["language"],
                        max_chunk_size=file["chunking_params"]["chunk_size"],
                        chunk_overlap=file["chunking_params"]["chunk_overlap"]
                    )
                    # Set chunking method for AST chunking
                    file["chunking_method_used"] = file["content"].transform(get_ast_tree_sitter_chunking_method)
                else:
                    # Fallback to basic chunking if AST operation is not available
                    # Skip transformation when AST chunking not available
                    raw_chunks = cast(Any, file["content"])
                    # Set chunking method for AST fallback
                    file["chunking_method_used"] = file["content"].transform(get_ast_fallback_chunking_method)
                # Ensure unique locations for AST chunking (safety measure)
                file["chunks"] = raw_chunks.transform(ensure_unique_chunk_locations)

            # Choose embedding method based on configuration
            use_smart_embedding = _global_flow_config.get('use_smart_embedding', False)
            LOGGER.debug(
                f"Embedding config: use_smart_embedding={use_smart_embedding}, SMART_EMBEDDING_AVAILABLE={SMART_EMBEDDING_AVAILABLE}")

            # Add model group information for smart embedding
            if use_smart_embedding and SMART_EMBEDDING_AVAILABLE:
                with file["chunks"].row() as chunk:
                    chunk["model_group"] = file["language"].transform(get_embedding_model_group)

            with file["chunks"].row() as chunk:
                # Smart embedding with language-aware model selection
                if use_smart_embedding and SMART_EMBEDDING_AVAILABLE:
                    model_group: Any = chunk["model_group"]
                    if model_group == "graphcodebert":
                        LOGGER.info(f"Using GraphCodeBERT for {file['language']}")
                        chunk["embedding"] = chunk["content"].call(graphcodebert_embedding)
                    elif model_group == "unixcoder":
                        LOGGER.info(f"Using UniXcode for {file['language']}")
                        chunk["embedding"] = chunk["content"].call(unixcoder_embedding)
                    else:  # fallback
                        LOGGER.info(f"Using fallback model for {file['language']}")
                        chunk["embedding"] = chunk["content"].call(fallback_embedding)
                else:
                    LOGGER.info("Using default embedding")
                    chunk["embedding"] = chunk["content"].call(code_to_embedding)

                # Extract metadata using appropriate method based on configuration
                use_default_language_handler = _global_flow_config.get('use_default_language_handler', False)

                if use_default_language_handler:
                    LOGGER.info("Using default language handler (--default-language-handler flag set)")
                    # Use transform function to create default metadata properly
                    chunk["extracted_metadata"] = chunk["content"].transform(create_default_metadata)
                else:
                    LOGGER.info("Using custom language handler extension")
                    chunk["extracted_metadata"] = chunk["content"].transform(
                        extract_code_metadata,
                        language=file["language"],
                        filename=file["filename"]
                    )

                # Promote all metadata fields from JSON to top-level fields using individual extractors
                # (We need individual extractors for CocoIndex to properly type and transform each field)
                chunk["functions"] = chunk["extracted_metadata"].transform(extract_functions_field)
                chunk["classes"] = chunk["extracted_metadata"].transform(extract_classes_field)
                chunk["imports"] = chunk["extracted_metadata"].transform(extract_imports_field)
                chunk["complexity_score"] = chunk["extracted_metadata"].transform(extract_complexity_score_field)
                chunk["has_type_hints"] = chunk["extracted_metadata"].transform(extract_has_type_hints_field)
                chunk["has_async"] = chunk["extracted_metadata"].transform(extract_has_async_field)
                chunk["has_classes"] = chunk["extracted_metadata"].transform(extract_has_classes_field)
                chunk["docstring"] = chunk["extracted_metadata"].transform(extract_docstring_field)
                
                # Additional promoted metadata fields
                chunk["analysis_method"] = chunk["extracted_metadata"].transform(extract_analysis_method_field)
                chunk["chunking_method"] = chunk["extracted_metadata"].transform(get_chunking_method_from_metadata)
                
                chunk["tree_sitter_chunking_error"] = chunk["extracted_metadata"].transform(extract_tree_sitter_chunking_error_field)
                chunk["tree_sitter_analyze_error"] = chunk["extracted_metadata"].transform(extract_tree_sitter_analyze_error_field)
                chunk["decorators_used"] = chunk["extracted_metadata"].transform(extract_decorators_used_field)
                chunk["dunder_methods"] = chunk["extracted_metadata"].transform(extract_dunder_methods_field)
                chunk["success"] = chunk["extracted_metadata"].transform(extract_success_field)
                chunk["parse_errors"] = chunk["extracted_metadata"].transform(extract_parse_errors_field)
                chunk["char_count"] = chunk["extracted_metadata"].transform(extract_char_count_field)
                
                # Language-specific fields (will be empty for non-matching languages)
                chunk["nodes_with_errors"] = chunk["extracted_metadata"].transform(extract_nodes_with_errors_field)  # Haskell
                chunk["data_types"] = chunk["extracted_metadata"].transform(extract_data_types_field)  # Haskell
                chunk["instances"] = chunk["extracted_metadata"].transform(extract_instances_field)  # Haskell
                chunk["type_classes"] = chunk["extracted_metadata"].transform(extract_type_classes_field)  # Haskell
                chunk["modules"] = chunk["extracted_metadata"].transform(extract_modules_field)  # Haskell, Rust
                chunk["has_module"] = chunk["extracted_metadata"].transform(extract_has_module_field)  # Haskell
                chunk["function_details"] = chunk["extracted_metadata"].transform(extract_function_details_field)  # Haskell
                chunk["data_type_details"] = chunk["extracted_metadata"].transform(extract_data_type_details_field)  # Haskell
                chunk["structs"] = chunk["extracted_metadata"].transform(extract_structs_field)  # Rust
                chunk["traits"] = chunk["extracted_metadata"].transform(extract_traits_field)  # Rust
                chunk["impls"] = chunk["extracted_metadata"].transform(extract_impls_field)  # Rust
                chunk["exports"] = chunk["extracted_metadata"].transform(extract_exports_field)  # TypeScript/JavaScript
                chunk["types"] = chunk["extracted_metadata"].transform(extract_types_field)  # TypeScript
                chunk["enums"] = chunk["extracted_metadata"].transform(extract_enums_field)  # TypeScript
                chunk["namespaces"] = chunk["extracted_metadata"].transform(extract_namespaces_field)  # TypeScript/JavaScript/C++
                
                # Promoted metadata fields are now handled automatically by generalized promotion in main_mcp_server.py
                # No need for chunk-level assignments - fields from metadata_json get promoted to top-level automatically

                ###################################################
                # Important:
                # This has to be updated for every new column.
                # An attempt to automate this from CONST_FIELD_MAPPINGS has failed.
                # 
                # TODO:
                # Might be a limitation of cocoindex, we should recheck this.
                ###################################################
                code_embeddings.collect(
                    filename=file["filename"],
                    language=file["language"],
                    location=chunk["location"],
                    code=chunk["content"].transform(convert_dataslice_to_string),
                    embedding=chunk["embedding"],
                    start=chunk["start"],
                    end=chunk["end"],
                    source_name=source_name,  # Add source name for identification
                    metadata_json=chunk["extracted_metadata"],  # Store full JSON
                    # Individual metadata fields (properly extracted from JSON)
                    functions=chunk["functions"],
                    classes=chunk["classes"],
                    imports=chunk["imports"],
                    complexity_score=chunk["complexity_score"],
                    has_type_hints=chunk["has_type_hints"],
                    has_async=chunk["has_async"],
                    has_classes=chunk["has_classes"],
                    # Promoted metadata fields using static extractors
                    analysis_method=chunk["analysis_method"],
                    chunking_method=chunk["chunking_method"],
                    tree_sitter_chunking_error=chunk["tree_sitter_chunking_error"],
                    tree_sitter_analyze_error=chunk["tree_sitter_analyze_error"],
                    decorators_used=chunk["decorators_used"],
                    dunder_methods=chunk["dunder_methods"],
                    success=chunk["success"],
                    parse_errors=chunk["parse_errors"],
                    char_count=chunk["char_count"],
                    # Language-specific fields
                    nodes_with_errors=chunk["nodes_with_errors"],
                    data_types=chunk["data_types"],
                    instances=chunk["instances"],
                    type_classes=chunk["type_classes"],
                    modules=chunk["modules"],
                    has_module=chunk["has_module"],
                    function_details=chunk["function_details"],
                    data_type_details=chunk["data_type_details"],
                    structs=chunk["structs"],
                    traits=chunk["traits"],
                    impls=chunk["impls"],
                    exports=chunk["exports"],
                    types=chunk["types"],
                    enums=chunk["enums"],
                    namespaces=chunk["namespaces"],
                    docstring=chunk["docstring"],
                )

    code_embeddings.export(
        "code_embeddings",
        # very important settings here
        # table_name="name" to customize the DB table to use
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename", "location", "source_name"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
            )
        ],
    )


def scale_chunking_params(chunk_factor_percent: int) -> None:
    """Scale all chunking parameters by the given percentage factor."""
    global EFFECTIVE_CHUNKING_PARAMS
    
    if chunk_factor_percent == 100:
        # No scaling needed, use original parameters
        EFFECTIVE_CHUNKING_PARAMS = copy.deepcopy(CHUNKING_PARAMS)
        return
    
    # Create scaled versions of all chunking parameters based on original values
    scaled_params = {}
    for language, params in CHUNKING_PARAMS.items():
        scaled_params[language] = ChunkingParams(
            chunk_size=params.chunk_size * chunk_factor_percent // 100,
            min_chunk_size=params.min_chunk_size * chunk_factor_percent // 100,
            chunk_overlap=params.chunk_overlap * chunk_factor_percent // 100,
            max_chunk_size=max(params.max_chunk_size * chunk_factor_percent // 100, params.chunk_size * 2) if params.max_chunk_size > 0 else params.chunk_size * 2
        )
    
    # Update the global EFFECTIVE_CHUNKING_PARAMS
    EFFECTIVE_CHUNKING_PARAMS = scaled_params
    
    LOGGER.info(f"Scaled chunking parameters by {chunk_factor_percent}%")


def update_flow_config(paths: Union[List[str], None] = None, enable_polling: bool = False, poll_interval: int = 30,
                       use_default_embedding: bool = False, use_default_chunking: bool = False,
                       use_default_language_handler: bool = False, chunk_factor_percent: int = 100) -> None:
    """Update the global flow configuration."""
    global _global_flow_config
    
    # Scale chunking parameters if needed
    scale_chunking_params(chunk_factor_percent)
    
    _global_flow_config.update({
        'paths': paths or ["cocoindex"],
        'enable_polling': enable_polling,
        'poll_interval': poll_interval,
        'use_default_embedding': use_default_embedding,
        'use_default_chunking': use_default_chunking,
        'use_default_language_handler': use_default_language_handler
    })


def run_flow_update(live_update: bool = False, poll_interval: int = 30) -> None:
    """Run the flow update (one-time or live)."""
    if live_update:
        LOGGER.info("🔄 Starting live update mode...")
        if poll_interval > 0:
            LOGGER.info(f"📊 File polling enabled: {poll_interval} seconds")
        else:
            LOGGER.info("📊 Event-based monitoring (no polling)")

        flow = code_embedding_flow

        # Setup the flow first
        flow.setup()

        # Initial update
        LOGGER.info("🚀 Initial index build...")
        stats = flow.update()
        LOGGER.info("Initial index built: %s", stats)

        # Start live updater
        LOGGER.info("👁️  Starting live file monitoring...")
        live_options = cocoindex.FlowLiveUpdaterOptions(
            live_mode=True,
            print_stats=True
        )

        with cocoindex.FlowLiveUpdater(flow, live_options) as updater:
            LOGGER.info("✅ Live update mode active. Press Ctrl+C to stop.")
            try:
                updater.wait()
            except KeyboardInterrupt:
                LOGGER.info("\n⏹️  Stopping live update mode...")

    else:
        # Regular one-time update mode
        stats = code_embedding_flow.update()
        LOGGER.info("Updated index: %s", stats)
