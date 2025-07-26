#!/usr/bin/env python3

"""
CocoIndex configuration and flow definitions.
"""

# Temporarily disabled due to cocoindex compatibility
# from __future__ import annotations  
import os
import datetime
import json
import torch
import traceback
from dataclasses import dataclass

from typing import List, Dict, Literal
from numpy.typing import NDArray
import numpy as np
import cocoindex
from cocoindex.typing import Vector
from lang.haskell.haskell_ast_chunker import get_haskell_language_spec
from lang.python.python_code_analyzer import analyze_python_code
from cocoindex_code_mcp_server import LOGGER
# from sentence_transformers import SentenceTransformer  # Use cocoindex.functions.SentenceTransformerEmbed instead
from ast_chunking import Chunk, ASTChunkOperation
from smart_code_embedding import LanguageModelSelector

# Models will be instantiated directly (HuggingFace handles caching)

DEFAULT_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STACKTRACE = False

# Import our custom extensions
try:
    from smart_code_embedding import LanguageModelSelector
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


# Tree-sitter supported languages mapping (from CocoIndex implementation)
TREE_SITTER_LANGUAGE_MAP = {
    # Core languages with tree-sitter support
    ".c": "C",
    ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".h": "C++", ".hpp": "C++",
    ".cs": "C#",
    ".css": "CSS", ".scss": "CSS",
    ".f": "Fortran", ".f90": "Fortran", ".f95": "Fortran", ".f03": "Fortran",
    ".go": "Go",
    ".html": "HTML", ".htm": "HTML",
    ".java": "Java",
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".json": "JSON",
    ".md": "Markdown", ".mdx": "Markdown",
    ".pas": "Pascal", ".dpr": "Pascal",
    ".php": "PHP",
    ".py": "Python", ".pyi": "Python",
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
    # ".yaml": "YAML", ".yml": "YAML",
    ".hs": "Haskell", ".lhs": "Haskell",
}

# Language-specific chunking parameters
CHUNKING_PARAMS = {
    # Larger chunks for documentation and config files
    "Markdown": ChunkingParams(chunk_size=2000, min_chunk_size=500, chunk_overlap=200),
    #"YAML": ChunkingParams(chunk_size=800, min_chunk_size=200, chunk_overlap=100),
    "JSON": ChunkingParams(chunk_size=1500, min_chunk_size=300, chunk_overlap=200),
    "XML": ChunkingParams(chunk_size=1500, min_chunk_size=300, chunk_overlap=200),
    "TOML": ChunkingParams(chunk_size=800, min_chunk_size=200, chunk_overlap=100),
    
    # Smaller chunks for dense code
    "C": ChunkingParams(chunk_size=800, min_chunk_size=200, chunk_overlap=150),
    "C++": ChunkingParams(chunk_size=800, min_chunk_size=200, chunk_overlap=150),
    "Rust": ChunkingParams(chunk_size=1000, min_chunk_size=250, chunk_overlap=200),
    "Go": ChunkingParams(chunk_size=1000, min_chunk_size=250, chunk_overlap=200),
    "Java": ChunkingParams(chunk_size=1200, min_chunk_size=300, chunk_overlap=250),
    "C#": ChunkingParams(chunk_size=1200, min_chunk_size=300, chunk_overlap=250),
    "Scala": ChunkingParams(chunk_size=1000, min_chunk_size=250, chunk_overlap=200),
    
    # Medium chunks for scripting languages
    "Python": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=250),
    "JavaScript": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=250),
    "TypeScript": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=250),
    "TSX": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=250),
    "Ruby": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=250),
    "PHP": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=250),
    
    # Web and styling
    "HTML": ChunkingParams(chunk_size=1500, min_chunk_size=400, chunk_overlap=200),
    "CSS": ChunkingParams(chunk_size=1000, min_chunk_size=250, chunk_overlap=150),
    
    # Data and scientific
    "SQL": ChunkingParams(chunk_size=1200, min_chunk_size=300, chunk_overlap=200),
    "R": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=200),
    "Fortran": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=200),
    
    # Others
    "Pascal": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=200),
    "Swift": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=200),
    "Haskell": ChunkingParams(chunk_size=1200, min_chunk_size=300, chunk_overlap=200),
    
    # Default fallback
    "_DEFAULT": ChunkingParams(chunk_size=1000, min_chunk_size=300, chunk_overlap=200),
}


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
    basename = os.path.basename(filename)
    
    # Handle special files without extensions
    if basename.lower() in ["makefile", "dockerfile", "jenkinsfile"]:
        return basename.lower()
    
    # Handle special patterns
    if basename.lower().startswith("cmakelists"):
        return "cmake"
    if basename.lower().startswith("build.gradle"):
        return "gradle"
    if basename.lower().startswith("pom.xml"):
        return "maven"
    if "docker-compose" in basename.lower():
        return "dockerfile"
    if basename.startswith("go."):
        return "go"
    if basename.lower() in ["stack.yaml", "cabal.project"]:
        return "haskell"
    
    # Get extension
    ext = os.path.splitext(filename)[1].lower()
    
    # Map to tree-sitter language
    return TREE_SITTER_LANGUAGE_MAP.get(ext, ext)


@cocoindex.op.function()
def get_chunking_params(language: str) -> ChunkingParams:
    """Get language-specific chunking parameters."""
    return CHUNKING_PARAMS.get(language, CHUNKING_PARAMS["_DEFAULT"])


@cocoindex.op.function()
def extract_code_metadata(text: str, language: str, filename: str = "") -> str:
    """Extract rich metadata from code chunks based on language and return as JSON string."""
    # Check if we should use default language handler
    use_default_handler = _global_flow_config.get('use_default_language_handler', False)
    
    try:
        if language == "Python" and PYTHON_HANDLER_AVAILABLE and not use_default_handler:
            # Use our advanced Python handler through the tree-sitter analyzer
            try:
                from lang.python.tree_sitter_python_analyzer import TreeSitterPythonAnalyzer
                LOGGER.debug("Using TreeSitterPythonAnalyzer with integrated PythonNodeHandler")
                analyzer = TreeSitterPythonAnalyzer(prefer_tree_sitter=True)
                metadata = analyzer.analyze_code(text, filename)
            except Exception as e:
                LOGGER.debug(f"TreeSitterPythonAnalyzer failed, falling back to basic analysis: {e}")
                metadata = analyze_python_code(text, filename)
        elif language == "Python":
            metadata = analyze_python_code(text, filename)
        else:
            # For non-Python languages, return basic metadata
            metadata = {
                "language": language,
                "analysis_method": "basic",
                "functions": [],
                "classes": [],
                "imports": [],
                "complexity_score": 0,
                "has_type_hints": False,
                "has_async": False,
                "has_classes": False,
                "decorators_used": [],
            }
        
        # Return just the JSON string for now
        result = {
            "functions": metadata.get("functions", []),
            "classes": metadata.get("classes", []),
            "imports": metadata.get("imports", []),
            "complexity_score": metadata.get("complexity_score", 0),
            "has_type_hints": metadata.get("has_type_hints", False),
            "has_async": metadata.get("has_async", False),
            "has_classes": metadata.get("has_classes", False),
            "decorators_used": metadata.get("decorators_used", []),
            "analysis_method": metadata.get("analysis_method", "basic"),
        }
        return json.dumps(result)
        
    except Exception as e:
        # Fallback to empty metadata if everything fails
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
        }
        return json.dumps(fallback_result)


@cocoindex.op.function()
def extract_functions_field(metadata_json: str) -> str:
    """Extract functions field from metadata JSON."""
    try:
        if not metadata_json or metadata_json.strip() == "":
            return "[]"
        metadata_dict = json.loads(metadata_json)
        functions = metadata_dict.get("functions", [])
        # Ensure it's a list and convert to string representation
        if isinstance(functions, list):
            return str(functions)
        else:
            return str([functions]) if functions else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for functions: {e}")
        return "[]"


@cocoindex.op.function()
def extract_classes_field(metadata_json: str) -> str:
    """Extract classes field from metadata JSON."""
    try:
        if not metadata_json or metadata_json.strip() == "":
            return "[]"
        metadata_dict = json.loads(metadata_json)
        classes = metadata_dict.get("classes", [])
        if isinstance(classes, list):
            return str(classes)
        else:
            return str([classes]) if classes else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for classes: {e}")
        return "[]"


@cocoindex.op.function()
def extract_imports_field(metadata_json: str) -> str:
    """Extract imports field from metadata JSON."""
    try:
        if not metadata_json or metadata_json.strip() == "":
            return "[]"
        metadata_dict = json.loads(metadata_json)
        imports = metadata_dict.get("imports", [])
        if isinstance(imports, list):
            return str(imports)
        else:
            return str([imports]) if imports else "[]"
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for imports: {e}")
        return "[]"


@cocoindex.op.function()
def extract_complexity_score_field(metadata_json: str) -> int:
    """Extract complexity_score field from metadata JSON."""
    try:
        if not metadata_json or metadata_json.strip() == "":
            return 0
        metadata_dict = json.loads(metadata_json)
        score = metadata_dict.get("complexity_score", 0)
        return int(score) if isinstance(score, (int, float, str)) and str(score).isdigit() else 0
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for complexity_score: {e}")
        return 0


@cocoindex.op.function()
def extract_has_type_hints_field(metadata_json: str) -> bool:
    """Extract has_type_hints field from metadata JSON."""
    try:
        if not metadata_json or metadata_json.strip() == "":
            return False
        metadata_dict = json.loads(metadata_json)
        return bool(metadata_dict.get("has_type_hints", False))
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for has_type_hints: {e}")
        return False


@cocoindex.op.function()
def extract_has_async_field(metadata_json: str) -> bool:
    """Extract has_async field from metadata JSON."""
    try:
        if not metadata_json or metadata_json.strip() == "":
            return False
        metadata_dict = json.loads(metadata_json)
        return bool(metadata_dict.get("has_async", False))
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for has_async: {e}")
        return False


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
            text = chunk.text
            start = chunk.start
            end = chunk.end
        elif isinstance(chunk, dict):
            # Dictionary format from SplitRecursively - convert to Chunk
            base_loc = chunk.get("location", f"chunk_{i}")
            text = chunk.get("text", "")
            start = chunk.get("start", 0)
            end = chunk.get("end", 0)
        else:
            # Fallback for unexpected types
            base_loc = f"chunk_{i}"
            text = str(chunk) if chunk else ""
            start = 0
            end = 0
        
        # Make location unique
        unique_loc = base_loc
        suffix = 0
        while unique_loc in seen_locations:
            suffix += 1
            unique_loc = f"{base_loc}#{suffix}"
        
        seen_locations.add(unique_loc)
        
        # Always create Chunk dataclass with unique location
        unique_chunk = Chunk(
            text=text,
            location=unique_loc,
            start=start,
            end=end
        )
        unique_chunks.append(unique_chunk)
    
    return unique_chunks


@cocoindex.op.function()
def extract_has_classes_field(metadata_json: str) -> bool:
    """Extract has_classes field from metadata JSON."""
    try:
        if not metadata_json or metadata_json.strip() == "":
            return False
        metadata_dict = json.loads(metadata_json)
        return bool(metadata_dict.get("has_classes", False))
    except Exception as e:
        LOGGER.debug(f"Failed to parse metadata JSON for has_classes: {e}")
        return False


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
    
    for i, path in enumerate(paths):
        source_name = f"files_{i}" if len(paths) > 1 else "files"
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
            else:
                LOGGER.info("Using AST chunking extension")
                raw_chunks = file["content"].transform(
                    ASTChunkOperation,
                    language=file["language"],
                    max_chunk_size=file["chunking_params"]["chunk_size"],
                    chunk_overlap=file["chunking_params"]["chunk_overlap"]
                )
                # Ensure unique locations for AST chunking (safety measure)
                file["chunks"] = raw_chunks.transform(ensure_unique_chunk_locations)
            
            # Choose embedding method based on configuration
            use_smart_embedding = _global_flow_config.get('use_smart_embedding', False)
            LOGGER.debug(f"Embedding config: use_smart_embedding={use_smart_embedding}, SMART_EMBEDDING_AVAILABLE={SMART_EMBEDDING_AVAILABLE}")
            
            # Add model group information for smart embedding
            if use_smart_embedding and SMART_EMBEDDING_AVAILABLE:
                with file["chunks"].row() as chunk:
                    chunk["model_group"] = file["language"].transform(get_embedding_model_group)
            
            with file["chunks"].row() as chunk:
                # Smart embedding with language-aware model selection
                if use_smart_embedding and SMART_EMBEDDING_AVAILABLE:
                    model_group = chunk["model_group"]
                    if model_group == "graphcodebert":
                        LOGGER.info(f"Using GraphCodeBERT for {file['language']}")
                        chunk["embedding"] = chunk["text"].call(graphcodebert_embedding)
                    elif model_group == "unixcoder":
                        LOGGER.info(f"Using UniXcode for {file['language']}")
                        chunk["embedding"] = chunk["text"].call(unixcoder_embedding)
                    else:  # fallback
                        LOGGER.info(f"Using fallback model for {file['language']}")
                        chunk["embedding"] = chunk["text"].call(fallback_embedding)
                else:
                    LOGGER.info("Using default embedding")
                    chunk["embedding"] = chunk["text"].call(code_to_embedding)
                
                # Extract metadata using appropriate method based on configuration
                use_default_language_handler = _global_flow_config.get('use_default_language_handler', False)
                
                if use_default_language_handler:
                    LOGGER.info("Using default language handler (--default-language-handler flag set)")
                    # Use simple default metadata (no custom processing)
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
                    chunk["metadata"] = json.dumps(default_metadata)
                else:
                    LOGGER.info("Using custom language handler extension")
                    chunk["metadata"] = chunk["text"].transform(
                        extract_code_metadata, 
                        language=file["language"], 
                        filename=file["filename"]
                    )
                
                # Extract individual metadata fields using CocoIndex transforms
                chunk["functions"] = chunk["metadata"].transform(extract_functions_field)
                chunk["classes"] = chunk["metadata"].transform(extract_classes_field)
                chunk["imports"] = chunk["metadata"].transform(extract_imports_field)
                chunk["complexity_score"] = chunk["metadata"].transform(extract_complexity_score_field)
                chunk["has_type_hints"] = chunk["metadata"].transform(extract_has_type_hints_field)
                chunk["has_async"] = chunk["metadata"].transform(extract_has_async_field)
                chunk["has_classes"] = chunk["metadata"].transform(extract_has_classes_field)
                
                code_embeddings.collect(
                    filename=file["filename"],
                    language=file["language"],
                    location=chunk["location"],
                    code=chunk["text"],
                    embedding=chunk["embedding"],
                    start=chunk["start"],
                    end=chunk["end"],
                    source_name=source_name,  # Add source name for identification
                    metadata_json=chunk["metadata"],  # Store full JSON
                    # Individual metadata fields (properly extracted from JSON)
                    functions=chunk["functions"],
                    classes=chunk["classes"],
                    imports=chunk["imports"],
                    complexity_score=chunk["complexity_score"],
                    has_type_hints=chunk["has_type_hints"],
                    has_async=chunk["has_async"],
                    has_classes=chunk["has_classes"],
                )

    code_embeddings.export(
        "code_embeddings",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename", "location", "source_name"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
            )
        ],
    )



def update_flow_config(paths: List[str] = None, enable_polling: bool = False, poll_interval: int = 30,
                      use_default_embedding: bool = False, use_default_chunking: bool = False, 
                      use_default_language_handler: bool = False):
    """Update the global flow configuration."""
    global _global_flow_config
    _global_flow_config.update({
        'paths': paths or ["cocoindex"],
        'enable_polling': enable_polling,
        'poll_interval': poll_interval,
        'use_default_embedding': use_default_embedding,
        'use_default_chunking': use_default_chunking,
        'use_default_language_handler': use_default_language_handler
    })


def run_flow_update(live_update: bool = False, poll_interval: int = 30):
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
