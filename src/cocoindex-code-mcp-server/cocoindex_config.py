#!/usr/bin/env python3

"""
CocoIndex configuration and flow definitions.
"""

import os
import datetime
from dataclasses import dataclass
from typing import List, Dict, Literal
from numpy.typing import NDArray
import numpy as np
import cocoindex
from cocoindex.typing import Vector
from lang.haskell.haskell_support import get_haskell_language_spec
from lang.python.python_code_analyzer import analyze_python_code
from __init__ import LOGGER

# Import our custom extensions
try:
    from smart_code_embedding import create_smart_code_embedding
    SMART_EMBEDDING_AVAILABLE = True
    LOGGER.info("Smart code embedding extension loaded")
except ImportError as e:
    SMART_EMBEDDING_AVAILABLE = False
    LOGGER.warning(f"Smart code embedding not available: {e}")

try:
    from ast_chunking import create_ast_chunking_operation
    AST_CHUNKING_AVAILABLE = True
    LOGGER.info("AST chunking extension loaded")
except ImportError as e:
    AST_CHUNKING_AVAILABLE = False
    LOGGER.warning(f"AST chunking not available: {e}")

try:
    from language_handlers.python_handler import PythonNodeHandler
    from language_handlers import get_handler_for_language
    PYTHON_HANDLER_AVAILABLE = True
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
    ".yaml": "YAML", ".yml": "YAML",
    ".hs": "Haskell", ".lhs": "Haskell",
}

# Language-specific chunking parameters
CHUNKING_PARAMS = {
    # Larger chunks for documentation and config files
    "Markdown": ChunkingParams(chunk_size=2000, min_chunk_size=500, chunk_overlap=200),
    "YAML": ChunkingParams(chunk_size=800, min_chunk_size=200, chunk_overlap=100),
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
    
    if lang == "Python" and PYTHON_HANDLER_AVAILABLE and not use_default_handler:
        # Use our advanced Python handler extension
        try:
            handler = PythonNodeHandler()
            # Note: This is a simplified integration - the handler expects AST nodes
            # For now, fall back to basic analysis but log that the handler is available
            LOGGER.info("Python handler available but needs AST integration")
            metadata = analyze_python_code(code_text, file_path)
        except Exception as e:
            LOGGER.warning(f"Python handler failed, falling back to basic analysis: {e}")
            metadata = analyze_python_code(code_text, file_path)
    elif lang == "Python":
        metadata = analyze_python_code(code_text, file_path)
    else:
        # For non-Python languages, return basic metadata
        metadata = {
            "language": lang,
            "analysis_method": "basic",
            "functions": [],
            "classes": [],
            "imports": [],
            "complexity_score": 0,
            "has_type_hints": False,
            "has_async": False,
            "has_classes": False,
        }
    
    # Return just the JSON string for now
    import json
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


@cocoindex.transform_flow()
def code_to_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[NDArray[np.float32]]:
    """
    Default embedding using a SentenceTransformer model with caching.
    """
    @cocoindex.op.function()
    def cached_embed_text(text: str) -> Vector[np.float32, Literal[384]]:
        """Embed text using cached SentenceTransformer model."""
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model = get_cached_sentence_transformer(model_name)
        embedding = model.encode(text)
        return embedding.astype(np.float32)
    
    return text.transform(cached_embed_text)


@cocoindex.transform_flow()
def smart_code_to_embedding(
    text: cocoindex.DataSlice[str],
    language: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[NDArray[np.float32]]:
    """
    Smart embedding that selects model based on language.
    """
    if not SMART_EMBEDDING_AVAILABLE:
        LOGGER.warning("Smart embedding not available, falling back to default")
        return code_to_embedding(text)
    
    @cocoindex.op.function()
    def embed_with_language_selection(text: str, language: str) -> Vector[np.float32, Literal[384]]:
        """Embed text using language-specific embedding model selection."""
        # Temporarily force use of basic model to avoid HuggingFace rate limits
        # TODO: Re-enable smart model selection once rate limiting is resolved
        LOGGER.debug(f"Using basic embedding model for language: {language} (smart embedding disabled)")
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model = get_cached_sentence_transformer(model_name)
        embedding = model.encode(text)
        return embedding.astype(np.float32)
    
    return text.transform(embed_with_language_selection, language=language)


# Global configuration for flow parameters  
_global_flow_config = {
    'paths': ["."],  # Use current directory for testing
    'enable_polling': False,
    'poll_interval': 30
}

# Try using functools.lru_cache for persistent model caching
from functools import lru_cache

@lru_cache(maxsize=3)
def get_cached_sentence_transformer(model_name: str):
    """Get cached SentenceTransformer model using lru_cache."""
    LOGGER.info(f"🔄 Loading model {model_name} via lru_cache")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    LOGGER.info(f"✅ Model {model_name} loaded and will be cached by lru_cache")
    return model


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
                "*.hs", "*.lhs", "*.cabal", "*.yaml", "*.yml", "stack.yaml",
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
                "Dockerfile", "*.dockerfile", "docker-compose.yml", "docker-compose.yaml",
                # CI/CD
                "*.gitlab-ci.yml", ".github/workflows/*.yml", ".github/workflows/*.yaml",
                "Jenkinsfile", "*.jenkinsfile", "azure-pipelines.yml",
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
            
            # Temporarily use only default chunking to avoid factory name collisions
            LOGGER.info("Using default recursive splitting (AST chunking temporarily disabled)")
                
            file["chunks"] = file["content"].transform(
                cocoindex.functions.SplitRecursively(),
                language=file["language"],
                chunk_size=file["chunking_params"]["chunk_size"],
                min_chunk_size=file["chunking_params"]["min_chunk_size"],
                chunk_overlap=file["chunking_params"]["chunk_overlap"],
            )
            
            with file["chunks"].row() as chunk:
                # Choose embedding method based on configuration
                use_default_embedding = _global_flow_config.get('use_default_embedding', False)
                
                if use_default_embedding or not SMART_EMBEDDING_AVAILABLE:
                    if not use_default_embedding and not SMART_EMBEDDING_AVAILABLE:
                        LOGGER.info("Smart embedding not available, using default embedding")
                    else:
                        LOGGER.info("Using default embedding (--default-embedding flag set)")
                    chunk["embedding"] = chunk["text"].call(code_to_embedding)
                else:
                    LOGGER.info("Using smart code embedding extension")
                    chunk["embedding"] = chunk["text"].call(smart_code_to_embedding, language=file["language"])
                
                # Extract metadata using appropriate method based on configuration
                use_default_language_handler = _global_flow_config.get('use_default_language_handler', False)
                
                if use_default_language_handler:
                    LOGGER.info("Using default language handler (--default-language-handler flag set)")
                    # Use simple default metadata (no custom processing)
                    import json
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
                
                # Parse metadata JSON to extract fields for collection
                import json
                try:
                    metadata_dict = json.loads(chunk["metadata"])
                except:
                    metadata_dict = {}
                
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
                    # Include parsed metadata fields
                    functions=str(metadata_dict.get("functions", [])),  # Convert to string for storage
                    classes=str(metadata_dict.get("classes", [])),
                    imports=str(metadata_dict.get("imports", [])),
                    complexity_score=metadata_dict.get("complexity_score", 0),
                    has_type_hints=metadata_dict.get("has_type_hints", False),
                    has_async=metadata_dict.get("has_async", False),
                    has_classes=metadata_dict.get("has_classes", False),
                )

    code_embeddings.export(
        "code_embeddings",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename", "location"],
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
