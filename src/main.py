from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from typing import Any
import cocoindex
import os
import argparse
import datetime
from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass
import haskell_tree_sitter


@dataclass
class ChunkingParams:
    """Parameters for chunking code."""
    chunk_size: int
    min_chunk_size: int
    chunk_overlap: int


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
# CUSTOM_LANGUAGES = []
CUSTOM_LANGUAGES = [
    # Build systems
    cocoindex.functions.CustomLanguageSpec(
        language_name="Makefile",
        aliases=[".makefile"],
        separators_regex=[r"\n\n+", r"\n\w+:", r"\n"]  # Remove (?=...)
    ),
    cocoindex.functions.CustomLanguageSpec(
        language_name="CMake",
        aliases=[".cmake"],
        separators_regex=[r"\n\n+", r"\n\w+\(", r"\n"]  # Remove (?=...)
    ),
    cocoindex.functions.CustomLanguageSpec(
        language_name="Dockerfile",
        aliases=[".dockerfile"],
        separators_regex=[r"\n\n+",
        r"\n(FROM|RUN|COPY|ADD|EXPOSE|ENV|CMD|ENTRYPOINT)", r"\n"]  # Remove (?=...)
    ),
    cocoindex.functions.CustomLanguageSpec(
        language_name="Gradle",
        aliases=[".gradle"],
        separators_regex=[r"\n\n+", r"\n\w+\s*\{", r"\n"]  # Remove (?=...)
    ),
    cocoindex.functions.CustomLanguageSpec(
        language_name="Maven",
        aliases=[".maven"],
        separators_regex=[r"</\w+>\s*<\w+>", r"\n\n+", r"\n"]  # This one was OK
    ),
    # Shell scripts
    cocoindex.functions.CustomLanguageSpec(
        language_name="Shell",
        aliases=[".sh", ".bash"],
        separators_regex=[r"\n\n+", r"\nfunction\s+\w+", r"\n\w+\(\)", r"\n"]  # Remove (?=...)
    ),
    # Configuration files
    cocoindex.functions.CustomLanguageSpec(
        language_name="Config",
        aliases=[".ini", ".cfg", ".conf"],
        separators_regex=[r"\n\n+", r"\n\[.*\]", r"\n"]  # Remove (?=...)
    ),
    # Haskell - using our custom tree-sitter parser
    cocoindex.functions.CustomLanguageSpec(
        language_name="Haskell",
        aliases=[".hs", ".lhs"],
        separators_regex=haskell_tree_sitter.get_haskell_separators()
    ),
    # Kotlin
    cocoindex.functions.CustomLanguageSpec(
        language_name="Kotlin",
        aliases=["kt", ".kt", "kts", ".kts"],
        separators_regex=[r"\n\n+", r"\nfun\s+", r"\nclass\s+", r"\nobject\s+",
        r"\ninterface\s+", r"\n"]  # Remove (?=...)
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


@cocoindex.transform_flow()
def code_to_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[NDArray[np.float32]]:
    """
    Embed the text using a SentenceTransformer model.
    """
    # You can also switch to Voyage embedding model:
    #    return text.transform(
    #        cocoindex.functions.EmbedText(
    #            api_type=cocoindex.LlmApiType.VOYAGE,
    #            model="voyage-code-3",
    #        )
    #    )
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    )


# Global configuration for flow parameters
_global_flow_config = {
    'paths': ["cocoindex"],
    'enable_polling': False,
    'poll_interval': 30
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
        print(f"Adding source: {path} as '{source_name}'")
        
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
        
        # Add polling configuration if enabled
        if enable_polling:
            source_config["recent_changes_poll_interval"] = datetime.timedelta(seconds=poll_interval)
            source_config["refresh_interval"] = datetime.timedelta(minutes=max(1, poll_interval // 60))
            print(f"  Polling enabled: {poll_interval}s interval")
        
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
            
            # Use language-specific chunking parameters with improved tree-sitter support
            file["chunks"] = file["content"].transform(
                cocoindex.functions.SplitRecursively(
                    custom_languages=CUSTOM_LANGUAGES
                ),
                language=file["language"],
                chunk_size=file["chunking_params"]["chunk_size"],
                min_chunk_size=file["chunking_params"]["min_chunk_size"],
                chunk_overlap=file["chunking_params"]["chunk_overlap"],
                )
            
            with file["chunks"].row() as chunk:
                chunk["embedding"] = chunk["text"].call(code_to_embedding)
                code_embeddings.collect(
                    filename=file["filename"],
                    language=file["language"],
                    location=chunk["location"],
                    code=chunk["text"],
                    embedding=chunk["embedding"],
                    start=chunk["start"],
                    end=chunk["end"],
                    source_name=source_name,  # Add source name for identification
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


def search(pool: ConnectionPool, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    # Get the table name, for the export target in the code_embedding_flow above.
    table_name = cocoindex.utils.get_target_default_name(
        code_embedding_flow, "code_embeddings"
    )
    # Evaluate the transform flow defined above with the input query, to get the embedding.
    query_vector = code_to_embedding.eval(query)
    # Run the query and get the results.
    with pool.connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT filename, language, code, embedding <=> %s AS distance, start, "end", source_name
                FROM {table_name} ORDER BY distance LIMIT %s
            """,
                (query_vector, top_k),
            )
            return [
                {
                    "filename": row[0],
                    "language": row[1],
                    "code": row[2],
                    "score": 1.0 - row[3],
                    "start": row[4],
                    "end": row[5],
                    "source": row[6] if len(row) > 6 else "unknown",
                }
                for row in cur.fetchall()
            ]


def _main(paths: list[str] = None, live_update: bool = False, poll_interval: int = 30) -> None:
    # Configure the global flow parameters
    global _global_flow_config
    _global_flow_config.update({
        'paths': paths or ["cocoindex"],
        'enable_polling': poll_interval > 0,  # Enable polling if interval is specified
        'poll_interval': poll_interval
    })
    
    if live_update:
        print("🔄 Starting live update mode...")
        if poll_interval > 0:
            print(f"📊 File polling enabled: {poll_interval} seconds")
        else:
            print("📊 Event-based monitoring (no polling)")
        
        flow = code_embedding_flow
        
        # Setup the flow first
        flow.setup()
        
        # Initial update
        print("🚀 Initial index build...")
        stats = flow.update()
        print("Initial index built:", stats)
        
        # Start live updater
        print("👁️  Starting live file monitoring...")
        live_options = cocoindex.FlowLiveUpdaterOptions(
            live_mode=True,
            print_stats=True
        )
        
        with cocoindex.FlowLiveUpdater(flow, live_options) as updater:
            print("✅ Live update mode active. Press Ctrl+C to stop.")
            try:
                updater.wait()
            except KeyboardInterrupt:
                print("\n⏹️  Stopping live update mode...")
                
    else:
        # Regular one-time update mode
        stats = code_embedding_flow.update()
        print("Updated index: ", stats)

        # Initialize the database connection pool.
        pool = ConnectionPool(os.getenv("COCOINDEX_DATABASE_URL"))
        print("\n🔍 Interactive search mode. Type queries to search the code index.")
        print("Press Enter with empty query to quit.\n")
        
        # Run queries in a loop to demonstrate the query capabilities.
        while True:
            try:
                query = input("Search query: ")
                if query == "":
                    break
                # Run the query function with the database connection pool and the query.
                results = search(pool, query)
                print(f"\n📊 Found {len(results)} results:")
                for result in results:
                    source_info = f" [{result['source']}]" if result.get('source') and result['source'] != 'files' else ""
                    print(
                        f"[{result['score']:.3f}] {result['filename']}{source_info} ({result['language']}) (L{result['start']['line']}-L{result['end']['line']})"
                    )
                    print(f"    {result['code']}")
                    print("---")
                print()
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Code embedding pipeline with Haskell tree-sitter support and live updates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                           # Use default path (cocoindex)
  python src/main.py /path/to/code             # Index single directory
  python src/main.py /path/to/code1 /path/to/code2  # Index multiple directories
  python src/main.py --paths /path/to/code     # Explicit paths argument
  
  # Live update mode
  python src/main.py --live                    # Live updates with event monitoring
  python src/main.py --live --poll 10         # Live updates with 10s polling
  python src/main.py --live --poll 60 /path/to/code  # Custom path with polling
        """
    )
    
    parser.add_argument(
        "paths", 
        nargs="*", 
        help="Code directory paths to index (default: cocoindex)"
    )
    
    parser.add_argument(
        "--paths",
        dest="explicit_paths",
        nargs="+",
        help="Alternative way to specify paths"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live update mode with continuous monitoring"
    )
    
    parser.add_argument(
        "--poll",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Enable file polling with specified interval in seconds (default: event-based monitoring)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    cocoindex.init()
    
    args = parse_args()
    
    # Determine paths to use
    paths = None
    if args.explicit_paths:
        paths = args.explicit_paths
    elif args.paths:
        paths = args.paths
    
    # Display configuration
    if paths:
        if len(paths) == 1:
            print(f"📁 Indexing path: {paths[0]}")
        else:
            print(f"📁 Indexing {len(paths)} paths:")
            for i, path in enumerate(paths, 1):
                print(f"  {i}. {path}")
    else:
        print("📁 Using default path: cocoindex")
    
    # Display mode
    if args.live:
        print("🔴 Mode: Live updates")
        if args.poll > 0:
            print(f"⏰ Polling: {args.poll} seconds")
        else:
            print("⚡ Monitoring: Event-based")
    else:
        print("🟢 Mode: One-time indexing")
    
    print()  # Empty line for readability
    
    _main(paths, live_update=args.live, poll_interval=args.poll)
