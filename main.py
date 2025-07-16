from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from typing import Any
import cocoindex
import os
from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass


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
    # Haskell
    cocoindex.functions.CustomLanguageSpec(
        language_name="Haskell",
        aliases=[".hs", ".lhs"],
        separators_regex=[r"\n\n+", r"\n\w+\s*::", r"\nimport\s+",
        r"\nmodule\s+", r"\n"]  # Remove (?=...)
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


@cocoindex.flow_def(name="CodeEmbedding")
def code_embedding_flow(
    flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
) -> None:
    """
    Define an improved flow that embeds files with syntax-aware chunking.
    """
    data_scope["files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            # path="../..",
            path="cocoindex",
            included_patterns=[
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
            excluded_patterns=[
                "**/.*", "target", "**/node_modules", "**/build", "**/dist",
                "**/__pycache__", "**/bin", "**/obj", "**/out", "**/venv",
                "**/env", "**/.gradle", "**/.idea", "**/.vscode",
                "**/target/debug", "**/target/release", "**/*.class",
                "**/*.jar", "**/*.war", "**/*.ear", "**/*.pyc", "**/*.pyo",
            ],
        )
    )
    code_embeddings = data_scope.add_collector()

    with data_scope["files"].row() as file:
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
                SELECT filename, language, code, embedding <=> %s AS distance, start, "end"
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
                }
                for row in cur.fetchall()
            ]


def _main() -> None:
    # Make sure the flow is built and up-to-date.
    stats = code_embedding_flow.update()
    print("Updated index: ", stats)

    # Initialize the database connection pool.
    pool = ConnectionPool(os.getenv("COCOINDEX_DATABASE_URL"))
    # Run queries in a loop to demonstrate the query capabilities.
    while True:
        query = input("Enter search query (or Enter to quit): ")
        if query == "":
            break
        # Run the query function with the database connection pool and the query.
        results = search(pool, query)
        print("\nSearch results:")
        for result in results:
            print(
                f"[{result['score']:.3f}] {result['filename']} ({result['language']}) (L{result['start']['line']}-L{result['end']['line']})"
            )
            print(f"    {result['code']}")
            print("---")
        print()


if __name__ == "__main__":
    load_dotenv()
    cocoindex.init()
    _main()
