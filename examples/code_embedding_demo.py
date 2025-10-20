#!/usr/bin/env python3

"""
Demonstration of language-aware code embeddings in CocoIndex.

This example shows how to use the new CodeEmbedding and SmartCodeEmbedding
functions to automatically select the best embedding model based on programming language.
"""

import os
import tempfile
from pathlib import Path

# Note: This is a demonstration script showing the API
# The actual execution would require the CocoIndex engine to be built


def create_sample_code_files():
    """Create sample code files in different languages for testing."""
    sample_files = {
        "python_example.py": '''
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """A simple calculator class."""

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

if __name__ == "__main__":
    calc = Calculator()
    print(f"Fibonacci(10) = {fibonacci(10)}")
    print(f"5 + 3 = {calc.add(5, 3)}")
''',
        "rust_example.rs": """
/// Calculate the nth Fibonacci number
fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

/// A simple calculator struct
pub struct Calculator;

impl Calculator {
    /// Add two numbers
    pub fn add(&self, a: i32, b: i32) -> i32 {
        a + b
    }

    /// Multiply two numbers
    pub fn multiply(&self, a: i32, b: i32) -> i32 {
        a * b
    }
}

fn main() {
    let calc = Calculator;
    println!("Fibonacci(10) = {}", fibonacci(10));
    println!("5 + 3 = {}", calc.add(5, 3));
}
""",
        "javascript_example.js": """
/**
 * Calculate the nth Fibonacci number
 */
function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

/**
 * A simple calculator class
 */
class Calculator {
    /**
     * Add two numbers
     */
    add(a, b) {
        return a + b;
    }

    /**
     * Multiply two numbers
     */
    multiply(a, b) {
        return a * b;
    }
}

// Main execution
const calc = new Calculator();
console.log(`Fibonacci(10) = ${fibonacci(10)}`);
console.log(`5 + 3 = ${calc.add(5, 3)}`);
""",
        "typescript_example.ts": """
/**
 * Calculate the nth Fibonacci number
 */
function fibonacci(n: number): number {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

/**
 * A simple calculator interface
 */
interface ICalculator {
    add(a: number, b: number): number;
    multiply(a: number, b: number): number;
}

/**
 * Calculator implementation
 */
class Calculator implements ICalculator {
    /**
     * Add two numbers
     */
    add(a: number, b: number): number {
        return a + b;
    }

    /**
     * Multiply two numbers
     */
    multiply(a: number, b: number): number {
        return a * b;
    }
}

// Main execution
const calc: Calculator = new Calculator();
console.log(`Fibonacci(10) = ${fibonacci(10)}`);
console.log(`5 + 3 = ${calc.add(5, 3)}`);
""",
        "haskell_example.hs": """
-- | Calculate the nth Fibonacci number
fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n - 1) + fibonacci (n - 2)

-- | A simple calculator data type
data Calculator = Calculator

-- | Add two numbers
add :: Calculator -> Int -> Int -> Int
add _ a b = a + b

-- | Multiply two numbers
multiply :: Calculator -> Int -> Int -> Int
multiply _ a b = a * b

-- | Main function
main :: IO ()
main = do
    let calc = Calculator
    putStrLn $ "Fibonacci(10) = " ++ show (fibonacci 10)
    putStrLn $ "5 + 3 = " ++ show (add calc 5 3)
""",
    }

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="cocoindex_demo_")

    for filename, content in sample_files.items():
        file_path = Path(temp_dir) / filename
        with open(file_path, "w") as f:
            f.write(content)

    return temp_dir


def demonstrate_model_selection():
    """Demonstrate automatic model selection for different languages."""
    print("=== Language-Aware Embedding Model Selection ===\n")

    # This would be the actual CocoIndex import when available
    # import cocoindex

    language_examples = [
        ("python", ".py", "microsoft/graphcodebert-base"),
        ("rust", ".rs", "microsoft/unixcoder-base"),
        ("javascript", ".js", "microsoft/graphcodebert-base"),
        ("typescript", ".ts", "microsoft/unixcoder-base"),
        ("java", ".java", "microsoft/graphcodebert-base"),
        ("kotlin", ".kt", "microsoft/unixcoder-base"),
        ("haskell", ".hs", "sentence-transformers/all-MiniLM-L6-v2"),  # fallback
    ]

    print("Language → Extension → Selected Model")
    print("-" * 50)

    for language, extension, expected_model in language_examples:
        print(f"{language:10} → {extension:4} → {expected_model}")

    print("\n")


def demonstrate_api_usage():
    """Demonstrate the API usage patterns."""
    print("=== API Usage Examples ===\n")

    # Example 1: Basic SmartCodeEmbedding usage
    print("1. SmartCodeEmbedding (Automatic Detection):")
    print("""
chunk["embedding"] = chunk["text"].transform(
    cocoindex.functions.SmartCodeEmbedding(
        file_extension=file["extension"]
    )
)
""")

    # Example 2: Manual language specification
    print("2. CodeEmbedding (Manual Language):")
    print("""
chunk["embedding"] = chunk["text"].transform(
    cocoindex.functions.CodeEmbedding(language="python")
)
""")

    # Example 3: Force specific model
    print("3. Force Specific Model:")
    print("""
chunk["embedding"] = chunk["text"].transform(
    cocoindex.functions.CodeEmbedding(
        language="rust",
        force_model="microsoft/unixcoder-base"
    )
)
""")

    # Example 4: Custom fallback
    print("4. Custom Fallback Model:")
    print("""
chunk["embedding"] = chunk["text"].transform(
    cocoindex.functions.CodeEmbedding(
        language="haskell",
        fallback_model="sentence-transformers/all-mpnet-base-v2"
    )
)
""")


def demonstrate_complete_flow():
    """Demonstrate a complete code embedding flow."""
    print("=== Complete Code Embedding Flow ===\n")

    flow_code = '''
@cocoindex.flow_def
def code_embedding_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """Complete flow for language-aware code embedding."""

    with flow_builder.read_files(data_scope.input_directory) as file:
        # Skip non-code files
        file = file.filter(lambda f: f["extension"] in [
            ".py", ".rs", ".js", ".ts", ".java", ".kt", ".go", ".cpp"
        ])

        # Chunk code using Tree-sitter (language-aware)
        file["chunks"] = file["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language=file["extension"],
            chunk_size=1000,
            chunk_overlap=200,
        )

        with file["chunks"].row() as chunk:
            # Language-aware embedding
            chunk["embedding"] = chunk["text"].transform(
                cocoindex.functions.SmartCodeEmbedding(
                    file_extension=file["extension"]
                )
            )

            # Add metadata about the embedding model used
            chunk["embedding_info"] = {
                "language": file["extension"],
                "model_type": "auto-selected",
            }

        # Store in vector database with metadata
        file["chunks"].save(cocoindex.targets.QdrantTarget(
            collection_name="code_embeddings",
            metadata_fields=["file_path", "language", "embedding_info"]
        ))
'''

    print(flow_code)


def demonstrate_advanced_patterns():
    """Demonstrate advanced usage patterns."""
    print("=== Advanced Usage Patterns ===\n")

    print("1. Multi-Model Embeddings:")
    multi_model_code = """
with file["chunks"].row() as chunk:
    # General-purpose embedding for broad semantic search
    chunk["embedding_general"] = chunk["text"].transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # Code-specific embedding for precise code search
    chunk["embedding_code"] = chunk["text"].transform(
        cocoindex.functions.SmartCodeEmbedding(
            file_extension=file["extension"]
        )
    )

    # Store both embeddings for hybrid search
"""
    print(multi_model_code)

    print("\n2. Language-Conditional Processing:")
    conditional_code = '''
def get_embedding_config(language: str):
    """Get embedding configuration based on language."""
    if language in ["python", "javascript", "java"]:
        # Use GraphCodeBERT for well-supported languages
        return {
            "model": cocoindex.functions.CodeEmbedding(
                language=language,
                force_model="microsoft/graphcodebert-base"
            ),
            "chunk_size": 1200,  # Larger chunks for these languages
        }
    elif language in ["rust", "typescript"]:
        # Use UniXcode for these languages
        return {
            "model": cocoindex.functions.CodeEmbedding(
                language=language,
                force_model="microsoft/unixcoder-base"
            ),
            "chunk_size": 1000,
        }
    else:
        # Fallback configuration
        return {
            "model": cocoindex.functions.SentenceTransformerEmbed(
                model="sentence-transformers/all-MiniLM-L6-v2"
            ),
            "chunk_size": 800,  # Smaller chunks for less supported languages
        }

# Usage in flow
config = get_embedding_config(detected_language)
chunk["embedding"] = chunk["text"].transform(config["model"])
'''
    print(conditional_code)


def main():
    """Main demonstration function."""
    print("CocoIndex Language-Aware Code Embeddings Demonstration")
    print("=" * 60)
    print()

    # Create sample files
    temp_dir = create_sample_code_files()
    print(f"Sample code files created in: {temp_dir}\n")

    # List created files
    print("Created sample files:")
    for file in sorted(os.listdir(temp_dir)):
        file_path = Path(temp_dir) / file
        size = file_path.stat().st_size
        print(f"  {file} ({size} bytes)")
    print()

    # Demonstrate model selection
    demonstrate_model_selection()

    # Demonstrate API usage
    demonstrate_api_usage()

    # Demonstrate complete flow
    demonstrate_complete_flow()

    # Demonstrate advanced patterns
    demonstrate_advanced_patterns()

    print("=== Performance Notes ===\n")
    print("Model Memory Usage (approximate):")
    print("  • GraphCodeBERT: ~500MB VRAM")
    print("  • UniXcode: ~1.2GB VRAM")
    print("  • SentenceTransformers: ~200-500MB VRAM")
    print()
    print("Features:")
    print("  • Automatic model caching after first load")
    print("  • GPU acceleration when available")
    print("  • Graceful fallback to CPU if GPU unavailable")
    print("  • Language-specific optimization")
    print()

    print("=== Getting Started ===\n")
    print("1. Install dependencies:")
    print("   pip install 'cocoindex[embeddings]'")
    print()
    print("2. Use in your flows:")
    print("   Replace SentenceTransformerEmbed with SmartCodeEmbedding")
    print()
    print("3. Benefits:")
    print("   • Better code understanding")
    print("   • Improved semantic search")
    print("   • Language-specific optimizations")
    print("   • No configuration required")
    print()

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
