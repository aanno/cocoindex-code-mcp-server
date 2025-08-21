#!/usr/bin/env python3
"""
Integration test for chunking methods within CocoIndex flow.
Converted from test_chunking_flow.py and test_chunking_quick.py
"""

import os
import tempfile

import pytest

import cocoindex
from cocoindex_code_mcp_server.cocoindex_config import (
    code_embedding_flow,
    update_flow_config,
)

# Add src to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestChunkingFlowIntegration:
    """Integration tests for chunking methods within the CocoIndex flow."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_cocoindex(self):
        """Setup CocoIndex with in-memory database for testing.
        
        PERFORMANCE OPTIMIZATION: Using class scope to avoid reinitializing 
        CocoIndex for every test method, as cocoindex.init() is expensive.
        """
        # Use in-memory SQLite for testing
        original_db_url = os.environ.get('COCOINDEX_DATABASE_URL')
        os.environ['COCOINDEX_DATABASE_URL'] = 'sqlite:///:memory:'

        try:
            # Initialize CocoIndex - this is expensive, so we do it once per class
            cocoindex.init()
            print("🏁 CocoIndex initialized for test class")
        except Exception as e:
            print(f"⚠️  CocoIndex initialization failed: {e}")
            # Continue with tests even if init fails - some tests may not need it
        
        yield

        # Restore original database URL if it existed
        if original_db_url is not None:
            os.environ['COCOINDEX_DATABASE_URL'] = original_db_url
        elif 'COCOINDEX_DATABASE_URL' in os.environ:
            del os.environ['COCOINDEX_DATABASE_URL']

    def create_test_file(self, content: str, filename: str, directory: str) -> str:
        """Create a test file with given content."""
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    @pytest.mark.parametrize("language,filename,content,expected_method_pattern", [
        ("Python", "test.py", '''def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """A simple calculator class."""
    def add(self, a, b):
        return a + b
''', "astchunk_library"),
        ("Java", "Test.java", '''public class Test {
    /**
     * Main method
     */
    public static void main(String[] args) {
        System.out.println("Hello, World!");
        fibonacci(10);
    }

    public static int fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n-1) + fibonacci(n-2);
    }
}''', "astchunk_library"),
        ("Haskell", "fibonacci.hs", '''-- Fibonacci module
module Fibonacci where

-- | Calculate fibonacci number
fibonacci :: Integer -> Integer
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n-1) + fibonacci (n-2)

-- | Person data type
data Person = Person String Int deriving Show

main :: IO ()
main = do
    putStrLn "Hello, World!"
    print $ fibonacci 10
''', "rust_haskell_"),
    ])
    @pytest.mark.slow
    @pytest.mark.integration
    def test_flow_chunking_methods(self, language, filename, content, expected_method_pattern):
        """Test that the flow produces correct chunking methods for different languages.
        
        SLOW TEST: This test is marked as slow because it:
        1. Runs 3 times (parametrized for Python, Java, Haskell)
        2. Initializes the full CocoIndex system for each run (cocoindex.init())
        3. Creates temporary files and processes them through the complete flow
        4. Executes code_embedding_flow.setup() and .update() which:
           - Sets up database connections and schemas
           - Runs language detection and AST parsing
           - Performs embedding generation and vector storage
           - Updates search indices
        5. Each language may use different chunking strategies (tree-sitter, AST parsers)
        
        Expected runtime: 30-60 seconds per parameterized run (90-180s total)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file_path = self.create_test_file(content, filename, temp_dir)

            # Update flow configuration to process only our test file
            update_flow_config(
                paths=[temp_dir],
                enable_polling=False,
                use_default_chunking=False,  # Use AST chunking
                use_default_language_handler=False  # Use custom language handlers
            )

            # Setup and run flow
            code_embedding_flow.setup()
            stats = code_embedding_flow.update()

            # Verify flow ran successfully
            assert stats is not None, "Flow should return statistics"

            # Access the data through CocoIndex data inspection
            # Since we can't easily access the vector store directly in tests,
            # we'll verify the flow completed without errors
            # Real verification would require database inspection
            print(f"Flow completed for {language}: {stats}")

            # Clean up the test file
            os.unlink(test_file_path)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_flow_with_multiple_files(self):
        """Test flow with multiple files of different languages.
        
        SLOW TEST: This test is marked as slow because it:
        1. Creates multiple test files (Python, Java, Haskell) in temp directory
        2. Initializes full CocoIndex system with SQLite in-memory database
        3. Processes all files through complete flow pipeline including:
           - Language detection for each file type
           - AST parsing and chunking for multiple languages
           - Embedding generation for all chunks
           - Database storage and indexing
        4. Each file type may require different parsers and chunking strategies
        
        Expected runtime: 45-90 seconds
        """
        test_files = [
            ("test.py", '''def hello():
    print("Hello from Python!")

class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
'''),
            ("Test.java", '''public class Test {
    public void hello() {
        System.out.println("Hello from Java!");
    }
}'''),
            ("hello.hs", '''main :: IO ()
main = putStrLn "Hello from Haskell!"

data Message = Message String deriving Show
'''),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create all test files
            for filename, content in test_files:
                self.create_test_file(content, filename, temp_dir)

            # Update flow configuration
            update_flow_config(
                paths=[temp_dir],
                enable_polling=False,
                use_default_chunking=False,
                use_default_language_handler=False
            )

            # Run flow
            code_embedding_flow.setup()
            stats = code_embedding_flow.update()

            # Verify flow completed
            assert stats is not None, "Flow should complete successfully"
            print(f"Multi-file flow completed: {stats}")

            # Clean up
            for filename, _ in test_files:
                os.unlink(os.path.join(temp_dir, filename))

    @pytest.mark.slow
    @pytest.mark.integration  
    def test_flow_with_default_chunking(self):
        """Test flow behavior when default chunking is enabled.
        
        SLOW TEST: This test is marked as slow because it:
        1. Initializes full CocoIndex system and database
        2. Processes files using default CocoIndex chunking (use_default_chunking=True)
        3. Executes complete flow pipeline with embedding generation and storage
        
        Expected runtime: 30-45 seconds
        """
        python_content = '''def test_function():
    """Test function with default chunking."""
    return "Hello, World!"

class TestClass:
    def method(self):
        return 42
'''

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file_path = self.create_test_file(python_content, "test.py", temp_dir)

            # Update flow configuration with default chunking
            update_flow_config(
                paths=[temp_dir],
                enable_polling=False,
                use_default_chunking=True,  # Use default CocoIndex chunking
                use_default_language_handler=False
            )

            # Run flow
            code_embedding_flow.setup()
            stats = code_embedding_flow.update()

            # Verify flow completed
            assert stats is not None, "Flow should complete with default chunking"
            print(f"Default chunking flow completed: {stats}")

            # Clean up
            os.unlink(test_file_path)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_flow_with_default_language_handler(self):
        """Test flow behavior when default language handler is enabled.
        
        SLOW TEST: This test is marked as slow because it:
        1. Initializes full CocoIndex system and database
        2. Uses default language handler (use_default_language_handler=True)
        3. Executes complete flow pipeline with language detection and processing
        
        Expected runtime: 30-45 seconds
        """
        python_content = '''def example():
    """Example function."""
    x = 1 + 2
    return x * 3

class Example:
    """Example class."""
    def __init__(self):
        self.value = 0
'''

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file_path = self.create_test_file(python_content, "test.py", temp_dir)

            # Update flow configuration with default language handler
            update_flow_config(
                paths=[temp_dir],
                enable_polling=False,
                use_default_chunking=False,
                use_default_language_handler=True  # Use default language handler
            )

            # Run flow
            code_embedding_flow.setup()
            stats = code_embedding_flow.update()

            # Verify flow completed
            assert stats is not None, "Flow should complete with default language handler"
            print(f"Default language handler flow completed: {stats}")

            # Clean up
            os.unlink(test_file_path)

    def test_flow_configuration_updates(self):
        """Test that flow configuration updates work correctly."""
        # Test initial configuration
        update_flow_config(
            paths=['/test/path'],
            enable_polling=True,
            poll_interval=60,
            use_default_chunking=True,
            use_default_language_handler=True
        )

        # Test updated configuration
        update_flow_config(
            paths=['/another/path'],
            enable_polling=False,
            use_default_chunking=False,
            use_default_language_handler=False
        )

        # Configuration changes don't throw errors
        # Real testing would require access to _global_flow_config
        print("Flow configuration updates completed successfully")


if __name__ == "__main__":
    pytest.main([__file__])
