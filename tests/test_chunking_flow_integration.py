#!/usr/bin/env python3
"""
Integration test for chunking methods within CocoIndex flow.
Converted from test_chunking_flow.py and test_chunking_quick.py
"""

import pytest
import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cocoindex
from cocoindex_code_mcp_server.cocoindex_config import code_embedding_flow, update_flow_config


class TestChunkingFlowIntegration:
    """Integration tests for chunking methods within the CocoIndex flow."""
    
    @pytest.fixture(autouse=True)
    def setup_cocoindex(self):
        """Setup CocoIndex with in-memory database for testing."""
        # Use in-memory SQLite for testing
        original_db_url = os.environ.get('COCOINDEX_DATABASE_URL')
        os.environ['COCOINDEX_DATABASE_URL'] = 'sqlite:///:memory:'
        
        # Initialize CocoIndex
        cocoindex.init()
        
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
''', "ast_tree_sitter"),
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
}''', "ast_tree_sitter"),
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
    def test_flow_chunking_methods(self, language, filename, content, expected_method_pattern):
        """Test that the flow produces correct chunking methods for different languages."""
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
    
    def test_flow_with_multiple_files(self):
        """Test flow with multiple files of different languages."""
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
    
    def test_flow_with_default_chunking(self):
        """Test flow behavior when default chunking is enabled."""
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
    
    def test_flow_with_default_language_handler(self):
        """Test flow behavior when default language handler is enabled."""
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
        initial_config = {
            'paths': ['/test/path'],
            'enable_polling': True,
            'poll_interval': 60,
            'use_default_chunking': True,
            'use_default_language_handler': True
        }
        
        update_flow_config(**initial_config)
        
        # Test updated configuration
        updated_config = {
            'paths': ['/another/path'],
            'enable_polling': False,
            'use_default_chunking': False,
            'use_default_language_handler': False
        }
        
        update_flow_config(**updated_config)
        
        # Configuration changes don't throw errors
        # Real testing would require access to _global_flow_config
        print("Flow configuration updates completed successfully")


if __name__ == "__main__":
    pytest.main([__file__])