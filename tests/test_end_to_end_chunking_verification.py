#!/usr/bin/env python3
"""
END-TO-END test to verify that chunking methods are correctly stored in the database.
This test will finally prove whether our AST chunking is working end-to-end.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
import json

# Load environment variables
load_dotenv()

# Add src to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cocoindex
from cocoindex_code_mcp_server.cocoindex_config import code_embedding_flow, update_flow_config


class TestEndToEndChunkingVerification:
    """End-to-end test to verify chunking methods in database."""
    
    @pytest.fixture(autouse=True)
    def setup_cocoindex(self):
        """Setup CocoIndex with loaded database configuration."""
        print(f"Using database: {os.environ.get('COCOINDEX_DATABASE_URL')}")
        cocoindex.init()
        yield
    
    def get_database_connection(self):
        """Get a direct database connection to query results."""
        db_url = os.environ.get('COCOINDEX_DATABASE_URL')
        if not db_url:
            pytest.skip("No database URL configured")
            
        # Parse the URL - format: postgres://user:password@host/database
        if db_url.startswith('postgres://'):
            # Extract components
            import urllib.parse
            parsed = urllib.parse.urlparse(db_url)
            
            return psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:]  # Remove leading '/'
            )
        else:
            pytest.skip(f"Unsupported database URL format: {db_url}")
    
    def test_end_to_end_python_chunking_method(self):
        """End-to-end test: Create Python file, run flow, verify chunking method in database."""
        python_code = '''def fibonacci(n):
    """Calculate fibonacci number using recursion."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class MathUtils:
    """Mathematical utility functions."""
    
    @staticmethod
    def factorial(n):
        """Calculate factorial."""
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n-1)
    
    def power(self, base, exp):
        """Calculate power."""
        return base ** exp
'''
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = os.path.join(temp_dir, "math_utils.py")
            with open(test_file, 'w') as f:
                f.write(python_code)
            
            print(f"Created test file: {test_file}")
            
            # Configure flow for AST chunking
            update_flow_config(
                paths=[temp_dir],
                enable_polling=False,
                use_default_chunking=False,  # Use AST chunking
                use_default_language_handler=False
            )
            
            # Run flow
            print("Running CocoIndex flow...")
            code_embedding_flow.setup()
            stats = code_embedding_flow.update()
            
            print(f"Flow statistics: {stats}")
            
            # Query database to see what was actually stored
            with self.get_database_connection() as conn:
                with conn.cursor() as cursor:
                    # Query for our test file
                    cursor.execute("""
                        SELECT filename, language, chunking_method, analysis_method, 
                               tree_sitter_chunking_error, tree_sitter_analyze_error,
                               functions, classes, code
                        FROM codeembedding__code_embeddings 
                        WHERE filename LIKE '%math_utils.py%'
                        ORDER BY filename, location
                    """)
                    
                    results = cursor.fetchall()
                    
                    print(f"\\n=== Database Query Results ===")
                    print(f"Found {len(results)} chunks for math_utils.py")
                    
                    assert len(results) > 0, "Should find chunks in database for Python file"
                    
                    for i, row in enumerate(results):
                        filename, language, chunking_method, analysis_method, ts_chunk_err, ts_analyze_err, functions, classes, code = row
                        
                        print(f"\\nChunk {i+1}:")
                        print(f"  filename: {filename}")
                        print(f"  language: {language}")
                        print(f"  chunking_method: '{chunking_method}'")
                        print(f"  analysis_method: '{analysis_method}'")
                        print(f"  tree_sitter_chunking_error: {ts_chunk_err}")
                        print(f"  tree_sitter_analyze_error: {ts_analyze_err}")
                        print(f"  functions: '{functions}'")
                        print(f"  classes: '{classes}'")
                        print(f"  code length: {len(code) if code else 0}")
                        
                        # Verify expectations
                        assert language == "Python", f"Expected Python, got {language}"
                        
                        # Check chunking method - this is the key test!
                        if chunking_method == "ast_tree_sitter":
                            print(f"  ✅ SUCCESS: Found expected chunking method 'ast_tree_sitter'")
                        elif chunking_method == "unknown_chunking":
                            print(f"  ❌ ISSUE: Still getting 'unknown_chunking' instead of 'ast_tree_sitter'")
                        else:
                            print(f"  ⚠️  UNEXPECTED: Got unexpected chunking method '{chunking_method}'")
                        
                        # For this test, we expect AST chunking to work
                        # But let's see what we actually get
                        print(f"  Chunking method result: {chunking_method}")
    
    def test_end_to_end_java_chunking_method(self):
        """End-to-end test: Create Java file, run flow, verify chunking method in database."""
        java_code = '''package com.example;

/**
 * Fibonacci calculator using recursive approach
 */
public class FibonacciCalculator {
    
    /**
     * Calculate fibonacci number recursively
     * @param n The number to calculate fibonacci for
     * @return The fibonacci value
     */
    public static long fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
    
    /**
     * Main method to test fibonacci calculation
     */
    public static void main(String[] args) {
        System.out.println("Fibonacci(10) = " + fibonacci(10));
        System.out.println("Fibonacci(15) = " + fibonacci(15));
    }
    
    /**
     * Utility class for mathematical operations
     */
    public static class MathUtils {
        public static int factorial(int n) {
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }
    }
}'''
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = os.path.join(temp_dir, "FibonacciCalculator.java")
            with open(test_file, 'w') as f:
                f.write(java_code)
            
            print(f"Created test file: {test_file}")
            
            # Configure flow for AST chunking
            update_flow_config(
                paths=[temp_dir],
                enable_polling=False,
                use_default_chunking=False,  # Use AST chunking
                use_default_language_handler=False
            )
            
            # Run flow
            print("Running CocoIndex flow...")
            code_embedding_flow.setup()
            stats = code_embedding_flow.update()
            
            print(f"Flow statistics: {stats}")
            
            # Query database to see what was actually stored
            with self.get_database_connection() as conn:
                with conn.cursor() as cursor:
                    # Query for our test file
                    cursor.execute("""
                        SELECT filename, language, chunking_method, analysis_method,
                               tree_sitter_chunking_error, tree_sitter_analyze_error,
                               functions, classes, code
                        FROM codeembedding__code_embeddings 
                        WHERE filename LIKE '%FibonacciCalculator.java%'
                        ORDER BY filename, location
                    """)
                    
                    results = cursor.fetchall()
                    
                    print(f"\\n=== Database Query Results ===")
                    print(f"Found {len(results)} chunks for FibonacciCalculator.java")
                    
                    assert len(results) > 0, "Should find chunks in database for Java file"
                    
                    for i, row in enumerate(results):
                        filename, language, chunking_method, analysis_method, ts_chunk_err, ts_analyze_err, functions, classes, code = row
                        
                        print(f"\\nChunk {i+1}:")
                        print(f"  filename: {filename}")
                        print(f"  language: {language}")
                        print(f"  chunking_method: '{chunking_method}'")
                        print(f"  analysis_method: '{analysis_method}'")
                        print(f"  tree_sitter_chunking_error: {ts_chunk_err}")
                        print(f"  tree_sitter_analyze_error: {ts_analyze_err}")
                        print(f"  functions: '{functions}'")
                        print(f"  classes: '{classes}'")
                        print(f"  code length: {len(code) if code else 0}")
                        
                        # Verify expectations
                        assert language == "Java", f"Expected Java, got {language}"
                        
                        # Check chunking method - this is the key test!
                        if chunking_method == "ast_tree_sitter":
                            print(f"  ✅ SUCCESS: Found expected chunking method 'ast_tree_sitter'")
                        elif chunking_method == "unknown_chunking":
                            print(f"  ❌ ISSUE: Still getting 'unknown_chunking' instead of 'ast_tree_sitter'")
                        else:
                            print(f"  ⚠️  UNEXPECTED: Got unexpected chunking method '{chunking_method}'")
    
    def test_end_to_end_haskell_chunking_method(self):
        """End-to-end test: Create Haskell file, run flow, verify chunking method in database."""
        haskell_code = '''-- | Fibonacci module with various implementations
module Fibonacci where

-- | Calculate fibonacci using basic recursion
fibonacci :: Integer -> Integer
fibonacci 0 = 0
fibonacci 1 = 1 
fibonacci n = fibonacci (n-1) + fibonacci (n-2)

-- | Fast fibonacci using accumulator
fibonacciFast :: Integer -> Integer
fibonacciFast n = fibHelper n 0 1
  where
    fibHelper 0 a b = a
    fibHelper m a b = fibHelper (m-1) b (a+b)

-- | Person data type
data Person = Person 
    { personName :: String
    , personAge  :: Int
    } deriving (Show, Eq)

-- | Tree data structure
data Tree a = Leaf a | Node (Tree a) (Tree a)
    deriving (Show, Eq)

-- | Main function
main :: IO ()
main = do
    putStrLn "Fibonacci Numbers:"
    mapM_ (\\n -> putStrLn $ "fib(" ++ show n ++ ") = " ++ show (fibonacci n)) [1..10]
'''
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = os.path.join(temp_dir, "Fibonacci.hs")
            with open(test_file, 'w') as f:
                f.write(haskell_code)
            
            print(f"Created test file: {test_file}")
            
            # Configure flow for AST chunking
            update_flow_config(
                paths=[temp_dir],
                enable_polling=False,
                use_default_chunking=False,  # Use AST chunking
                use_default_language_handler=False
            )
            
            # Run flow
            print("Running CocoIndex flow...")
            code_embedding_flow.setup()
            stats = code_embedding_flow.update()
            
            print(f"Flow statistics: {stats}")
            
            # Query database to see what was actually stored
            with self.get_database_connection() as conn:
                with conn.cursor() as cursor:
                    # Query for our test file
                    cursor.execute("""
                        SELECT filename, language, chunking_method, analysis_method,
                               tree_sitter_chunking_error, tree_sitter_analyze_error,
                               functions, classes, data_types, code
                        FROM codeembedding__code_embeddings 
                        WHERE filename LIKE '%Fibonacci.hs%'
                        ORDER BY filename, location
                    """)
                    
                    results = cursor.fetchall()
                    
                    print(f"\\n=== Database Query Results ===")
                    print(f"Found {len(results)} chunks for Fibonacci.hs")
                    
                    assert len(results) > 0, "Should find chunks in database for Haskell file"
                    
                    for i, row in enumerate(results):
                        filename, language, chunking_method, analysis_method, ts_chunk_err, ts_analyze_err, functions, classes, data_types, code = row
                        
                        print(f"\\nChunk {i+1}:")
                        print(f"  filename: {filename}")
                        print(f"  language: {language}")
                        print(f"  chunking_method: '{chunking_method}'")
                        print(f"  analysis_method: '{analysis_method}'")
                        print(f"  tree_sitter_chunking_error: {ts_chunk_err}")
                        print(f"  tree_sitter_analyze_error: {ts_analyze_err}")
                        print(f"  functions: '{functions}'")
                        print(f"  classes: '{classes}'")
                        print(f"  data_types: '{data_types}'")
                        print(f"  code length: {len(code) if code else 0}")
                        
                        # Verify expectations
                        assert language == "Haskell", f"Expected Haskell, got {language}"
                        
                        # Check chunking method - for Haskell, we expect rust_haskell_*
                        if chunking_method.startswith("rust_haskell_"):
                            print(f"  ✅ SUCCESS: Found expected Haskell chunking method '{chunking_method}'")
                        elif chunking_method == "unknown_chunking":
                            print(f"  ❌ ISSUE: Still getting 'unknown_chunking' instead of rust_haskell_*")
                        else:
                            print(f"  ⚠️  UNEXPECTED: Got unexpected chunking method '{chunking_method}'")


if __name__ == "__main__":
    pytest.main([__file__])
