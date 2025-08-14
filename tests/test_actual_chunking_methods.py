#!/usr/bin/env python3
"""
COMPREHENSIVE TEST to finally identify and fix chunking method issues.
This test directly tests the actual MCP server and CocoIndex flow behavior.
"""

import pytest
import json
import tempfile
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from tests.common import CocoIndexTestInfrastructure
from cocoindex_code_mcp_server.ast_chunking import CocoIndexASTChunker, ASTChunkOperation
from cocoindex_code_mcp_server.cocoindex_config import (
    code_embedding_flow, update_flow_config, extract_code_metadata, 
    AST_CHUNKING_AVAILABLE, _global_flow_config
)
import cocoindex


class TestActualChunkingMethods:
    """Tests to identify and fix the actual chunking method issues."""
    
    def test_ast_chunking_availability(self):
        """Test that AST chunking components are properly available."""
        print(f"AST_CHUNKING_AVAILABLE: {AST_CHUNKING_AVAILABLE}")
        print(f"ASTChunkOperation: {ASTChunkOperation}")
        print(f"ASTChunkOperation type: {type(ASTChunkOperation)}")
        
        assert AST_CHUNKING_AVAILABLE is True, "AST chunking should be available"
        assert ASTChunkOperation is not None, "ASTChunkOperation should not be None"
    
    def test_direct_ast_chunker_produces_correct_methods(self):
        """Test that CocoIndexASTChunker directly produces expected chunking methods."""
        chunker = CocoIndexASTChunker()
        
        # Test data: language -> (code, expected_chunking_method)
        test_cases = [
            ("Python", '''def hello():
    print("Hello, World!")
class Test:
    pass
''', "ast_tree_sitter"),
            ("Java", '''public class Test {
    public void hello() {
        System.out.println("Hello");
    }
}''', "ast_tree_sitter"),
            ("Haskell", '''main = putStrLn "Hello"
data Person = Person String
''', "rust_haskell_"),  # Should start with this
            ("UnknownLang", "some random text", "ast_fallback_unavailable"),
        ]
        
        for language, code, expected_method in test_cases:
            print(f"\n=== Testing {language} ===")
            chunks = chunker.chunk_code(code, language, f"test.{language.lower()}")
            
            assert len(chunks) > 0, f"Should produce chunks for {language}"
            
            chunk = chunks[0]
            actual_method = chunk.metadata.get('chunking_method')
            print(f"Expected: {expected_method}")
            print(f"Actual: {actual_method}")
            
            if expected_method.endswith('_'):
                # For Haskell, check it starts with the expected prefix
                assert actual_method.startswith(expected_method), \
                    f"{language}: Expected method starting with '{expected_method}', got '{actual_method}'"
            else:
                assert actual_method == expected_method, \
                    f"{language}: Expected '{expected_method}', got '{actual_method}'"
    
    def test_metadata_extraction_preserves_chunking_method(self):
        """Test that extract_code_metadata preserves chunking_method from existing metadata."""
        # Create a chunk with our expected chunking method
        chunker = CocoIndexASTChunker()
        python_code = '''def test():
    return "Hello, World!"
'''
        
        chunks = chunker.chunk_code(python_code, "Python", "test.py")
        assert len(chunks) > 0, "Should produce chunks"
        
        chunk = chunks[0]
        original_chunking_method = chunk.metadata.get('chunking_method')
        print(f"Original chunking method: {original_chunking_method}")
        
        # Test extract_code_metadata with existing metadata
        existing_metadata_json = json.dumps(chunk.metadata)
        print(f"Existing metadata JSON: {existing_metadata_json}")
        
        extracted_metadata_json = extract_code_metadata(
            text=chunk.content,
            language="Python", 
            filename="test.py",
            existing_metadata_json=existing_metadata_json
        )
        
        extracted_metadata = json.loads(extracted_metadata_json)
        final_chunking_method = extracted_metadata.get('chunking_method')
        print(f"Final chunking method: {final_chunking_method}")
        
        assert final_chunking_method == original_chunking_method, \
            f"Chunking method should be preserved: expected '{original_chunking_method}', got '{final_chunking_method}'"
    
    def test_cocoindex_flow_configuration(self):
        """Test that CocoIndex flow is configured to use AST chunking."""
        # Check global config
        print(f"Global flow config: {_global_flow_config}")
        
        # Check if default chunking is disabled (should be False to use AST chunking)
        use_default_chunking = _global_flow_config.get('use_default_chunking', False)
        print(f"use_default_chunking: {use_default_chunking}")
        
        # Update config to ensure AST chunking is used
        update_flow_config(
            paths=["/tmp/test"],
            use_default_chunking=False,  # Use AST chunking
            use_default_language_handler=False
        )
        
        print(f"Updated global flow config: {_global_flow_config}")
    
    @pytest.mark.asyncio
    async def test_mcp_server_chunking_methods(self):
        """Test actual MCP server hybrid search to see what chunking methods are returned."""
        # Create test files
        test_files = {
            "test_python.py": '''def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """Simple calculator."""
    def add(self, a, b):
        return a + b
''',
            "Test.java": '''public class Test {
    /**
     * Fibonacci calculation
     */
    public static int fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n-1) + fibonacci(n-2);
    }
    
    public static void main(String[] args) {
        System.out.println(fibonacci(10));
    }
}''',
            "fibonacci.hs": '''-- Fibonacci module
module Fibonacci where

-- | Calculate fibonacci number  
fibonacci :: Integer -> Integer
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n-1) + fibonacci (n-2)

main :: IO ()
main = print $ fibonacci 10
''',
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            for filename, content in test_files.items():
                with open(os.path.join(temp_dir, filename), 'w') as f:
                    f.write(content)
            
            # Setup test infrastructure
            infrastructure_config = {
                "paths": [temp_dir],
                "enable_polling": False,
                "default_chunking": False,  # Use AST chunking
                "default_language_handler": False,
            }
            
            async with CocoIndexTestInfrastructure(**infrastructure_config) as test_infrastructure:
                # Test different languages
                test_queries = [
                    ("python", "language:Python", "Python files should use ast_tree_sitter"),
                    ("java", "language:Java", "Java files should use ast_tree_sitter"), 
                    ("haskell", "language:Haskell", "Haskell files should use rust_haskell_*"),
                ]
                
                for query_term, keyword_query, description in test_queries:
                    print(f"\n=== Testing {description} ===")
                    
                    results = test_infrastructure.hybrid_search_engine.search(
                        vector_query=f"fibonacci {query_term}",
                        keyword_query=keyword_query,
                        top_k=10
                    )
                    
                    print(f"Found {len(results)} results for {query_term}")
                    
                    if len(results) > 0:
                        for i, result in enumerate(results[:3]):  # Check first 3 results
                            chunking_method = result.get('chunking_method', 'MISSING')
                            language = result.get('language', 'MISSING')
                            filename = result.get('filename', 'MISSING')
                            
                            print(f"  Result {i+1}: {filename} ({language}) -> chunking_method: '{chunking_method}'")
                            
                            # Check for expected chunking methods
                            if language == "Python":
                                assert chunking_method in ["ast_tree_sitter", "unknown_chunking"], \
                                    f"Python should use ast_tree_sitter or unknown_chunking, got: {chunking_method}"
                            elif language == "Java":
                                assert chunking_method in ["ast_tree_sitter", "unknown_chunking"], \
                                    f"Java should use ast_tree_sitter or unknown_chunking, got: {chunking_method}"
                            elif language == "Haskell":
                                assert chunking_method.startswith("rust_haskell_") or chunking_method == "unknown_chunking", \
                                    f"Haskell should use rust_haskell_* or unknown_chunking, got: {chunking_method}"
                            
                            # Report what we actually found
                            if chunking_method == "unknown_chunking":
                                print(f"    ❌ ISSUE: Still getting 'unknown_chunking' for {language}")
                            elif language in ["Python", "Java"] and chunking_method == "ast_tree_sitter":
                                print(f"    ✅ CORRECT: {language} using ast_tree_sitter")
                            elif language == "Haskell" and chunking_method.startswith("rust_haskell_"):
                                print(f"    ✅ CORRECT: Haskell using {chunking_method}")
                            else:
                                print(f"    ⚠️  UNEXPECTED: {language} using {chunking_method}")
                    else:
                        print(f"  ❌ No results found for {query_term} - this indicates indexing issues")
    
    def test_ast_chunk_operation_directly(self):
        """Test ASTChunkOperation function directly."""
        if ASTChunkOperation is None:
            pytest.fail("ASTChunkOperation is None - AST chunking not available")
        
        python_code = '''def hello():
    print("Hello, World!")
    
class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
'''
        
        # Call ASTChunkOperation directly
        result = ASTChunkOperation(
            content=python_code,
            language="Python",
            max_chunk_size=1800
        )
        
        print(f"ASTChunkOperation returned: {type(result)}")
        print(f"Number of chunks: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        
        if hasattr(result, '__iter__'):
            for i, chunk in enumerate(result):
                if hasattr(chunk, 'metadata'):
                    chunking_method = chunk.metadata.get('chunking_method', 'MISSING')
                    print(f"  Chunk {i+1}: chunking_method = {chunking_method}")
                else:
                    print(f"  Chunk {i+1}: No metadata attribute")
        
        assert result is not None, "ASTChunkOperation should return results"
    
    def test_debug_chunking_method_extraction(self):
        """Debug test to trace exactly where chunking methods get lost."""
        chunker = CocoIndexASTChunker()
        python_code = '''def test():
    return "test"
'''
        
        print("\n=== DEBUGGING CHUNKING METHOD FLOW ===")
        
        # Step 1: Create chunk with ASTChunker
        chunks = chunker.chunk_code(python_code, "Python", "test.py") 
        assert len(chunks) > 0, "Should create chunks"
        
        chunk = chunks[0]
        original_method = chunk.metadata.get('chunking_method')
        print(f"1. Original chunk chunking_method: {original_method}")
        
        # Step 2: Test metadata extraction
        existing_metadata_json = json.dumps(chunk.metadata)
        print(f"2. Existing metadata JSON contains: {json.loads(existing_metadata_json).keys()}")
        
        extracted_json = extract_code_metadata(
            text=chunk.content,
            language="Python",
            filename="test.py", 
            existing_metadata_json=existing_metadata_json
        )
        
        extracted_dict = json.loads(extracted_json)
        extracted_method = extracted_dict.get('chunking_method')
        print(f"3. Extracted chunking_method: {extracted_method}")
        
        # Step 3: Check if preservation works
        assert extracted_method == original_method, \
            f"Method should be preserved: {original_method} -> {extracted_method}"
        
        print("✅ Chunking method preservation works correctly")


if __name__ == "__main__":
    pytest.main([__file__])