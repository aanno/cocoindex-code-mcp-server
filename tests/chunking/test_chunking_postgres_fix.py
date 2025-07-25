#!/usr/bin/env python3

"""
Pytest tests to systematically diagnose and verify fixes for chunking failures 
that were causing PostgreSQL duplicate key errors.

These tests verify:
1. CocoIndex SplitRecursively usage is correct
2. AST chunking produces unique location identifiers 
3. No duplicate primary keys (filename, location) are generated
"""

import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'src', 'cocoindex_code_mcp_server'))

import cocoindex
from cocoindex_config import (
    extract_language, get_chunking_params, ASTChunkOperation, AST_CHUNKING_AVAILABLE,
    _global_flow_config
)

# Sample code for testing
SAMPLE_PYTHON_CODE = '''#!/usr/bin/env python3

"""
Sample Python code for testing chunking behavior.
"""

import os
import sys
from typing import List, Dict

def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return "success"

class SampleClass:
    """A sample class for testing."""
    
    def __init__(self, name: str):
        self.name = name
    
    def greet(self) -> str:
        return f"Hello, {self.name}!"
    
    def process_data(self, data: List[Dict]) -> List[str]:
        results = []
        for item in data:
            if "name" in item:
                results.append(item["name"])
        return results

if __name__ == "__main__":
    obj = SampleClass("Test")
    print(obj.greet())
    hello_world()
'''

SAMPLE_GO_CODE = '''package main

import (
    "fmt"
    "log"
    "net/http"
)

type User struct {
    ID   int    `json:"id"`
    Name string `json:"name"`
    Email string `json:"email"`
}

func main() {
    http.HandleFunc("/users", handleUsers)
    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}

func handleUsers(w http.ResponseWriter, r *http.Request) {
    switch r.Method {
    case "GET":
        getUsers(w, r)
    case "POST":
        createUser(w, r)
    default:
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
    }
}

func getUsers(w http.ResponseWriter, r *http.Request) {
    // Implementation here
    fmt.Fprintf(w, "Getting users")
}

func createUser(w http.ResponseWriter, r *http.Request) {
    // Implementation here  
    fmt.Fprintf(w, "Creating user")
}
'''


class TestLanguageDetection:
    """Test language detection functionality."""
    
    def test_python_detection(self):
        """Test Python file language detection."""
        language = extract_language("test.py")
        assert language == "Python"
    
    def test_go_detection(self):
        """Test Go file language detection.""" 
        language = extract_language("main.go")
        assert language == "Go"
    
    def test_special_files(self):
        """Test special file detection."""
        assert extract_language("Dockerfile") == "dockerfile"
        assert extract_language("Makefile") == "makefile"


class TestChunkingParams:
    """Test chunking parameter retrieval."""
    
    def test_python_params(self):
        """Test Python chunking parameters."""
        params = get_chunking_params("Python")
        assert params.chunk_size == 1000
        assert params.min_chunk_size == 300
        assert params.chunk_overlap == 250
    
    def test_go_params(self):
        """Test Go chunking parameters."""
        params = get_chunking_params("Go")
        assert params.chunk_size == 1000
        assert params.min_chunk_size == 250  
        assert params.chunk_overlap == 200
    
    def test_default_params(self):
        """Test default chunking parameters."""
        params = get_chunking_params("UnknownLanguage")
        assert params.chunk_size == 1000
        assert params.min_chunk_size == 300
        assert params.chunk_overlap == 200


class TestCocoIndexSplitRecursively:
    """Test CocoIndex's built-in SplitRecursively function."""
    
    def test_split_recursively_function_exists(self):
        """Test that SplitRecursively function exists."""
        assert hasattr(cocoindex.functions, 'SplitRecursively')
        split_func = cocoindex.functions.SplitRecursively()
        assert callable(split_func)
    
    def test_split_recursively_with_python_code(self):
        """Test SplitRecursively with Python code."""
        split_func = cocoindex.functions.SplitRecursively()
        
        # Create a mock record similar to what CocoIndex uses
        record = {"content": SAMPLE_PYTHON_CODE}
        
        try:
            # Test the function call with parameters similar to the flow
            chunks = split_func(
                record,
                language="Python",
                chunk_size=1000,
                min_chunk_size=300, 
                chunk_overlap=250
            )
            
            print(f"SplitRecursively produced {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i}: keys={list(chunk.keys())}")
                if isinstance(chunk, dict) and 'text' in chunk:
                    print(f"  Text length: {len(chunk['text'])}")
                    print(f"  Location: {chunk.get('location', 'NO_LOCATION')}")
                    print(f"  First 100 chars: {chunk['text'][:100]}...")
                
            # Basic validation
            assert len(chunks) > 0, "Should produce at least one chunk"
            
            # Check if chunks have expected structure
            for chunk in chunks:
                if isinstance(chunk, dict):
                    assert 'text' in chunk or 'content' in chunk, f"Chunk missing text field: {chunk.keys()}"
                    # Check for location field
                    if 'location' in chunk:
                        print(f"Chunk location: {chunk['location']}")
                        
        except Exception as e:
            pytest.fail(f"SplitRecursively failed: {e}")
    
    def test_split_recursively_with_go_code(self):
        """Test SplitRecursively with Go code."""
        split_func = cocoindex.functions.SplitRecursively()
        record = {"content": SAMPLE_GO_CODE}
        
        try:
            chunks = split_func(
                record,
                language="Go",
                chunk_size=1000,
                min_chunk_size=250,
                chunk_overlap=200
            )
            
            print(f"SplitRecursively (Go) produced {len(chunks)} chunks")
            assert len(chunks) > 0, "Should produce at least one chunk for Go code"
            
        except Exception as e:
            pytest.fail(f"SplitRecursively failed with Go code: {e}")


class TestASTChunking:
    """Test AST chunking functionality."""
    
    def test_ast_chunking_availability(self):
        """Test AST chunking availability."""
        print(f"AST_CHUNKING_AVAILABLE: {AST_CHUNKING_AVAILABLE}")
        if AST_CHUNKING_AVAILABLE:
            assert ASTChunkOperation is not None
            assert callable(ASTChunkOperation)
        else:
            print("AST chunking not available - skipping related tests")
    
    @pytest.mark.skipif(not AST_CHUNKING_AVAILABLE, reason="AST chunking not available")
    def test_ast_chunking_with_python_code(self):
        """Test AST chunking with Python code."""
        try:
            # Test calling ASTChunkOperation directly
            chunks = ASTChunkOperation(
                content=SAMPLE_PYTHON_CODE,
                language="Python",
                max_chunk_size=1000,
                chunk_overlap=250
            )
            
            print(f"AST chunking produced {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i}: {type(chunk)} - {chunk}")
                if hasattr(chunk, 'text'):
                    print(f"  Text length: {len(chunk.text)}")
                    print(f"  Location: {chunk.location}")
                    print(f"  Start: {chunk.start}, End: {chunk.end}")
                    print(f"  First 100 chars: {chunk.text[:100]}...")
            
            assert len(chunks) > 0, "AST chunking should produce at least one chunk"
            
            # Check chunk structure
            for chunk in chunks:
                assert hasattr(chunk, 'text'), "Chunk should have text attribute"
                assert hasattr(chunk, 'location'), "Chunk should have location attribute"
                assert hasattr(chunk, 'start'), "Chunk should have start attribute"
                assert hasattr(chunk, 'end'), "Chunk should have end attribute"
                
                # Validate location uniqueness
                print(f"Chunk location: '{chunk.location}'")
                assert chunk.location != "", "Location should not be empty"
                
        except Exception as e:
            pytest.fail(f"AST chunking failed: {e}")
    
    @pytest.mark.skipif(not AST_CHUNKING_AVAILABLE, reason="AST chunking not available")  
    def test_ast_chunking_location_uniqueness(self):
        """Test that AST chunking produces unique locations."""
        chunks = ASTChunkOperation(
            content=SAMPLE_PYTHON_CODE,
            language="Python", 
            max_chunk_size=500,  # Smaller chunks to force multiple
            chunk_overlap=0
        )
        
        locations = [chunk.location for chunk in chunks]
        print(f"Generated locations: {locations}")
        
        # Check for duplicates
        unique_locations = set(locations)
        assert len(locations) == len(unique_locations), f"Duplicate locations found: {locations}"


class TestChunkingIntegration:
    """Test chunking integration issues that might cause PostgreSQL conflicts."""
    
    def test_chunking_produces_unique_keys(self):
        """Test that chunking produces unique (filename, location) combinations."""
        filename = "test.py"
        
        # Test both chunking methods if available
        chunking_methods = []
        
        # Default chunking
        split_func = cocoindex.functions.SplitRecursively()
        record = {"content": SAMPLE_PYTHON_CODE}
        try:
            default_chunks = split_func(
                record,
                language="Python",
                chunk_size=800,
                min_chunk_size=200,
                chunk_overlap=150
            )
            chunking_methods.append(("SplitRecursively", default_chunks))
        except Exception as e:
            print(f"Default chunking failed: {e}")
        
        # AST chunking
        if AST_CHUNKING_AVAILABLE:
            try:
                ast_chunks = ASTChunkOperation(
                    content=SAMPLE_PYTHON_CODE,
                    language="Python",
                    max_chunk_size=800,
                    chunk_overlap=150
                )
                # Convert to similar format for comparison
                ast_chunk_dicts = []
                for chunk in ast_chunks:
                    ast_chunk_dicts.append({
                        "text": chunk.text,
                        "location": chunk.location,
                        "start": chunk.start,
                        "end": chunk.end
                    })
                chunking_methods.append(("ASTChunking", ast_chunk_dicts))
            except Exception as e:
                print(f"AST chunking failed: {e}")
        
        # Analyze results
        for method_name, chunks in chunking_methods:
            print(f"\n=== {method_name} Results ===")
            print(f"Number of chunks: {len(chunks)}")
            
            # Extract locations and check for duplicates
            locations = []
            for i, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    location = chunk.get('location', f'chunk_{i}')
                elif hasattr(chunk, 'location'):
                    location = chunk.location
                else:
                    location = f'unknown_{i}'
                
                locations.append(location)
                primary_key = (filename, location)
                print(f"  Chunk {i}: location='{location}' -> primary_key={primary_key}")
            
            # Check for duplicate locations within the same file
            unique_locations = set(locations)
            duplicates = [loc for loc in locations if locations.count(loc) > 1]
            
            if duplicates:
                print(f"  ❌ DUPLICATE LOCATIONS FOUND: {set(duplicates)}")
                print(f"  This would cause PostgreSQL 'ON CONFLICT DO UPDATE' error!")
                assert False, f"{method_name} produced duplicate locations: {duplicates}"
            else:
                print(f"  ✅ All locations unique ({len(unique_locations)} unique locations)")


if __name__ == "__main__":
    # Initialize CocoIndex for testing
    cocoindex.init()
    
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
