#!/usr/bin/env python3

"""
Test to verify the unique location post-processing fix works correctly.
"""

import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'src', 'cocoindex-code-mcp-server'))

import cocoindex
from cocoindex_config import ensure_unique_chunk_locations
from ast_chunking import Chunk

class TestUniqueLocationsFix:
    """Test the ensure_unique_chunk_locations function."""
    
    def test_ensure_unique_locations_with_duplicate_dict_chunks(self):
        """Test the function with duplicate dictionary chunks (default chunking format)."""
        # Simulate what SplitRecursively might produce - multiple chunks with same location
        duplicate_chunks = [
            {"text": "chunk 1 content", "location": "line:0", "start": 0, "end": 10},
            {"text": "chunk 2 content", "location": "line:0", "start": 10, "end": 20},
            {"text": "chunk 3 content", "location": "line:0", "start": 20, "end": 30},
            {"text": "chunk 4 content", "location": "line:5", "start": 50, "end": 60},
            {"text": "chunk 5 content", "location": "line:5", "start": 60, "end": 70},
        ]
        
        print(f"Input chunks with duplicate locations:")
        for i, chunk in enumerate(duplicate_chunks):
            print(f"  Chunk {i}: location='{chunk['location']}'")
        
        # Apply the fix
        unique_chunks = ensure_unique_chunk_locations(duplicate_chunks)
        
        print(f"\nOutput chunks with unique locations:")
        locations = []
        for i, chunk in enumerate(unique_chunks):
            location = chunk.location  # Now returns Chunk dataclass objects
            locations.append(location)
            print(f"  Chunk {i}: location='{location}'")
        
        # Verify all locations are unique
        unique_locations = set(locations)
        assert len(locations) == len(unique_locations), f"Duplicate locations found: {locations}"
        
        # Verify the expected pattern (function now converts everything to Chunk format)
        expected_locations = ["line:0", "line:0#1", "line:0#2", "line:5", "line:5#1"]
        assert locations == expected_locations, f"Expected {expected_locations}, got {locations}"
        
        print("✅ Dictionary chunks made unique successfully")
    
    def test_ensure_unique_locations_with_duplicate_dataclass_chunks(self):
        """Test the function with duplicate Chunk dataclass objects (AST chunking format)."""
        # Simulate what AST chunking might produce if it had a bug
        duplicate_chunks = [
            Chunk(text="chunk 1", location="line:0#0", start=0, end=10),
            Chunk(text="chunk 2", location="line:0#0", start=10, end=20),  # duplicate location
            Chunk(text="chunk 3", location="line:0#1", start=20, end=30),
            Chunk(text="chunk 4", location="line:0#1", start=30, end=40),  # duplicate location
        ]
        
        print(f"Input Chunk objects with duplicate locations:")
        for i, chunk in enumerate(duplicate_chunks):
            print(f"  Chunk {i}: location='{chunk.location}'")
        
        # Apply the fix
        unique_chunks = ensure_unique_chunk_locations(duplicate_chunks)
        
        print(f"\nOutput Chunk objects with unique locations:")
        locations = []
        for i, chunk in enumerate(unique_chunks):
            location = chunk.location
            locations.append(location)
            print(f"  Chunk {i}: location='{location}'")
        
        # Verify all locations are unique
        unique_locations = set(locations)
        assert len(locations) == len(unique_locations), f"Duplicate locations found: {locations}"
        
        # Verify the expected pattern
        expected_locations = ["line:0#0", "line:0#0#1", "line:0#1", "line:0#1#1"]
        assert locations == expected_locations, f"Expected {expected_locations}, got {locations}"
        
        print("✅ Chunk dataclass objects made unique successfully")
    
    def test_ensure_unique_locations_with_already_unique_chunks(self):
        """Test that already unique chunks are not modified."""
        unique_chunks = [
            {"text": "chunk 1", "location": "line:0", "start": 0, "end": 10},
            {"text": "chunk 2", "location": "line:5", "start": 50, "end": 60},
            {"text": "chunk 3", "location": "line:10", "start": 100, "end": 110},
        ]
        
        original_locations = [chunk["location"] for chunk in unique_chunks]
        print(f"Input already unique locations: {original_locations}")
        
        # Apply the fix
        result_chunks = ensure_unique_chunk_locations(unique_chunks)
        result_locations = [chunk.location for chunk in result_chunks]
        
        print(f"Output locations: {result_locations}")
        
        # Should be unchanged
        assert original_locations == result_locations, f"Unique chunks were modified: {original_locations} -> {result_locations}"
        
        print("✅ Already unique chunks preserved correctly")
    
    def test_ensure_unique_locations_with_empty_input(self):
        """Test edge case with empty input."""
        result = ensure_unique_chunk_locations([])
        assert result == [], "Empty input should return empty output"
        
        # Note: None input also returns empty list for CocoIndex compatibility
        result = ensure_unique_chunk_locations(None)
        assert result == [], "None input should return empty list"
        
        print("✅ Empty/None input handled correctly")
    
    def test_ensure_unique_locations_with_mixed_formats(self):
        """Test with mixed chunk formats (shouldn't happen in practice but good to test)."""
        mixed_chunks = [
            {"text": "dict chunk", "location": "line:0"},
            Chunk(text="dataclass chunk", location="line:0", start=0, end=10),
        ]
        
        print("Input mixed format chunks:")
        for i, chunk in enumerate(mixed_chunks):
            if isinstance(chunk, dict):
                print(f"  Chunk {i}: dict location='{chunk['location']}'")
            else:
                print(f"  Chunk {i}: dataclass location='{chunk.location}'")
        
        result_chunks = ensure_unique_chunk_locations(mixed_chunks)
        
        print("Output mixed format chunks:")
        locations = []
        for i, chunk in enumerate(result_chunks):
            # All chunks are now Chunk dataclass objects
            location = chunk.location
            print(f"  Chunk {i}: Chunk location='{location}'")
            locations.append(location)
        
        # Should have unique locations
        unique_locations = set(locations)
        assert len(locations) == len(unique_locations), f"Mixed format chunks not made unique: {locations}"
        
        print("✅ Mixed format chunks handled correctly")


if __name__ == "__main__":
    cocoindex.init()
    pytest.main([__file__, "-v", "-s"])
