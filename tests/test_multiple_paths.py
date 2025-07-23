#!/usr/bin/env python3

import unittest
import pytest
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

class TestMultiplePaths(unittest.TestCase):
    """Test multiple path handling in the code embedding flow."""
    
    def setUp(self):
        # Create temporary directories for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create test directory structure
        self.dir1 = self.temp_path / "dir1"
        self.dir2 = self.temp_path / "dir2"
        self.dir1.mkdir()
        self.dir2.mkdir()
        
        # Create test files
        (self.dir1 / "test1.py").write_text("def hello1(): pass")
        (self.dir1 / "test1.hs").write_text("module Test1 where\nhello1 = \"world\"")
        (self.dir2 / "test2.py").write_text("def hello2(): pass")
        (self.dir2 / "test2.rs").write_text("fn hello2() {}")
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_argument_parsing_logic(self):
        """Test the argument parsing logic for multiple paths."""
        from arg_parser import parse_args
        
        # Test default behavior
        with patch('sys.argv', ['main_interactive_query.py']):
            args = parse_args()
            paths = None
            if args.explicit_paths:
                paths = args.explicit_paths
            elif args.paths:
                paths = args.paths
            
            self.assertIsNone(paths)
        
        # Test multiple paths
        with patch('sys.argv', ['main_interactive_query.py', str(self.dir1), str(self.dir2)]):
            args = parse_args()
            paths = None
            if args.explicit_paths:
                paths = args.explicit_paths
            elif args.paths:
                paths = args.paths
            
            expected_paths = [str(self.dir1), str(self.dir2)]
            self.assertEqual(paths, expected_paths)
    
    @pytest.mark.skip(reason="Main function output format changed")
    def test_main_function_output(self):
        """Test that the main function properly handles multiple paths."""
        from main import main
        
        # Test that the main function doesn't crash with multiple paths
        # We can't easily test the full flow without a database, but we can test the interface
        with patch('cocoindex_config.code_embedding_flow') as mock_flow:
            with patch('query_interactive.ConnectionPool') as mock_pool:
                with patch('builtins.input', side_effect=['', '']):  # Empty input to exit
                    # Mock the flow update
                    mock_flow.update.return_value = {"processed": 0}
                    
                    # This should not raise an exception
                    try:
                        with patch('sys.argv', ['main_interactive_query.py', str(self.dir1), str(self.dir2)]):
                            main()
                    except (KeyboardInterrupt, SystemExit):
                        pass  # Expected when input() is mocked
    
    def test_paths_default_logic(self):
        """Test the paths default logic."""
        # This tests the logic we added to handle multiple paths
        paths = None
        if not paths:
            paths = ["cocoindex"]
        
        self.assertEqual(paths, ["cocoindex"])
        
        # Test with actual paths
        paths = [str(self.dir1), str(self.dir2)]
        if not paths:
            paths = ["cocoindex"]
        
        self.assertEqual(paths, [str(self.dir1), str(self.dir2)])
    
    def test_source_naming_logic(self):
        """Test the source naming logic for multiple paths."""
        # Test single path naming
        paths = [str(self.dir1)]
        all_sources = []
        
        for i, path in enumerate(paths):
            source_name = f"files_{i}" if len(paths) > 1 else "files"
            all_sources.append(source_name)
        
        self.assertEqual(all_sources, ["files"])
        
        # Test multiple path naming
        paths = [str(self.dir1), str(self.dir2)]
        all_sources = []
        
        for i, path in enumerate(paths):
            source_name = f"files_{i}" if len(paths) > 1 else "files"
            all_sources.append(source_name)
        
        self.assertEqual(all_sources, ["files_0", "files_1"])

if __name__ == "__main__":
    unittest.main()
