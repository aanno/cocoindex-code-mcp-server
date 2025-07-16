#!/usr/bin/env python3

import unittest
import sys
from unittest.mock import patch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

class TestCliArguments(unittest.TestCase):
    """Test command-line argument parsing for main.py."""
    
    def test_default_args(self):
        """Test that no arguments defaults to cocoindex."""
        with patch('sys.argv', ['main.py']):
            from main import parse_args
            args = parse_args()
            self.assertEqual(args.paths, [])
            self.assertIsNone(args.explicit_paths)
    
    def test_single_path_argument(self):
        """Test single positional path argument."""
        with patch('sys.argv', ['main.py', '/path/to/code']):
            from main import parse_args
            args = parse_args()
            self.assertEqual(args.paths, ['/path/to/code'])
            self.assertIsNone(args.explicit_paths)
    
    def test_multiple_path_arguments(self):
        """Test multiple positional path arguments."""
        with patch('sys.argv', ['main.py', '/path/to/code1', '/path/to/code2']):
            from main import parse_args
            args = parse_args()
            self.assertEqual(args.paths, ['/path/to/code1', '/path/to/code2'])
            self.assertIsNone(args.explicit_paths)
    
    def test_explicit_paths_argument(self):
        """Test --paths argument."""
        with patch('sys.argv', ['main.py', '--paths', '/path/to/code1', '/path/to/code2']):
            from main import parse_args
            args = parse_args()
            self.assertEqual(args.paths, [])
            self.assertEqual(args.explicit_paths, ['/path/to/code1', '/path/to/code2'])
    
    def test_mixed_arguments(self):
        """Test both positional and --paths arguments (--paths takes precedence)."""
        with patch('sys.argv', ['main.py', '/positional/path', '--paths', '/explicit/path']):
            from main import parse_args
            args = parse_args()
            self.assertEqual(args.paths, ['/positional/path'])
            self.assertEqual(args.explicit_paths, ['/explicit/path'])
    
    def test_path_determination_logic(self):
        """Test the logic for determining which paths to use."""
        # Test default (no paths)
        with patch('sys.argv', ['main.py']):
            from main import parse_args
            args = parse_args()
            
            paths = None
            if args.explicit_paths:
                paths = args.explicit_paths
            elif args.paths:
                paths = args.paths
            
            self.assertIsNone(paths)
        
        # Test explicit paths take precedence
        with patch('sys.argv', ['main.py', '/pos/path', '--paths', '/exp/path']):
            from main import parse_args
            args = parse_args()
            
            paths = None
            if args.explicit_paths:
                paths = args.explicit_paths
            elif args.paths:
                paths = args.paths
            
            self.assertEqual(paths, ['/exp/path'])
        
        # Test positional paths when no explicit
        with patch('sys.argv', ['main.py', '/pos/path']):
            from main import parse_args
            args = parse_args()
            
            paths = None
            if args.explicit_paths:
                paths = args.explicit_paths
            elif args.paths:
                paths = args.paths
            
            self.assertEqual(paths, ['/pos/path'])

if __name__ == "__main__":
    unittest.main()