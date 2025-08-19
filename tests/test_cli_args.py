#!/usr/bin/env python3

import pytest


class TestCliArguments:
    """Test command-line argument parsing for main_interactive_query.py."""

    def test_default_args(self):
        """Test that no arguments defaults to cocoindex."""
        with patch('sys.argv', ['main_interactive_query.py']):
            from cocoindex_code_mcp_server.arg_parser_old import parse_args
            args = parse_args()
            assert args.paths == []
            assert args.explicit_paths is None

    def test_single_path_argument(self):
        """Test single positional path argument."""
        with patch('sys.argv', ['main_interactive_query.py', '/path/to/code']):
            from cocoindex_code_mcp_server.arg_parser_old import parse_args
            args = parse_args()
            assert args.paths == ['/path/to/code']
            assert args.explicit_paths is None

    def test_multiple_path_arguments(self):
        """Test multiple positional path arguments."""
        with patch('sys.argv', ['main_interactive_query.py', '/path/to/code1', '/path/to/code2']):
            from cocoindex_code_mcp_server.arg_parser_old import parse_args
            args = parse_args()
            assert args.paths == ['/path/to/code1', '/path/to/code2']
            assert args.explicit_paths is None

    def test_explicit_paths_argument(self):
        """Test --paths argument."""
        with patch('sys.argv', ['main_interactive_query.py', '--paths', '/path/to/code1', '/path/to/code2']):
            from cocoindex_code_mcp_server.arg_parser_old import parse_args
            args = parse_args()
            assert args.paths == []
            assert args.explicit_paths == ['/path/to/code1', '/path/to/code2']

    def test_mixed_arguments(self):
        """Test both positional and --paths arguments (--paths takes precedence)."""
        with patch('sys.argv', ['main_interactive_query.py', '/positional/path', '--paths', '/explicit/path']):
            from cocoindex_code_mcp_server.arg_parser_old import parse_args
            args = parse_args()
            assert args.paths == ['/positional/path']
            assert args.explicit_paths == ['/explicit/path']

    def test_path_determination_logic(self):
        """Test the logic for determining which paths to use."""
        # Test default (no paths)
        with patch('sys.argv', ['main_interactive_query.py']):
            from cocoindex_code_mcp_server.arg_parser_old import parse_args
            args = parse_args()

            paths = None
            if args.explicit_paths:
                paths = args.explicit_paths
            elif args.paths:
                paths = args.paths

            assert paths is None

        # Test explicit paths take precedence
        with patch('sys.argv', ['main_interactive_query.py', '/pos/path', '--paths', '/exp/path']):
            from cocoindex_code_mcp_server.arg_parser_old import parse_args
            args = parse_args()

            paths = None
            if args.explicit_paths:
                paths = args.explicit_paths
            elif args.paths:
                paths = args.paths

            assert paths == ['/exp/path']

        # Test positional paths when no explicit
        with patch('sys.argv', ['main_interactive_query.py', '/pos/path']):
            from cocoindex_code_mcp_server.arg_parser_old import parse_args
            args = parse_args()

            paths = None
            if args.explicit_paths:
                paths = args.explicit_paths
            elif args.paths:
                paths = args.paths

            assert paths == ['/pos/path']


if __name__ == "__main__":
    pytest.main()
