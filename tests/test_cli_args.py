#!/usr/bin/env python3

import pytest


class TestCliArguments:
    """Test command-line argument parsing for main_interactive_query.py."""

    def test_default_args(self, mocker):
        """Test that no arguments defaults to cocoindex."""
        mocker.patch('sys.argv', ['main_interactive_query.py'])
        from cocoindex_code_mcp_server.arg_parser_old import parse_args
        args = parse_args()
        assert args.paths == []
        assert args.explicit_paths is None

    def test_single_path_argument(self, mocker):
        """Test single positional path argument."""
        mocker.patch('sys.argv', ['main_interactive_query.py', '/path/to/code'])
        from cocoindex_code_mcp_server.arg_parser_old import parse_args
        args = parse_args()
        assert args.paths == ['/path/to/code']
        assert args.explicit_paths is None

    def test_multiple_path_arguments(self, mocker):
        """Test multiple positional path arguments."""
        mocker.patch('sys.argv', ['main_interactive_query.py', '/path/to/code1', '/path/to/code2'])
        from cocoindex_code_mcp_server.arg_parser_old import parse_args
        args = parse_args()
        assert args.paths == ['/path/to/code1', '/path/to/code2']
        assert args.explicit_paths is None

    def test_explicit_paths_argument(self, mocker):
        """Test --paths argument."""
        mocker.patch('sys.argv', ['main_interactive_query.py', '--paths', '/path/to/code1', '/path/to/code2'])
        from cocoindex_code_mcp_server.arg_parser_old import parse_args
        args = parse_args()
        assert args.paths == []
        assert args.explicit_paths == ['/path/to/code1', '/path/to/code2']

    def test_mixed_arguments(self, mocker):
        """Test both positional and --paths arguments (--paths takes precedence)."""
        mocker.patch('sys.argv', ['main_interactive_query.py', '/positional/path', '--paths', '/explicit/path'])
        from cocoindex_code_mcp_server.arg_parser_old import parse_args
        args = parse_args()
        assert args.paths == ['/positional/path']
        assert args.explicit_paths == ['/explicit/path']

    def test_path_determination_logic(self, mocker):
        """Test the logic for determining which paths to use."""
        from cocoindex_code_mcp_server.arg_parser_old import parse_args

        # Test default (no paths)
        mocker.patch('sys.argv', ['main_interactive_query.py'])
        args = parse_args()

        paths = args.explicit_paths or args.paths
        assert not paths  # Could be None or empty list

        # Test explicit paths take precedence
        mocker.patch('sys.argv', ['main_interactive_query.py', '/pos/path', '--paths', '/exp/path'])
        args = parse_args()

        paths = args.explicit_paths or args.paths
        assert paths == ['/exp/path']

        # Test positional paths when no explicit
        mocker.patch('sys.argv', ['main_interactive_query.py', '/pos/path'])
        args = parse_args()

        paths = args.explicit_paths or args.paths
        assert paths == ['/pos/path']


if __name__ == "__main__":
    pytest.main()
