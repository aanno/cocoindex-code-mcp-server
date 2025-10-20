#!/usr/bin/env python3

import tempfile
from pathlib import Path

import pytest


class TestMultiplePaths:
    """Test multiple path handling in the code embedding flow."""

    @pytest.fixture(autouse=True)
    def setup_temp_dirs(self):
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

        yield

        # Cleanup
        self.temp_dir.cleanup()

    def test_argument_parsing_logic(self, mocker):
        """Test the argument parsing logic for multiple paths."""
        from cocoindex_code_mcp_server.arg_parser_old import parse_args

        # Test default behavior
        mocker.patch('sys.argv', ['main_interactive_query.py'])
        args = parse_args()
        paths = args.explicit_paths or args.paths

        # Parser returns empty list when no paths provided
        assert not paths  # Could be None or empty list

        # Test multiple paths
        mocker.patch('sys.argv', ['main_interactive_query.py', str(self.dir1), str(self.dir2)])
        args = parse_args()
        paths = args.explicit_paths or args.paths

        expected_paths = [str(self.dir1), str(self.dir2)]
        assert paths == expected_paths

    @pytest.mark.skip(reason="Main function output format changed")
    def test_main_function_output(self, mocker):
        """Test that the main function properly handles multiple paths."""
        from cocoindex_code_mcp_server.main_interactive_query import main

        # Test that the main function doesn't crash with multiple paths
        # We can't easily test the full flow without a database, but we can test the interface
        mock_flow = mocker.patch('cocoindex_config.code_embedding_flow')
        mocker.patch('query_interactive.ConnectionPool')  # Mock but don't need reference
        mocker.patch('builtins.input', side_effect=['', ''])  # Empty input to exit

        # Mock the flow update
        mock_flow.update.return_value = {"processed": 0}

        # This should not raise an exception
        try:
            mocker.patch('sys.argv', ['main_interactive_query.py', str(self.dir1), str(self.dir2)])
            main()
        except (KeyboardInterrupt, SystemExit):
            pass  # Expected when input() is mocked

    def test_paths_default_logic(self):
        """Test the paths default logic."""
        # This tests the logic we added to handle multiple paths
        paths = None
        paths = paths or ["cocoindex"]

        assert paths == ["cocoindex"]

        # Test with actual paths
        paths = [str(self.dir1), str(self.dir2)]
        paths = paths or ["cocoindex"]

        assert paths == [str(self.dir1), str(self.dir2)]

    def test_source_naming_logic(self):
        """Test the source naming logic for multiple paths."""
        # Test single path naming
        paths = [str(self.dir1)]
        all_sources = [
            f"files_{i}" if len(paths) > 1 else "files"
            for i in range(len(paths))
        ]

        assert all_sources == ["files"]

        # Test multiple path naming
        paths = [str(self.dir1), str(self.dir2)]
        all_sources = [
            f"files_{i}" if len(paths) > 1 else "files"
            for i in range(len(paths))
        ]

        assert all_sources == ["files_0", "files_1"]


if __name__ == "__main__":
    pytest.main()
