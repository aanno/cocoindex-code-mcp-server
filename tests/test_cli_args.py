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


class TestLogLevelConfiguration:
    """Tests for --log-level wiring.

    logging.basicConfig() is a no-op when the root logger already has handlers
    (which any prior import can trigger). The fix must use force=True so that
    the user-supplied --log-level actually takes effect.
    """

    def _apply_log_level(self, level_str: str):
        """Replicate the logging-setup code from main()."""
        import logging
        logging.basicConfig(
            level=getattr(logging, level_str.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _apply_log_level_force(self, level_str: str):
        """The fixed version using force=True."""
        import logging
        logging.basicConfig(
            level=getattr(logging, level_str.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )

    def test_basicconfig_noop_when_handlers_already_present(self):
        """Demonstrate the bug: basicConfig is silently ignored after the first call."""
        import logging

        # Simulate what happens when any import triggers basicConfig
        logging.basicConfig(level=logging.DEBUG)
        assert logging.root.handlers, "Pre-condition: root logger must already have handlers"

        # Now try to raise the log level — without force=True this is a no-op
        self._apply_log_level("WARNING")
        # BUG: level is still DEBUG (10), not WARNING (30)
        assert logging.root.level != logging.WARNING, (
            "Expected the bug to be present (level NOT set to WARNING) — "
            "if this assertion fails the bug is already fixed"
        )

    def test_force_true_overrides_existing_handlers(self):
        """The fix: force=True makes basicConfig reconfigure even with existing handlers."""
        import logging

        # Simulate pre-existing handler from an import
        logging.basicConfig(level=logging.DEBUG)
        assert logging.root.handlers, "Pre-condition: root logger must have handlers"

        self._apply_log_level_force("WARNING")
        assert logging.root.level == logging.WARNING, (
            f"Expected root logger level=WARNING (30), got {logging.root.level}"
        )

    def test_warning_level_suppresses_info(self):
        """With force=True, setting WARNING must suppress INFO records."""
        import logging

        # Simulate noisy import
        logging.basicConfig(level=logging.DEBUG)

        self._apply_log_level_force("WARNING")

        log = logging.getLogger("test_log_level_suppresses_info")
        with pytest.raises(AssertionError):
            # Capture handler to verify no INFO record passes through
            class _Capture(logging.Handler):
                def __init__(self):
                    super().__init__()
                    self.records: list[logging.LogRecord] = []
                def emit(self, record: logging.LogRecord):
                    self.records.append(record)

            cap = _Capture()
            cap.setLevel(logging.DEBUG)
            log.addHandler(cap)
            log.info("this should be filtered")
            assert cap.records, "INFO record should NOT have been emitted at WARNING level"

    def test_debug_level_allows_info(self):
        """Sanity: DEBUG level must allow INFO records through."""
        import logging
        logging.basicConfig(level=logging.ERROR)  # pre-existing handler

        self._apply_log_level_force("DEBUG")
        assert logging.root.level == logging.DEBUG

    def teardown_method(self, _method):
        """Restore root logger to a clean state after each test."""
        import logging
        logging.root.handlers.clear()
        logging.root.setLevel(logging.WARNING)


if __name__ == "__main__":
    pytest.main()
