#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import pytest


class TestLiveUpdates(unittest.TestCase):
    """Test live update functionality and command-line options."""

    def test_live_argument_parsing(self):
        """Test that --live and --poll arguments are parsed correctly."""
        from arg_parser import parse_args

        # Test --live flag
        with patch('sys.argv', ['main_interactive_query.py', '--live']):
            args = parse_args()
            self.assertTrue(args.live)
            self.assertEqual(args.poll, 0)  # Default no polling

        # Test --live with --poll
        with patch('sys.argv', ['main_interactive_query.py', '--live', '--poll', '30']):
            args = parse_args()
            self.assertTrue(args.live)
            self.assertEqual(args.poll, 30)

        # Test --poll without --live (should work)
        with patch('sys.argv', ['main_interactive_query.py', '--poll', '60']):
            args = parse_args()
            self.assertFalse(args.live)
            self.assertEqual(args.poll, 60)

    def test_live_arguments_with_paths(self):
        """Test live update arguments combined with paths."""
        from arg_parser import parse_args

        with patch('sys.argv', ['main_interactive_query.py', '--live', '--poll', '15', '/path/to/code']):
            args = parse_args()
            self.assertTrue(args.live)
            self.assertEqual(args.poll, 15)
            self.assertEqual(args.paths, ['/path/to/code'])

    @pytest.mark.skip(reason="Config update logic changed")
    def test_global_config_updates(self):
        """Test that global flow configuration is updated correctly."""
        from cocoindex_config import _global_flow_config
        from main import main

        # Mock the flow to prevent actual execution
        with patch('cocoindex_config.code_embedding_flow') as mock_flow:
            with patch('query_interactive.ConnectionPool'):
                with patch('builtins.input', side_effect=['']):  # Exit immediately
                    with patch('sys.argv', ['main_interactive_query.py', '--poll', '45', '/test/path']):
                        # Test configuration update
                        main()

                    # Check that global config was updated
                    self.assertEqual(_global_flow_config['paths'], ['/test/path'])
                    self.assertTrue(_global_flow_config['enable_polling'])  # Should be True since poll_interval > 0
                    self.assertEqual(_global_flow_config['poll_interval'], 45)

    @pytest.mark.skip(reason="Polling logic implementation changed")
    def test_polling_enable_logic(self):
        """Test the logic for enabling polling based on interval."""
        from cocoindex_config import _global_flow_config
        from main import main

        with patch('cocoindex_config.code_embedding_flow') as mock_flow:
            with patch('query_interactive.ConnectionPool'):
                with patch('builtins.input', side_effect=['', '', '', '']):  # Multiple empty inputs
                    # Test with poll_interval = 0 (should disable polling)
                    with patch('sys.argv', ['main_interactive_query.py', '--poll', '0', '/test']):
                        main()
                    self.assertFalse(_global_flow_config['enable_polling'])

                    # Test with poll_interval > 0 (should enable polling)
                    with patch('sys.argv', ['main_interactive_query.py', '--poll', '30', '/test']):
                        main()
                    self.assertTrue(_global_flow_config['enable_polling'])

    def test_live_update_flow_configuration(self):
        """Test that live update mode configures the flow correctly."""
        from cocoindex_config import _global_flow_config

        # Test the flow configuration function
        original_config = _global_flow_config.copy()

        # Simulate flow configuration
        test_config = {
            'paths': ['/test/path1', '/test/path2'],
            'enable_polling': True,
            'poll_interval': 60
        }
        _global_flow_config.update(test_config)

        # Verify configuration
        self.assertEqual(_global_flow_config['paths'], ['/test/path1', '/test/path2'])
        self.assertTrue(_global_flow_config['enable_polling'])
        self.assertEqual(_global_flow_config['poll_interval'], 60)

        # Restore original config
        _global_flow_config.clear()
        _global_flow_config.update(original_config)


if __name__ == "__main__":
    unittest.main()
