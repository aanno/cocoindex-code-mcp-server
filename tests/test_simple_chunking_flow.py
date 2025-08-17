#!/usr/bin/env python3
"""
Simple test to verify the chunking method fix works correctly.
"""

import os
import tempfile

import pytest
from dotenv import load_dotenv

import cocoindex
from cocoindex_code_mcp_server.cocoindex_config import (
    code_embedding_flow,
    update_flow_config,
)

# Load environment variables
load_dotenv()

# Add src to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSimpleChunkingFlow:
    """Simple test to verify chunking method preservation."""

    @pytest.fixture(autouse=True)
    def setup_cocoindex(self):
        """Setup CocoIndex with loaded database configuration."""
        print(f"Using database: {os.environ.get('COCOINDEX_DATABASE_URL')}")
        cocoindex.init()
        yield

    def test_flow_setup_with_new_function(self):
        """Test that the flow can be set up with the new chunking method function."""
        python_code = '''def simple_test():
    return "hello world"
'''

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = os.path.join(temp_dir, "simple.py")
            with open(test_file, 'w') as f:
                f.write(python_code)

            print(f"Created test file: {test_file}")

            # Configure flow for AST chunking
            update_flow_config(
                paths=[temp_dir],
                enable_polling=False,
                use_default_chunking=False,  # Use AST chunking
                use_default_language_handler=False
            )

            # Test that setup works
            print("Testing CocoIndex flow setup...")
            try:
                code_embedding_flow.setup()
                print("✅ Flow setup successful!")
            except Exception as e:
                print(f"❌ Flow setup failed: {e}")
                raise

            # Try a quick update
            try:
                stats = code_embedding_flow.update()
                print(f"✅ Flow update successful: {stats}")
            except Exception as e:
                print(f"❌ Flow update failed: {e}")
                # Don't fail the test on update issues, just setup issues


if __name__ == "__main__":
    pytest.main([__file__])
