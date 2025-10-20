#!/usr/bin/env python3

"""
Minimal test to debug the CocoIndex flow with a single file.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, "src")


def test_minimal_flow():
    print("🧪 Testing Minimal CocoIndex Flow")
    print("=" * 50)

    # Create a temporary test file
    test_content = '''#!/usr/bin/env python3

def test_function():
    """A simple test function."""
    return "Hello, World!"

class TestClass:
    """A simple test class."""

    def method(self):
        return 42
'''

    # Create temporary directory and file
    temp_dir = tempfile.mkdtemp(prefix="cocoindex_test_")
    test_file = os.path.join(temp_dir, "test_file.py")

    try:
        # Write test content
        with open(test_file, "w") as f:
            f.write(test_content)

        print(f"📁 Created test file: {test_file}")
        print(f"📊 Content size: {len(test_content)} characters")
        print()

        # Initialize CocoIndex with database connection
        import cocoindex

        # Set database URL if not already set
        if not os.getenv("COCOINDEX_DATABASE_URL"):
            os.environ["COCOINDEX_DATABASE_URL"] = "postgres://cocoindex:cocoindex@host.docker.internal/cocoindex"

        cocoindex.init()

        # Configure CocoIndex for this single file
        from cocoindex_code_mcp_server.cocoindex_config import (
            code_embedding_flow,
            update_flow_config,
        )

        update_flow_config(
            paths=[temp_dir],  # Use the temp directory
            use_default_chunking=False,  # Use AST chunking
            use_default_language_handler=False,  # Use proper language handler
        )

        print("✅ Configured CocoIndex for single file test")
        print()

        # Test the flow directly
        print("🔄 Setting up and running flow...")
        try:
            # Setup the flow first (CRITICAL!)
            code_embedding_flow.setup()
            print("✅ Flow setup completed")

            # Now run the update
            stats = code_embedding_flow.update()
            print(f"📈 Flow stats: {stats}")
            print()

            # Check what was processed
            if hasattr(stats, "updated_rows"):
                print(f"📊 Updated rows: {stats.updated_rows}")
            if hasattr(stats, "source_stats"):
                print(f"📊 Source stats: {stats.source_stats}")

            return True

        except Exception as e:
            print(f"❌ Flow operation failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


if __name__ == "__main__":
    success = test_minimal_flow()
    sys.exit(0 if success else 1)
