#!/usr/bin/env python3
"""
Direct CocoIndex flow test to identify where chunking methods get lost.
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


class TestCocoIndexFlowDirect:
    """Test CocoIndex flow directly to identify chunking method issues."""

    @pytest.fixture(autouse=True)
    def setup_cocoindex(self):
        """Setup CocoIndex with loaded database configuration."""
        # Use the database URL from .env file (already loaded)
        print(f"Using database: {os.environ.get('COCOINDEX_DATABASE_URL')}")

        # Initialize CocoIndex
        cocoindex.init()

        # Import the flow to register it
        from cocoindex_code_mcp_server.cocoindex_config import code_embedding_flow

        yield

    def test_flow_with_single_python_file(self):
        """Test flow with a single Python file to see what chunking method is used."""
        python_code = '''def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """Simple calculator."""
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
'''

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = os.path.join(temp_dir, "fibonacci.py")
            with open(test_file, 'w') as f:
                f.write(python_code)

            print(f"Created test file: {test_file}")

            # Update flow configuration
            update_flow_config(
                paths=[temp_dir],
                enable_polling=False,
                use_default_chunking=False,  # Use AST chunking
                use_default_language_handler=False
            )

            # Setup flow
            print("Setting up CocoIndex flow...")
            code_embedding_flow.setup()

            # Run flow
            print("Running CocoIndex flow...")
            stats = code_embedding_flow.update()

            print(f"Flow statistics: {stats}")

            # Try to inspect what was created
            # In a real test, we'd need to query the database or access internal structures
            # For now, let's just verify the flow completed without errors
            assert stats is not None, "Flow should return statistics"

            print("✅ Flow completed successfully")

    @pytest.mark.xfail(reason="CocoIndex flow configuration issue: Flow instance not found for Java files")
    def test_flow_with_java_file(self):
        """Test flow with a Java file."""
        java_code = '''public class Fibonacci {
    /**
     * Calculate fibonacci number
     */
    public static int fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n-1) + fibonacci(n-2);
    }

    /**
     * Main method
     */
    public static void main(String[] args) {
        System.out.println("Fibonacci(10) = " + fibonacci(10));
    }
}'''

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = os.path.join(temp_dir, "Fibonacci.java")
            with open(test_file, 'w') as f:
                f.write(java_code)

            print(f"Created test file: {test_file}")

            # Update flow configuration
            update_flow_config(
                paths=[temp_dir],
                enable_polling=False,
                use_default_chunking=False,  # Use AST chunking
                use_default_language_handler=False
            )

            # Setup and run flow
            print("Setting up and running CocoIndex flow...")
            code_embedding_flow.setup()
            stats = code_embedding_flow.update()

            print(f"Flow statistics: {stats}")
            assert stats is not None, "Flow should return statistics"

            print("✅ Java flow completed successfully")

    def test_flow_configuration_check(self):
        """Test that flow configuration is correct."""
        from cocoindex_code_mcp_server.cocoindex_config import (
            AST_CHUNKING_AVAILABLE,
            ASTChunkOperation,
            _global_flow_config,
        )

        print("=== Flow Configuration Check ===")
        print(f"AST_CHUNKING_AVAILABLE: {AST_CHUNKING_AVAILABLE}")
        print(f"ASTChunkOperation: {ASTChunkOperation}")
        print(f"Global flow config: {_global_flow_config}")

        # Update config to use AST chunking
        update_flow_config(
            paths=["/tmp/test"],
            use_default_chunking=False,
            use_default_language_handler=False
        )

        print(f"Updated global flow config: {_global_flow_config}")

        # Verify AST chunking should be used
        use_default_chunking = _global_flow_config.get('use_default_chunking', False)
        assert not use_default_chunking, "Should not use default chunking"
        assert AST_CHUNKING_AVAILABLE, "AST chunking should be available"
        assert ASTChunkOperation is not None, "ASTChunkOperation should not be None"

        print("✅ Configuration is correct for AST chunking")


if __name__ == "__main__":
    pytest.main([__file__])
