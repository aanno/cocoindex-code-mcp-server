#!/usr/bin/env python3

"""
Common test helpers for CocoIndex MCP server testing.

This module provides shared functionality for test fixture processing,
result comparison, and test result saving.
"""

import datetime
import json
import logging
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Load environment variables at module level
from dotenv import load_dotenv



load_dotenv()

# CocoIndex and MCP infrastructure imports
try:
    import cocoindex
    from cocoindex_code_mcp_server.backends import BackendFactory, VectorStoreBackend
    from cocoindex_code_mcp_server.cocoindex_config import (
        run_flow_update,
        update_flow_config,
    )
    from cocoindex_code_mcp_server.db.pgvector.hybrid_search import HybridSearchEngine
    from cocoindex_code_mcp_server.keyword_search_parser_lark import KeywordSearchParser
    from cocoindex_code_mcp_server.main_mcp_server import safe_embedding_function
    COCOINDEX_AVAILABLE = True
except ImportError as e:
    logging.warning(f"CocoIndex infrastructure not available: {e}")
    COCOINDEX_AVAILABLE = False


def generate_test_timestamp() -> str:
    """Generate a timestamp for test run identification."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds


def copy_directory_structure(source_dir: Path, target_dir: Path) -> None:
    """
    Copy complete directory structure from source to target.

    Args:
        source_dir: Source directory path
        target_dir: Target directory path
    """
    # Ensure target directory exists
    target_dir.mkdir(exist_ok=True)

    print(f"📁 Copying directory structure from {source_dir} to {target_dir}...")

    if source_dir.exists():
        # Remove existing content in target directory first
        if target_dir.exists():
            shutil.rmtree(target_dir)

        # Copy the entire directory structure
        shutil.copytree(source_dir, target_dir)
        print("  ✅ Copied complete directory structure")
    else:
        print(f"  ❌ Source directory {source_dir} does not exist")


def copy_test_files_legacy(source_dir: Path, target_dir: Path, test_files: List[str]) -> None:
    """
    Legacy method to copy individual test files.

    Args:
        source_dir: Source directory path
        target_dir: Target directory path
        test_files: List of filenames to copy
    """
    # Ensure target directory exists
    target_dir.mkdir(exist_ok=True)

    print(f"📁 Copying test fixtures from {source_dir} to {target_dir}...")
    for test_file in test_files:
        src = source_dir / test_file
        dst = target_dir / test_file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✅ Copied {test_file}")
        else:
            print(f"  ⚠️  Warning: {test_file} not found in source directory")


def parse_jsonc_file(file_path: Path) -> Dict[str, Any]:
    """
    Parse a JSONC (JSON with comments) file.

    Args:
        file_path: Path to the JSONC file

    Returns:
        Parsed JSON data as dictionary

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the JSON is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"JSONC file not found: {file_path}")

    # Read file content
    fixture_content = file_path.read_text()

    # Remove comments for JSON parsing
    lines = []
    for line in fixture_content.split('\n'):
        stripped = line.strip()
        if not stripped.startswith('//'):
            lines.append(line)

    clean_json = '\n'.join(lines)
    return json.loads(clean_json)


def save_search_results(
    test_name: str,
    query: Dict[str, Any],
    search_data: Dict[str, Any],
    run_timestamp: str,
    results_base_dir: str = "/workspaces/rust/test-results"
) -> None:
    """
    Save search results to test-results directory with unique naming.

    Args:
        test_name: Name of the test
        query: The search query that was executed
        search_data: The search results data
        run_timestamp: Timestamp for consistent naming across test run
        results_base_dir: Base directory for test results
    """
    # Use the provided run timestamp for consistent naming across the test run
    filename = f"{test_name}_{run_timestamp}.json"

    # Ensure directory exists
    results_dir = os.path.join(results_base_dir, "search-hybrid")
    os.makedirs(results_dir, exist_ok=True)

    # Prepare complete result data
    result_data = {
        "test_name": test_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "search_results": search_data
    }

    # Save to file with proper JSON serialization
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"💾 Saved search results: {filepath}")


def compare_expected_vs_actual(
    expected_item: Dict[str, Any],
    result_item: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Compare expected metadata with actual result metadata.

    Args:
        expected_item: Expected result specification
        result_item: Actual search result item

    Returns:
        Tuple of (match_found, list_of_errors)
    """
    errors = []

    # Check filename pattern if specified
    if "filename_pattern" in expected_item:
        pattern = expected_item["filename_pattern"]
        filename = result_item.get("filename", "")
        if not re.match(pattern, filename):
            return False, [f"Filename '{filename}' does not match pattern '{pattern}'"]

    # Check expected metadata
    if "expected_metadata" in expected_item:
        metadata_errors = []
        expected_metadata = expected_item["expected_metadata"]

        # Get metadata from both flattened fields and metadata_json
        combined_metadata = dict(result_item)
        if "metadata_json" in result_item and isinstance(result_item["metadata_json"], dict):
            combined_metadata.update(result_item["metadata_json"])

        for field, expected_value in expected_metadata.items():
            actual_value = combined_metadata.get(field)

            # Handle special comparison operators
            if isinstance(expected_value, str):
                if expected_value == "!empty":
                    # Not empty check
                    if not actual_value or (isinstance(actual_value, list) and len(actual_value) == 0):
                        metadata_errors.append(f"{field}: expected non-empty, got '{actual_value}'")
                elif expected_value.startswith("!"):
                    # Not equal comparison
                    not_expected = expected_value[1:]
                    if str(actual_value) == not_expected:
                        metadata_errors.append(f"{field}: expected not '{not_expected}', got '{actual_value}'")
                elif expected_value.startswith(">"):
                    # Greater than comparison
                    try:
                        threshold = float(expected_value[1:])
                        if not (isinstance(actual_value, (int, float)) and actual_value > threshold):
                            metadata_errors.append(f"{field}: expected > {threshold}, got '{actual_value}'")
                    except ValueError:
                        metadata_errors.append(f"{field}: invalid threshold '{expected_value}'")
                else:
                    # Direct equality
                    if str(actual_value) != expected_value:
                        metadata_errors.append(f"{field}: expected '{expected_value}', got '{actual_value}'")

            elif isinstance(expected_value, bool):
                if actual_value != expected_value:
                    metadata_errors.append(f"{field}: expected {expected_value}, got {actual_value}")
            elif isinstance(expected_value, list):
                if actual_value != expected_value:
                    metadata_errors.append(f"{field}: expected {expected_value}, got {actual_value}")

        if metadata_errors:
            errors.extend(metadata_errors)

    # Check should_not_be_empty fields
    if "should_not_be_empty" in expected_item:
        # Get metadata from both flattened fields and metadata_json
        combined_metadata = dict(result_item)
        if "metadata_json" in result_item and isinstance(result_item["metadata_json"], dict):
            combined_metadata.update(result_item["metadata_json"])

        empty_fields = []
        for field in expected_item["should_not_be_empty"]:
            field_value = combined_metadata.get(field)
            if not field_value or (isinstance(field_value, list) and len(field_value) == 0):
                empty_fields.append(field)

        if empty_fields:
            errors.append(f"Fields should not be empty: {empty_fields}")

    return len(errors) == 0, errors


def validate_search_results(
    test_cases: List[Dict[str, Any]],
    execute_search_func,
    run_timestamp: str
) -> List[Dict[str, Any]]:
    """
    Validate search results against expected outcomes.

    Args:
        test_cases: List of test case definitions
        execute_search_func: Async function to execute search queries
        run_timestamp: Timestamp for result saving

    Returns:
        List of failed test cases with error details
    """
    failed_tests = []

    for test_case in test_cases:
        test_name = test_case["name"]
        description = test_case["description"]
        query = test_case["query"]
        expected_results = test_case["expected_results"]

        logging.info(f"Running hybrid search test: {test_name}")
        logging.info(f"Description: {description}")

        try:
            # Execute search (this should be provided by caller)
            search_data = execute_search_func(query)

            results = search_data.get("results", [])
            total_results = len(results)

            # Save search results to test-results directory
            save_search_results(test_name, query, search_data, run_timestamp)

            # Check minimum results requirement
            min_results = expected_results.get("min_results", 1)
            if total_results < min_results:
                failed_tests.append({
                    "test": test_name,
                    "error": f"Expected at least {min_results} results, got {total_results}",
                    "query": query
                })
                continue

            # Check expected results
            if "should_contain" in expected_results:
                for expected_item in expected_results["should_contain"]:
                    found_match = False

                    for result_item in results:
                        match_found, errors = compare_expected_vs_actual(expected_item, result_item)
                        if match_found:
                            found_match = True
                            break

                    if not found_match:
                        failed_tests.append({
                            "test": test_name,
                            "error": f"No matching result found for expected item: {expected_item}",
                            "query": query,
                            "actual_results": [{
                                "filename": r.get("filename"),
                                "metadata_summary": {
                                    "classes": r.get("classes", []),
                                    "functions": r.get("functions", []),
                                    "imports": r.get("imports", []),
                                    "analysis_method": r.get("metadata_json", {}).get("analysis_method", "unknown")
                                }
                            } for r in results[:3]]  # Show first 3 results for debugging
                        })

        except Exception as e:
            failed_tests.append({
                "test": test_name,
                "error": f"Test execution failed: {str(e)}",
                "query": query
            })

    return failed_tests


def clear_test_tables(test_type: Optional[str] = None) -> None:
    """
    Clear embeddings and tracking tables for integration tests.

    This is critical for forcing CocoIndex to re-index files after code changes.
    Without clearing tracking tables, CocoIndex will skip unchanged files even
    if embeddings tables are empty.

    Args:
        test_type: Specific test type to clear ('keyword', 'vector', 'hybrid'),
                  or None to clear all test tables
    """
    import psycopg

    database_url = os.getenv("DATABASE_URL") or os.getenv("COCOINDEX_DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL or COCOINDEX_DATABASE_URL not found in environment")

    # Define table mappings
    embeddings_tables = {
        'keyword': 'keywordsearchtest_code_embeddings',
        'vector': 'vectorsearchtest_code_embeddings',
        'hybrid': 'hybridsearchtest_code_embeddings'
    }

    tracking_tables = {
        'keyword': 'searchtest_keyword__cocoindex_tracking',
        'vector': 'searchtest_vector__cocoindex_tracking',
        'hybrid': 'searchtest_hybrid__cocoindex_tracking'
    }

    # Determine which tables to clear
    if test_type:
        if test_type not in embeddings_tables:
            raise ValueError(f"Unknown test type: {test_type}. Must be one of {list(embeddings_tables.keys())}")
        # Only clear tables for this specific test type
        tables_to_clear = {
            'embeddings': [embeddings_tables[test_type]],
            'tracking': [tracking_tables[test_type]]
        }
        logging.info(f"📋 Clearing ONLY {test_type} test tables (not affecting other test types)")
    else:
        # Clear all test tables (only used for manual cleanup, not by tests)
        tables_to_clear = {
            'embeddings': list(embeddings_tables.values()),
            'tracking': list(tracking_tables.values())
        }
        logging.info(f"📋 Clearing ALL test tables (keyword, vector, hybrid)")

    # Clear tables using SQL TRUNCATE (faster and resets auto-increment)
    conn = psycopg.connect(database_url)
    cur = conn.cursor()

    try:
        # Clear embeddings tables
        for table in tables_to_clear['embeddings']:
            # Check if table exists first
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = %s
                );
            """, (table,))
            if cur.fetchone()[0]:
                # Get count before truncating (for logging)
                cur.execute(f"SELECT COUNT(*) FROM {table};")
                count = cur.fetchone()[0]
                # TRUNCATE is faster than DELETE and resets auto-increment
                cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")
                logging.info(f"✅ Truncated {table} ({count} records removed)")
            else:
                logging.info(f"⚠️  Table {table} does not exist, skipping")

        # Clear tracking tables (critical for re-indexing!)
        for table in tables_to_clear['tracking']:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = %s
                );
            """, (table,))
            if cur.fetchone()[0]:
                # Get count before truncating (for logging)
                cur.execute(f"SELECT COUNT(*) FROM {table};")
                count = cur.fetchone()[0]
                # TRUNCATE is faster than DELETE and resets auto-increment
                cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")
                logging.info(f"✅ Truncated {table} ({count} records removed)")
            else:
                logging.info(f"⚠️  Table {table} does not exist, skipping")

        conn.commit()
        logging.info("✅ Database cleared - ready for fresh indexing")

    except Exception as e:
        conn.rollback()
        logging.error(f"❌ Failed to clear tables: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def format_test_failure_report(failed_tests: List[Dict[str, Any]]) -> str:
    """
    Format a comprehensive failure report for failed tests.

    Args:
        failed_tests: List of failed test dictionaries

    Returns:
        Formatted error message string
    """
    if not failed_tests:
        return ""

    error_msg = f"Hybrid search validation failed for {len(failed_tests)} test(s):\n"

    for failure in failed_tests:
        error_msg += f"\n  Test: {failure['test']}\n"
        error_msg += f"  Query: {failure['query']}\n"
        error_msg += f"  Error: {failure['error']}\n"
        if "actual_results" in failure:
            error_msg += f"  Sample Results: {json.dumps(failure['actual_results'], indent=2)}\n"

    return error_msg


# CocoIndex Infrastructure Setup
class CocoIndexTestInfrastructure:
    """
    Test infrastructure for running CocoIndex tests directly without integration server.

    This class sets up the complete CocoIndex infrastructure including:
    - Flow configuration and updates
    - Database backend initialization
    - Search engine setup
    - Background processes
    """

    def __init__(
        self,
        paths: Optional[List[str]] = None,
        default_embedding: bool = False,
        default_chunking: bool = False,
        default_language_handler: bool = False,
        chunk_factor_percent: int = 100,
        enable_polling: bool = False,
        poll_interval: int = 30,
        test_type: Optional[str] = None
    ):
        if not COCOINDEX_AVAILABLE:
            raise RuntimeError("CocoIndex infrastructure not available. Check imports.")

        self.paths = paths or ["tmp"]  # Default to tmp directory for tests
        self.default_embedding = default_embedding
        self.default_chunking = default_chunking
        self.default_language_handler = default_language_handler
        self.chunk_factor_percent = chunk_factor_percent
        self.enable_polling = enable_polling
        self.poll_interval = poll_interval
        self.test_type = test_type  # 'keyword', 'vector', 'hybrid', or None for main flow

        # Infrastructure components
        self.hybrid_search_engine: Optional[HybridSearchEngine] = None
        self.backend: Optional[VectorStoreBackend] = None
        self.shutdown_event = threading.Event()
        self.background_thread: Optional[threading.Thread] = None
        self.flow_def = None
        self.table_name: Optional[str] = None

        # Logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def setup(self) -> None:
        """Set up the complete CocoIndex infrastructure."""
        try:
            self.logger.info("🚀 Setting up CocoIndex test infrastructure...")

            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv()

            # Initialize CocoIndex library with database settings
            database_url = os.getenv("DATABASE_URL") or os.getenv("COCOINDEX_DATABASE_URL")
            if not database_url:
                raise ValueError("DATABASE_URL or COCOINDEX_DATABASE_URL not found in environment")

            # Set COCOINDEX_DATABASE_URL for CocoIndex if not already set
            if not os.getenv("COCOINDEX_DATABASE_URL"):
                os.environ["COCOINDEX_DATABASE_URL"] = database_url

            cocoindex.init()
            self.logger.info("✅ CocoIndex library initialized with database")

            # Clear test tables to force re-indexing (critical for fresh test data)
            if self.test_type:
                self.logger.info(f"🗑️  Clearing {self.test_type} test tables for fresh indexing...")
                clear_test_tables(self.test_type)

            if self.test_type:
                # Use parameterized test flows for isolation
                from .search_test_flows import get_search_test_flow, get_test_table_name
                
                self.flow_def = get_search_test_flow(self.test_type)
                self.table_name = get_test_table_name(self.test_type)
                self.logger.info(f"🔧 Using {self.test_type} test flow with table: {self.table_name}")
            else:
                # Use the main flow
                from cocoindex_code_mcp_server.cocoindex_config import code_embedding_flow
                self.flow_def = code_embedding_flow
                # Use default table name
                from .cocoindex_util import get_default_db_name
                self.table_name = get_default_db_name()
                self.logger.info("🔧 Using main CodeEmbedding flow")
                
            # Update flow configuration (only for main flow)
            if not self.test_type:
                from cocoindex_code_mcp_server.cocoindex_config import update_flow_config
                update_flow_config(
                    paths=self.paths,
                    enable_polling=self.enable_polling,
                    poll_interval=self.poll_interval,
                    use_default_embedding=self.default_embedding,
                    use_default_chunking=self.default_chunking,
                    use_default_language_handler=self.default_language_handler,
                    chunk_factor_percent=self.chunk_factor_percent
                )

            # Log configuration
            self.logger.info(f"📁 Paths: {self.paths}")
            self.logger.info(f"🔴 Live updates: {'ENABLED' if self.enable_polling else 'DISABLED'}")
            if self.enable_polling:
                self.logger.info(f"⏰ Polling interval: {self.poll_interval} seconds")
            if self.chunk_factor_percent != 100:
                self.logger.info(f"📏 Chunk size scaling: {self.chunk_factor_percent}%")

            # Run initial flow update to process files
            self.logger.info("🔄 Running initial flow update...")
            self.logger.info(f"🔧 Setting up flow...")
            self.flow_def.setup()
            self.logger.info("✅ Flow setup completed")
            
            self.logger.info("🔄 Running flow update...")
            stats = self.flow_def.update()
            self.logger.info(f"📊 Flow update stats: {stats}")
            self.logger.info("✅ Flow update completed")
            self.logger.info("📚 CocoIndex indexing completed and ready for searches")

            # Initialize backend and search engine
            await self._initialize_backend()
            await self._initialize_search_engine()

            # Start background processes if needed
            if self.enable_polling:
                await self._start_background_processes()

            self.logger.info("✅ CocoIndex test infrastructure initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to set up CocoIndex infrastructure: {e}")
            raise

    async def _initialize_backend(self) -> None:
        """Initialize the database backend."""
        # Get database configuration from environment (should already be loaded)
        database_url = os.getenv("DATABASE_URL") or os.getenv("COCOINDEX_DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL or COCOINDEX_DATABASE_URL not found in environment")

        backend_type = os.getenv("BACKEND_TYPE", "postgres")

        # Use the table name determined during setup
        table_name = self.table_name
        if not table_name:
            raise ValueError("Table name not set. Ensure setup() was called first.")

        self.logger.info(f"🔧 Initializing {backend_type} backend with table: {table_name}")

        # Create backend based on type
        if backend_type == "postgres":
            from pgvector.psycopg import register_vector
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(database_url)
            # Register pgvector extensions
            with pool.connection() as conn:
                register_vector(conn)

            self.backend = BackendFactory.create_backend(
                backend_type,
                pool=pool,
                table_name=table_name
            )
        else:
            # For other backends that might expect connection_string
            self.backend = BackendFactory.create_backend(
                backend_type,
                connection_string=database_url,
                table_name=table_name
            )

        self.logger.info(f"✅ Backend initialized: {backend_type}")

    async def _initialize_search_engine(self) -> None:
        """Initialize the hybrid search engine."""
        # Create parser
        parser = KeywordSearchParser()

        # Get table name from test infrastructure configuration
        table_name = self.table_name
        if not table_name:
            raise ValueError("Table name not set. Ensure setup() was called first.")

        # Initialize hybrid search engine
        self.hybrid_search_engine = HybridSearchEngine(
            table_name=table_name,
            parser=parser,
            backend=self.backend,
            embedding_func=safe_embedding_function
        )

        self.logger.info("✅ Hybrid search engine initialized")

    async def _start_background_processes(self) -> None:
        """Start background processes for live updates."""
        # For direct testing, we'll handle background processes differently
        # In integration tests, this would start the full background initialization
        # For now, we'll just log that background processes would start
        self.logger.info("✅ Background processes would start (skipped in direct tests)")

    async def perform_hybrid_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform hybrid search using the initialized infrastructure.

        Args:
            arguments: Search arguments containing vector_query, keyword_query, etc.

        Returns:
            Search results dictionary
        """
        if not self.hybrid_search_engine:
            raise RuntimeError("Hybrid search engine not initialized")

        vector_query = arguments["vector_query"]
        keyword_query = arguments["keyword_query"]
        top_k = arguments.get("top_k", 10)
        vector_weight = arguments.get("vector_weight", 0.7)
        keyword_weight = arguments.get("keyword_weight", 0.3)

        try:
            # Use HybridSearchEngine which already converts to dictionaries
            results = self.hybrid_search_engine.search(
                vector_query=vector_query,
                keyword_query=keyword_query,
                top_k=top_k,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight
            )

            return {
                "query": {
                    "vector_query": vector_query,
                    "keyword_query": keyword_query,
                    "top_k": top_k,
                    "vector_weight": vector_weight,
                    "keyword_weight": keyword_weight
                },
                "results": results,
                "total_results": len(results)
            }

        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            raise

    async def perform_vector_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform vector-only search using the initialized infrastructure.

        Args:
            arguments: Search arguments containing vector_query, etc.

        Returns:
            Search results dictionary
        """
        if not self.backend:
            raise RuntimeError("Backend not initialized")

        vector_query = arguments["vector_query"]
        top_k = arguments.get("top_k", 10)

        try:
            # Convert text query to vector using embedding function
            from cocoindex_code_mcp_server.main_mcp_server import safe_embedding_function
            query_vector = safe_embedding_function(vector_query)

            # Call backend method directly 
            search_results = self.backend.vector_search(
                query_vector=query_vector,
                top_k=top_k
            )

            # Convert SearchResult objects to dictionaries (like HybridSearchEngine does)
            results = [self._search_result_to_dict(result) for result in search_results]

            return {
                "query": {
                    "vector_query": vector_query,
                    "top_k": top_k
                },
                "results": results,
                "total_results": len(results)
            }

        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            raise

    async def perform_keyword_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform keyword-only search using the initialized infrastructure.

        Args:
            arguments: Search arguments containing keyword_query, etc.

        Returns:
            Search results dictionary
        """
        if not self.backend:
            raise RuntimeError("Backend not initialized")

        keyword_query = arguments["keyword_query"]
        top_k = arguments.get("top_k", 10)

        try:
            # Parse keyword query using the keyword search parser
            from cocoindex_code_mcp_server.keyword_search_parser_lark import KeywordSearchParser
            from cocoindex_code_mcp_server.backends import QueryFilters

            parser = KeywordSearchParser()
            search_group = parser.parse(keyword_query)
            
            # Convert SearchGroup to QueryFilters format
            filters = QueryFilters(conditions=search_group.conditions, operator=search_group.operator.value)

            # Call backend method directly
            search_results = self.backend.keyword_search(
                filters=filters,
                top_k=top_k
            )

            # Convert SearchResult objects to dictionaries (like HybridSearchEngine does)
            results = [self._search_result_to_dict(result) for result in search_results]

            return {
                "query": {
                    "keyword_query": keyword_query,
                    "top_k": top_k
                },
                "results": results,
                "total_results": len(results)
            }

        except Exception as e:
            self.logger.error(f"Keyword search failed: {e}")
            raise

    def _search_result_to_dict(self, result) -> Dict[str, Any]:
        """Convert SearchResult to dict format for backward compatibility."""
        result_dict = {
            "filename": result.filename,
            "language": result.language,
            "code": result.code,
            "score": result.score,
            "start": result.start,
            "end": result.end,
            "source": result.source,
            "score_type": result.score_type
        }

        # Add metadata fields if available
        if result.metadata:
            result_dict.update(result.metadata)

        return result_dict

    async def cleanup(self) -> None:
        """Clean up the infrastructure."""
        self.logger.info("🧹 Cleaning up CocoIndex test infrastructure...")

        # Signal shutdown
        self.shutdown_event.set()

        # Wait for background thread to finish
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=5)

        # Clean up backend
        if self.backend:
            # Backend cleanup if needed
            pass

        # Stop CocoIndex library
        try:
            cocoindex.stop()
            self.logger.info("✅ CocoIndex library stopped")
        except Exception as e:
            self.logger.warning(f"Error stopping CocoIndex: {e}")

        self.logger.info("✅ CocoIndex test infrastructure cleaned up")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


async def run_cocoindex_hybrid_search_tests(
    test_cases: List[Dict[str, Any]],
    infrastructure: CocoIndexTestInfrastructure,
    run_timestamp: str
) -> List[Dict[str, Any]]:
    """
    Run hybrid search tests using CocoIndex infrastructure directly.

    Args:
        test_cases: List of test case definitions
        infrastructure: Initialized CocoIndex infrastructure
        run_timestamp: Timestamp for result saving

    Returns:
        List of failed test cases with error details
    """
    failed_tests = []

    for test_case in test_cases:
        test_name = test_case["name"]
        description = test_case["description"]
        query = test_case["query"]
        expected_results = test_case["expected_results"]

        logging.info(f"Running hybrid search test: {test_name}")
        logging.info(f"Description: {description}")

        try:
            # Execute search using infrastructure
            search_data = await infrastructure.perform_hybrid_search(query)

            results = search_data.get("results", [])
            total_results = len(results)

            # Save search results to test-results directory
            save_search_results(test_name, query, search_data, run_timestamp)

            # Check minimum results requirement
            min_results = expected_results.get("min_results", 1)
            if total_results < min_results:
                failed_tests.append({
                    "test": test_name,
                    "error": f"Expected at least {min_results} results, got {total_results}",
                    "query": query
                })
                continue

            # Check expected results using common helper
            if "should_contain" in expected_results:
                for expected_item in expected_results["should_contain"]:
                    found_match = False

                    for result_item in results:
                        match_found, errors = compare_expected_vs_actual(expected_item, result_item)
                        if match_found:
                            found_match = True
                            break

                    if not found_match:
                        # Enhanced error reporting with database comparison for hybrid search
                        try:
                            from .db_comparison import compare_test_with_database
                            db_comparison = await compare_test_with_database(
                                test_name, query, expected_item, results
                            )
                            db_report = f"\n🔍 Database Comparison Analysis (HYBRID SEARCH):\n"
                            for discrepancy in db_comparison.discrepancies:
                                db_report += f"  ❌ {discrepancy}\n"
                            
                            if db_comparison.matching_db_records:
                                db_report += f"\n📋 Database has {len(db_comparison.matching_db_records)} matching records\n"
                                # Show sample DB record metadata
                                if db_comparison.matching_db_records:
                                    sample_record = db_comparison.matching_db_records[0]
                                    db_report += f"  Sample DB record: complexity_score={sample_record.get('complexity_score', 'N/A')}, "
                                    db_report += f"has_classes={sample_record.get('has_classes', 'N/A')}, "
                                    db_report += f"language={sample_record.get('language', 'N/A')}, "
                                    db_report += f"functions='{sample_record.get('functions', 'N/A')[:50]}...'\n"
                            
                            error_with_db_analysis = f"No matching result found for expected item: {expected_item}{db_report}"
                        except Exception as db_error:
                            logging.warning(f"Database comparison failed for hybrid search: {db_error}")
                            error_with_db_analysis = f"No matching result found for expected item: {expected_item}"
                        
                        failed_tests.append({
                            "test": test_name,
                            "error": error_with_db_analysis,
                            "query": query,
                            "actual_results": [{
                                "filename": r.get("filename"),
                                "metadata_summary": {
                                    "classes": r.get("classes", []),
                                    "functions": r.get("functions", []),
                                    "imports": r.get("imports", []),
                                    "analysis_method": r.get("metadata_json", {}).get("analysis_method", "unknown")
                                }
                            } for r in results[:3]]  # Show first 3 results for debugging
                        })

        except Exception as e:
            failed_tests.append({
                "test": test_name,
                "error": f"Test execution failed: {str(e)}",
                "query": query
            })

    return failed_tests

async def run_cocoindex_vector_search_tests(
      test_cases: List[Dict[str, Any]],
      infrastructure: CocoIndexTestInfrastructure,
      run_timestamp: str
  ) -> List[Dict[str, Any]]:
      """
      Run vector-only search tests using CocoIndex infrastructure directly.

      Args:
          test_cases: List of test case definitions
          infrastructure: Initialized CocoIndex infrastructure
          run_timestamp: Timestamp for result saving

      Returns:
          List of failed test cases with error details
      """
      failed_tests = []

      for test_case in test_cases:
          test_name = test_case["name"]
          description = test_case["description"]
          query = test_case["query"]
          expected_results = test_case["expected_results"]

          logging.info(f"Running vector search test: {test_name}")
          logging.info(f"Description: {description}")

          try:
              # Execute vector-only search using infrastructure backend
              search_data = await infrastructure.perform_vector_search(query)

              results = search_data.get("results", [])
              total_results = len(results)

              # Save search results to test-results directory
              save_search_results(test_name, query, search_data, run_timestamp,
  "search-vector")

              # Check minimum results requirement
              min_results = expected_results.get("min_results", 1)
              if total_results < min_results:
                  failed_tests.append({
                      "test": test_name,
                      "error": f"Expected at least {min_results} results, got {total_results}",
                      "query": query
                  })
                  continue

              # Check expected results using common helper
              if "should_contain" in expected_results:
                  for expected_item in expected_results["should_contain"]:
                      found_match = False

                      for result_item in results:
                          match_found, _ = compare_expected_vs_actual(expected_item, result_item)
                          if match_found:
                              found_match = True
                              break

                      if not found_match:
                          # Enhanced error reporting with database comparison for vector search
                          try:
                              from .db_comparison import compare_test_with_database
                              db_comparison = await compare_test_with_database(
                                  test_name, query, expected_item, results
                              )
                              db_report = f"\n🔍 Database Comparison Analysis (VECTOR SEARCH):\n"
                              for discrepancy in db_comparison.discrepancies:
                                  db_report += f"  ❌ {discrepancy}\n"

                              if db_comparison.matching_db_records:
                                  db_report += f"\n📋 Database has {len(db_comparison.matching_db_records)} matching records\n"
                                  # Show sample DB record metadata
                                  if db_comparison.matching_db_records:
                                      sample_record = db_comparison.matching_db_records[0]
                                      db_report += f"  Sample DB record: complexity_score={sample_record.get('complexity_score', 'N/A')}, "
                                      db_report += f"has_classes={sample_record.get('has_classes', 'N/A')}, "
                                      db_report += f"language={sample_record.get('language', 'N/A')}, "
                                      db_report += f"functions='{sample_record.get('functions', 'N/A')[:50]}...'\n"

                              error_with_db_analysis = f"No matching result found  for expected item: {expected_item}{db_report}"
                          except Exception as db_error:
                              logging.warning(f"Database comparison failed for  vector search: {db_error}")
                              error_with_db_analysis = f"No matching result found  for expected item: {expected_item}"

                          failed_tests.append({
                              "test": test_name,
                              "error": error_with_db_analysis,
                              "query": query,
                              "actual_results": [{
                                  "filename": r.get("filename"),
                                  "metadata_summary": {
                                      "classes": r.get("classes", []),
                                      "functions": r.get("functions", []),
                                      "imports": r.get("imports", []),
                                      "analysis_method": r.get("metadata_json",{}).get("analysis_method", "unknown")
                                  }
                              } for r in results[:3]]  # Show first 3 results for debugging
                          })
          except Exception as e:
              failed_tests.append({
                  "test": test_name,
                  "error": f"Test execution failed: {str(e)}",
                  "query": query
              })

      return failed_tests
