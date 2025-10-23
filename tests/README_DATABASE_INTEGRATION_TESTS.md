# Database Integration Tests

This document describes the database integration tests and their requirements.

## Overview

Some tests in this project require a PostgreSQL database with pgvector extension to be available. These tests are marked with the `@pytest.mark.db_integration` marker and are not run by default during unit testing.

## Database Requirements

### Prerequisites

- PostgreSQL database server (version 12+)
- pgvector extension installed and enabled
- Environment variables configured for database connection

### Required Environment Variables

- `DATABASE_URL` or individual database connection parameters
- Connection pool settings (if applicable)

### Database Schema

The tests expect:

- A table for storing embeddings (typically named with fields for filename, language, code, embedding, start, end, source_name)
- pgvector extension enabled for vector similarity search
- Proper indexes for performance

## Test Categories

### Files with Database Integration Tests

#### `/test_hybrid_search_engine.py`

- **Classes**: `TestHybridSearchEngine`, `TestMockDatabase`
- **Purpose**: Test hybrid search engine functionality with mocked and real database connections
- **Requirements**: Mock database setup or real PostgreSQL with pgvector

#### `/test_hybrid_search_integration.py`

- **Classes**: All test classes in this file
- **Purpose**: Integration tests for the complete hybrid search workflow
- **Requirements**: Full database setup with proper schema and data

#### `/test_hybrid_search_e2e.py`

- **Purpose**: End-to-end tests with real database operations
- **Requirements**: Fully configured PostgreSQL database with test data
- **Note**: All tests in this file are marked with `pytestmark = pytest.mark.db_integration`

## Running Database Integration Tests

### Run Only Unit Tests (Default)

```bash
# This will exclude db_integration tests
pytest tests/
```

### Run Only Database Integration Tests

```bash
pytest tests/ -m db_integration
```

### Run All Tests (Including Database Integration)

```bash
pytest tests/ -m "not db_integration or db_integration"
# OR
pytest tests/ --ignore-markers
```

### Skip Database Integration Tests Explicitly

```bash
pytest tests/ -m "not db_integration"
```

## Test Environment Setup

### Development Environment

1. Ensure PostgreSQL is running with pgvector extension
2. Set up environment variables for database connection
3. Initialize database schema if needed
4. Run integration tests: `pytest tests/ -m db_integration`

### CI/CD Environment

- Database integration tests should only run when a test database is available
- Consider using Docker containers for PostgreSQL + pgvector
- Use conditional test execution based on environment availability

## Troubleshooting

### Common Issues

1. **pgvector not found**: Ensure pgvector extension is installed and enabled
2. **Connection errors**: Verify DATABASE_URL and connection parameters
3. **Schema errors**: Ensure database schema matches expected structure
4. **Permission errors**: Check database user permissions for test operations

### Mock vs Real Database

- Some tests use mocked database connections for unit testing
- Integration tests require real database connections
- E2E tests perform actual database operations and require full setup

## Contributing

When adding new database-dependent tests:

1. Add the `@pytest.mark.db_integration` marker
2. Document database requirements in test docstrings
3. Ensure tests can run independently
4. Consider adding both mocked unit tests and integration tests for coverage
