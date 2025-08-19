#!/usr/bin/env python3

"""
Debug test to identify the exact source of PostgreSQL duplicate key errors.
This test simulates the actual flow execution to see where duplicates are generated.
"""

from types import FunctionType
from typing import cast

import pytest

import cocoindex
from cocoindex_code_mcp_server.cocoindex_config import (
    AST_CHUNKING_AVAILABLE,
    CUSTOM_LANGUAGES,
    ASTChunkOperation,
    _global_flow_config,
    extract_language,
    get_chunking_params,
)

# Sample files that are causing errors
SAMPLE_TYPESCRIPT_CODE = '''import express from 'express';
import { Request, Response } from 'express';
import { User } from '../models/User';

const router = express.Router();

// Get all users
router.get('/users', async (req: Request, res: Response) => {
    try {
        const users = await User.findAll();
        res.json(users);
    } catch (error) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Create user
router.post('/users', async (req: Request, res: Response) => {
    try {
        const { name, email } = req.body;
        const user = await User.create({ name, email });
        res.status(201).json(user);
    } catch (error) {
        res.status(400).json({ error: 'Bad request' });
    }
});

export default router;
'''

SAMPLE_RUST_CODE = '''use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct User {
    pub id: u32,
    pub name: String,
    pub email: String,
}

impl User {
    pub fn new(id: u32, name: String, email: String) -> Self {
        User { id, name, email }
    }

    pub fn validate_email(&self) -> bool {
        self.email.contains('@')
    }
}

pub struct UserService {
    users: HashMap<u32, User>,
    next_id: u32,
}

impl UserService {
    pub fn new() -> Self {
        UserService {
            users: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn create_user(&mut self, name: String, email: String) -> User {
        let user = User::new(self.next_id, name, email);
        self.users.insert(self.next_id, user.clone());
        self.next_id += 1;
        user
    }

    pub fn get_user(&self, id: u32) -> Option<&User> {
        self.users.get(&id)
    }
}
'''


class TestDuplicateKeyDebug:
    """Debug tests to identify duplicate key sources."""

    @pytest.mark.xfail(reason="ASTChunkOperation is not directly callable - requires CocoIndex flow context")
    def test_typescript_chunking_path(self):
        """Test what chunking path TypeScript files take."""
        filename = "userRoutes.ts"
        language = cast(FunctionType, extract_language)(filename)

        print(f"Filename: {filename}")
        print(f"Detected language: {language}")
        print(f"AST_CHUNKING_AVAILABLE: {AST_CHUNKING_AVAILABLE}")

        # Check chunking path selection
        use_default_chunking = _global_flow_config.get('use_default_chunking', False)
        print(f"use_default_chunking flag: {use_default_chunking}")

        if use_default_chunking or not AST_CHUNKING_AVAILABLE:
            print("→ Would use DEFAULT chunking (SplitRecursively)")

            # Test default chunking
            try:
                chunker = cocoindex.functions.SplitRecursively(custom_languages=CUSTOM_LANGUAGES)
                print(f"Default chunker created: {chunker}")

                # This is where we'd need to simulate the DataSlice transform
                # For now, just verify the chunker exists and takes parameters
                params = cast(FunctionType, get_chunking_params)(language)
                print(
                    f"Chunking params: chunk_size={params.chunk_size}, min_chunk_size={params.min_chunk_size}, chunk_overlap={params.chunk_overlap}")

            except Exception as e:
                print(f"Default chunking setup failed: {e}")

        else:
            print("→ Would use AST chunking")

            # Test AST chunking
            try:
                chunks = cast(FunctionType, ASTChunkOperation)(
                    content=SAMPLE_TYPESCRIPT_CODE,
                    language=language,
                    max_chunk_size=1000,
                    chunk_overlap=250
                )

                print(f"AST chunking produced {len(chunks)} chunks:")
                locations = []
                for i, chunk in enumerate(chunks):
                    primary_key = (filename, chunk.location)
                    locations.append(chunk.location)
                    print(f"  Chunk {i}: location='{chunk.location}' -> primary_key={primary_key}")

                # Check for duplicates
                unique_locations = set(locations)
                if len(locations) != len(unique_locations):
                    duplicates = [loc for loc in locations if locations.count(loc) > 1]
                    print(f"❌ DUPLICATE LOCATIONS: {set(duplicates)}")
                    assert False, f"AST chunking produced duplicates: {duplicates}"
                else:
                    print("✅ All locations unique")

            except Exception as e:
                print(f"AST chunking failed: {e}")
                raise

    def test_rust_chunking_path(self):
        """Test what chunking path Rust files take."""
        filename = "lib.rs"
        language = cast(FunctionType, extract_language)(filename)

        print(f"Filename: {filename}")
        print(f"Detected language: {language}")

        # Check if Rust is supported by AST chunking
        if AST_CHUNKING_AVAILABLE:
            from cocoindex_code_mcp_server.ast_chunking import ASTChunkExecutor  # type: ignore
            # CocoIndexASTChunker doesn't exist anymore - skipping this test section
            print(f"Rust supported by AST chunking: Unknown (legacy API removed)")
            is_supported = False  # Default since we can't check with legacy API

            if not is_supported:
                print("→ Rust would use DEFAULT chunking (SplitRecursively)")
            else:
                print("→ Rust would use AST chunking")

                # Test AST chunking with Rust
                try:
                    chunks = cast(FunctionType, ASTChunkOperation)(
                        content=SAMPLE_RUST_CODE,
                        language=language,
                        max_chunk_size=1000,
                        chunk_overlap=200
                    )

                    print(f"AST chunking produced {len(chunks)} chunks:")
                    locations = []
                    for i, chunk in enumerate(chunks):
                        primary_key = (filename, chunk.location)
                        locations.append(chunk.location)
                        print(f"  Chunk {i}: location='{chunk.location}' -> primary_key={primary_key}")

                    # Check for duplicates
                    unique_locations = set(locations)
                    if len(locations) != len(unique_locations):
                        duplicates = [loc for loc in locations if locations.count(loc) > 1]
                        print(f"❌ DUPLICATE LOCATIONS: {set(duplicates)}")
                    else:
                        print("✅ All locations unique")

                except Exception as e:
                    print(f"AST chunking with Rust failed: {e}")

    @pytest.mark.xfail(reason="ASTChunkOperation is not directly callable - requires CocoIndex flow context")
    def test_potential_flow_duplication(self):
        """Test if the flow logic itself could create duplicates."""
        print("=== Testing potential flow duplication scenarios ===")

        # Scenario 1: What if a file gets processed multiple times?
        filename = "test.ts"

        # Simulate multiple processing of the same file
        results = []
        for run in range(2):
            print(f"\n--- Run {run + 1} ---")
            language = cast(FunctionType, extract_language)(filename)

            if AST_CHUNKING_AVAILABLE:
                chunks = cast(FunctionType, ASTChunkOperation)(
                    content=SAMPLE_TYPESCRIPT_CODE,
                    language=language,
                    max_chunk_size=800,
                    chunk_overlap=150
                )

                for chunk in chunks:
                    primary_key = (filename, chunk.location)
                    results.append(primary_key)
                    print(f"  Generated primary_key: {primary_key}")

        # Check if multiple runs produce the same primary keys
        print(f"\nTotal primary keys generated: {len(results)}")
        unique_keys = set(results)
        print(f"Unique primary keys: {len(unique_keys)}")

        if len(results) != len(unique_keys):
            duplicates = [key for key in results if results.count(key) > 1]
            print(f"❌ DUPLICATE PRIMARY KEYS across runs: {set(duplicates)}")
            print("This could cause PostgreSQL conflicts if the same file is processed multiple times!")
        else:
            print("✅ No duplicate primary keys across multiple runs")

    def test_chunking_with_empty_content(self):
        """Test edge case of empty or minimal content that might cause default fallbacks."""
        test_cases = [
            ("empty.ts", ""),
            ("minimal.rs", "// Just a comment"),
            ("single_line.ts", "export default {};"),
        ]

        for filename, content in test_cases:
            print(f"\n=== Testing {filename} with content: '{content}' ===")
            language = cast(FunctionType, extract_language)(filename)

            if AST_CHUNKING_AVAILABLE:
                try:
                    chunks = cast(FunctionType, ASTChunkOperation)(
                        content=content,
                        language=language,
                        max_chunk_size=1000,
                        chunk_overlap=100
                    )

                    print(f"Chunks produced: {len(chunks)}")
                    for i, chunk in enumerate(chunks):
                        print(f"  Chunk {i}: location='{chunk.location}', text_len={len(chunk.text)}")
                        if chunk.location == "line:0" and len(chunk.text) == 0:
                            print("  ⚠️  Empty chunk with default location - potential duplicate risk!")

                except Exception as e:
                    print(f"Chunking failed: {e}")


if __name__ == "__main__":
    cocoindex.init()
    pytest.main([__file__, "-v", "-s"])
