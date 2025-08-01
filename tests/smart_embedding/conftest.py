#!/usr/bin/env python3
"""
Pytest configuration and fixtures for smart embedding tests.
"""

import os
import shutil
import tempfile

import pytest

# Package should be installed via maturin develop or pip install -e .
# No need to manually add src to path


@pytest.fixture(scope="session")
def test_code_samples():
    """Provide sample code files for different programming languages."""
    return {
        'python': '''
def fibonacci(n: int) -> int:
    """Calculate Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class MathUtils:
    @staticmethod
    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True
''',

        'rust': '''
#[derive(Debug, Clone)]
pub struct Person {
    pub name: String,
    pub age: u32,
}

impl Person {
    pub fn new(name: String, age: u32) -> Self {
        Self { name, age }
    }

    pub fn is_adult(&self) -> bool {
        self.age >= 18
    }
}

fn fibonacci(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}
''',

        'javascript': '''
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

class Calculator {
    constructor() {
        this.history = [];
    }

    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }
}

const isPrime = (num) => {
    if (num < 2) return false;
    for (let i = 2; i <= Math.sqrt(num); i++) {
        if (num % i === 0) return false;
    }
    return true;
};
''',

        'typescript': '''
interface User {
    id: number;
    name: string;
    isActive: boolean;
}

class UserService {
    private users: Map<number, User> = new Map();

    createUser(name: string): User {
        const user: User = {
            id: Date.now(),
            name,
            isActive: true
        };
        this.users.set(user.id, user);
        return user;
    }

    getActiveUsers(): User[] {
        return Array.from(this.users.values())
            .filter(user => user.isActive);
    }
}
''',

        'haskell': '''
data Person = Person
    { personName :: String
    , personAge  :: Int
    } deriving (Show, Eq)

fibonacci :: Int -> Int
fibonacci n
    | n <= 1    = n
    | otherwise = fibonacci (n - 1) + fibonacci (n - 2)

sumList :: [Int] -> Int
sumList []     = 0
sumList (x:xs) = x + sumList xs

main :: IO ()
main = do
    let person = Person "Alice" 30
    putStrLn $ "Person: " ++ show person
'''
    }


@pytest.fixture(scope="session")
def fixture_files():
    """Provide paths to test fixture files for different languages."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), '../fixtures')

    return {
        'python': os.path.join(fixtures_dir, 'test_python.py'),
        'rust': os.path.join(fixtures_dir, 'test_rust.rs'),
        'javascript': os.path.join(fixtures_dir, 'test_javascript.js'),
        'typescript': os.path.join(fixtures_dir, 'test_typescript.ts'),
        'haskell': os.path.join(fixtures_dir, 'test_haskell.hs'),
    }


@pytest.fixture(scope="function")
def temp_test_files(fixture_files):
    """Create temporary copies of fixture files for testing."""
    temp_dir = tempfile.mkdtemp(prefix="smart_embedding_test_")

    test_files = {}

    try:
        for language, source_path in fixture_files.items():
            if os.path.exists(source_path):
                filename = os.path.basename(source_path)
                dest_path = os.path.join(temp_dir, filename)
                shutil.copy2(source_path, dest_path)
                test_files[language] = dest_path

        yield test_files

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def expected_model_mappings():
    """Provide expected language to model mappings."""
    return {
        # GraphCodeBERT languages
        'python': 'microsoft/graphcodebert-base',
        'java': 'microsoft/graphcodebert-base',
        'javascript': 'microsoft/graphcodebert-base',
        'php': 'microsoft/graphcodebert-base',
        'ruby': 'microsoft/graphcodebert-base',
        'go': 'microsoft/graphcodebert-base',
        'c': 'microsoft/graphcodebert-base',
        'c++': 'microsoft/graphcodebert-base',

        # UniXcode languages
        'rust': 'microsoft/unixcoder-base',
        'typescript': 'microsoft/unixcoder-base',
        'tsx': 'microsoft/unixcoder-base',
        'c#': 'microsoft/unixcoder-base',
        'kotlin': 'microsoft/unixcoder-base',
        'scala': 'microsoft/unixcoder-base',
        'swift': 'microsoft/unixcoder-base',
        'dart': 'microsoft/unixcoder-base',

        # Fallback languages
        'haskell': 'sentence-transformers/all-MiniLM-L6-v2',
        'ocaml': 'sentence-transformers/all-MiniLM-L6-v2',
        'fortran': 'sentence-transformers/all-MiniLM-L6-v2',
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Ensure smart embedding is available for tests
    try:
        from cocoindex_code_mcp_server.cocoindex_config import SMART_EMBEDDING_AVAILABLE
        if not SMART_EMBEDDING_AVAILABLE:
            pytest.skip("Smart embedding not available - check configuration")
    except ImportError as e:
        pytest.skip(f"Could not import cocoindex_config: {e}")
