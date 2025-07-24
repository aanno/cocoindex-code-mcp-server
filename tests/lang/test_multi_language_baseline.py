#!/usr/bin/env python3

"""
Multi-language baseline integration test following the CocoIndex baseline pattern.

This test evaluates language support capabilities across multiple programming languages
to create a comprehensive matrix of what works out-of-the-box including:
- Chunking strategy and quality
- Metadata extraction capabilities  
- Language-specific features
- Smart embedding model selection
- AST visitor support

Based on tests/lang/python/test_cocoindex_baseline_comparison.py pattern.
"""

import pytest
import json
import sys
import os
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'cocoindex-code-mcp-server'))

from cocoindex_config import (
    extract_code_metadata, 
    extract_language,
    get_embedding_model_group,
    LANGUAGE_MODEL_GROUPS,
    _global_flow_config
)

@dataclass
class LanguageCapabilities:
    """Data class to track language support capabilities."""
    language: str
    extension: str
    chunking_quality: str
    metadata_extraction: List[str]
    metadata_quality: str
    ast_visitor_support: bool
    embedding_model: str
    language_specific_features: List[str]
    analysis_method: str
    error_message: str = ""

class MultiLanguageExtractor:
    """Multi-language metadata extractor for comprehensive testing."""
    
    def __init__(self, use_smart_embedding: bool = True):
        self.use_smart_embedding = use_smart_embedding
        self.capabilities_matrix: List[LanguageCapabilities] = []
    
    def test_language_support(self, language: str, code: str, filename: str) -> LanguageCapabilities:
        """Test comprehensive language support capabilities."""
        extension = os.path.splitext(filename)[1] if '.' in filename else filename
        
        capabilities = LanguageCapabilities(
            language=language,
            extension=extension,
            chunking_quality="unknown",
            metadata_extraction=[],
            metadata_quality="unknown", 
            ast_visitor_support=False,
            embedding_model="unknown",
            language_specific_features=[],
            analysis_method="unknown"
        )
        
        try:
            # Test language detection
            detected_language = extract_language(filename)
            if detected_language.lower() != language.lower():
                capabilities.error_message = f"Language detection mismatch: expected {language}, got {detected_language}"
            
            # Test embedding model selection
            if self.use_smart_embedding:
                model_group = get_embedding_model_group(detected_language)
                capabilities.embedding_model = LANGUAGE_MODEL_GROUPS[model_group]['model']
            else:
                capabilities.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            
            # Test metadata extraction
            metadata_json = extract_code_metadata(code, detected_language, filename)
            metadata_dict = json.loads(metadata_json)
            
            capabilities.analysis_method = metadata_dict.get("analysis_method", "unknown")
            
            # Analyze metadata capabilities
            metadata_fields = []
            try:
                if metadata_dict.get("functions") and len(eval(str(metadata_dict["functions"]))) > 0:
                    metadata_fields.append("functions")
            except:
                if metadata_dict.get("functions"):
                    metadata_fields.append("functions")
            
            try:
                if metadata_dict.get("classes") and len(eval(str(metadata_dict["classes"]))) > 0:
                    metadata_fields.append("classes")
            except:
                if metadata_dict.get("classes"):
                    metadata_fields.append("classes")
                    
            try:
                if metadata_dict.get("imports") and len(eval(str(metadata_dict["imports"]))) > 0:
                    metadata_fields.append("imports")
            except:
                if metadata_dict.get("imports"):
                    metadata_fields.append("imports")
                    
            if metadata_dict.get("has_type_hints"):
                metadata_fields.append("type_hints")
            if metadata_dict.get("has_async"):
                metadata_fields.append("async")
                
            try:
                if metadata_dict.get("decorators_used") and len(eval(str(metadata_dict["decorators_used"]))) > 0:
                    metadata_fields.append("decorators")
            except:
                if metadata_dict.get("decorators_used"):
                    metadata_fields.append("decorators")
                    
            if metadata_dict.get("complexity_score", 0) > 0:
                metadata_fields.append("complexity")
            
            capabilities.metadata_extraction = metadata_fields
            
            # Assess metadata quality based on analysis method and field richness
            if capabilities.analysis_method in ["tree_sitter_enhanced", "tree_sitter_python"]:
                capabilities.metadata_quality = "high"
                capabilities.ast_visitor_support = True
            elif capabilities.analysis_method in ["python_ast", "tree_sitter"]:
                capabilities.metadata_quality = "medium" if len(metadata_fields) >= 3 else "basic"
                # Python AST analysis uses our enhanced AST visitor pattern
                if capabilities.analysis_method == "python_ast":
                    capabilities.ast_visitor_support = True
                    capabilities.metadata_quality = "high" if len(metadata_fields) >= 4 else "medium"
            elif capabilities.analysis_method == "basic":
                capabilities.metadata_quality = "basic"
            else:
                capabilities.metadata_quality = "unknown"
            
            # Assess chunking quality based on analysis method
            if capabilities.analysis_method in ["tree_sitter_enhanced", "ast_chunking"]:
                capabilities.chunking_quality = "ast_aware"
            elif capabilities.analysis_method in ["tree_sitter", "python_ast"]:
                capabilities.chunking_quality = "syntax_aware"
            else:
                capabilities.chunking_quality = "text_based"
            
            # Detect language-specific features
            language_features = []
            if metadata_dict.get("has_async"):
                language_features.append("async_support")
            if metadata_dict.get("has_type_hints"):
                language_features.append("type_annotations")
            if metadata_dict.get("decorators_used"):
                language_features.append("decorators")
            if "tree_sitter" in capabilities.analysis_method:
                language_features.append("syntax_tree_parsing")
            if capabilities.analysis_method == "tree_sitter_enhanced":
                language_features.append("enhanced_ast_analysis")
            
            capabilities.language_specific_features = language_features
            
        except Exception as e:
            capabilities.error_message = str(e)
            capabilities.metadata_quality = "error"
            capabilities.chunking_quality = "error"
        
        return capabilities
    
    def generate_capabilities_matrix(self, test_cases: List[Tuple[str, str, str]]) -> List[LanguageCapabilities]:
        """Generate comprehensive capabilities matrix for all test languages."""
        for language, code, filename in test_cases:
            capabilities = self.test_language_support(language, code, filename)
            self.capabilities_matrix.append(capabilities)
        
        return self.capabilities_matrix

@pytest.fixture
def language_test_cases():
    """Provide comprehensive test cases for multiple programming languages."""
    return [
        # Python - should have full support
        ("Python", '''
import asyncio
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class DataProcessor:
    """A data processor with async capabilities."""
    name: str
    
    @property
    def status(self) -> str:
        return "active"
    
    async def process_data(self, items: List[str]) -> List[str]:
        """Process data asynchronously."""
        return [f"processed_{item}" for item in items]

def helper_function(x: int, y: int) -> int:
    """Helper function with type hints."""
    return x + y
''', "test_processor.py"),

        # Rust - should have medium support via tree-sitter
        ("Rust", '''
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct DataProcessor {
    pub name: String,
    pub config: HashMap<String, String>,
}

impl DataProcessor {
    pub fn new(name: String) -> Self {
        Self {
            name,
            config: HashMap::new(),
        }
    }
    
    pub fn process(&self, data: &str) -> String {
        format!("processed_{}", data)
    }
}

fn main() {
    let processor = DataProcessor::new("test".to_string());
    println!("{:?}", processor);
}
''', "data_processor.rs"),

        # JavaScript - should have medium support via tree-sitter
        ("JavaScript", '''
class DataProcessor {
    constructor(name) {
        this.name = name;
        this.config = {};
    }
    
    get status() {
        return 'active';
    }
    
    async processData(items) {
        return items.map(item => `processed_${item}`);
    }
}

function helperFunction(x, y) {
    return x + y;
}

const processor = new DataProcessor('test');
console.log(processor.status);
''', "data_processor.js"),

        # TypeScript - should have medium support via tree-sitter
        ("TypeScript", '''
interface Config {
    maxItems: number;
    timeout: number;
}

class DataProcessor {
    private name: string;
    private config: Config;
    
    constructor(name: string, config: Config) {
        this.name = name;
        this.config = config;
    }
    
    public async processData(items: string[]): Promise<string[]> {
        return items.map(item => `processed_${item}`);
    }
    
    public get status(): string {
        return 'active';
    }
}

function helperFunction(x: number, y: number): number {
    return x + y;
}
''', "data_processor.ts"),

        # Java - should have medium support via tree-sitter
        ("Java", '''
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;

public class DataProcessor {
    private String name;
    private Map<String, String> config;
    
    public DataProcessor(String name) {
        this.name = name;
        this.config = new HashMap<>();
    }
    
    public String getStatus() {
        return "active";
    }
    
    public CompletableFuture<List<String>> processData(List<String> items) {
        return CompletableFuture.supplyAsync(() -> {
            List<String> result = new ArrayList<>();
            for (String item : items) {
                result.add("processed_" + item);
            }
            return result;
        });
    }
    
    public static int helperFunction(int x, int y) {
        return x + y;
    }
}
''', "DataProcessor.java"),

        # Go - should have medium support via tree-sitter
        ("Go", '''
package main

import (
    "fmt"
    "sync"
)

type DataProcessor struct {
    Name   string
    Config map[string]string
    mu     sync.RWMutex
}

func NewDataProcessor(name string) *DataProcessor {
    return &DataProcessor{
        Name:   name,
        Config: make(map[string]string),
    }
}

func (dp *DataProcessor) GetStatus() string {
    return "active"
}

func (dp *DataProcessor) ProcessData(items []string) []string {
    result := make([]string, len(items))
    for i, item := range items {
        result[i] = fmt.Sprintf("processed_%s", item)
    }
    return result
}

func helperFunction(x, y int) int {
    return x + y
}

func main() {
    processor := NewDataProcessor("test")
    fmt.Printf("Status: %s\\n", processor.GetStatus())
}
''', "data_processor.go"),

        # C++ - should have basic support via tree-sitter
        ("C++", '''
#include <iostream>
#include <vector>
#include <string>
#include <memory>

class DataProcessor {
private:
    std::string name;
    
public:
    DataProcessor(const std::string& name) : name(name) {}
    
    std::string getStatus() const {
        return "active";
    }
    
    std::vector<std::string> processData(const std::vector<std::string>& items) {
        std::vector<std::string> result;
        for (const auto& item : items) {
            result.push_back("processed_" + item);
        }
        return result;
    }
};

int helperFunction(int x, int y) {
    return x + y;
}

int main() {
    auto processor = std::make_unique<DataProcessor>("test");
    std::cout << "Status: " << processor->getStatus() << std::endl;
    return 0;
}
''', "data_processor.cpp"),

        # Haskell - should have basic support via custom chunking
        ("Haskell", '''
module DataProcessor where

import Control.Concurrent.Async
import Data.List

data DataProcessor = DataProcessor
    { name :: String
    , config :: [(String, String)]
    } deriving (Show, Eq)

newDataProcessor :: String -> DataProcessor
newDataProcessor n = DataProcessor n []

getStatus :: DataProcessor -> String
getStatus _ = "active"

processData :: DataProcessor -> [String] -> IO [String]
processData _ items = do
    async $ return $ map (\\item -> "processed_" ++ item) items
    wait =<< async (return $ map (\\item -> "processed_" ++ item) items)

helperFunction :: Int -> Int -> Int
helperFunction x y = x + y

main :: IO ()
main = do
    let processor = newDataProcessor "test"
    putStrLn $ "Status: " ++ getStatus processor
''', "DataProcessor.hs"),
    ]

@pytest.fixture
def multi_language_extractor():
    """Create multi-language extractor with smart embedding enabled."""
    return MultiLanguageExtractor(use_smart_embedding=True)

class TestMultiLanguageBaseline:
    """Comprehensive multi-language baseline test suite."""
    
    def test_language_detection_accuracy(self, language_test_cases):
        """Test that language detection works correctly for all test files."""
        for expected_language, code, filename in language_test_cases:
            detected_language = extract_language(filename)
            assert detected_language.lower() == expected_language.lower(), \
                f"Language detection failed for {filename}: expected {expected_language}, got {detected_language}"
    
    def test_smart_embedding_model_selection(self, language_test_cases):
        """Test that smart embedding selects appropriate models for each language."""
        expected_models = {
            'python': 'microsoft/graphcodebert-base',
            'javascript': 'microsoft/graphcodebert-base', 
            'java': 'microsoft/graphcodebert-base',
            'go': 'microsoft/graphcodebert-base',
            'c++': 'microsoft/graphcodebert-base',
            'rust': 'microsoft/unixcoder-base',
            'typescript': 'microsoft/unixcoder-base',
            'haskell': 'sentence-transformers/all-MiniLM-L6-v2',
        }
        
        for language, code, filename in language_test_cases:
            detected_language = extract_language(filename)
            model_group = get_embedding_model_group(detected_language)
            actual_model = LANGUAGE_MODEL_GROUPS[model_group]['model']
            expected_model = expected_models[language.lower()]
            
            assert actual_model == expected_model, \
                f"Embedding model mismatch for {language}: expected {expected_model}, got {actual_model}"
    
    def test_metadata_extraction_capabilities(self, multi_language_extractor, language_test_cases):
        """Test metadata extraction capabilities across all languages."""
        capabilities_matrix = multi_language_extractor.generate_capabilities_matrix(language_test_cases)
        
        # Ensure we got results for all languages
        assert len(capabilities_matrix) == len(language_test_cases)
        
        # Check that each language has some level of support
        for capabilities in capabilities_matrix:
            assert capabilities.language, f"Language should be detected"
            assert capabilities.embedding_model, f"Embedding model should be selected for {capabilities.language}"
            
            if capabilities.error_message:
                print(f"Warning: {capabilities.language} had error: {capabilities.error_message}")
            else:
                # Should have at least basic metadata extraction
                assert capabilities.analysis_method != "unknown", \
                    f"{capabilities.language} should have known analysis method"
    
    def test_python_enhanced_capabilities(self, multi_language_extractor, language_test_cases):
        """Test that Python has enhanced capabilities compared to other languages."""
        capabilities_matrix = multi_language_extractor.generate_capabilities_matrix(language_test_cases)
        
        python_capabilities = next((c for c in capabilities_matrix if c.language == "Python"), None)
        assert python_capabilities, "Python capabilities should be found"
        
        # Python should have enhanced analysis
        assert python_capabilities.ast_visitor_support, "Python should have AST visitor support"
        assert python_capabilities.metadata_quality in ["high", "medium"], \
            f"Python should have high/medium quality metadata, got {python_capabilities.metadata_quality}"
        
        # Python should detect multiple metadata types
        assert len(python_capabilities.metadata_extraction) >= 4, \
            f"Python should extract multiple metadata types, got {python_capabilities.metadata_extraction}"
        
        # Python should detect language-specific features
        expected_features = ["async_support", "type_annotations", "decorators"]
        found_features = sum(1 for feature in expected_features 
                           if feature in python_capabilities.language_specific_features)
        assert found_features >= 2, \
            f"Python should detect at least 2 language-specific features, found {found_features}"
    
    def test_generate_capabilities_report(self, multi_language_extractor, language_test_cases):
        """Generate and validate comprehensive capabilities report."""
        capabilities_matrix = multi_language_extractor.generate_capabilities_matrix(language_test_cases)
        
        print("\n" + "="*80)
        print("MULTI-LANGUAGE CAPABILITIES MATRIX")
        print("="*80)
        
        # Print header
        print(f"{'Language':<12} {'Model':<25} {'Chunking':<12} {'Metadata':<20} {'Quality':<8} {'AST':<5} {'Features'}")
        print("-" * 110)
        
        # Print results for each language
        for capabilities in capabilities_matrix:
            model_short = capabilities.embedding_model.split('/')[-1] if '/' in capabilities.embedding_model else capabilities.embedding_model
            metadata_count = len(capabilities.metadata_extraction)
            features_str = ','.join(capabilities.language_specific_features[:2])  # Show first 2 features
            
            print(f"{capabilities.language:<12} {model_short:<25} {capabilities.chunking_quality:<12} "
                  f"{metadata_count} fields{'':<11} {capabilities.metadata_quality:<8} "
                  f"{'✓' if capabilities.ast_visitor_support else '✗':<5} {features_str}")
            
            if capabilities.error_message:
                print(f"             ERROR: {capabilities.error_message}")
        
        print("-" * 110)
        
        # Summary statistics
        high_quality = sum(1 for c in capabilities_matrix if c.metadata_quality == "high")
        medium_quality = sum(1 for c in capabilities_matrix if c.metadata_quality == "medium")
        ast_support = sum(1 for c in capabilities_matrix if c.ast_visitor_support)
        smart_models = sum(1 for c in capabilities_matrix if "graphcodebert" in c.embedding_model.lower() or "unixcoder" in c.embedding_model.lower())
        
        print(f"SUMMARY:")
        print(f"- High quality metadata: {high_quality}/{len(capabilities_matrix)} languages")
        print(f"- Medium+ quality metadata: {high_quality + medium_quality}/{len(capabilities_matrix)} languages")  
        print(f"- AST visitor support: {ast_support}/{len(capabilities_matrix)} languages")
        print(f"- Smart embedding models: {smart_models}/{len(capabilities_matrix)} languages")
        
        # Validate that we have reasonable coverage
        assert high_quality >= 1, "At least one language should have high quality metadata"
        assert high_quality + medium_quality >= len(capabilities_matrix) // 2, \
            "At least half the languages should have medium or better metadata quality"
        assert smart_models >= len(capabilities_matrix) // 2, \
            "At least half the languages should use specialized embedding models"
    
    def test_haskell_capabilities_for_enhancement(self, multi_language_extractor, language_test_cases):
        """Test Haskell capabilities to identify enhancement opportunities."""
        capabilities_matrix = multi_language_extractor.generate_capabilities_matrix(language_test_cases)
        
        haskell_capabilities = next((c for c in capabilities_matrix if c.language == "Haskell"), None)
        assert haskell_capabilities, "Haskell capabilities should be found"
        
        print(f"\nHaskell Current Capabilities:")
        print(f"- Analysis method: {haskell_capabilities.analysis_method}")
        print(f"- Metadata quality: {haskell_capabilities.metadata_quality}")
        print(f"- Chunking quality: {haskell_capabilities.chunking_quality}")
        print(f"- AST support: {haskell_capabilities.ast_visitor_support}")
        print(f"- Metadata fields: {haskell_capabilities.metadata_extraction}")
        print(f"- Embedding model: {haskell_capabilities.embedding_model}")
        
        # Document current state for enhancement planning
        assert haskell_capabilities.embedding_model == "sentence-transformers/all-MiniLM-L6-v2", \
            "Haskell should use fallback embedding model"
        
        # This test documents current limitations to guide enhancement
        if not haskell_capabilities.ast_visitor_support:
            print("Enhancement opportunity: Haskell needs AST visitor support")
        
        if haskell_capabilities.metadata_quality in ["basic", "unknown"]:
            print("Enhancement opportunity: Haskell needs better metadata extraction")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])