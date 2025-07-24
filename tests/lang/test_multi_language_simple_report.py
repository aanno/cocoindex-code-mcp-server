#!/usr/bin/env python3

"""
Simple multi-language capabilities report.
Quick test to see what works out-of-the-box without complex logic.
"""

import sys
import os
import json

# Add src to path 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'cocoindex-code-mcp-server'))

from cocoindex_config import (
    extract_code_metadata, 
    extract_language,
    get_embedding_model_group,
    LANGUAGE_MODEL_GROUPS
)

def test_language(language_name, code, filename):
    """Test a single language and return results."""
    try:
        # Language detection
        detected_language = extract_language(filename)
        
        # Embedding model selection
        model_group = get_embedding_model_group(detected_language)
        embedding_model = LANGUAGE_MODEL_GROUPS[model_group]['model']
        
        # Metadata extraction
        metadata_json = extract_code_metadata(code, detected_language, filename)
        metadata = json.loads(metadata_json)
        
        # Count meaningful metadata fields
        field_count = 0
        fields_found = []
        
        if metadata.get("functions") and str(metadata["functions"]) not in ["[]", "['']", ""]:
            field_count += 1
            fields_found.append("functions")
            
        if metadata.get("classes") and str(metadata["classes"]) not in ["[]", "['']", ""]:
            field_count += 1
            fields_found.append("classes")
            
        if metadata.get("imports") and str(metadata["imports"]) not in ["[]", "['']", ""]:
            field_count += 1  
            fields_found.append("imports")
            
        if metadata.get("has_type_hints"):
            field_count += 1
            fields_found.append("type_hints")
            
        if metadata.get("has_async"):
            field_count += 1
            fields_found.append("async")
            
        if metadata.get("decorators_used") and str(metadata["decorators_used"]) not in ["[]", "['']", ""]:
            field_count += 1
            fields_found.append("decorators")
            
        if metadata.get("complexity_score", 0) > 0:
            field_count += 1
            fields_found.append("complexity")
        
        analysis_method = metadata.get("analysis_method", "unknown")
        
        return {
            "language": language_name,
            "detected": detected_language,
            "model": embedding_model.split('/')[-1],
            "analysis_method": analysis_method,
            "field_count": field_count,
            "fields": fields_found,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "language": language_name,
            "detected": "error",
            "model": "unknown",
            "analysis_method": "error",
            "field_count": 0,
            "fields": [],
            "success": False,
            "error": str(e)
        }

def main():
    """Run multi-language capability test."""
    
    test_cases = [
        ("Python", '''
import asyncio
from typing import List
from dataclasses import dataclass

@dataclass
class Processor:
    name: str
    
    async def process(self, items: List[str]) -> List[str]:
        return [f"processed_{item}" for item in items]

def helper(x: int) -> int:
    return x * 2
''', "processor.py"),

        ("Rust", '''
pub struct Processor {
    pub name: String,
}

impl Processor {
    pub fn new(name: String) -> Self {
        Self { name }
    }
    
    pub fn process(&self, item: &str) -> String {
        format!("processed_{}", item)
    }
}
''', "processor.rs"),

        ("JavaScript", '''
class Processor {
    constructor(name) {
        this.name = name;
    }
    
    process(item) {
        return `processed_${item}`;
    }
}
''', "processor.js"),

        ("TypeScript", '''
interface Config {
    timeout: number;
}

class Processor {
    private name: string;
    
    constructor(name: string) {
        this.name = name;
    }
    
    process(item: string): string {
        return `processed_${item}`;
    }
}
''', "processor.ts"),

        ("Haskell", '''
data Processor = Processor { name :: String }

processItem :: Processor -> String -> String
processItem _ item = "processed_" ++ item
''', "Processor.hs")
    ]
    
    print("="*90)
    print("MULTI-LANGUAGE CAPABILITIES REPORT")
    print("="*90)
    print(f"{'Language':<12} {'Model':<20} {'Analysis':<15} {'Fields':<8} {'Detected Fields'}")
    print("-"*90)
    
    results = []
    for language, code, filename in test_cases:
        result = test_language(language, code, filename)
        results.append(result)
        
        model_short = result["model"]
        fields_str = ",".join(result["fields"][:3])  # Show first 3 fields
        
        if result["success"]:
            print(f"{result['language']:<12} {model_short:<20} {result['analysis_method']:<15} "
                  f"{result['field_count']:<8} {fields_str}")
        else:
            print(f"{result['language']:<12} {'ERROR':<20} {'ERROR':<15} {0:<8} {result['error'][:30]}")
    
    print("-"*90)
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    python_fields = next((r["field_count"] for r in results if r["language"] == "Python"), 0)
    smart_models = sum(1 for r in results if r["success"] and "graphcodebert" in r["model"].lower() or "unixcoder" in r["model"].lower())
    
    print(f"SUMMARY:")
    print(f"- Successful extractions: {successful}/{len(results)}")
    print(f"- Python metadata fields: {python_fields}")
    print(f"- Languages with smart models: {smart_models}/{len(results)}")
    print(f"- Languages with metadata (>0 fields): {sum(1 for r in results if r['field_count'] > 0)}/{len(results)}")

if __name__ == "__main__":
    main()