#!/usr/bin/env python3
"""
Sample Python file for testing CocoIndex functionality.

This file provides realistic Python code for testing chunking,
metadata extraction, and other CocoIndex operations.
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SampleClass:
    """A sample dataclass for testing."""
    name: str
    value: int = 0
    
    def get_info(self) -> Dict[str, Any]:
        """Return information about this instance."""
        return {
            "name": self.name,
            "value": self.value,
            "type": "SampleClass"
        }


def process_data(data: List[Dict[str, Any]]) -> List[SampleClass]:
    """Process a list of data dictionaries into SampleClass instances.
    
    Args:
        data: List of dictionaries containing 'name' and optionally 'value'
        
    Returns:
        List of SampleClass instances
    """
    results = []
    
    for item in data:
        if "name" not in item:
            continue
            
        instance = SampleClass(
            name=item["name"],
            value=item.get("value", 0)
        )
        results.append(instance)
    
    return results


async def async_function(delay: float = 1.0) -> str:
    """An async function for testing async detection."""
    import asyncio
    await asyncio.sleep(delay)
    return "async_result"


def complex_function_with_types(
    items: List[str], 
    mapping: Dict[str, int], 
    callback=None
) -> Dict[str, List[int]]:
    """A more complex function with type hints for testing complexity analysis.
    
    This function demonstrates various Python features that should be
    detected by metadata extraction:
    - Type hints
    - Default parameters
    - Complex return types
    - List/dict comprehensions
    - Conditional logic
    """
    result = {}
    
    for item in items:
        if item in mapping:
            values = [mapping[item] * i for i in range(3)]
            
            if callback:
                values = [callback(v) for v in values]
                
            result[item] = values
    
    return result


# Module-level variables
CONSTANT_VALUE = 42
configuration = {
    "debug": True,
    "max_items": 100,
    "processors": ["default", "advanced"]
}


if __name__ == "__main__":
    # Example usage
    sample_data = [
        {"name": "item1", "value": 10},
        {"name": "item2", "value": 20},
        {"name": "item3"}  # No value, should default to 0
    ]
    
    instances = process_data(sample_data)
    
    for instance in instances:
        print(json.dumps(instance.get_info(), indent=2))