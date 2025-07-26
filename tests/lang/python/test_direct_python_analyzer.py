#!/usr/bin/env python3

"""
Test the enhanced Python analyzer directly to see if the new metadata fields are working.
"""

import sys

def test_direct_python_analyzer():
    """Test the PythonCodeAnalyzer directly."""
    
    try:
        from cocoindex_code_mcp_server.lang.python.python_code_analyzer import PythonCodeAnalyzer
    except ImportError as e:
        print(f"❌ Could not import analyzer: {e}")
        return False
    
    # Simple test code
    test_code = """
import os
from typing import List

class TestClass:
    '''A test class.'''
    
    def method(self, x: int) -> str:
        '''A test method.'''
        return str(x)

def function() -> bool:
    '''A test function.'''
    return True
"""
    
    print("🧪 Testing Direct Python Analyzer")
    print("=" * 50)
    
    try:
        analyzer = PythonCodeAnalyzer()
        metadata = analyzer.analyze_code(test_code, "test_direct.py")
        
        print(f"✅ Analysis completed successfully!")
        print(f"📊 Analysis Method: {metadata.get('analysis_method', 'unknown')}")
        
        # Check for enhanced fields
        enhanced_fields = [
            'file', 'node_type', 'lines_of_code', 'hash', 'content_hash',
            'node_relationships', 'additional_metadata', 'start_line', 'end_line'
        ]
        
        print(f"\n📋 Enhanced Fields Check:")
        print("-" * 30)
        
        for field in enhanced_fields:
            if field in metadata:
                print(f"✅ {field}: {type(metadata[field]).__name__}")
            else:
                print(f"❌ {field}: MISSING")
        
        # Check function details
        if 'function_details' in metadata:
            func_details = metadata['function_details']
            if func_details:
                print(f"\n🔍 Function Details Check:")
                print("-" * 30)
                sample_func = func_details[0]
                enhanced_func_fields = ['end_line', 'column', 'end_column', 'lines_of_code']
                for field in enhanced_func_fields:
                    if field in sample_func:
                        print(f"✅ function.{field}: {sample_func[field]}")
                    else:
                        print(f"❌ function.{field}: MISSING")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_direct_python_analyzer()
    if success:
        print(f"\n✅ Direct Python analyzer test completed!")
    else:
        print(f"\n❌ Direct Python analyzer test failed!")
        sys.exit(1)
