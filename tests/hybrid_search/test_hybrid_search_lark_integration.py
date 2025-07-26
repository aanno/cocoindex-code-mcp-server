#!/usr/bin/env python3

"""
Test that hybrid search works with the new Lark-based keyword parser.
"""

import sys
import os

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'cocoindex_code_mcp_server'))

def test_hybrid_search_lark_integration():
    """Test that hybrid search imports and initializes with Lark parser."""
    
    try:
        # Test the imports work correctly
        from db.pgvector.hybrid_search import HybridSearchEngine
        from keyword_search_parser_lark import KeywordSearchParser, build_sql_where_clause
        
        print("✅ Successfully imported hybrid search with Lark parser")
        
        # Test parser initialization
        parser = KeywordSearchParser()
        print("✅ Lark parser initialized successfully")
        
        # Test some parsing functionality
        test_queries = [
            "language:python",
            "language:python and exists(embedding)",
            "(language:python or language:rust) and exists(embedding)",
            'value_contains(filename, "test") and language:python'
        ]
        
        print(f"\n🧪 Testing Lark Parser Functionality:")
        print("-" * 40)
        
        for query in test_queries:
            try:
                result = parser.parse(query)
                where_clause, params = build_sql_where_clause(result)
                print(f"✅ '{query}' -> SQL: {where_clause}")
            except Exception as e:
                print(f"❌ '{query}' -> ERROR: {e}")
                return False
        
        # Test that value_contains works (this is the new operator)
        value_contains_query = 'value_contains(code, "async")'
        try:
            result = parser.parse(value_contains_query)
            where_clause, params = build_sql_where_clause(result)
            expected_sql = "code ILIKE %s"
            expected_params = ["%async%"]
            
            if where_clause == expected_sql and params == expected_params:
                print(f"✅ value_contains operator working correctly")
                print(f"   SQL: {where_clause}")
                print(f"   Params: {params}")
            else:
                print(f"❌ value_contains operator not working correctly")
                print(f"   Expected SQL: {expected_sql}, got: {where_clause}")
                print(f"   Expected params: {expected_params}, got: {params}")
                return False
                
        except Exception as e:
            print(f"❌ value_contains test failed: {e}")
            return False
        
        print(f"\n🎉 All Lark parser integration tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nested_parentheses_with_value_contains():
    """Test the complex nested parentheses with value_contains that was previously failing."""
    
    try:
        from keyword_search_parser_lark import KeywordSearchParser, build_sql_where_clause
        
        parser = KeywordSearchParser()
        
        # This was the test that was previously skipped
        complex_query = "((language:python or language:rust) and exists(embedding)) or filename:test.py"
        
        print(f"\n🔍 Testing Complex Nested Query:")
        print("-" * 40)
        print(f"Query: {complex_query}")
        
        result = parser.parse(complex_query)
        where_clause, params = build_sql_where_clause(result)
        
        print(f"✅ Parsed successfully!")
        print(f"SQL: {where_clause}")
        print(f"Params: {params}")
        
        # Test with value_contains in complex query
        complex_value_contains = "(value_contains(code, \"async\") and language:python) or exists(embedding)"
        print(f"\nQuery with value_contains: {complex_value_contains}")
        
        result2 = parser.parse(complex_value_contains)
        where_clause2, params2 = build_sql_where_clause(result2)
        
        print(f"✅ Complex value_contains query parsed successfully!")
        print(f"SQL: {where_clause2}")
        print(f"Params: {params2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Complex query test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Hybrid Search Lark Integration Test")
    print("=" * 50)
    
    success1 = test_hybrid_search_lark_integration()
    success2 = test_nested_parentheses_with_value_contains()
    
    if success1 and success2:
        print(f"\n🎉 ALL HYBRID SEARCH LARK INTEGRATION TESTS PASSED!")
        print(f"✅ Hybrid search successfully updated to use Lark parser")
        print(f"✅ New value_contains operator working correctly")
        print(f"✅ Complex nested queries with parentheses working")
    else:
        print(f"\n❌ Some hybrid search integration tests failed")
        sys.exit(1)
