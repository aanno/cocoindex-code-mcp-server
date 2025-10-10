#!/usr/bin/env python3

"""
Database Comparison Utility for Search Tests

This module provides functionality to compare search test results with actual
PostgreSQL database contents to help debug discrepancies between expected
test outcomes and stored data.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import asyncpg  # type: ignore[import-untyped]


@dataclass
class DBComparisonResult:
    """Result of comparing test expectations with database reality."""
    test_name: str
    query: Dict[str, Any]
    expected_item: Dict[str, Any]
    matching_db_records: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    discrepancies: List[str]


class DatabaseComparator:
    """Compare search test results with PostgreSQL database contents."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize with database connection details."""
        load_dotenv()
        self.database_url = database_url or os.getenv('COCOINDEX_DATABASE_URL')
        if not self.database_url:
            raise ValueError("Database URL not provided and COCOINDEX_DATABASE_URL not set in .env")
        
        self.connection: Optional[asyncpg.Connection] = None
        
    async def connect(self):
        """Connect to the PostgreSQL database."""
        if not self.connection:
            self.connection = await asyncpg.connect(self.database_url)
            logging.info(f"✅ Connected to PostgreSQL database")
    
    async def disconnect(self):
        """Disconnect from the PostgreSQL database."""
        if self.connection:
            await self.connection.close()
            self.connection = None
            logging.info("✅ Disconnected from PostgreSQL database")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def query_database(self, sql: str, *args) -> List[Dict[str, Any]]:
        """Execute SQL query and return results as list of dictionaries."""
        if not self.connection:
            await self.connect()

        assert self.connection is not None, "Connection must be established"
        rows = await self.connection.fetch(sql, *args)
        return [dict(row) for row in rows]
    
    async def get_code_embeddings_for_file(self, filename_pattern: str) -> List[Dict[str, Any]]:
        """Get all code embeddings records matching a filename pattern."""
        sql = """
        SELECT 
            filename, location, language, 
            functions, classes, imports,
            complexity_score, has_classes, has_async, has_type_hints,
            analysis_method, chunking_method,
            success, parse_errors, char_count,
            metadata_json::text as metadata_json_text
        FROM codeembedding__code_embeddings 
        WHERE filename LIKE $1
        ORDER BY filename, location
        """
        return await self.query_database(sql, f"%{filename_pattern}%")
    
    async def get_code_embeddings_by_metadata(self, **filters) -> List[Dict[str, Any]]:
        """Get code embeddings records matching metadata filters."""
        conditions = []
        args = []
        arg_num = 1
        
        for field, value in filters.items():
            if field == 'language':
                conditions.append(f"language = ${arg_num}")
                args.append(value)
            elif field == 'has_classes':
                conditions.append(f"has_classes = ${arg_num}")
                args.append(value)
            elif field == 'complexity_score_gt':
                conditions.append(f"complexity_score > ${arg_num}")
                args.append(value)
            elif field == 'functions_contains':
                conditions.append(f"functions LIKE ${arg_num}")
                args.append(f"%{value}%")
            arg_num += 1
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        sql = f"""
        SELECT 
            filename, location, language, 
            functions, classes, imports,
            complexity_score, has_classes, has_async, has_type_hints,
            analysis_method, chunking_method,
            success, parse_errors, char_count,
            metadata_json::text as metadata_json_text
        FROM codeembedding__code_embeddings 
        WHERE {where_clause}
        ORDER BY filename, location
        LIMIT 50
        """
        
        return await self.query_database(sql, *args)
    
    async def analyze_test_failure(
        self, 
        test_name: str, 
        query: Dict[str, Any], 
        expected_item: Dict[str, Any],
        search_results: List[Dict[str, Any]]
    ) -> DBComparisonResult:
        """Analyze a failing test by comparing with database contents."""
        
        logging.info(f"🔍 Analyzing test failure: {test_name}")
        
        # Extract search conditions from keyword query
        keyword_query = query.get('keyword_query', '')
        
        # Try to find matching records in database
        matching_db_records = []
        
        # Parse common query patterns
        if 'language:java AND has_classes:true' in keyword_query:
            matching_db_records = await self.get_code_embeddings_by_metadata(
                language='Java', 
                has_classes=True
            )
        elif 'language:' in keyword_query:
            language = self._extract_language_from_query(keyword_query)
            if language:
                matching_db_records = await self.get_code_embeddings_by_metadata(language=language)
        elif 'functions:' in keyword_query:
            function_name = self._extract_function_from_query(keyword_query)
            if function_name:
                matching_db_records = await self.get_code_embeddings_by_metadata(
                    functions_contains=function_name
                )
        
        # Analyze discrepancies
        discrepancies = self._identify_discrepancies(expected_item, matching_db_records, search_results)
        
        return DBComparisonResult(
            test_name=test_name,
            query=query,
            expected_item=expected_item,
            matching_db_records=matching_db_records,
            search_results=search_results,
            discrepancies=discrepancies
        )
    
    def _extract_language_from_query(self, query: str) -> Optional[str]:
        """Extract language from keyword query string."""
        import re
        match = re.search(r'language:(\w+)', query.lower())
        if match:
            lang = match.group(1)
            # Map common language names to database values
            lang_map = {
                'python': 'Python',
                'java': 'Java', 
                'rust': 'Rust',
                'javascript': 'JavaScript',
                'typescript': 'TypeScript',
                'c': 'C',
                'cpp': 'C++',
                'haskell': 'Haskell',
                'kotlin': 'Kotlin'
            }
            return lang_map.get(lang, lang.capitalize())
        return None
    
    def _extract_function_from_query(self, query: str) -> Optional[str]:
        """Extract function name from keyword query string."""
        import re
        match = re.search(r'functions:(\w+)', query)
        return match.group(1) if match else None
    
    def _identify_discrepancies(
        self, 
        expected_item: Dict[str, Any], 
        db_records: List[Dict[str, Any]], 
        search_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify discrepancies between expected results and database reality."""
        
        discrepancies = []
        
        # Check if we have any database records at all
        if not db_records:
            discrepancies.append("No matching records found in database")
            return discrepancies
        
        # Check expected metadata against database records
        expected_metadata = expected_item.get('expected_metadata', {})
        
        for field, expected_value in expected_metadata.items():
            if field == 'complexity_score':
                if expected_value.startswith('>'):
                    threshold = float(expected_value[1:])
                    max_complexity = max((r.get('complexity_score', 0) or 0) for r in db_records)
                    if max_complexity <= threshold:
                        discrepancies.append(
                            f"Expected complexity_score > {threshold}, but max found in DB is {max_complexity}"
                        )
            elif field == 'has_classes':
                db_has_classes_values = [r.get('has_classes', False) for r in db_records]
                if expected_value not in db_has_classes_values:
                    discrepancies.append(
                        f"Expected has_classes={expected_value}, but DB values are: {set(db_has_classes_values)}"
                    )
            elif field == 'language':
                db_languages = [r.get('language') for r in db_records]
                if expected_value.capitalize() not in [lang.capitalize() if lang else None for lang in db_languages]:
                    discrepancies.append(
                        f"Expected language={expected_value}, but DB languages are: {set(db_languages)}"
                    )
            elif field == 'functions' and expected_value == '!empty':
                empty_function_records = [r for r in db_records
                                         if not r.get('functions') or
                                         (isinstance(r.get('functions'), str) and r.get('functions').strip() == '')]  # type: ignore[union-attr]
                if empty_function_records:
                    discrepancies.append(
                        f"Expected non-empty functions, but {len(empty_function_records)} records have empty functions"
                    )
            elif field == 'classes' and expected_value == '!empty':
                empty_class_records = [r for r in db_records
                                      if not r.get('classes') or
                                      (isinstance(r.get('classes'), str) and r.get('classes').strip() == '')]  # type: ignore[union-attr]
                if empty_class_records:
                    discrepancies.append(
                        f"Expected non-empty classes, but {len(empty_class_records)} records have empty classes"
                    )
        
        return discrepancies
    
    def format_comparison_report(self, result: DBComparisonResult) -> str:
        """Format a detailed comparison report."""
        
        report = f"""
=== Database Comparison Report ===
Test: {result.test_name}
Query: {result.query}
Expected: {result.expected_item}

🔍 Database Records Found: {len(result.matching_db_records)}
📊 Search Results Found: {len(result.search_results)}

"""
        
        if result.discrepancies:
            report += "❌ Discrepancies Found:\n"
            for i, discrepancy in enumerate(result.discrepancies, 1):
                report += f"  {i}. {discrepancy}\n"
        else:
            report += "✅ No discrepancies found\n"
        
        if result.matching_db_records:
            report += f"\n📋 Sample Database Records ({min(3, len(result.matching_db_records))}):\n"
            for i, record in enumerate(result.matching_db_records[:3]):
                report += f"  Record {i+1}:\n"
                report += f"    Filename: {record.get('filename', 'N/A')}\n"
                report += f"    Language: {record.get('language', 'N/A')}\n"
                report += f"    Functions: {record.get('functions', 'N/A')}\n"
                report += f"    Classes: {record.get('classes', 'N/A')}\n"
                report += f"    Complexity: {record.get('complexity_score', 'N/A')}\n"
                report += f"    Has Classes: {record.get('has_classes', 'N/A')}\n"
                report += f"    Analysis Method: {record.get('analysis_method', 'N/A')}\n"
                report += "\n"
        
        return report


async def compare_test_with_database(
    test_name: str,
    query: Dict[str, Any], 
    expected_item: Dict[str, Any],
    search_results: List[Dict[str, Any]]
) -> DBComparisonResult:
    """Compare a single test result with database contents."""
    
    async with DatabaseComparator() as comparator:
        return await comparator.analyze_test_failure(
            test_name, query, expected_item, search_results
        )


if __name__ == "__main__":
    # Example usage
    async def main():
        test_name = "complexity_filter_high"
        query = {"keyword_query": "language:java AND has_classes:true"}
        expected_item = {
            "expected_metadata": {
                "complexity_score": ">2",
                "classes": "!empty", 
                "functions": "!empty",
                "has_classes": True,
                "analysis_method": "!unknown"
            }
        }
        search_results: List[Dict[str, Any]] = []  # Would come from actual search
        
        result = await compare_test_with_database(test_name, query, expected_item, search_results)
        print(DatabaseComparator().format_comparison_report(result))
    
    asyncio.run(main())