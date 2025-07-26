#!/usr/bin/env python3

"""
Script to check what's actually in the database.
"""

import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

def check_database():
    """Check database schema and sample data."""
    load_dotenv()
    pool = ConnectionPool(os.getenv("COCOINDEX_DATABASE_URL"))
    
    with pool.connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # Get table name
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE '%code_embeddings%'
            """)
            tables = cur.fetchall()
            print("📋 Tables containing 'code_embeddings':")
            for table in tables:
                print(f"  - {table[0]}")
            
            if not tables:
                print("❌ No code_embeddings table found!")
                return
            
            table_name = tables[0][0]
            print(f"\n🔍 Using table: {table_name}")
            
            # Get column names
            cur.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """)
            columns = cur.fetchall()
            print(f"\n📊 Columns in {table_name}:")
            for col_name, col_type in columns:
                print(f"  - {col_name}: {col_type}")
            
            # Get sample row
            cur.execute(f"SELECT * FROM {table_name} LIMIT 1")
            sample = cur.fetchone()
            if sample:
                print(f"\n📝 Sample row:")
                column_names = [desc[0] for desc in cur.description]
                for i, (col_name, value) in enumerate(zip(column_names, sample)):
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"  {col_name}: {value}")
            
            # Check for language field specifically
            cur.execute(f"""
                SELECT DISTINCT language 
                FROM {table_name} 
                WHERE language IS NOT NULL 
                LIMIT 10
            """)
            languages = cur.fetchall()
            if languages:
                print(f"\n🗣️ Languages found:")
                for lang in languages:
                    print(f"  - {lang[0]}")
            else:
                print(f"\n❌ No language field data found")
            
            # Count total rows
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
            print(f"\n📈 Total rows: {count}")

if __name__ == "__main__":
    check_database()