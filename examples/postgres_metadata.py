#!/usr/bin/env python3
import os

from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg import connect

# Lade Umgebungsvariablen
load_dotenv()
database_url = os.getenv("COCOINDEX_DATABASE_URL")
if not database_url:
    raise ValueError(".env fehlt oder COCOINDEX_DATABASE_URL fehlt!")

# Connection nutzen
with connect(database_url, autocommit=True) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        # Alle Datenbanken auflisten
        cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
        print("Datenbanken:")
        for db in cur.fetchall():
            print("-", db[0])

        # Tabellen auflisten
        cur.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type='BASE TABLE'
              AND table_schema NOT IN ('pg_catalog','information_schema');
        """)
        print("\nTABELLEN:")
        for schema, table in cur.fetchall():
            print(f"* {schema}.{table}")

        # Spalten, Typen etc.
        cur.execute("""
            SELECT table_schema, table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog','information_schema');
        """)
        print("\nSPALTEN:")
        for row in cur.fetchall():
            print(row)

        # Primärschlüssel
        cur.execute("""
            SELECT kcu.table_schema, kcu.table_name, tco.constraint_name, kcu.column_name
            FROM information_schema.table_constraints tco
            JOIN information_schema.key_column_usage kcu ON kcu.constraint_name = tco.constraint_name
                AND kcu.constraint_schema = tco.constraint_schema
            WHERE tco.constraint_type = 'PRIMARY KEY'
              AND kcu.table_schema NOT IN ('pg_catalog','information_schema')
            ORDER BY kcu.table_schema, kcu.table_name;
        """)
        print("\nPrimärschlüssel:")
        for row in cur.fetchall():
            print(row)

        # Indices (inkl. Details)
        cur.execute("""
            SELECT schemaname, tablename, indexname, indexdef
            FROM pg_indexes
            WHERE schemaname NOT IN ('pg_catalog','information_schema');
        """)
        print("\nINDICES:")
        for row in cur.fetchall():
            print(row)

        # pgvector-Spalten
        cur.execute("""
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE data_type = 'vector';
        """)
        print("\npgvector-Spalten:")
        for row in cur.fetchall():
            print(row)

        # JSONB-Spalten
        cur.execute("""
            SELECT table_schema, table_name, column_name
            FROM information_schema.columns
            WHERE data_type = 'jsonb';
        """)
        print("\nJSONB-Spalten:")
        for row in cur.fetchall():
            print(row)

        # Fulltext-Indizes (tsvector über GIN/GIN-Trigram)
        cur.execute("""
            SELECT schemaname, tablename, indexname, indexdef
            FROM pg_indexes
            WHERE (indexdef ~* 'gin' AND indexdef ~* 'to_tsvector')
               AND schemaname NOT IN ('pg_catalog','information_schema');
        """)
        print("\nFulltext-Indizes:")
        for row in cur.fetchall():
            print(row)
