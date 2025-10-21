#!/usr/bin/env python3

import psycopg

conn = psycopg.connect(
    "postgresql://cocoindex:cocoindex@host.docker.internal/cocoindex"
)

# psycopg3 also supports autocommit at connection level
conn.autocommit = True

with conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

conn.close()
