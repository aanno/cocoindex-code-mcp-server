# embedding metadata

1. Our MCP is able to use 'smart embeddings'. This means that the embedding used depends on the programming language of the code being processed, see docs/claude/Embedding-Selection.md .
2. Because of some problems (see below), we currently do not use this feature, and hence start the MCP server with the --default-embedding flag, which uses a single embedding model for all languages.
3. To overcome the problem, we need to add the embedding used to the metadata of each chunk. See docs/cocoindex/metadata.md for details about metadata handling. We want a database field 'embedding_model' to store the embedding model used for each chunk.
4. The problem is that you must not compare embedding vectors created with different models.
5. Hence, after we get embedding_model column, we have to modify the search code to only compare chunks with the same embedding_model as the one used for the query. The easiest way is to require the language or the embedding_model field for each query.

We could easily test this with starting the MCP server, wait for the end of scanning, and then look into the DB if the column is appropriate.
