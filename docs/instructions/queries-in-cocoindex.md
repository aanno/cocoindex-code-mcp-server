<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# In cocoindex, is there support for queries as well? Is that support abstracting over the different queries possiblities in the supported vector db implementations?

**CocoIndex provides indexing support and query interoperability, but does not directly implement or abstract over underlying vector database query languages.**

## Query Support in CocoIndex

**CocoIndex** is fundamentally a **data transformation and indexing framework**, designed to make it easy to extract, embed, and index structured and unstructured data—especially for use in AI and retrieval-augmented generation (RAG) pipelines[^1_1][^1_4][^1_7]. Its core focus is on defining repeatable, incremental indexing workflows (flows) and storing the resulting embeddings and metadata in vector databases or other storage backends[^1_1][^1_4].

When it comes to **querying**, CocoIndex does *not* provide a unified, database-agnostic query (SQL, graph, etc.) abstraction layer. Instead, it offers *integration points* for you to implement your own query handlers using libraries or frameworks of your choice[^1_3]. For example, in its tutorials, you write a simple interactive loop to accept natural language queries, compute embeddings, and then call a search function (defined in your code) that leverages the target database’s own query engine (e.g., Postgres’s native vector search capabilities)[^1_1]. **CocoIndex does not provide or enforce a single query API across all supported backends**.

## Examples of Query Handling in CocoIndex

- **Custom Query Handlers:** You can define your own query logic in Python, using the same embedding model as used during indexing, to search your indexed data. For instance, the `SimpleSemanticsQueryHandler` allows you to search via semantic similarity, but the actual vector search is implemented by your code and the underlying database[^1_4].
- **Interoperability:** CocoIndex’s workflows are composable and data-lineage-aware, so you can share transformations and logic between indexing and querying, and easily reference the correct tables (e.g., using default naming conventions)[^1_3].
- **No Database-Agnostic Abstraction:** There is no indication in the documentation or examples that CocoIndex provides a single query API that works identically across all supported vector databases (e.g., Postgres, Pinecone, Redis, etc.); rather, you write your own adapter code for each backend[^1_1][^1_3].
- **Terminal Interaction:** Tutorials show interactive loops in the terminal for simple semantic search, but this is a convenience, not a database abstraction—each backend’s query capabilities are exposed as-is[^1_1][^1_5].

## Summary Table

| Feature | CocoIndex Support | Details |
| :-- | :-- | :-- |
| Indexing workflows | Yes | Define flows to extract, embed, and store data in various backends[^1_1][^1_4][^1_7]. |
| Generic query language/abstraction | No | No unified query API; you implement queries using the target database’s capabilities[^1_3]. |
| Custom query handlers | Yes | Example handlers shown for semantic search; you provide the actual query logic[^1_4]. |
| Database-agnostic vector search | No | You write backend-specific code for vector querying; no abstraction layer provided[^1_1][^1_3]. |

**In summary:**
CocoIndex is a powerful framework for defining and orchestrating indexing workflows, and it provides hooks for you to build your own query capabilities, but it does not abstract away the differences between vector database query languages or provide a single, unified query interface across all backends[^1_1][^1_3][^1_4]. You must implement the query side yourself, tailored to each storage engine you use.

<div style="text-align: center">⁂</div>

[^1_1]: <https://substack.com/home/post/p-164288038>

[^1_3]: <https://cocoindex.io/docs/query>

[^1_4]: <https://cocoindexio.substack.com/p/index-codebase-with-tree-sitter-and>

[^1_5]: <https://www.youtube.com/watch?v=gv5R8nOXsWU>

[^1_7]: <https://github.com/cocoindex-io/cocoindex>

---

# If I want to support several vector dbs as storage engine in a cocoindex application, what is the best approach to that?

To **support several vector DBs as storage backends in a CocoIndex application**, the best approach is to **design your indexing and transformation logic to be storage-agnostic**, using CocoIndex’s extensible dataflow model to export results to multiple target systems as needed. Here’s how to achieve this in practice, based on the available documentation:

## Key Principles

- **Define indexing flows in a declarative, backend-agnostic manner.**
Your core data transformations, chunking, and embedding logic should be specified without direct ties to a specific vector DB. CocoIndex’s flow-based pipeline lets you express these operations in Python, with data types and schema inferred from your transformations[^2_3].
- **Use CocoIndex’s storage export interface for each backend.**
When you are ready to store the results, you can route data to multiple vector stores using CocoIndex’s storage adapters. For example, you can call `.export()` with different storage targets, such as `Qdrant`, `Postgres`, `Neo4j`, or `Kuzu`[^2_2][^2_3]. Each call specifies connection details for that particular backend.
- **Schema inference and automatic setup.**
CocoIndex can automate schema setup for certain integrations, such as Qdrant, so you don’t have to manually configure collections, tables, or indexes. But you still must specify which storage backend you want to use at export time[^2_3].
Be aware that Qdrant, for example, only requires schema for vector fields, while relational backends like Postgres may need more detailed configuration[^2_3]. CocoIndex handles these differences by deriving the schema from your flow definition[^2_3].
- **Separate target configuration from transformation logic.**
Keep your core transformation logic (Python functions defining the flow) separate from the storage configuration. This makes it easier to switch, add, or update backends without rewriting your data pipeline.
- **Migration and maintenance.**
CocoIndex supports incremental processing and can handle backends with different capabilities (e.g., some support in-place schema updates, others require drop-and-recreate)[^2_3]. The `cocoindex setup` and `cocoindex update` commands help manage migrations and schema changes across backends[^2_3].

## Example Workflow

- **Write a CocoIndex flow** that processes your data, embeds it, and collects the results.
- **At export time, select your target vector DBs** by invoking the appropriate storage adapter for each backend. For instance, you could export to both Qdrant and Postgres by adding two export steps, each with its own connection details.
- **Handle schema differences** by relying on CocoIndex’s schema inference and setup commands. Know your backends’ limitations—for example, Qdrant requires the vector size and distance metric at collection creation time, while relational databases may need more schema information[^2_3].
- **Deploy and maintain** with CocoIndex’s CLI tools (`cocoindex setup`, `cocoindex update`), which will manage schema migrations and data synchronization for each backend[^2_3].

## Recommendations

- **For maximum flexibility**, consider using a configuration file or environment variables to specify backend connection details, so you can switch between dev, staging, and production environments (and between different stores) without code changes.
- **For some use cases**, you may want to build your own thin abstraction layer around CocoIndex’s export interface, to make it even easier to add or remove backends, but this is not strictly necessary—CocoIndex’s current architecture is already designed for multi-backend support[^2_3].
- **Monitor and test** each backend’s performance and capabilities, as schema handling, indexing speed, and query features may vary.

## Summary

**The best approach is to use CocoIndex’s flow definition to create storage-agnostic transformation logic, then use its export system to send data to multiple vector databases as needed.**
This keeps your application flexible, maintains clear data lineage, and leverages CocoIndex’s schema inference and migration tools to handle the differences between backend systems[^2_3]. No single, built-in query abstraction is provided, but the indexing and export pipeline is designed for multi-backend operation out of the box.

<div style="text-align: center">⁂</div>

[^2_2]: <https://qdrant.tech/documentation/data-management/cocoindex/>

[^2_3]: <https://cocoindex.io/blogs/schema-inference-for-qdrant/>

---

# I heart there is (beta) support for hybrid search in cocoindex. What does that mean? Small code examples would be good

There is **no direct evidence of hybrid search support in CocoIndex** in the provided search results. The results that discuss **hybrid search** are not about CocoIndex at all, but rather about platforms like Couchbase[^3_1], Google Vertex AI[^3_2], and Pinecone[^3_4]. Hybrid search, in the context of vector databases, typically means **combining vector (semantic) search with keyword (BM25, full-text, term-matching) search** in a way that improves recall and relevance of results[^3_5][^3_8].

If a CocoIndex beta does indeed offer hybrid search, it would logically imply that **your CocoIndex workflow can support, in some form, both semantic (vector) and keyword (text-based) search operations in a single query**—though the provided results do not confirm or detail how this works specifically in CocoIndex.

## What Hybrid Search Typically Involves

**Hybrid search** architectures generally require:

- **Both embeddings and keyword metadata:** Each document or chunk should have both vector embeddings and textual fields (for keyword/BM25 search).
- **A unified query pipeline:** Your application, or the database itself, needs to run both types of search (vector and keyword) and fuse (rerank, interleave, or otherwise combine) the results for the user.
- **Configuration or declarative setup:** The search indexes need to be configured for both text and vector search, often with special indexing rules.

### Example Workflow (Generic, Not CocoIndex-Specific)

Suppose you have a **CocoIndex flow** that processes and embeds documents, indexing both the embeddings and the source text in a supported backend (e.g., PostgreSQL, Qdrant). To **add hybrid search** (based on how other platforms work[^3_1][^3_2][^3_5]), your application code might:

- **Store both embeddings and full-text fields** in the database.
- **Define a hybrid query function** that runs a vector search and a separate keyword search, then combines the results.
- **Optionally, normalize and rerank the combined result sets** for relevance.

#### Example Python Snippet (Pseudocode, Hypothetical CocoIndex Integration)

```python
import cocoindex as ci
from pyvi import HybridSearchClient  # Hypothetical, not a real lib

# 1. Define a CocoIndex flow to process and store both embeddings and text
flow = ci.Flow(...)
flow \
    .chunk() \
    .embed(model='text-embedding-3-small', output_field='embedding') \
    .save(text_fields=['full_text'], vector_field='embedding', backend='postgres')

# 2. Export to target backend(s)
flow.export(storage='postgres://...')

# 3. Define a hybrid query handler
def hybrid_search(query: str, num_results: int = 10):

    # Perform vector (semantic) search
    semantic_results = ci.vector_search(query, field='embedding', top_k=num_results)

    # Perform keyword (text) search
    keyword_results = ci.text_search(query, field='full_text', top_k=num_results)

    # Combine, rerank, and deduplicate
    all_results = semantic_results + keyword_results
    reranked = rerank(all_results, query)  # You define rerank() logic
    return reranked[:num_results]
```

But **this is a synthetic example**—neither `ci.vector_search` nor `ci.text_search` are confirmed CocoIndex APIs, nor is `HybridSearchClient` a real library. If CocoIndex has true hybrid search support, it would **likely provide a built-in way to define hybrid indexing and querying**, similar to how Couchbase[^3_1] or Vertex AI[^3_2] do, but specific syntax and supported backends would need to be confirmed from CocoIndex’s official documentation.

## What to Do Until Official Documentation Is Available

- **Monitor CocoIndex documentation** for beta features, especially around search APIs and query capabilities.
- **Consider a hybrid architecture in your logic:** If CocoIndex does not abstract this, you can still build it yourself by combining its export/query features with custom backend query logic.
- **Check for community threads or discussions** about hybrid search in CocoIndex—if it exists, there may be user-provided examples or workarounds.

## Summary Table (Generic Hybrid, Not CocoIndex-Specific)

| Step | What you do | Where to look in CocoIndex (hypothetical) |
| :-- | :-- | :-- |
| Indexing | Store both embeddings and text | `.save(text_fields=[...], vector_field=...)` |
| Querying | Run both vector and keyword search | Custom query handlers or (maybe) beta API |
| Reranking | Combine/fuse/rerank results | Application logic or (maybe) built-in |

## Conclusion

**If CocoIndex offers beta hybrid search, expect built-in support for running both vector and keyword searches and combining results.** Until official documentation is available, you can manually implement this pattern by combining CocoIndex’s embedding workflow with custom query logic in your target backend. **No real CocoIndex hybrid search code examples are confirmed in the provided results.** If you have access to beta documentation or example repos, refer to those for accurate syntax and usage.

<div style="text-align: center">⁂</div>

[^3_1]: <https://github.com/couchbase-examples/hybrid-search-demo>

[^3_2]: <https://cloud.google.com/vertex-ai/docs/vector-search/about-hybrid-search>

[^3_4]: <https://www.pinecone.io/learn/hybrid-search-intro/>

[^3_5]: <https://learn.microsoft.com/en-us/fabric/database/cosmos-db/hybrid-search>

[^3_8]: <https://towardsdatascience.com/rag-hybrid-search-based-on-two-indexes-2b35444fa96d/>

---

# No, I think the hybrid search support in cocoindex is only on the indexing site (not for querying)

Based on the available documentation and code examples, **CocoIndex is primarily an advanced ETL (Extract, Transform, Load) framework for AI indexing workloads**. It excels at defining indexing flows—extracting, transforming, and embedding data, then exporting it to various storage backends (e.g., Qdrant, Postgres)[^4_3][^4_4]. Its design centers on **incremental, schema-aware data processing for indexing**, with built-in support for maintaining data freshness and lineage[^4_4][^4_7].

### What “Hybrid Search Support” Likely Means in CocoIndex

- **Current documentation does not mention hybrid search**—either on the indexing or querying side—by name or by example[^4_3][^4_4].
- **All official examples focus on exporting embeddings (vector data) to supported backends**[^4_3]. For instance, you can configure an export to Qdrant, specifying the vector field and primary key, but there is no indication of built-in support for also exporting full-text or keyword fields necessary for hybrid search (e.g., BM25 or FTS + vector search)[^4_3].
- **CocoIndex’s own query utilities** (for retrieval) are referenced but not documented in detail; the emphasis is on **exporting for later retrieval**, not on implementing search logic directly within CocoIndex[^4_4].
- **If hybrid search is supported “only on the indexing side,”** the most plausible interpretation is that **CocoIndex allows you to define flows that extract, transform, and export both textual content and vector embeddings** to your backend, effectively enabling you to build a hybrid search-capable index (because the raw content and embeddings are present). However, CocoIndex does not implement the search fusion or reranking logic itself; **you must handle that in your application or using the capabilities of your backend**.

### Code Implications

```python
# Hypothetical example: exporting both text and embeddings for hybrid search
# (This is not a real CocoIndex API—just illustrative.)

flow = (
    cocoindex.Flow()
    .import_from(...)                 # Bring in your documents
    .transform(...)                   # Clean, chunk, etc.
    .embed(text_field="content", output_field="embedding")  # Create embeddings
    .keep_fields(["content", "embedding", "id"])  # Keep both text and vector
    .export(
        "hybrid_index",
        cocoindex.storages.Qdrant(
            collection_name="hybrid_docs",
            grpc_url="...",
            api_key="...",
        ),
        primary_key_fields=["id"],
        setup_by_user=True,
    )
)
```

**In this flow** (synthetic, for illustration), you ensure both the raw text (`content`) and embeddings (`embedding`) are exported. To perform hybrid search, your application (or backend) must then implement a query that uses both fields.

### What Is *Not* Present

- **No built-in reranking or fusion of BM25 and vector search results** in CocoIndex itself (unlike, e.g., LanceDB, which provides reranking utilities[^4_2]).
- **No query DSL or abstraction for hybrid search**—you export the data, then query using your backend’s capabilities.
- **No explicit documentation of hybrid indexing workflows**—just the general ability to export multiple fields, which *could* include those needed for hybrid search, if you design your flow accordingly.

### Summary Table

| Feature | CocoIndex Support | Details/Implications |
| :-- | :-- | :-- |
| Export both text \& embeddings | Yes (possible via flow) | Lets you build a hybrid-search-capable index |
| Query fusion/reranking | No | Must implement in your app or backend |
| Hybrid search abstraction | No | Export only—search logic not in CocoIndex |

### Conclusion

**If hybrid search “support” exists in CocoIndex, it means you can export both textual content and embeddings to your backend via its flexible ETL flows—enabling hybrid search infrastructure—but CocoIndex itself does not handle querying, result fusion, or reranking.**
You must implement those components separately, using the capabilities of your chosen storage backend (e.g., Qdrant, Postgres, Elasticsearch). There is no evidence in the documentation of a built-in hybrid search query API or automatic result fusion within CocoIndex[^4_3][^4_4].

If you’d like to see a real, minimal hybrid indexing flow example for a specific backend, clarify which database you’re using, and I can adapt the illustration accordingly. But as of now, **CocoIndex’s role is to facilitate hybrid index *construction* (via export), not hybrid query *execution***.

<div style="text-align: center">⁂</div>

[^4_2]: <https://lancedb.github.io/lancedb/hybrid_search/hybrid_search/>

[^4_3]: <https://qdrant.tech/documentation/data-management/cocoindex/>

[^4_4]: <https://cocoindex.io/docs/core/basics>

[^4_7]: <https://github.com/cocoindex-io/cocoindex>

---

# Please elaborate on 'CocoIndex’s own query utilities' a bit. Perhaps you could refer to code with in cocoindex to start my own research

CocoIndex’s **core focus is on ETL and indexing**—transforming data, embedding it, and exporting it to backend stores like Postgres or Qdrant[^5_1][^5_3]. For **querying**, CocoIndex *explicitly does not provide a fully abstracted, database-agnostic query language or API*—you use the native capabilities of your backend or wrap those in your application logic[^5_2].

However, **CocoIndex does offer some utilities and patterns to make the transition between indexing and querying smoother**, especially for common tasks like ensuring consistency between how embeddings are generated during indexing and how they are generated during querying.

## CocoIndex Query Utilities: Highlights

- **No high-level query abstraction**: You write your own query logic, using libraries like `psycopg` for Postgres or SDKs for other backends[^5_1][^5_2].
- **Shared transformation logic**: CocoIndex lets you *reuse the same embedding logic* for both indexing and querying, so you get the same vector representations at query time as you did during indexing—crucial for accurate semantic search[^5_1].
- **Table name resolution**: Utilities like `cocoindex.utils.get_target_default_name()` help you reference the correct index/table by name, even if you didn’t specify it explicitly in your export[^5_1].
- **Transform flow evaluation**: You can evaluate CocoIndex transform flows (e.g., embedding a query string) directly in your query logic, using the same functions as the indexing pipeline[^5_1].
- **Seamless integration**: The framework is design to let you share code and config between indexing and querying, making maintenance and iteration easier[^5_2].

## Example: Building a Query Handler with CocoIndex Utilities

Here’s a condensed version of the **quickstart.py** example[^5_1], illustrating how CocoIndex’s utilities fit into a typical query workflow:

```python
import cocoindex
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
import os
from numpy.typing import NDArray
import numpy as np

# 1. Define a shared embedding transform (used in both indexing and querying)
@cocoindex.transform_flow()
def text_to_embedding(text: cocoindex.DataSlice[str]) -> cocoindex.DataSlice[NDArray[np.float32]]:
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"))

# 2. Get the default table name for the export target in your flow
table_name = cocoindex.utils.get_target_default_name(text_embedding_flow, "doc_embeddings")

# 3. Build a query function that uses the same embedding logic
def search(pool: ConnectionPool, query: str, top_k: int = 5):
    query_vector = text_to_embedding.eval(query)
    with pool.connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT filename, text, embedding <=> %s AS distance
                FROM {table_name} ORDER BY distance LIMIT %s
            """, (query_vector, top_k))
            return [
                {"filename": row[^5_0], "text": row[^5_1], "score": 1.0 - row[^5_2]}
                for row in cur.fetchall()
            ]

# 4. Main loop
if __name__ == "__main__":
    cocoindex.init()
    pool = ConnectionPool(os.getenv("COCOINDEX_DATABASE_URL"))
    while True:
        query = input("Enter search query: ")
        if not query: break
        results = search(pool, query)
        for r in results:
            print(f"[{r['score']:.3f}] {r['filename']}\n{r['text']}\n---")
```

**Key points from this architecture**:

- **Both chunking and embedding** (during indexing) and **embedding the query string** (during search) use the exact same `SentenceTransformerEmbed` model and parameters, thanks to the `text_to_embedding` transform flow[^5_1].
- **Table name resolution** is handled by `get_target_default_name`, so your query logic does not hardcode table names[^5_1].
- **Query logic itself** is standard SQL (in this case, for pgvector/Postgres), with no CocoIndex-specific query DSL—just utilities for consistency and maintenance[^5_1].
- **You are expected to write your own query loop**, handle connection pooling, and process results as you see fit[^5_1].

## What CocoIndex Does Not Provide

- **There is no query DSL, no unified query API across backends, and no automatic hybrid (BM25 + vector) search fusion**[^5_2].
- **All search logic must be implemented by you, using your backend’s native query language and capabilities**[^5_2].
- **CocoIndex’s utilities are for reuse, consistency, and maintainability, not for hiding database differences**.

## Where to Start Your Research

- **Study the official quickstart**, especially the sections that show how to share embedding logic between indexing and querying[^5_1].
- **Look at how `transform_flow` is used for both indexing and querying** to keep embeddings consistent.
- **Explore the `utils` module** for helpers like table naming, especially if you use default export targets.
- **Write your own query handlers** for each backend, using the utilities as glue between your indexing and querying logic.

## Summary Table

| Utility/Pattern | Purpose | Example Code Reference |
| :-- | :-- | :-- |
| `transform_flow` | Reuse embeddings for index \& query | `text_to_embedding.eval(query)` |
| `get_target_default_name` | Resolve default table/collection name | `utils.get_target_default_name` |
| Direct backend query | Perform actual searches | `psycopg`/SQL in example |

**In summary:**
CocoIndex’s “query utilities” are **small, focused helpers for maintaining consistency and reducing boilerplate** between indexing and querying, but **all substantive query logic is your responsibility**[^5_1][^5_2]. Start with the quickstart example to see these patterns in action, then adapt them to your backend and use case.

<div style="text-align: center">⁂</div>

[^5_1]: <https://cocoindex.io/docs/getting_started/quickstart>

[^5_2]: <https://cocoindex.io/docs/query>

[^5_3]: <https://github.com/cocoindex-io/cocoindex>

---

## Key Naming Differences: ASTChunk vs CocoIndex

When working with both **ASTChunk** (used for chunking code via abstract syntax trees) and **CocoIndex** (as the broader ETL/indexing framework), it’s common to notice that **key naming conventions can diverge** between these two tools. This often means that the output from one (like ASTChunk) doesn’t align directly with what CocoIndex expects as its canonical field or key names during indexing or export, necessitating a mapping step.

### Why Do Key Naming Conflicts Occur?

- **ASTChunk** tools and code parsers often generate detailed tree-based outputs, introducing fields such as `node_type`, `start_line`, `end_line`, `tokens`, or context-specific chunk identifiers.
- **CocoIndex**, by contrast, leans toward a more general, schema-inferred approach favoring unambiguous and standard names like `id`, `content`, `chunk`, and `embedding`.
- When integrating both, you frequently encounter **name mismatches between ASTChunk’s detailed, code-centric output and CocoIndex’s expected standard fields**, especially for primary keys, content, or metadata.

### Typical Solution: Field Mapping

To reconcile these differences:

- Most workflows introduce a **mapping or renaming operation** to align ASTChunk-generated fields with the expected CocoIndex schema before export or further transformation.
- This usually happens in a post-processing or pre-export step, ensuring, for instance, that what ASTChunk calls `node_id` maps to CocoIndex’s `id`, or `code` or `text` fields are renamed to `content`.
- The mapping step is essential for:
  + Primary key consistency (e.g., mapping the chunk or node key to `id`).
  + Embeddings association (matching ASTChunk representations to CocoIndex’s `embedding` field).
  + Compatibility with backend storage schema that CocoIndex will infer or create automatically[^6_1][^6_2].

### Practical Workflow Illustration

Here’s what a typical pipeline might look like:

1. **ASTChunk parses codebase → produces rich chunk objects with custom keys.**
2. **Map/rename fields**: e.g.,
    + `node_id` → `id`
    + `code` or `chunk_text` → `content`
    + `ast_embedding` → `embedding`
3. **Pass to CocoIndex flow**, which now receives objects with standard field names.
4. **CocoIndex handles further processing and export**, leveraging its schema inference and automation features[^6_1][^6_2].

### Considerations \& Best Practices

- Use mapping functions or explicit schema declarations in your pipeline code to minimize confusion, especially if you intend to switch chunking strategies or integrate multiple parsers.
- Refer to CocoIndex’s inference system for supported/expected field names before mapping, so as to avoid schema setup failures or data loss.
- The need for mapping is a reflection of **CocoIndex’s flexible, flow-based model**—it assumes diverse sources and encourages standardizing structure before indexing or storage[^6_1][^6_2].

## Summary Table: Why Mapping Is Needed

| Origin | Field Example | Standard CocoIndex Name | Requires Mapping? |
| :-- | :-- | :-- | :-- |
| ASTChunk | node_id | id | Yes |
| ASTChunk | code | content | Yes |
| CocoIndex | id, content | id, content | No |
| ASTChunk | custom tokens | Not used directly | Maybe |

### Conclusion

**Key name mapping is a standard step when using ASTChunk in combination with CocoIndex, due to their differing field conventions.** This ensures compatibility with CocoIndex’s schema automation, smooth downstream processing, and reliable storage across supported backends[^6_1][^6_2].

<div style="text-align: center">⁂</div>

[^6_1]: <https://hackernoon.com/this-open-source-tool-could-save-your-data-team-hundreds-of-hours>

[^6_2]: <https://qdrant.tech/documentation/data-management/cocoindex/>
