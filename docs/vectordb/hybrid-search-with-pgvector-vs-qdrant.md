<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# I have a few questions regarding qdrant hybrid search and full text search. so far I have understood the following: text for full text is stored separately as metadata. what search operators are available for full text? what operators are available for hybrid search? what operators are available for metadata search?

**Full text search**, **hybrid search**, and **metadata search** in Qdrant use different query mechanisms and available operators:

### Full Text Search Operators

- In Qdrant, true full text search (with operators like AND, OR, NOT, phrase queries, wildcards, etc.) is typically not natively implemented at the payload level. Payload (metadata) indexes support only basic matching and filtering[^1_2].
- As of early 2025, users report that payload indexing does not support complex full text search operators and is limited in capability[^1_2].
- Advanced full text search capabilities (such as phrase search, wildcard matching, stemming) are not core to Qdrant; for rich full text search, most implementations use an external search service or combine sparse vector models like BM25 or SPLADE for keyword matching[^1_1][^1_9].

### Hybrid Search Operators

- **Hybrid search** in Qdrant (v1.10+) natively supports combining:
  + **Dense vector search** (semantic/embedding-based)
  + **Sparse vector search** (lexical/keyword-based, e.g., BM25, Splade)[^1_1][^1_3][^1_5].
- The primary *operator* for combining results is the **fusion type**:
  + `rrf` (Reciprocal Rank Fusion): Combines ranks from different queries, giving priority to items appearing high in both[^1_5].
  + `dbsf` (Distribution-Based Score Fusion, v1.11+): Normalizes and sums scores across queries[^1_5].
- Hybrid search queries can use the **prefetch** parameter to chain queries and nest them to build complex, multi-stage searches[^1_5].

### Metadata Search Operators

- Qdrant supports **payload filtering** using basic operators:
  + **Equality and inequality**: `==`, `!=`
  + **Range**: `<`, `>`, `<=`, `>=`
  + **Set membership**: `IN`, `NOT IN`
  + **Exists/Not Exists** (field presence)
- Logical operators: `AND`, `OR`, `NOT` can be applied to combine filters[^1_5].
- These filters can be used in combination with vector or hybrid search queries to restrict results[^1_5].

#### Summary Table

| Search Type | Operators Supported |
| :-- | :-- |
| Full Text Search | Basic keyword filtering via metadata. No native advanced text operators (AND, OR, NOT, phrase, etc.)[^1_2]. |
| Hybrid Search | Dense and sparse vector search; fusion via RRF, DBSF; prefetch and nested queries[^1_1][^1_5]. |
| Metadata Search | Equality/inequality, range, set membership, exists/not exists; logical AND, OR, NOT[^1_5]. |

For complex full text search, you usually need to use external search services or combine with sparse vector models and handle operators at the application level[^1_1][^1_2]. Qdrant’s strength is in vector and hybrid (vector + metadata) search with simple payload filtering.

<div style="text-align: center">⁂</div>

[^1_1]: <https://docs.llamaindex.ai/en/stable/examples/vector_stores/qdrant_hybrid/>

[^1_2]: <https://github.com/qdrant/qdrant/issues/5817>

[^1_3]: <https://qdrant.tech/articles/hybrid-search/>

[^1_5]: <https://qdrant.tech/documentation/concepts/hybrid-queries/>

[^1_9]: <https://github.com/qdrant/workshop-ultimate-hybrid-search>

---

# hence metadata could contain key to scalar or key to sequence?

Yes, **metadata** (called **payload** in Qdrant) can contain mappings from a **key** to a **scalar value** (such as a string, integer, or boolean) or a **key** to a **sequence** (such as an array/list of values)[^2_5]. You can store:

- Key-value pairs where the value is a single scalar (e.g., `"category": "news"` or `"year": 2024`)
- Key-value pairs where the value is a list/array (e.g., `"tags": ["AI", "ML", "Databases"]`)

This means Qdrant supports both **scalar values** and **arrays/sequences** as payload values for metadata attached to each point[^2_5].

<div style="text-align: center">⁂</div>

[^2_5]: <https://qdrant.tech/documentation/concepts/points/>

---

# I have not understood the BM25 or SPLADE trick for having more full featured full text search. please elaborate on this. what are the pros and cons?

The **BM25** and **SPLADE** "trick" in Qdrant and similar vector databases refers to using these algorithms to generate **sparse vectors** from text. These sparse vectors let you conduct more advanced, keyword-aware searches—approaching the capabilities of "full text search"—while still benefiting from vector-based (and potentially hybrid) retrieval.

### How it Works

- **BM25** is a classic keyword-based ranking model for information retrieval. It produces a score for each document based on the frequency and uniqueness of terms (TF-IDF-like), normalized by document length. In vector databases, BM25 is usually implemented as a way to compute "sparse vectors," where each dimension corresponds to a term[^3_3][^3_6].
- **SPLADE** uses transformer models (like BERT) but creates *sparse* output vectors, where each dimension still represents a vocabulary term and its importance in a given context[^3_1][^3_3][^3_7]. Unlike BM25, SPLADE can expand queries and encode richer semantics, helping match on synonyms and rephrasing while retaining keyword explainability.

You index your documents by converting them into sparse vectors using BM25 or SPLADE, then index those vectors alongside your dense (semantic) vectors. Query text is similarly converted into sparse vectors at search time.

### What "More Full-Featured" Full Text Search Means

- **BM25** and **SPLADE** *approximate* classic full text search features:
  + **Keyword matching** with relevance ranking.
  + **Handling of field weights** (BM25F).
  + Some token handling (e.g., stemming, lemmatization, stopword removal if you pre-process).
  + **Query expansion** (SPLADE can add related terms that aren't explicitly in the query)[^3_1][^3_7].
- They **do not natively support**:
  + Complex Boolean logic (AND/OR/NOT with parentheses and precedence).
  + Wildcards and fuzzy searching.
  + Phrase or proximity queries.
  + Full query language features typical in Elasticsearch or Lucene.
- But, they *do* allow **hybrid search**—combining keyword recall with neural semantic search in a single workflow.

### Pros and Cons

| Method | Pros | Cons |
| :-- | :-- | :-- |
| BM25 | - Fast and lightweight<br>- Well-understood<br>- Great for exact keyword matches | - Only keyword aware<br>- No semantics or context<br>- Limited handling of synonyms/rephrasings |
| SPLADE | - Keyword+context aware<br>- More explainable than dense vectors<br>- Handles expansions (synonyms, context) | - Slower than BM25 due to more non-zero entries and transformer complexity<br>- Requires more memory and compute<br>- Not natively supported by all databases, pre/post-processing steps needed[^3_2] |
| Both | - Can combine with dense vectors in hybrid search for better recall and relevance<br>- Sparse vectors are interpretable | - Not a 1:1 replacement for advanced full text features<br>- Loss of full query language flexibility |

### Key Takeaways

- BM25/SPLADE **let you go beyond simple payload filters**—they provide ranking and scoring, able to consider all relevant documents rather than just filtering.
- You gain much of the *recall and ranking power* of a dedicated full text engine, and can blend it with neural search results.
- Still, some advanced features like proximity, complex Boolean queries, or custom analyzers require an external text search engine or heavier application logic[^3_1][^3_3][^3_6][^3_7].

If you need transparent, explainable, hybrid keyword and semantic search—and can work within some query limitations—this trick is powerful, but doesn't fully replace a feature-complete text search engine.

<div style="text-align: center">⁂</div>

[^3_1]: <https://qdrant.tech/articles/sparse-vectors/>

[^3_2]: <https://www.pinecone.io/learn/splade/>

[^3_3]: <https://weaviate.io/blog/hybrid-search-explained>

[^3_6]: <https://docs.paradedb.com/documentation/concepts/bm25>

[^3_7]: <https://infiniflow.org/blog/best-hybrid-search-solution>

---

# hence using BM25 or SPLADE maps some features of full text search to a vector DB. it has nothing to do with native full text in qdrant and it available with all vector DBs

Your understanding is correct: **using BM25 or SPLADE with a vector database like Qdrant allows you to add keyword-aware, somewhat full text-like search capabilities by mapping text into sparse vectors. This is not the same as "native" full text search capability in Qdrant or similar databases**.

**Key points:**

- **BM25 and SPLADE generate sparse vectors** based on text content. Each dimension represents a token/word, and non-zero values indicate relevance or importance for that term. This allows you to do efficient retrieval and ranking based on keyword overlap and (with SPLADE) context expansion[^4_5][^4_2][^4_7].
- **Qdrant itself does not have robust, native full text search** features like phrase, fuzzy, wildcard, or Boolean queries[^4_2]. Instead, BM25/SPLADE trick lets you bring in some of these capabilities as vector operations, primarily keyword relevance and (for SPLADE) some query expansion.
- **This approach is not unique to Qdrant**; it works in other vector databases as well (Milvus, Weaviate, Pinecone, etc.), provided they support sparse or hybrid retrieval[^4_1][^4_3][^4_4][^4_5].

### Pros

- **No need for a separate full text search engine** (like Elasticsearch) for decent keyword-based search[^4_6].
- **Supports hybrid search:** You can blend these sparse (keyword) vectors with dense (semantic) vectors to improve both recall and relevancy[^4_3][^4_5].
- **Explainability:** Sparse vectors are more interpretable; you see which keywords contributed most to the score[^4_2][^4_5].
- **Works across many vector DBs:** Not tied to Qdrant; the trick is general-purpose[^4_1][^4_3][^4_5].

### Cons

- **Lacks advanced full text features:** Can’t do complex Boolean logic, wildcards, fuzzy, phrase, or proximity queries within Qdrant itself[^4_2][^4_5].
- **Performance:** SPLADE, in particular, produces vectors with more non-zero elements, which can be slower to retrieve and need more memory/storage[^4_4][^4_5].
- **Compatibility/complexity:** Not always natively supported or as efficient as classic inverted indexes; sometimes requires custom pre/post-processing, especially for SPLADE[^4_4].
- **Does not fully replace full text engines** for heavy text analytics or use cases needing advanced text query language[^4_2][^4_5].

In summary: **BM25/SPLADE lets you map some "full text" features into the vector space for retrieval—making it possible to do useful keyword-based ranking and hybrid semantic search without needing Qdrant’s native full text search, and this pattern works with all major vector databases**[^4_1][^4_3][^4_4][^4_5][^4_6].

<div style="text-align: center">⁂</div>

[^4_1]: <https://zilliz.com/learn/mastering-bm25-a-deep-dive-into-the-algorithm-and-application-in-milvus>

[^4_2]: <https://qdrant.tech/articles/sparse-vectors/>

[^4_3]: <https://weaviate.io/blog/hybrid-search-explained>

[^4_4]: <https://www.pinecone.io/learn/splade/>

[^4_5]: <https://zilliz.com/learn/comparing-splade-sparse-vectors-with-bm25>

[^4_6]: <https://infiniflow.org/blog/sparse-embedding-bm25>

[^4_7]: <https://qdrant.tech/documentation/fastembed/fastembed-splade/>

---

# ok. now let's focus on qdrant native full text and hybrid search support only. what operators are available. small code example would be great

**Qdrant's native full text search and hybrid search** features provide several operators, but their capabilities are more limited than traditional full text engines.

### Operators for Native Full Text Search

- The main operator is the `match` condition with `text` type.
  + **If a full text index exists on a field**: the query checks if each word (token) in the search string exists in the field, regardless of order. All words must be present (logical AND)[^5_7][^5_5].
    - Example: searching `"good cheap"` matches if both "good" and "cheap" appear in the field, anywhere.
  + **If no full text index exists**: Qdrant falls back to checking for an exact substring match—less flexible and usually much slower[^5_7][^5_5][^5_1].
- Tokenizer can be customized (`word`, `whitespace`, `prefix`) at index creation—this affects the granularity of token matching, not the runtime query logic[^5_5].
- **No native support for phrase matching, wildcards, partial match, or fuzzy queries** (phrase search is planned but not available as of July 2025)[^5_6].
- No synonym expansion: queries match only on the literal tokens (e.g., searching "UK" will not match "United Kingdom")[^5_8].

### Operators for Native Hybrid Search

- Hybrid search = combining **vector similarity** with **full text or metadata filtering** in a single query, using logical AND for overall filtering[^5_3].
- You can apply any available filter (like `match`, range, set membership, etc.) on metadata alongside vector search.
- The query results are first filtered according to the filter, then the vector search is applied on the filtered subset.

### Example (Python Pseudocode) — Hybrid Full Text + Vector Search

```python
# Example query: vector search filtered to points where 'description' contains both 'good' and 'cheap'
filter = {
    "must": [
        {
            "key": "description",
            "match": {"text": "good cheap"}
        }
    ]
}

query_vector = [0.2, 0.1, 0.9, 0.7]  # Example embedding

results = client.search(
    collection_name="your_collection",
    query_vector=query_vector,
    filter=filter,
    limit=5
)
```

### Summary Table

| Search Type | Operators/Behavior |
| :-- | :-- |
| Full Text Search (native) | `match: {text: "word1 word2"}` — returns if **all words are present** (AND). No phrase/wildcard/fuzzy support. Customizable tokenizer at index time. |
| Hybrid Search (native) | Combine vector search with any metadata/full text filter using logical AND (filter+vector query). |
| Metadata Filtering | `==`, `!=`, `<`, `>`, `IN`, `MATCH`, etc. Combined with AND, OR, NOT logic in filters. |

**Limitations:**

- No phrase search, no synonym match, no wildcards or fuzzy matching.
- Full text queries are essentially logical AND of tokens, not full Lucene-style search.

For advanced text queries, you’ll need to combine Qdrant with a dedicated text search system.

<div style="text-align: center">⁂</div>

[^5_1]: <https://qdrant.tech/articles/qdrant-introduces-full-text-filters-and-indexes/>

[^5_3]: <https://qdrant.tech/documentation/concepts/search/>

[^5_5]: <https://blog.qdrant.tech/qdrant-introduces-full-text-filters-and-indexes-9a032fcb5fa?gi=1378f151b6d8>

[^5_6]: <https://github.com/qdrant/qdrant/issues/2955>

[^5_7]: <https://qdrant.tech/documentation/concepts/filtering/>

[^5_8]: <https://stackoverflow.com/questions/79689817/qdrant-full-text-payload-index-langchain-selfqueryretriever-how-to-match-syno>

---

# you now mentioned filter. please define it a bit. so far I have no idea what a range (filter) can do

A **filter** in Qdrant is a way to limit search results based on conditions applied to the metadata (payload) of each vector point. Filters can be combined in flexible and expressive ways to support a wide variety of business needs, such as restricting results to particular categories, price ranges, availability statuses, locations, and more[^6_1][^6_3][^6_5][^6_7].

### Filter Structure and Operators

**Filter** objects use logical clauses:

- `must`: All contained conditions must be true (logical AND).
- `should`: At least one contained condition must be true (logical OR).
- `must_not`: Exclude results matching contained conditions (logical NOT)[^6_1][^6_7].

#### Simple Filtering Operators

- **match** – exact value match (string, int, bool, etc.)
- **range** – numeric or date ranges (e.g., greater than, less than, between)[^6_1][^6_7]
- **geo** – filter by geographic location (bounding box, polygon, radius)
- **full_text_match** – search for presence of a word (if full text index present)
- **nested_key/object** – operate on nested fields

#### Range Filter Examples

A **range filter** works on numerical or datetime fields, supporting:

- `gt`, `gte` (greater than, greater/equal)
- `lt`, `lte` (less than, less/equal)[^6_7]

**Example:**

```json
{
  "key": "price",
  "range": {
    "gte": 100,
    "lte": 500
  }
}
```

This filter restricts results to records where `price` is between 100 and 500, inclusive.

### Example: Combined Filter

Request vectors with:

- `price` between 100 and 500,
- `category` is `"clothing"`,
- `brand` is `"nike"` or `"adidas"`,
- and **not** `out_of_stock`.

```json
{
  "must": [
    { "key": "price", "range": { "gte": 100, "lte": 500 } },
    { "key": "category", "match": { "value": "clothing" } }
  ],
  "should": [
    { "key": "brand", "match": { "value": "nike" } },
    { "key": "brand", "match": { "value": "adidas" } }
  ],
  "must_not": [
    { "key": "out_of_stock", "match": { "value": true } }
  ]
}
```

This filter selects items in the desired price/category, with either `nike` or `adidas` as the brand, and excludes those that are out of stock[^6_7].

### Key Points

- **Range filters** allow you to select values within a numeric or date interval.
- Filters work efficiently even on large datasets, provided relevant payload indexes are set up[^6_5].
- Filters are applied during search—results are computed from only those records satisfying filter conditions, not after-the-fact[^6_5].

For even more detailed filter types and code samples, refer to Qdrant's filtering documentation[^6_1][^6_3][^6_7].

<div style="text-align: center">⁂</div>

[^6_1]: <https://qdrant.tech/articles/vector-search-filtering/>

[^6_3]: <https://qdrant.tech/documentation/concepts/filtering/>

[^6_5]: <https://www.cohorte.co/blog/a-developers-friendly-guide-to-qdrant-vector-database>

[^6_7]: <https://zenn.dev/taksnr/articles/a18ff137e23256>

---

# ok. I've got the impression that pgvector may also support full text and hybrid search, but different from qdrant

Your impression is accurate: **pgvector** enables **vector similarity search** in PostgreSQL, but its “full text” and “hybrid” search operate differently from Qdrant.

## Full Text Search in PostgreSQL + pgvector

- **Native full text search in PostgreSQL** uses:
  + The `tsvector` and `tsquery` types for tokenization, stemming, and advanced query parsing (`AND`, `OR`, `NOT`, phrase, prefix, weighting, language dictionaries, and ranking).
  + GIN/GIST indexes for performance.
- **Operators supported:**
  + `@@` operator (match against query),
  + Boolean logic (`&`, `|`, `!` corresponding to `AND`, `OR`, `NOT`),
  + Phrase and proximity search,
  + Prefix search (`:*`),
  + Ranking functions like `ts_rank`[^7_6][^7_7][^7_5].
- **Example:**

```sql
-- Match rows where the 'content' field contains both 'pizza' and 'review'
SELECT * FROM articles
WHERE to_tsvector('english', content) @@ to_tsquery('pizza & review');
```

## Hybrid Search (Combining Full Text + Vector Search)

Hybrid search is typically done in **two steps**:

1. **Filter and/or rank with full text search** (using `tsvector`, `tsquery` features).
2. **Rank or re-rank with vector similarity** (using `pgvector` extension and embedding columns).

This gives you both precise keyword filtering and semantic ranking[^7_1][^7_8][^7_2][^7_4][^7_9].

**Example:**

```sql
-- Example hybrid search: filter by full-text, then order by embedding similarity
SELECT *,
    1 - (embedding <=> '[0.2, 0.3, ...]') AS embedding_score
FROM articles
WHERE to_tsvector('english', content) @@ to_tsquery('pizza & review')
ORDER BY embedding_score DESC
LIMIT 10;
```

- `<=>` is pgvector's vector distance operator (cosine distance, Euclidean, or inner product, depending on setup)[^7_3].

**You can combine filter conditions, too:**

- Range filters (`price BETWEEN 100 AND 500`)
- Metadata exact or pattern match
- Joins, etc.

## Feature Comparison: pgvector+Postgres vs. Qdrant

| Feature | pgvector + PostgreSQL | Qdrant |
| :-- | :-- | :-- |
| Full Text Search | Advanced: Boolean logic, phrase, ranking, prefix, etc. | Limited: Only AND of tokens in field; no phrase/OR/wildcard |
| Hybrid Search | Full text + vector search in same SQL query | Metadata/full text filters + vector similarity |
| Indexing | GIN/GIST for FTS, custom for vector | Payload (metadata) and HNSW for vectors |
| Range/Other Filters | Standard SQL WHERE clause supports all typical cases | Supported via payload filters |

## Key Takeaways

- **Postgres with pgvector** offers classic full text search features and arbitrary hybrid queries using standard SQL[^7_5][^7_6][^7_7].
- **You can combine advanced text search with semantic search and any SQL expressions or joins**—this flexibility is unique compared to specialized vector DBs.
- **Qdrant** provides simpler but more limited native full text and hybrid search.

Let me know if you need specific Postgres code combining more complex filters, or if you want to see "hybrid" code mixing SQL full text and vector search patterns.

<div style="text-align: center">⁂</div>

[^7_1]: <https://www.pedroalonso.net/blog/advanced-postgres-full-text-search-with-pgvector/>

[^7_2]: <https://www.timescale.com/blog/combining-semantic-search-and-full-text-search-in-postgresql-with-cohere-pgvector-and-pgai>

[^7_3]: <https://github.com/pgvector/pgvector>

[^7_4]: <https://www.tigerdata.com/blog/postgresql-hybrid-search-using-pgvector-and-cohere>

[^7_5]: <https://www.enterprisedb.com/blog/enhancing-search-capabilities-postgresql-standard-semantic>

[^7_6]: <https://www.postgresql.org/docs/current/textsearch-controls.html>

[^7_7]: <https://www.postgresql.org/docs/current/datatype-textsearch.html>

[^7_8]: <https://www.tigerdata.com/learn/full-text-search-vs-vector-search>

[^7_9]: <https://jkatz.github.io/post/postgres/hybrid-search-postgres-pgvector/>

---

# for pgvector you distinguish between rank, rerank, and filter. elaborate on this. what is the difference between tsvector and tsquery?

Here’s a clear breakdown answering both parts of your query:

## 1. Difference between **rank**, **rerank**, and **filter** (in context of pgvector/PostgreSQL)

- **Filter**:
  + *Definition*: Narrows down candidate database rows before any ranking.
  + *How*: Uses SQL `WHERE` clauses (for numbers, strings, Booleans, ranges, full text, etc.).
  + *Example*:

```sql
SELECT * FROM documents
WHERE to_tsvector('english', content) @@ to_tsquery('pizza & review')
      AND published_at > '2025-01-01'
```

Here, only documents with both 'pizza' and 'review' **AND** published after 2025-01-01 are returned.

- **Rank**:
  + *Definition*: Orders results according to a chosen relevance score.
  + *How*: Use full text ranking (such as `ts_rank`), vector similarity (e.g., `<=>` for embeddings), or other scoring.
  + *Example*:

```sql
SELECT *, ts_rank(to_tsvector('english', content), to_tsquery('pizza & review')) AS text_rank
FROM documents
WHERE ...
ORDER BY text_rank DESC
LIMIT 10
```

or with vectors:

```sql
SELECT *, 1 - (embedding <=> '[...]') AS embedding_score
FROM documents
WHERE ...
ORDER BY embedding_score DESC
LIMIT 10
```

- **Rerank**:
  + *Definition*: Applies a new, usually more expensive ranking step **after** an initial retrieval (often using a neural model on the top N).
  + *Purpose*: Boosts relevance by using more advanced models, sometimes combining semantic and lexical similarity.
  + *How*: Fetch initial candidates (`filter` + `rank`), then pass to a reranker and sort again.
  + *Example workflow*:

1. Retrieve top 30 by vector similarity.
2. For each, use a cross-encoder (or LLM or advanced model) to compute query-document relevance.
3. Sort by reranker score, select top 5[^8_1][^8_4][^8_5][^8_7].

## 2. Difference between **tsvector** and **tsquery** in PostgreSQL

| Type | Description | Example |
| :-- | :-- | :-- |
| tsvector | Preprocessed **document** representation for full text search; combines tokens, supports weighting, splitting, etc. | `to_tsvector('english', 'pizza reviews')` |
| tsquery | Preprocessed **search query** representation; can include boolean operators, wildcards, phrases, etc. | `to_tsquery('pizza & review')` |

- **tsvector:** Indexes and represents the content/text of your documents after tokenization, stemming, and normalization.
- **tsquery:** Encodes what to look for: search string(s) with operators (`&`, `|`, `!` for AND, OR, NOT), and supports phrase (`<->`), prefix (`:*`), and complex queries.

**Putting them together:**

```sql
-- Find docs about pizza AND review, order by similarity
SELECT *
FROM articles
WHERE to_tsvector('english', content) @@ to_tsquery('pizza & review')
ORDER BY ts_rank(to_tsvector('english', content), to_tsquery('pizza & review')) DESC;
```

- `to_tsvector('english', content)`: turns each document into a searchable vector.
- `to_tsquery('pizza & review')`: turns your query ("pizza and review") into an efficient query form.
- `@@`: returns true for matches between the document and the query.

### Summary Table

| Step | Purpose | Example (SQL) |
| :-- | :-- | :-- |
| Filter | Reduce dataset before ranking | `WHERE date > '2025-01-01'` |
| Rank | Order candidates by a score (text or vector) | `ORDER BY ts_rank(...)` or `ORDER BY 1 - (vector <=> ...)` |
| Rerank | Reorder retrieved candidates using a secondary model | Post-process with LLM or cross-encoder |

Let me know if you’d like a step-by-step rerank SQL/code sample or further examples for advanced full text matching!

<div style="text-align: center">⁂</div>

[^8_1]: <https://mastra.ai/examples/rag/rerank>

[^8_4]: <https://www.tigerdata.com/blog/postgresql-hybrid-search-using-pgvector-and-cohere>

[^8_5]: <https://unfoldai.com/rag-rerankers/>

[^8_7]: <https://www.pinecone.io/learn/series/rag/rerankers/>

---

# pgvector enables very sophisticated queries. what are the pros and cons on pgvector Vs qdrant? why to use qdrant at all?

**pgvector** and **Qdrant** serve different needs in the vector database ecosystem. Here’s a focused analysis on their pros and cons, and why you might choose Qdrant, even though pgvector enables sophisticated queries:

## Pros \& Cons: pgvector vs. Qdrant

| Aspect | pgvector (PostgreSQL) | Qdrant |
| :-- | :-- | :-- |
| Integration | Full SQL, native joins, transactional ACID, no new system to operate | REST/gRPC APIs, new system to run but focused purpose |
| Query Sophistication | Combines SQL, full text (`tsvector`, `tsquery`), relational joins, filtering, ranking | Advanced hybrid, payload, geo, quantization, complex filtering |
| Performance (Vector) | Slower and less accurate for large/high-dim datasets; drops sharply past ~100k docs[^9_2][^9_5] | Purpose-built for speed, scale (billion vectors), high recall and throughput[^9_1][^9_2][^9_5][^9_7] |
| Scalability | Vertical; limited by PostgreSQL's single-node nature | Horizontally scalable, sharding, high-concurrency |
| Ecosystem | Benefit from PostgreSQL’s mature tooling, backup, monitoring, etc. | Specialized, but less mature general-purpose ecosystem |
| Full Text Search | Very advanced; Boolean logic, ranking, phrase/proximity, languages, etc. | Basic: AND-over-tokens, no phrase/wildcard/fuzzy/search language |
| Hybrid Search | Extremely flexible: combine vectors with any SQL logic | Strong hybrid (vector + payload/full text), including advanced fusion |
| Operational Overhead | Low if you’re already using Postgres | Adds a new technology to stack |
| Indexing | Brute-force or IVF/HNSW (via pgvectorscale), can be slower as dataset grows[^9_6][^9_8] | HNSW and other ANN methods, optimized for high-throughput |

## Why Use Qdrant Instead of pgvector?

- **Speed \& Scale**: Qdrant outperforms pgvector by up to 15x on throughput and maintains higher recall/accuracy in high-dimensional or large-scale settings (100k+ vectors)[^9_1][^9_2][^9_5]. Benchmarks regularly show Qdrant’s purpose-built ANN search beating Postgres, especially as datasets scale.
- **Specialized Features**: Qdrant supports aggressive optimization: advanced quantization, payload indexing, sharding, distribution, and advanced hybrid fusion, not natively present or performant in Postgres[^9_1][^9_7].
- **Horizontal Scalability**: Qdrant is built from the ground up for clustering and distributed workloads, making it a better choice for use cases approaching billions of vectors.
- **Low Latency, High Recall**: For mission-critical applications (real-time recommendations, semantic search, etc.), Qdrant’s low tail latency and high recall reputation outperform what’s feasible on even optimized Postgres[^9_1][^9_2][^9_5][^9_7].
- **Geospatial and Multi-modal Filtering**: Qdrant includes optimized geo and hybrid payload filters out-of-the-box, making it easier to build AI and recommendation platforms.

## When is pgvector Enough (or Superior)?

- **You need complex SQL logic or advanced full text** (ranked, fuzzy, phrase, etc.)—Postgres is unbeaten here[^9_5][^9_6][^9_8].
- **No extra infra**: If you’re already running Postgres, adding pgvector is minimally disruptive and low on ops overhead.
- **Smaller datasets**: For datasets well under 100,000–500,000 vectors, Postgres+pgvector is often “fast enough”[^9_2][^9_5][^9_6][^9_8].
- **Transactional workloads**: When you want atomic updates across relational and vector data (e.g., in e-commerce).

## Typical Guidance

- For **AI/semantic search, RAG, LLM retrieval, large-scale recommendation** or anything >1M vectors: **Qdrant** offers stronger performance, scalability, filtering, and real-time capabilities[^9_1][^9_2][^9_5][^9_7].
- For **multi-purpose, transaction-heavy, and SQL-centric data stacks**, or if you need rich full text and it “just needs to work” at moderate scale: **pgvector** makes more sense[^9_5][^9_6][^9_8].

**In summary:**
**Qdrant** is designed for high-throughput, large-scale, real-time vector search and offers advanced hybrid/filtering/geo/quantization features.
**pgvector** is unbeatable for flexible, transactional hybrid queries inside PostgreSQL but hits scaling and accuracy limits quickly as data grows[^9_1][^9_2][^9_5][^9_8].
You’d choose Qdrant when speed, accuracy, and scale are critical for vector-heavy workloads, or you expect to grow beyond what a monolithic relational DB can offer.

<div style="text-align: center">⁂</div>

[^9_1]: <https://myscale.com/blog/comprehensive-comparison-pgvector-vs-qdrant-performance-vector-database-benchmarks/>

[^9_2]: <https://nirantk.com/writing/pgvector-vs-qdrant/>

[^9_5]: <https://zilliz.com/comparison/qdrant-vs-pgvector>

[^9_6]: <https://dev.to/tigerdata/postgresql-vs-qdrant-for-vector-search-50m-embedding-benchmark-3hhe>

[^9_7]: <https://airbyte.com/data-engineering-resources/qdrant-vs-pinecone>

[^9_8]: <https://dev.to/tigerdata/postgres-vs-qdrant-why-postgres-wins-for-ai-and-vector-workloads-3d71>

---

# can you elaborate on  purpose-built ANN search ?

**Purpose-built ANN (Approximate Nearest Neighbor) search** refers to the core technology in modern vector databases like Qdrant, Milvus, Pinecone, and Weaviate. These databases are specifically architected to make finding the “closest” (most similar) vectors to a query *extremely fast and scalable*, even in datasets with millions or billions of items.

### What is ANN Search?

- **Nearest neighbor search** finds the points in a dataset that are most similar (nearest) to a query vector in high-dimensional space.
- **“Brute-force” search** would require comparing the query to every vector in the set— which is slow and impractical at scale.
- **ANN algorithms** trade a bit of strict accuracy for vastly improved speed, returning “close-enough” results—**not always the mathematically true nearest, but nearly always close enough for practical retrieval**[^10_1][^10_5][^10_6].

### How Does Purpose-Built ANN Work in Vector Databases?

These databases use specialized algorithms and data structures for indexing and retrieving vectors, such as:

- **HNSW (Hierarchical Navigable Small World graphs):** Layered graph structures that make it possible to “jump” closer and closer to the query point with very few actual comparisons, resulting in rapid search times[^10_5].
- **IVF (Inverted File Index):** Partitions vectors into clusters and searches only the most promising ones[^10_5].
- **PQ (Product Quantization):** Compresses vectors into compact codes, reducing memory use and search costs[^10_2][^10_5].
- **LSH (Locality Sensitive Hashing):** Hashes similar items into the same buckets for speed[^10_2].

These methods enable:

- **Sublinear time searches (much faster than brute force), even over hundreds of millions or billions of vectors**
- **Real-time search applications**—retrieval can happen in milliseconds[^10_1][^10_3][^10_4].
- **Scalability and reliability:** Sharding, disk-based/indexed storage, persistence, and distributed computing, all unavailable or cumbersome with traditional DBs[^10_3][^10_4][^10_5].

### Why Not Just Use a Traditional Database?

Traditional databases (like SQL or NoSQL systems) are optimized for structured data, exact matches, and simple range queries—not high-dimensional similarity. Even when they support vectors, their indexes are not built for high-speed or large-scale nearest neighbor search[^10_4].

### Examples in Practice

- **Semantic document search:** Instantly retrieve the most relevant documents out of millions, not just those matching exact keywords[^10_1].
- **Image or audio similarity:** Quickly find similar photos or sound clips.
- **Recommendation engines:** Suggest products, movies, or songs “like” what a user already likes[^10_7].

### Key Benefits

- **Speed and scalability:** Sub-second response for similarity search, even with gigantic datasets[^10_1][^10_3][^10_4][^10_5].
- **Efficient storage and retrieval:** Lower memory usage, smarter indexing, disk-based scaling[^10_2][^10_3][^10_5].
- **Hybrid filtering:** Combine ANN search with metadata filters (e.g., “red shoes under \$100 that look like this”)—not possible with simple vector indices[^10_5].
- **Reliability and distribution:** Data persistence, failure recovery, horizontal scaling[^10_3].

**In summary:**
Purpose-built ANN search means using advanced algorithms and architecture—optimized specifically for similarity search in high-dimensional, unstructured data—delivering the speed, scale, and capabilities modern AI applications demand[^10_1][^10_3][^10_4][^10_5][^10_6].

<div style="text-align: center">⁂</div>

[^10_1]: <https://weaviate.io/blog/vector-search-explained>

[^10_2]: <https://www.pinecone.io/learn/vector-database/>

[^10_3]: <https://milvus.io/blog/why-and-when-you-need-a-purpose-built-vector-database.md>

[^10_4]: <https://bhavishyapandit9.substack.com/p/vector-search-explained-a-deep-dive>

[^10_5]: <https://milvus.io/ai-quick-reference/how-does-a-vector-database-support-vector-search>

[^10_6]: <https://glasp.co/hatch/5lgo21m6rKZSWY6oo1BlCprbxVZ2/p/SZiESro3r7igi6ElSXlo>

[^10_7]: <https://pub.towardsai.net/vector-databases-101-a-beginners-guide-to-building-ai-driven-applications-ad89749cac1c?gi=bdca202570d7>

---

### Qdrant: Queries on Nested Keys

**Qdrant** natively supports querying arrays of objects (i.e., nested payload fields), but **with some restrictions and a dedicated syntax**:

#### How Nested Filtering Works

- You must use the **`nested` condition type** to filter inside arrays of objects.
- The filter applies to a single element of the array at a time; the parent matches if **at least one element in the array satisfies all nested conditions**[^11_1][^11_2].
- Keys can be referenced with or without `[]` notation (e.g., `"data"` or `"data[]"`); best practice is to use the base key name.
- **You cannot use the `has_id` filter** inside a nested condition—it must be at the top level or a sibling to `nested`[^11_2].
- You can combine `nested` with other top-level filters using `must`, `should`, or `must_not`.

#### Example Qdrant Filter (JSON)

```json
{
  "filter": {
    "must": [
      {
        "nested": {
          "key": "diet",
          "filter": {
            "must": [
              { "key": "food", "match": { "value": "meat" } },
              { "key": "likes", "match": { "value": true } }
            ]
          }
        }
      }
    ]
  }
}
```

*This retrieves points where at least one `diet` object has `food="meat"` AND `likes=true`*[^11_1][^11_2].

***Limitations:***

- No support for `has_id` within a nested object filter—must go outside[^11_2].
- Cannot filter for relationships *between* different nested objects in the same array (each `must` applies to the same element)[^11_1][^11_2].
- Only arrays of objects are supported, not arbitrary deeply nested trees.

### pgvector and Nested (JSON) Queries

- **pgvector** itself is only the vector extension for PostgreSQL and does **not directly address nested payloads/objects**.
- **However**, PostgreSQL’s JSON/JSONB types and their operators let you **query deeply nested or arrayed data structures** with great flexibility.
- **You can filter on arrays of objects inside your table’s JSONB column using expressions like**:
  + `@>`, `#>`, `#>>`, `jsonb_array_elements`, lateral joins, etc.
- This allows complex queries against any shaped data, subject only to SQL expressiveness.

#### Example PostgreSQL Query on Nested Object (JSONB)

```sql
SELECT *
FROM products
WHERE EXISTS (
  SELECT 1
  FROM jsonb_array_elements(payload->'diet') as elem
  WHERE elem->>'food' = 'meat'
    AND (elem->>'likes')::boolean = true
)
```

*This retrieves rows with at least one `diet` object where `food` is `meat` and `likes` is `true`—functionally equivalent to Qdrant's nested filter.*

## **Summary Table**

| Feature | Qdrant | PostgreSQL (pgvector + JSONB) |
| :-- | :-- | :-- |
| Nested/object filtering | Supported on arrays of objects via `nested` type and restrictions | Fully supported on any JSON/array structure via SQL |
| Limitations | No `has_id` inside nested, only single-array-level nesting | Only limited by SQL/JSONB query expressiveness |
| Query flexibility | Good for array-of-object payloads, faster for ANN filtering use | Maximum (arbitrary depth/logic with JSON/SQL) |

**In short:**

- Qdrant supports array-of-object filtering with some restrictions.
- PostgreSQL supports arbitrary JSON nesting, making filters more flexible but relying on SQL and may be slower for very large datasets combining ANN and complex JSON logic[^11_1][^11_2][^11_3][^11_8].

<div style="text-align: center">⁂</div>

[^11_1]: <https://qdrant.tech/articles/vector-search-filtering/>

[^11_2]: <https://qdrant.tech/documentation/concepts/filtering/>

[^11_3]: <https://github.com/qdrant/qdrant/issues/2336>

[^11_8]: <https://stackoverflow.com/questions/76568932/search-points-by-payloads-nested-objects-in-qdrant>

---

# for pgvector, what is the difference between JSON and JSONB. are these the data types sorted for metadata?

The **difference between `JSON` and `JSONB`** in PostgreSQL centers on **storage format** and **query efficiency**—and this impacts your choice for storing metadata with pgvector.

### JSON vs JSONB

| Property | JSON | JSONB |
| :-- | :-- | :-- |
| Storage | Plain text, exact copy of input | Binary, decomposed/compressed |
| Query Efficiency | Slower (needs reparsing) | Much faster (pre-parsed, no reparsing on query) |
| Indexing | No native indexing support | Supports efficient indexing (GIN/GIST, etc.) |
| Formatting | Preserves input formatting \& key order | Loses insignificant whitespace, key order not kept |
| Write Performance | Faster (no conversion needed) | Slightly slower (needs conversion to binary) |
| Advanced Queries | Limited | Supports more functions and operators |

- **JSON**: Good if you simply need to store data as text and rarely run queries.
- **JSONB**: Optimal when you need to **query, index, and search the metadata efficiently**[^12_1][^12_2][^12_3][^12_5][^12_6][^12_8][^12_9].

### Which is Used for Metadata in pgvector?

- In PostgreSQL (and with pgvector), metadata is **most often stored as a `JSONB` column**, because `JSONB` supports indexing and enables very efficient, expressive queries—including filters on nested keys and arrays[^12_1][^12_2][^12_3][^12_5].
- Storing metadata as `JSONB` is the industry best practice for high-performance and complex queries.

### Summary

- **Use `JSONB`** for most pgvector applications that involve querying, filtering, or indexing metadata.
- `JSON` is only preferable if you need to preserve exact formatting and never plan to query/filter on the data.

**Bottom line:**
For scalable vector search with rich metadata queries (including sorting, filtering, and hybrid logic), always choose **`JSONB`** for your metadata column in PostgreSQL[^12_1][^12_2][^12_3][^12_5][^12_6][^12_8][^12_9].

<div style="text-align: center">⁂</div>

[^12_1]: <https://www.dbvis.com/thetable/json-vs-jsonb-in-postgresql-a-complete-comparison/>

[^12_2]: <https://risingwave.com/blog/optimal-scenarios-for-using-json-vs-jsonb-in-postgresql/>

[^12_3]: <https://www.postgresql.org/docs/current/datatype-json.html>

[^12_5]: <https://olake.io/blog/json-vs-bson-vs-jsonb>

[^12_6]: <https://www.educative.io/answers/what-is-the-difference-between-json-and-jsonb-in-postgresql>

[^12_8]: <https://www.baeldung.com/sql/postgresql-json-vs-jsonb-data-types>

[^12_9]: <https://risingwave.com/blog/exploring-json-vs-jsonb-postgresqls-dilemma/>
