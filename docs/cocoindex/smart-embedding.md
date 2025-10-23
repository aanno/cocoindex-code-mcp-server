# I'm inside a @cocoindex.transform_flow() function. However, I want to dynamically select my     cocoindex.functions.SentenceTra

nsformerEmbed model based on the value of str inside DataSlice[str]. What to do?

cocoindex.functions.SentenceTransformerEmbed expects the model name to be fixed when constructing the transform, and there is no direct public API to dynamically change the model on a *per-row* or *per-call* basis inside a transform_flow. The model parameter is bound at function construction time, not run time[^4_1][^4_2][^4_8].

**What you can do:**

- If you must choose the model dynamically based on the text, you need to:
  + Split your data ahead of time by which model should be used, then run the transform_flow separately for each subset with the appropriate model.
  + Or, bypass the built-in SentenceTransformerEmbed and write your own embedding operator that loads models conditionally inside its function, but be aware of the heavy loading overhead and possible inefficiency[^4_1][^4_2][^4_8].

**Why:**
The built-in embedding transform is optimized for batching and reproducibility—changing models dynamically per row would break batching and is not supported in the official design or examples[^4_1][^4_2][^4_8].

**Patterns:**

- *Split by model need:*

```python
model_a_texts = [t for t in texts if choose_model(t) == "model_a"]
model_b_texts = [t for t in texts if choose_model(t) == "model_b"]

# Run the transform per model, then merge results
emb_a = text_to_embedding_a(model_a_texts)
emb_b = text_to_embedding_b(model_b_texts)
```

- *Or, if you really must have a single flow:*
  + Implement a custom embedding op using @cocoindex.op.function, but understand this will be much less efficient if you switch models per call.

```python
@cocoindex.transform_flow()
def dynamic_embed(texts: cocoindex.DataSlice[str]) -> cocoindex.DataSlice[list[float]]:
    @cocoindex.op.function()
    def embedding_op(text: str) -> list[float]:
        model_name = choose_model(text)  # implement your rule here
        model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        return model.encode(text).astype(np.float32).tolist()
    return texts.transform(embedding_op)
```

*This pattern is slow and not recommended unless absolutely necessary, as models are reloaded for every call.*

**Summary:**
You cannot change the SentenceTransformerEmbed model dynamically inside a single cocoindex transform_flow. Prefer splitting your data ahead of time and run separate flows per model, or implement a custom operator with performance caveats if you need true per-row model switching[^4_1][^4_2][^4_8].

<div style="text-align: center">⁂</div>

[^4_1]: <https://dev.to/cocoindex/how-to-build-index-with-text-embeddings-j3m>

[^4_2]: <https://hackernoon.com/this-open-source-project-turns-text-into-instant-ai-answers>

[^4_8]: <https://github.com/cocoindex-io/cocoindex/blob/main/examples/gdrive_text_embedding/main.py>

---

# Well, I only need to select the model based on the language of the chunk

If you only need to select the embedding model based on the language of each chunk, note that cocoindex's built-in SentenceTransformerEmbed does **not** support changing the model dynamically (per chunk) inside a single transform_flow[^5_1][^5_2]. The model is bound when constructing the transform, making per-row model selection unsupported by the core API.

**What you can do:**

- If you have multilingual content, use a multilingual model such as **all-MiniLM-L6-v2** or **distiluse-base-multilingual-cased-v2** from sentence-transformers, which will handle many languages well in a single pass[^5_3][^5_5][^5_6].
- If you require specialized models for different languages, partition your chunks by language, then process each language subset with a separately parameterized transform_flow, each using the best model for that language[^5_2].

**Pattern Example:**

```python
# Detect language of each chunk (pseudo-code)
lang_to_chunks = defaultdict(list)
for chunk in all_chunks:
    lang = detect_language(chunk)
    lang_to_chunks[lang].append(chunk)

# For each language, process with appropriate model
results = []
for lang, chunks in lang_to_chunks.items():
    model = LANG_MODEL_MAP[lang]    # Map language to appropriate model name
    @cocoindex.transform_flow()
    def embed_text(text: cocoindex.DataSlice[str]) -> cocoindex.DataSlice[list[float]]:
        return text.transform(
            cocoindex.functions.SentenceTransformerEmbed(model=model)
        )
    results.extend(embed_text(chunks))
```

This is efficient: you avoid model-load overhead, preserve batching, and stay within the intended use of cocoindex[^5_1][^5_2].

**Summary:**

- Use a single strong multilingual model if possible (preferred for most scenarios)[^5_3][^5_6].
- Otherwise, split your data by language, then embed each group with its respective SentenceTransformer model using dedicated transform flows.

No public cocoindex solution supports switching the model per chunk within a single flow. This constraint is shared by most production embedding pipelines.

<div style="text-align: center">⁂</div>

[^5_1]: <https://substack.com/home/post/p-164288038>

[^5_2]: <https://cocoindex.io/blogs/text-embeddings-101/>

[^5_3]: <https://github.com/siamakz/sentence-transformers-1>

[^5_5]: <https://github.com/UKPLab/sentence-transformers>

[^5_6]: <https://sbert.net>

---

# What is DataScope?

In cocoindex, a **DataScope** represents a logical view or "scope" of the data being processed or transformed in your flow. It is a core concept for data presentation and organization within cocoindex pipelines.

Key points about DataScope:

- **Unit of Data**: A DataScope is the container for data in a flow, representing all or part of your dataset at any given step. The top-level DataScope typically holds all the data for your flow, but you can also create subscopes (such as per document, per chunk, etc.) for more granular data handling[^6_2].
- **Indexed Like a Dict**: In flow definitions, you often access or define fields and tables using dictionary-like syntax on the DataScope, for example: `data_scope["files"] = ...`[^6_3][^6_7].
- **Hierarchical and Nested**: Subscopes can be nested, for example when you need a per-row or per-chunk context. This allows you to process or collect data at various levels of granularity.
- **Central to Transformations \& Collection**: Collectors (outputs) and transformations are typically attached to a DataScope, so the DataScope governs what data is available to each operation.

Example usage:

```python
@cocoindex.flow_def(name="CodeEmbedding")
def code_embedding_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    data_scope["files"] = flow_builder.add_source(...)
    code_embeddings = data_scope.add_collector()
```

In this example, `data_scope["files"]` adds a table of files to the scope, and a collector is added for further transformations or export[^6_3][^6_7].

To summarize:
**DataScope** is the structured data context you build on, work within, and transform as you define your cocoindex ETL/data flow, supporting both top-level and nested (subscope) data access[^6_2][^6_3].

<div style="text-align: center">⁂</div>

[^6_2]: <https://www.youtube.com/watch?v=gv5R8nOXsWU>

[^6_3]: <https://dev.to/badmonster0/want-ai-to-actually-understand-your-code-this-tool-says-it-can-help-530p>

[^6_7]: <https://github.com/cocoindex-io/cocoindex/blob/main/examples/pdf_embedding/main.py>

---

# So how to divide a DataScope into several DataSlices, one for each language?

To divide a `DataScope` into several `DataSlice`s, one per language, you need to:

1. Determine the language for each chunk (or row) inside the `DataScope`.
2. Filter or partition the data by language, creating separate `DataSlice` objects for each language.

Since `DataScope` and its inner tables (`KTable`/`DataSlice`) in cocoindex behave like structured sets of rows with fields, you often do this by:

- Adding a language detection step to annotate each chunk with its language.
- Then, within your flow, filter or extract slices of data where the language matches a given value.

### How to do this in cocoindex

Here is a conceptual example based on cocoindex patterns:

```python
from collections import defaultdict
import cocoindex

@cocoindex.op.function()
def detect_language(text: str) -> str:
    # Implement or call your language detection here
    # Example stub:
    if text.startswith("Bonjour"):
        return "fr"
    elif text.startswith("Hola"):
        return "es"
    else:
        return "en"

@cocoindex.flow_def(name="SplitByLanguage")
def split_by_language(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    # Assume you have a source table of documents or chunks
    data_scope["chunks"] = flow_builder.add_source(...)  # your input source

    # Add language field to each chunk
    with data_scope["chunks"].row() as chunk:
        chunk["language"] = chunk["text"].transform(detect_language)

    # Now, to get separate DataSlices per language,
    # you can filter rows in the "chunks" KTable.

    # Example: get English chunks
    english_chunks = data_scope["chunks"].filter(lambda row: row["language"] == "en")

    # Spanish chunks
    spanish_chunks = data_scope["chunks"].filter(lambda row: row["language"] == "es")

    # French chunks
    french_chunks = data_scope["chunks"].filter(lambda row: row["language"] == "fr")

    # These are DataSlices filtered by language, you can transform/embed separately:
    embeddings_en = english_chunks["text"].transform(
        cocoindex.functions.SentenceTransformerEmbed(model="model_for_english")
    )
    embeddings_es = spanish_chunks["text"].transform(
        cocoindex.functions.SentenceTransformerEmbed(model="model_for_spanish")
    )
    embeddings_fr = french_chunks["text"].transform(
        cocoindex.functions.SentenceTransformerEmbed(model="model_for_french")
    )

    # Collect or further process as needed
```

### Important notes

- `filter` on a table or `DataSlice` allows you to select rows matching a predicate, producing a new DataSlice with those rows.
- You can create one filtered DataSlice per language, then embed those slices separately with different models.
- This approach keeps your pipeline efficient, uses batching per language, and fits cocoindex dataflow paradigms.

### Summary

- Use an op.function to label language per chunk.
- Use `.filter` on the DataSlice or table to split the data per language.
- Process each filtered DataSlice separately with appropriate embedding models.

This fits naturally with cocoindex incremental flows and composable transformations.

If you want, I can provide a runnable minimal example for your specific use case.

<div style="text-align: center">⁂</div>
