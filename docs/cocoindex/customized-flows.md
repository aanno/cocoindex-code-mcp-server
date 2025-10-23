<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# I'm struggling with cocoindex custom flow definition. could you give me an overview?

Here’s an **overview of custom flow definition in CocoIndex**:

- **Purpose:** A custom indexing flow in CocoIndex allows you to define how raw data is imported, transformed, and indexed—tailoring the ETL (Extract, Transform, Load) process to your specific needs[^1_1][^1_5].

### Key Concepts

- **Indexing Flow:** The sequence of operations that extracts data from a source, applies transformations, and outputs to a target index for efficient retrieval. Flows can be updated once or run live for continuous updates[^1_1][^1_3].

#### Elements of a Flow

- **Data:** Structured via a schema at flow definition time. Data types include basic types, structs, tables (KTable/LTable), and these can be nested. All data for a flow is managed by a top-level struct.
- **Operations:** Flow is a chain of operations, each with an action (e.g., import, transform, export) and an operation spec (e.g., data source, transformation function, or export target)[^1_1].

#### Defining a Custom Flow

- **Entry Point:** Typically, you use Python and decorate your flow-defining function with `@cocoindex.flow_def(name="YourFlow")`. This function gets a `flow_builder` (to construct the flow) and a `data_scope` (managing data fields and collectors)[^1_9].

```python
@cocoindex.flow_def(name="DemoFlow")
def demo_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    ...
```

- **Alternative:** You can also use `cocoindex.add_flow_def()` for dynamic flow registration[^1_9].
- **Defining Operations:**
  + Use methods on `flow_builder` and `data_scope` to define sources, transformations, collectors, and outputs.
  + You connect source imports, transformations, and outputs through the data scope’s fields.
  + For collecting data (e.g., from processed rows), use a collector and define what fields to include[^1_9].

#### Running and Updating Flows

- **Run via CLI or Python API:** Once defined, flows are run or updated using the CLI or the `Flow.update()` API for batch or live/incremental processing[^1_3][^1_1].
- **Incremental Processing:** CocoIndex only processes data that has changed since the last run—making indexing efficient for large or evolving datasets[^1_1][^1_7].

### Practical Example (Python outline)

```python
@cocoindex.flow_def(name="CustomFlow")
def custom_flow(flow_builder, data_scope):
    data_scope["source"] = flow_builder.import_source(...)
    data_scope["transformed"] = flow_builder.transform(..., input=data_scope["source"])
    collector = data_scope.add_collector()
    for row in data_scope["transformed"]:
        collector.collect(id=cocoindex.GeneratedField.UUID, field1=row["field1"])
    flow_builder.export(..., input=collector)
```

- Here, you import data, transform it, collect the results, assign unique identifiers, and export to your desired index.

### Summary Table

| Concept | Description |
| :-- | :-- |
| Flow Definition | Python function using `@cocoindex.flow_def` or `add_flow_def()` |
| Data Scope | Manages input/output fields and collectors within the flow |
| Operations | Steps like import, transform, collect, and export with specific behavior |
| Incremental | Only changed data is reprocessed, supporting both batch and live modes |

**In short:** Define a flow function, add/import/transform/export steps using provided APIs, manage through a top-level data scope, and connect all steps to create your custom ETL pipeline[^1_1][^1_9]. CocoIndex takes care of dependencies, incremental updates, and makes flows straightforward to maintain and evolve.

<div style="text-align: center">⁂</div>

[^1_1]: <https://cocoindex.io/docs/core/basics>

[^1_3]: <https://cocoindex.io/docs/core/flow_methods>

[^1_5]: <https://www.aibase.com/tool/36560>

[^1_7]: <https://github.com/cocoindex-io/cocoindex>

[^1_9]: <https://cocoindex.io/docs/core/flow_def>

---

# ok, the are the cocoindex annotations and a type system in annotations

CocoIndex uses a **type system** that is distinct from any specific programming language but integrates smoothly with Python through type annotations[^2_1][^2_3]. Here’s how annotations and the type system work in CocoIndex:

- **Type System Overview:**
  + CocoIndex’s core is written in Rust and its types are language-agnostic, but integrations (such as the Python SDK) map CocoIndex types to Python types[^2_1].
  + The type of every piece of data in a flow is determined at definition time, ensuring a clear schema for all processed and indexed data[^2_3].
  + Common types include basic scalars, structs (represented by Python dataclasses), and collections like List and Table[^2_1][^2_3].
- **Annotations in Custom Functions:**
  + When writing **custom functions** (e.g., for transformations), you *annotate* both **arguments** and **return values** with Python type hints.
  + Type annotations are required for:
    - Arguments and return values of custom functions, which act as the contract between your Python code and CocoIndex’s engine[^2_1][^2_3].
    - Table types (e.g., KTable/LTable) always need type annotations when tied to Python variables[^2_1].

```python
from dataclasses import dataclass

@dataclass
class ModuleInfo:
    # fields here...

@dataclass
class ModuleSummary:
    # fields here...

@cocoindex.op.function()
def summarize_module(module_info: ModuleInfo) -> ModuleSummary:
    # processing...
    return ModuleSummary(...)
```

    - Here, `ModuleInfo` and `ModuleSummary` are Python dataclasses that mirror struct types in CocoIndex[^2_3].

- **Type Inference:**
  + CocoIndex will automatically infer the type of output generated by its built-in functions and sources.
  + Explicit annotation is only necessary when bridging values between Python and the CocoIndex engine, not for every intermediate step[^2_1][^2_3].
- **Composite and Collection Types:**
  + Support exists for complex Python types, including lists and tables as well as nested and custom datatype structures[^2_3].
  + The Python SDK translates these composite types directly to CocoIndex’s system, enabling rich data schemas in your flows[^2_1][^2_3].
- **Optional vs. Required Annotations:**
  + Annotations are **optional** for simple/basic and struct types when binding Python variables, but **required** for tables[^2_1].

This design provides **strong interoperability**, ensuring your Python code is type-safe and that the data schemas used throughout your CocoIndex flows are consistent and validated at definition time.

<div style="text-align: center">⁂</div>

[^2_1]: <https://cocoindex.io/docs/core/data_types>

[^2_3]: <https://cocoindex.io/blogs/cocoindex-changelog-2025-03-20>

---

# Which types are not allowed? I think any or generic lists. give me the full picture

Here is the full picture of **which types are not allowed in CocoIndex** annotations and type system:

### Allowed Types

- **Primitive types** supported by CocoIndex include `int` (as Int64), `str`, `bool`, floats (Float32, Float64), `bytes`, `uuid.UUID`, `datetime` types, and time ranges[^3_1].
- **Structs** must be defined as `@dataclass` Python classes with typed fields, mapping to Rust structs.
- **Tables** come in two flavors:
  + `KTable` - represented by `dict[K, V]` in Python, with *first key type being a key type* (e.g., string, int).
  + `LTable` - represented by `list[R]` with `R` a dataclass for each row[^3_2].
- Nested composite types of the above are also allowed.
- Custom types like `Vector(dim=N)` and specialized CocoIndex types have Python aliases.
- Python **collections are allowed only if they directly map to CocoIndex collections** like `dict` and `list` with strict typing.

### Types Not Allowed or Disallowed

- **`Any` type or untyped generic lists**:
CocoIndex requires *explicit typing* for tables and rejects untyped or generic collections like plain `list` or `dict` without type arguments. You *cannot use Python’s `Any` or generic unspecialized containers* because the engine must know the exact structure for serialization and indexing[^3_1][^3_5].
- **Generic Python `List` with mixed or unspecified element types**:
Python’s typing requires a single type argument like `List[int]`. CocoIndex requires a concrete dataclass or primitive type for list elements. Using `List[Any]` or `List` with heterogeneous types is disallowed because CocoIndex maps List to `LTable` rows, which must have a fixed schema[^3_2][^3_3].
- **Union types with incompatible members or loosely typed unions**:
Union types combining unsupported types or lacking concrete field definitions are disallowed (though limited unions of compatible types like `str | None` might be fine). Complex unions with unrelated types risk rejection[^3_1].
- **Arbitrary nested generic types not compatible with dataclasses or builtin collections**:
CocoIndex needs data types to be serializable and mappable to Rust types. So you cannot freely nest unsupported types, such as generic `Dict` or `List` with keys or values that are not dataclasses or primitive types.
- **Python-specific complex generics without direct Rust mapping**:
Types like `Generic[T]`, `TypeVar` with open bounds, or Python runtime-enforced generics are not supported (CocoIndex requires static, fixed schemas)[^3_5][^3_9].
- **Unhashable or non-key types as KTable keys**:
For `KTable` the key type must be a valid *key* type such as `int`, `str`, or other hashable primitive types. Using unhashable types or structs as keys is disallowed[^3_2].

### Summary Table

| Type Category | Allowed? | Condition / Explanation |
| :-- | :-- | :-- |
| Primitive types (int, str, bool, float, etc.) | Yes | Core scalar types are fully supported |
| `@dataclass` structs | Yes | Must define fixed fields with proper types |
| `dict[K, V]` as `KTable` | Yes | Key `K` must be key type (hashable, primitive) |
| `list[R]` as `LTable` | Yes | `R` is dataclass with fixed schema |
| Unparameterized `list` or `dict` | No | Must be fully typed for serialization and indexing |
| Python `Any` | No | Provides no static type info, disallowed |
| Generic `List` or `Dict` with `Any` or mixed types | No | Must specify exact contained type, no heterogeneity |
| Unhashable key type for `KTable` key | No | Key must be hashable and key-compatible |
| Complex unions or `Generic[T]` with open bounds | No | Must have concrete fixed types; open generics disallowed |

### In short

You **cannot use untyped or loosely typed collections or generic types like `Any`, generic Python lists without type argument, or open generics** because CocoIndex requires a **concrete, fixed, serializable schema** that maps directly to its Rust-based engine types for indexing and data correctness.

Explicit annotations using primitives, `@dataclass` structs, `dict[K, V]` with a valid key type, and `list[R]` with a dataclass row type are **mandatory for tables and collections** to ensure a robust and type-safe indexing flow[^3_1][^3_2][^3_5].

If you try to use unsupported types, CocoIndex will reject or fail type analysis during flow definition or runtime conversion.

<div style="text-align: center">⁂</div>

[^3_1]: <https://cocoindex.io/docs/core/data_types>

[^3_2]: <https://cocoindex.io/blogs/cocoindex-changelog-2025-04-30/>

[^3_3]: <https://docs.python.org/3/library/typing.html>

[^3_5]: <https://deepwiki.com/cocoindex-io/cocoindex/7-python-integration>

[^3_9]: <https://stackoverflow.com/questions/57706180/generict-base-class-how-to-get-type-of-t-from-within-instance>

---

# please elaborate a bit more on Ktable and Ltable. do they need an type parameter?

**KTable and LTable in CocoIndex** are two specific table types with distinct characteristics and strong typing requirements:

| Feature | KTable | LTable |
| :-- | :-- | :-- |
| **Type parameter needed?** | Yes, requires two type parameters: a **key type (K)** and a **value/row type (V)** | Yes, requires one type parameter: the **row type (R)** |
| **Python type representation** | `dict[K, V]` where `K` is a key type (immutable and hashable) and `V` is a `@dataclass` or `NamedTuple` struct | `list[R]` where `R` is a `@dataclass` struct representing a row |
| **Key column** | The **first column acts as the key** and must be a key type (e.g., str, int, or immutable dataclass) | No key column, only ordered rows |
| **Row order** | Row order **is not preserved** (unordered) | Row order **is preserved** |
| **Use case** | Represent a map from key to row, ideal for lookup by key, stateful data | Ordered collection of rows without keys, like simple tables or logs |
| **Key constraints** | Key must be *immutable* and *hashable*. For dataclasses, use `@dataclass(frozen=True)` to ensure immutability | No key constraints since no keys |
| **Value/Row constraints** | Value `V` must be a struct with fixed fields (dataclass or NamedTuple) | Row `R` must be a dataclass with fixed schema |

### Example usage in Python

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class PersonKey:
    id_kind: str
    id: str

@dataclass
class Person:
    first_name: str
    last_name: str
    dob: str  # or datetime.date

# KTable keyed by PersonKey with Person as value
persons_table: dict[PersonKey, Person] = {}

# LTable with rows as Person
persons_list: list[Person] = []
```

### Summary

- Both **KTable and LTable require explicit type parameters** describing the row and key (for KTable).
- **KTable = dict with typed key + struct value**, unordered, key is required and must be immutable.
- **LTable = list of structs**, ordered, no key.

This explicit typing enables CocoIndex to enforce correctness, serialization, and efficient indexing during flow definition and runtime[^4_1][^4_2].

<div style="text-align: center">⁂</div>

[^4_1]: <https://cocoindex.io/docs/core/data_types>

[^4_2]: <https://cocoindexio.substack.com/p/cocoindex-changelog-2025-04-30>

---

# and what's the heck with Vector?

**Vector in CocoIndex** refers to typed multi-dimensional numeric arrays used primarily for embeddings in AI workflows.

### What is a Vector in CocoIndex?

- It is a **fixed-dimension numeric array**, typically floats (`Float32`), representing dense vectors like embeddings from language models or other feature extractors.
- Vectors are a **first-class data type** designed for similarity search, indexing, and AI model integration.
- They are commonly used as the field type for embedding vectors in documents or chunks that CocoIndex exports to vector databases such as Postgres with PGVector or Qdrant.

### How Vector is used in CocoIndex flows?

- When defining a flow that produces embeddings, you add a **vector field** to your data structure.
- The vector field holds the embedding representation, e.g., output from `SentenceTransformerEmbed` function.
- Upon export, you specify the vector field so CocoIndex knows to treat it as a vector for indexing with similarity metrics like cosine similarity.

Example snippet of vector usage (from CocoIndex Quickstart):

```python
doc_embeddings.export(
    "doc_embeddings",
    cocoindex.targets.Postgres(),
    primary_key_fields=["filename", "location"],
    vector_indexes=[
        cocoindex.VectorIndexDef(
            field_name="embedding",
            metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY
        )
    ]
)
```

Here `"embedding"` is a vector-typed field that will be indexed for similarity search[^5_1].

### Do vectors need a type parameter?

- Yes, **vectors have a fixed dimension**, so conceptually a vector type includes a dimensionality parameter, e.g., `Vector(dim=384)`.
- This dimension must be known at flow definition or data ingestion time because:
  + It defines the size of the numeric array.
  + It ensures consistent vector shapes for indexing and similarity computations.
- In Python SDK usage, this is abstracted but crucial internally.

### Why vectors?

- They power **semantic search and retrieval-augmented generation (RAG)** by representing text chunks or documents in a continuous vector space.
- CocoIndex integrates tightly with vector DBs, making vectors a core primitive for AI-augmented indexing flows[^5_2][^5_3].

### Summary

| Aspect | Explanation |
| :-- | :-- |
| Vector type | Fixed-dimension numeric array of floats (e.g. embedding) |
| Use in flow | Holds embeddings created by transform functions like SentenceTransformerEmbed |
| Type parameter | Yes, dimension parameter required (e.g. Vector(dim=384)) |
| Purpose | Enable similarity search, vector indexing in vector DBs like Postgres PGVector, Qdrant |
| Example export | Export collector with vector field and similarity metric specified |

Vectors are essential for AI workflows in CocoIndex, enabling powerful semantic search and embedding-based indexing.

<div style="text-align: center">⁂</div>

[^5_1]: <https://cocoindex.io/docs/getting_started/quickstart>

[^5_2]: <https://cocoindex.io/blogs/index-code-base-for-rag/>

[^5_3]: <https://qdrant.tech/documentation/data-management/cocoindex/>

---

# ok. elaborate on @cocoindex.op.function() a bit more

The decorator `@cocoindex.op.function()` in CocoIndex is used to **define custom functions** within indexing flows, enabling you to add your own data processing or transformation logic that integrates smoothly into CocoIndex’s type system and execution engine.

### Key points about `@cocoindex.op.function()`

- **Purpose:** It marks a Python function as a *CocoIndex custom operation* that can be used inside flows for transformation or computation.
- **Type annotations are mandatory:** You must annotate all function arguments and the return type with supported CocoIndex types so the engine knows how to handle input/output data and serialize it properly.
- **Simple and flexible:** This decorator suits straightforward functions without extra setup or state. For more complex needs, CocoIndex also supports defining function specs and executors separately.
- **Optional parameters:** The decorator accepts optional arguments for customization (not always needed in basic use).

### How to use it (basic example)

```python
@cocoindex.op.function()
def extract_extension(filename: str) -> str:
    """Returns file extension."""
    return filename.split('.')[-1] if '.' in filename else ''
```

- The above defines a simple function that CocoIndex can call during an indexing flow.
- The annotations tell CocoIndex that it takes a `str` and returns a `str`.

### How CocoIndex uses this

- When you write flows, you can call these decorated functions as operators in the flow pipeline.
- CocoIndex inspects the type hints to generate schemas for inputs and outputs.
- The function runs inside the runtime environment seamlessly, bridging Python code with CocoIndex’s Rust core.

### Why use this decorator?

- Maintains **type safety** and **data schema consistency** across the indexing pipeline.
- Enables use of customized transformation logic that integrates with CocoIndex’s incremental updates and dependency tracking.
- Supports multiple argument types and optional args (via default parameter values).
- Easier testing and debugging by clearly isolating custom logic.

### Summary table

| Aspect | Description |
| :-- | :-- |
| Decorator name | `@cocoindex.op.function()` |
| Use case | Define simple custom functions that participate in flows |
| Requirements | Python function + full type annotations (args + return) |
| Supports | Optional arguments with defaults; basic customization params |
| Output | Registered custom function usable inside CocoIndex flow |
| Typical usage | Data transformations, computations, extractors etc. |

In essence, `@cocoindex.op.function()` is a lightweight, type-safe way to add your own Python logic into CocoIndex flows while keeping consistency with its data model and incremental processing system[^6_1].

<div style="text-align: center">⁂</div>

[^6_1]: <https://cocoindex.io/docs/core/custom_function>

---

# show an example with optional parameters

Here’s an example showing how to define a CocoIndex custom function with **optional parameters**, using Python’s default arguments and `@cocoindex.op.function()` decorator:

```python
import cocoindex

@cocoindex.op.function()
def greet(name: str, greeting: str = "Hello") -> str:
    """
    Custom function that optionally takes a greeting prefix.
    If no greeting is provided, defaults to "Hello".
    """
    return f"{greeting}, {name}!"
```

### Explanation

- `name: str` is a required argument.
- `greeting: str = "Hello"` is an optional argument with a default value.
- You *must* provide full type annotations (including for optional args).
- CocoIndex uses these annotations to infer data types for the flow’s schema.
- When calling this function inside your flow, you can omit `greeting` to get the default.

### Usage notes

- Optional arguments in custom functions **work naturally** by defining them as Python default parameters.
- CocoIndex recognizes the defaults but requires the type hints to enforce schema.
- The decorator itself can also accept parameters, but that's separate from function argument defaults.

This matches CocoIndex’s lightweight custom function style, making it easy to add flexible operations with optional parameters while retaining strong typing and schema validation[^7_2].

<div style="text-align: center">⁂</div>

[^7_2]: <https://cocoindex.io/docs/core/custom_function>

---

# For chunking, I would like to use the ASTChunk library. what to do?

To use **ASTChunk library** for chunking in your CocoIndex flow, you can integrate it as a **custom transformation function** that processes your source data (like code files or other text) and outputs AST-based semantic chunks.

### What to do

1. **Import and prepare ASTChunk** in your Python environment alongside CocoIndex.
2. **Define a custom transformer function** wrapped by `@cocoindex.op.function()` (or a Python function) that uses ASTChunk to split input source text into semantic chunks based on abstract syntax trees.
3. **In your CocoIndex flow, apply this transformer** on your source data to generate a `chunks` field containing the AST-based chunks.
4. **Embed and index the chunks** as usual with CocoIndex’s embedding functions and exporters.

### Conceptual example outline

```python
import cocoindex
from your_astchunk_lib import ast_chunk  # hypothetical import of ASTChunk function

@cocoindex.op.function()
def ast_chunking(source_code: str, language: str = "python") -> list[str]:
    # Use ASTChunk to semantically split source_code into chunks
    chunks = ast_chunk(source_code, language=language)
    return chunks

@cocoindex.flow_def(name="CodeASTChunkFlow")
def flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    # Add codebase files as source
    data_scope["files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="path/to/codebase",
            included_patterns=["*.py", "*.js", "*.ts"],
            excluded_patterns=["**/node_modules", "**/.git"]
        )
    )
    # Add collector for output chunks (embedding, etc.)
    chunk_collector = data_scope.add_collector()

    # Process each file
    with data_scope["files"].row() as file:
        # Apply AST chunking transform to the file content
        file["chunks"] = file["content"].transform(ast_chunking, language="python")

        # For each chunk, generate embedding and collect it
        with file["chunks"].row() as chunk:
            chunk["embedding"] = chunk["text"].transform(
                cocoindex.functions.SentenceTransformerEmbed(model="sentence-transformers/all-MiniLM-L6-v2")
            )
            chunk_collector.collect(
                filename=file["filename"],
                location=chunk["location"],
                text=chunk["text"],
                embedding=chunk["embedding"]
            )

    # Export collector to vector DB or other targets
    chunk_collector.export(
        "code_chunks",
        cocoindex.storages.Postgres(),
        primary_key_fields=["filename", "location"],
        vector_indexes=[
            cocoindex.VectorIndexDef(field_name="embedding", metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY)
        ]
    )
```

### Additional notes

- CocoIndex already has **built-in Tree-sitter chunking** support (which uses AST parsing), and tutorials show how to use it for code chunking as semantic units.
- If ASTChunk is a third-party or proprietary parser that better suits your needs, the key is to wrap it as a **custom function** and plug it into CocoIndex’s transformation step.
- You should ensure ASTChunk’s output matches the expected format (e.g., a list of chunks with text and location info) so you can further embed and index them.
- Incremental processing and state management will remain handled by CocoIndex automatically once the chunking is integrated into the flow.

This approach lets you **replace or complement CocoIndex’s built-in chunking** with your preferred AST-based semantic chunking from ASTChunk, enabling fine-grained, code-aware data indexing for RAG and semantic search.

<div style="text-align: center">⁂</div>

---

# I want custom embeddings as well. chosen by the language of the chunk

To implement **custom embeddings chosen by the language of each chunk** in CocoIndex, you can follow this approach:

### Overview

- After you split source documents into chunks (e.g., via ASTChunk or another chunker), each chunk has a language label (e.g., `"python"`, `"javascript"`, `"markdown"`).
- Define a **custom embedding function** that selects the embedding model based on the chunk’s language.
- Apply this function in your flow, so each chunk is embedded with the model appropriate for its language.
- Collect and export the embeddings as usual.

### How to do it concretely

1. **Define a typed transformer function with language-based embedding dispatch:**

```python
import cocoindex

@cocoindex.transform_flow()
def language_based_embedding(chunks: cocoindex.DataSlice) -> cocoindex.DataSlice[list[float]]:
    # Embeds each chunk using a different model depending on chunk["language"]
    def embed_chunk(chunk):
        lang = chunk["language"]
        text = chunk["text"]
        if lang == "python":
            model_name = "sentence-transformers/python-model"
        elif lang == "javascript":
            model_name = "sentence-transformers/javascript-model"
        else:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"  # default
        return text.transform(cocoindex.functions.SentenceTransformerEmbed(model=model_name))

    return chunks.transform(embed_chunk)
```

- This function examines each chunk’s language and picks a fitting model.

2. **Use this embedding function in your flow pipeline on the chunked data:**

```python
@cocoindex.flow_def(name="MultiLangEmbeddingFlow")
def flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    data_scope["files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(path="path/to/codebase", included_patterns=["*.py", "*.js", "*.md"])
    )

    embeddings_collector = data_scope.add_collector()

    with data_scope["files"].row() as file:
        # Chunk the file content (assume chunking also annotates chunk["language"])
        file["chunks"] = file["content"].transform(
            your_astchunking_function_or_coco_chunker,  # custom or builtin chunker
            language=file["language"]  # or infer language per file
        )

        # Embed chunks with language-aware embedder
        file["chunks"]["embedding"] = file["chunks"].call(language_based_embedding)

        # Collect embeddings
        with file["chunks"].row() as chunk:
            embeddings_collector.collect(
                filename=file["filename"],
                location=chunk["location"],
                language=chunk["language"],
                text=chunk["text"],
                embedding=chunk["embedding"]
            )

    embeddings_collector.export(
        "multilang_embeddings",
        cocoindex.storages.Postgres(),
        primary_key_fields=["filename", "location"],
        vector_indexes=[cocoindex.VectorIndexDef(field_name="embedding", metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY)]
    )
```

### Notes

- You can **define and reuse** the embedding logic with `@cocoindex.transform_flow()` to share it between indexing and query time.
- Make sure each chunk has a `"language"` field. This can come from metadata in your chunking step or be assigned per file.
- Pick models suitable for each language; CocoIndex supports thousands of SentenceTransformer models or custom models via `SentenceTransformerEmbed`.
- This pattern works for both built-in chunkers or custom chunking libs like ASTChunk.
- The example follows patterns seen in CocoIndex blogs and official docs for language-aware chunk embedding[^9_1][^9_2][^9_5].

This lets you apply **custom embeddings dynamically chosen by each chunk’s language**, optimizing index quality and search relevance across multilingual/multimodal data.

<div style="text-align: center">⁂</div>

[^9_1]: <https://dev.to/cocoindex/beyond-embeddings-building-metadata-rich-indexes-from-academic-pdfs-2cde>

[^9_2]: <https://dev.to/cocoindex/how-to-build-index-with-text-embeddings-j3m>

[^9_5]: <https://cocoindexio.substack.com/p/index-codebase-with-tree-sitter-and>

---

# I also want to enhance it replace the metadata extraction (also language dependant)

To enhance your CocoIndex flow by **replacing metadata extraction with language-dependent logic**, you can follow a pattern similar to your custom embeddings setup but focused on metadata fields.

### How to implement language-dependent metadata extraction

1. **Define a custom metadata extraction function** decorated with `@cocoindex.op.function()` that takes chunk text (and language) and returns metadata fields specific to that language.
2. **Branch your extraction logic inside the function based on the language**, so for each supported language you apply a different parsing or extraction approach (e.g., different AST parsers or regexes).
3. **In your flow, apply this metadata extraction function to each chunk after chunking and before embedding or collecting.**

### Example of a language-aware metadata extractor

```python
import cocoindex

@dataclass
class Metadata:
    author: str = ""
    complexity: int = 0
    # Add any fields relevant to your use case

@cocoindex.op.function()
def extract_metadata(text: str, language: str) -> Metadata:
    if language == "python":
        # Use Python-specific metadata extractor (e.g., ASTChunk, ast module)
        author = extract_python_author(text)
        complexity = measure_python_complexity(text)
    elif language == "typescript":
        # Use TypeScript-specific logic
        author = extract_ts_author(text)
        complexity = measure_ts_complexity(text)
    else:
        # Default fallback
        author = "unknown"
        complexity = 0
    return Metadata(author=author, complexity=complexity)
```

You would need to implement (or wrap) functions like `extract_python_author`, `measure_python_complexity`, etc., using relevant parsers or heuristics.

### Incorporate this in your flow

```python
@cocoindex.flow_def(name="EnhancedFlow")
def flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    data_scope["files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(path="path/to/codebase", included_patterns=["*.py", "*.ts"])
    )

    collector = data_scope.add_collector()

    with data_scope["files"].row() as file:
        file["chunks"] = file["content"].transform(your_chunking_function, language=file["language"])

        # Extract metadata per chunk
        file["chunks"]["metadata"] = file["chunks"].transform(
            extract_metadata,
            language=file["chunks"]["language"]
        )

        # Embed each chunk as before (using language-based embedding)
        file["chunks"]["embedding"] = file["chunks"].call(language_based_embedding)

        with file["chunks"].row() as chunk:
            collector.collect(
                filename=file["filename"],
                location=chunk["location"],
                language=chunk["language"],
                text=chunk["text"],
                embedding=chunk["embedding"],
                author=chunk["metadata"].author,
                complexity=chunk["metadata"].complexity,
            )

    collector.export(
        "enhanced_code_index",
        cocoindex.storages.Postgres(),
        primary_key_fields=["filename", "location"],
        vector_indexes=[cocoindex.VectorIndexDef(field_name="embedding", metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY)]
    )
```

### Benefits and notes

- This approach unifies embedding and metadata extraction workflows, making the pipeline modular and language-aware.
- Based on your memory, you can integrate **language-specific parsers** like Tree Sitter for JavaScript/TypeScript, or Haskell AST parsers for other languages.
- Metadata fields can be anything relevant: author info, complexity scores, function counts, module names, etc., depending on available parsers.
- Your chunking and metadata extraction remain incremental and efficient through CocoIndex’s framework.

This method lets you **enhance your indexing with rich, language-tailored metadata** that improves search context and retrieval relevance.

<div style="text-align: center">⁂</div>

---

# I heart about some elements of the flow have to be registered?

Yes, in CocoIndex certain **elements of a flow must be registered or set up** before you can run the flow for the first time or after making schema/structure changes. This includes creating and updating persistent backend entities like tables in relational databases or collections in vector databases used as export targets.

### What needs registration or setup?

- **The flow itself** must be *set up* so CocoIndex can provision all the required backend structures derived from your flow definition. For example:
  + Creating tables or collections in your Postgres/vector DB targets.
  + Adding/removing columns if you add or remove fields from collectors.
  + Preparing internal CocoIndex storage for tracking incremental state.
- **Collectors and exported targets** are implicitly registered via the flow definition but require backend setup to exist physically on the target system.

### How to register/setup flows and related elements?

You have several options:

| Method | Description |
| :-- | :-- |
| **CocoIndex CLI** | Use commands like `cocoindex setup`, `cocoindex update --setup`, or `cocoindex drop` to setup or drop flows and backends automatically. |
| **Python API** | After defining your flow, call `flow.setup()` or `flow.drop()` to create or drop backend resources programmatically. |
| **Async API variants** | Use `await flow.setup_async()` or `await flow.drop_async()` for asynchronous operation. |
| **Global setup/drop** | CocoIndex also provides `setup_all_flows()` and `drop_all_flows()` to manage multiple flows at once. |

### Typical usage example

```python
demo_flow.setup(report_to_stdout=True)   # Creates all backend tables and targets for this flow
# Run or update your flow after setup
demo_flow.update(report_to_stdout=True)
```

### Why is this needed?

- **Flow setup brings your flow definition into reality, creating necessary physical backend structures to hold and index your data.**
- It ensures consistency between your flow schema and downstream data stores.
- Without setup, your flow will not run properly or export data correctly.

### Summary

| Element | Registration/Setup Required? |
| :-- | :-- |
| Flow Definition | Yes, run setup to provision backend entities |
| Collectors \& Exports | Registered as part of flow setup |
| Internal CocoIndex State | Setup automatically with flow |

In short, **defining a flow in code registers it logically, but you must run setup commands or API calls to prepare backend storage and targets before running or updating the flow**[^11_1].

<div style="text-align: center">⁂</div>

[^11_1]: <https://cocoindex.io/docs/core/flow_methods>

---

# finally I saw in the examples that there is support for graph DBs. what must be done for a GraphRAG. what changes to the flow?

For implementing **GraphRAG (Graph + Retrieval Augmented Generation) with CocoIndex**, you will need to **adapt your flow to build and export a knowledge graph (property graph) alongside or instead of a vector index**, supporting graph queries and hierarchical summarization. Here's what changes and steps are involved:

### Key changes \& additions to your flow for GraphRAG

1. **Extract structured entities and relationships as graph elements**
    + Instead of (or alongside) just chunking and embedding text, you define transformations or custom functions that extract **triplets or tuples** representing entities and relationships from your documents or chunks.
    + These can be subject-predicate-object triples, mentions, or other graph data.
2. **Define collectors for graph nodes and edges**
    + Add collectors that store graph **nodes (entities, document nodes, chunks)** and **edges (relationships between entities, mentions, hierarchical community edges)** as separate collected tables/outputs.
    + This might mean multiple collectors, e.g., one for documents/nodes, one for entity relationships, one for mentions.
3. **Use a dedicated graph storage target**
    + Export those graph collectors to a **GraphDB or PropertyGraph target** supported by CocoIndex (e.g., Neo4j, JanusGraph, or CocoIndex’s PropertyGraph target).
    + This replaces or complements the vector index storage.
4. **Build a `PropertyGraphIndex` and graph community structure** (usually outside flow, in your app or query layer)
    + After data is ingested, use CocoIndex’s or LlamaIndex’s graph abstractions to:
        - Build the graph from collected nodes and relationships.
        - Perform community detection (e.g., Hierarchical Leiden clustering).
        - Generate summaries for graph communities.
5. **Query with a `GraphRAGQueryEngine` or equivalent**
    + At query time, queries involve graph traversals and summaries on communities, not just vector similarity search.
    + You combine global community summaries and local node neighborhoods.

### Example flow snippet (conceptual, based on [^12_5][^12_6])

```python
@cocoindex.flow_def(name="DocsToKG")
def docs_to_kg_flow(flow_builder, data_scope):
    # Source documents as a KTable keyed by filename
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="docs/core",
            included_patterns=["*.md", "*.mdx"]
        )
    )

    # Collectors for graph elements
    document_node = data_scope.add_collector()
    entity_relationship = data_scope.add_collector()
    entity_mention = data_scope.add_collector()

    # Process each document into graph components
    with data_scope["documents"].row() as doc:
        # Extract nodes from document (e.g., chunk, section, entity)
        # transform with your extractor returning entities & relationships
        entities, relationships, mentions = extract_graph_elements(doc["content"])

        # Collect documents/nodes
        document_node.collect(
            filename=doc["filename"],
            content=doc["content"],
            # add more metadata
        )

        # Collect extracted relationships between entities
        for (subject, predicate, obj) in relationships:
            entity_relationship.collect(subject=subject, predicate=predicate, object=obj)

        # Collect mentions of entities in documents
        for (entity, mention_location) in mentions:
            entity_mention.collect(entity=entity, location=mention_location, filename=doc["filename"])

    # Export collectors to graph DB targets (e.g., Neo4j)
    document_node.export("document_nodes", cocoindex.targets.Neo4j())
    entity_relationship.export("entity_relationships", cocoindex.targets.Neo4j())
    entity_mention.export("entity_mentions", cocoindex.targets.Neo4j())
```

### Summary of necessary changes for GraphRAG in your flow

| Aspect | Standard RAG Flow | GraphRAG Flow |
| :-- | :-- | :-- |
| Data extracted | Text chunks + embeddings | Entities, relationships, mentions + nodes |
| Collectors | Typically 1 for chunks and embeddings | Multiple collectors for nodes, edges, mentions |
| Storage targets | Vector DB (Postgres with PGVector, Qdrant) | GraphDB target (Neo4j, PropertyGraph) |
| Additional indexing steps | Embedding + vector indexing | Graph building, community detection, summary generation (often outside flow, in index/query layer) |
| Query handling | Vector-based similarity search | Graph traversal + community-based summarization |
| Required tooling | LLM embed models, chunkers | Graph extractor (triplets), community clustering (Leiden) |

### Practical tips

- You can still embed chunks as nodes if needed; GraphRAG fuses vector similarity with graph structure.
- Use or adapt open-source **KG extractors** or LLM-based extractors for relationship/triplet extraction inside your flow.
- CocoIndex’s documentation and examples on PropertyGraph targets show how to export graph elements and what collector schemas to use.
- At query time, leverage **graph-aware query engines** like `GraphRAGQueryEngine` (in LlamaIndex) or built-in CocoIndex graph query tools to deliver enhanced answers.

This setup realizes a **powerful hybrid indexing pipeline** that combines semantic indexing with rich graph structure and hierarchical summarization for complex, theme-aware query answering as shown in the GraphRAG approach from LlamaIndex and related projects[^12_1][^12_2][^12_3][^12_5][^12_6][^12_7].

<div style="text-align: center">⁂</div>

[^12_1]: <https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/>

[^12_2]: <https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v2/>

[^12_3]: <https://microsoft.github.io/graphrag/>

[^12_5]: <https://hackernoon.com/sick-of-reading-docs-this-open-source-tool-builds-a-smart-graph-so-you-dont-have-to>

[^12_6]: <https://dev.to/cocoindex/build-real-time-knowledge-graph-for-documents-with-llm-256h>

[^12_7]: <https://machinelearningmastery.com/building-graph-rag-system-step-by-step-approach/>

---

## Custom Parsers \& Extending Language Support in CocoIndex

**Yes, CocoIndex makes it possible to both replace an existing language parser with a custom one and to extend support for new languages.** This flexibility applies to chunking logic and parsing for source code or text files.

### Two Main Approaches

#### 1. Custom Regular Expression-Based Chunkers

- **CocoIndex offers a `custom_languages` option** when using the `SplitRecursively` function. You can **define your own chunking logic for a new or existing language** by specifying:
  + `language_name`: Name of the language (e.g., `"QBasic"` or `"CustomScript"`).
  + `aliases`: Alternate names/extensions for the language.
  + `separators_regex`: A list of regex patterns defining how to split the code/text into chunks—from high-level sections down to small statements.
- When you supply a matching `language` in your flow, CocoIndex will apply your provided chunker logic **instead of or in addition to built-in parsers**[^13_1].

**Example usage in a flow:**

```python
from cocoindex.functions import SplitRecursively

custom_spec = [{
    "language_name": "QBasic",
    "aliases": [".bas"],
    "separators_regex": [r"\nSUB ", r"\nFUNCTION ", r"\n\n", r"\."]
}]

file["chunks"] = file["content"].transform(
    SplitRecursively(custom_languages=custom_spec),
    language=file["extension"]
)
```

This would allow you to support a language CocoIndex doesn't natively recognize or override the behavior for an existing language.

#### 2. Pluggable or External Parsers

- For deeper customization (such as full AST parsing or integrating a third-party or proprietary parser), you can **wrap your own parser logic as a custom function** decorated with `@cocoindex.op.function()`. Use this function in your flow to transform file content into structured "chunks" or another schema.
- This approach works for both new languages and for enhancing/replacing the behavior for a supported language, as the flow will use your function whenever configured to do so.

### Supported Languages and Fallbacks

- The built-in system, which includes support for languages via tools like Tree-sitter, defaults to treating unknown or unsupported languages as plain text.
- **If you add a `custom_languages` entry that matches the extension or alias, your logic takes precedence** for files of that type[^13_2][^13_1].

### How It Works Internally

| Task | How to Extend or Replace |
| :-- | :-- |
| Add chunking for a new language | Provide a `custom_languages` spec with regexes |
| Override built-in chunker for an existing language | Supply a custom spec with the same language name |
| Use a non-regex chunker or parser (e.g., AST/LLM-based) | Wrap parser as a custom function via the API |
| Combine multi-language support in a single flow | List multiple language specs in `custom_languages` |

### Summary

- **You can add or replace language parsers and chunkers in CocoIndex** by using the `custom_languages` capability within chunking functions for regex-based rules, or by plugging your own parsing logic as a Python function in your flow.
- This approach is fully supported and documented in the official CocoIndex docs[^13_1].
- If the language is not built-in, or you want to do more advanced parsing (AST, semantics), author a custom function and wire it into your flow’s logic.

This makes extending CocoIndex with custom or domain-specific language support straightforward and robust for both code and text sources.

<div style="text-align: center">⁂</div>

[^13_1]: <https://cocoindex.io/docs/ops/functions>

[^13_2]: <https://cocoindexio.substack.com/p/index-codebase-with-tree-sitter-and>

---

# ok. now a bit on collect and export. when is it necessary to customise this as well. already for custom metadata?

You should customize **collectors and export configuration** in CocoIndex **whenever you want to**:

- **Add custom metadata fields** beyond defaults (e.g., author, complexity, language tags)
- **Map collected data to different storage targets** (Postgres, Qdrant, Neo4j, etc.) with specific table/collection/label names
- **Export multiple types of data**, e.g., vector embeddings *and* metadata, or graph nodes and relationships
- **Control primary keys, vector index fields, or similarity metrics** for exports
- **Support new storage backends or complex data models like graphs** requiring node and relationship specifications

### When customizing is necessary (including for custom metadata)

**1. Adding custom metadata fields**
If you extract additional metadata (e.g., language-dependent fields like author or complexity), you must:

- Add those fields in the `collector.collect(...)` call when collecting chunk or document data.
- Ensure your collector schema includes them so these fields become part of the collected rows.
- Include those fields in the export so the target storage has proper columns/properties for them.

**Example:**

```python
collector.collect(
    filename=chunk["filename"],
    location=chunk["location"],
    text=chunk["text"],
    embedding=chunk["embedding"],
    author=chunk["metadata"].author,        # custom metadata field
    complexity=chunk["metadata"].complexity # custom metadata field
)
```

**2. Defining exports to storage**
By default, collectors export to their default tables or collections, but:

- You usually specify a **target storage type**: Postgres, Qdrant, Neo4j, etc.
- For **vector exports**, define primary key fields and vector index config (vector field name and similarity metric).
- For **graph exports**, define node labels, relationship types, and field-to-property mappings explicitly.
- Set options like `setup_by_user=True` if manual collection or setup is needed.

**Example of exporting vector data with custom keys and indexes:**

```python
collector.export(
    "doc_embeddings",
    cocoindex.targets.Postgres(),
    primary_key_fields=["filename", "location"],
    vector_indexes=[
        cocoindex.VectorIndexDef(
            field_name="embedding",
            metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY
        )
    ]
)
```

**3. Exporting graph nodes and relationships**
If your flow builds a graph for GraphRAG:

- Use separate collectors for **nodes** and **relationships**.
- Export them with `cocoindex.storages.Neo4j(...)` or other graph target specifying node labels and relationship types.
- Provide mapping configs so CocoIndex knows how to map collected row fields to graph node properties or edge properties.

### Summary

| When to customize collect \& export | What to do |
| :-- | :-- |
| You add or change custom metadata fields | Add fields to `collector.collect()` calls and include them in exports |
| You use different storage backends | Choose target storage type and configure primary keys, indexes, and labels |
| You export complex data models (graphs) | Use separate collectors for nodes/edges and map properly to graph schema |
| You want control over primary keys or vector indexes | Specify keys and vector index definitions explicitly in `export()` |
| Your data model evolves | Adjust collector schema and exports to reflect new/changed fields |

**In short:**
Customizing collectors and exports is necessary **whenever you extend your data schema or switch from default storage/export behavior**—including adding custom metadata. This ensures your extracted data is fully represented and correctly stored/indexed in your target backend.

<div style="text-align: center">⁂</div>

---

## Exporting Data to Files in CocoIndex

CocoIndex is primarily designed to export data to various backends like vector databases (Postgres, Qdrant, Neo4j, Kuzu, etc.), but saving exports to a file (such as CSV, JSON, or Parquet) in a local or cloud filesystem is a common ETL requirement.

While the official documentation focuses on built-in targets like databases and graph stores, exporting to a file is possible with a custom target or by using standard Python with collectors. For example, after collecting data, you can write the collected result to disk using `pandas` or native file I/O in your custom flow steps. This pattern is typically adopted when you wish to save intermediate results locally before or in addition to pushing to an external database.

There is no built-in `FileTarget` listed in the core targets, but CocoIndex flows are Pythonic, so adding a final export step to write to a file is fully supported with a custom step[^15_1][^15_2].

### Example: Writing Exported Data to a File

```python
import pandas as pd

# After your collector is filled
df = pd.DataFrame(collected_data)
df.to_csv("/path/to/exported_data.csv", index=False)
```

Or, inside your flow’s export logic, you can create custom functions to serialize collected data as needed.

## Exporting Schemas

CocoIndex maintains a data schema at flow definition time, which is used to infer and synchronize backend storage schemas automatically[^15_3][^15_4]. To export or document your current schema:

- **Automatic backend setup:** When you run `flow.setup()`, CocoIndex uses the schema in your flow to create/update the target schema in your backend.
- **Schema extraction:** While there isn't a public "export schema to file" command documented, you can reconstruct your collector or table schemas (since they’re declared as Python dataclasses/types).
- If you need a file (e.g., for sharing schema with a DBA or mirroring index structure), you can serialize your schema manually (as JSON/YAML) based on your dataclass definition.

> Tip: This is a common pattern in Python ETL frameworks; you extract field names/types from your dataclasses and write them out as a JSON schema.

## Distributed Database Mapping/Driver Documentation

CocoIndex supports a range of distributed and cloud databases. Each target (Postgres, Qdrant, Neo4j, Kuzu, etc.) comes with its own mapping configuration. The mappings for graph DBs (nodes, relationships) or property graphs are well detailed in the official documentation:

- **Graph/database target mapping**: The [Targets documentation](https://cocoindex.io/docs/ops/targets) covers the mapping of collector data to graph nodes/relationships and describes required fields for each supported backend.
- For distributed graph DBs like Kuzu, the config fields (`api_server_url`, `KuzuConnectionSpec`, etc.) and mapping setup are also shown[^15_2].

This section also includes Py-specific export code snippets, requirements for mapping (like node/relationship label uniqueness), and the approach for primary key and index definitions on exports.

### Table: Where to Find What

| Requirement | Solution | Documentation/Link |
| :-- | :-- | :-- |
| Exporting data to files | Use Python code post-collection | [^15_1] |
| Exporting or documenting schemas | Manual via dataclass/collector schema | [^15_3][^15_4] |
| Mapping for distributed DBs | Official Targets documentation | [^15_2] |

## Key References

- CocoIndex Quickstart \& Basics: File, flow, collector, backend examples[^15_1][^15_3]
- CocoIndex Targets \& Mapping: Graph, property graph, Neo4j, Kuzu configurations[^15_2]
- Auto backend setup and schema management: Overview of flow-driven schema creation and updates[^15_4]

CocoIndex’s usage of Python for flow composition means you have *fine-grained control* over both exports and schemas, and the core documentation provides all mapping drivers and distributed backend configs for extending to new datastores or file-based output.

<div style="text-align: center">⁂</div>

[^15_1]: <https://cocoindex.io/docs/getting_started/quickstart>

[^15_2]: <https://cocoindex.io/docs/ops/targets>

[^15_3]: <https://cocoindex.io/docs/core/basics>

[^15_4]: <https://hackernoon.com/this-open-source-tool-could-save-your-data-team-hundreds-of-hours>
