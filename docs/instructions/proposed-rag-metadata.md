<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# for a RAG for coding, I'm looking for recommendations what code chunk metadata should be included, e.g. footpath, language, start line number, function name, class name. this seems a bit dependant on the concrete programming language. are there python libraries available to extract this information from tree sitter AST

For retrieval-augmented generation (RAG) systems focused on coding, extracting rich and accurate **metadata for each code chunk** is essential. Based on best practices and real-world implementations, recommended metadata fields typically include:

- **File path** (footpath)
- **Programming language**
- **Start and end line numbers**
- **Function name** (if applicable)
- **Class name** (if applicable)
- **Type of chunk** (module, class, function, method, code block, import)
- **Imports contained**
- **Comments and docstrings**
- **Hash or unique identifier**

These fields offer valuable context, grounding search and retrieval for more accurate code-related answers[^1_6][^1_8]. The ideal metadata schema can adjust depending on language specifics or project conventions; for example, languages with different function or class structures may require nuanced parsing rules.

### Python Libraries Using Tree-sitter AST

Several **Python libraries** help extract such metadata from code using Tree-sitter:

- **py-tree-sitter**: Provides Python bindings for Tree-sitter, letting you parse source files and walk the resulting ASTs to find nodes corresponding to classes, functions, etc. You can manually extract line numbers, names, and nested relationships using the AST[^1_3][^1_9].
- **asts (GrammaTech SEL)**: This library offers a generic API over Tree-sitter for multiple languages, abstracting some of the language-specific quirks and making extraction of cross-language metadata easier[^1_1].
- **rag-pychunk**: Specifically designed for RAG-style chunking in Python, this library leverages Python’s AST (with support for Tree-sitter planned or possible) and assigns metadata such as `file`, `node_type` (method/class/function/code/import), line numbers, and relationships (parent-child, etc.) to each code chunk. It stores metadata in structured schemas to facilitate downstream retrieval tasks[^1_8].
- **tree-sitter-type-provider**: Provides AST typing and traversal utilities, which can assist in designing metadata extraction pipelines once the AST is constructed[^1_11].

#### Example Schema (rag-pychunk)

The **rag-pychunk** approach structures metadata like this:

- **file**: file path
- **node_type**: ENUM (MODULE, CLASS, FUNCTION, METHOD, CODE, etc.)
- **lines_of_code**: tuple of (start, end) line numbers
- **node_relationships**: mapping of parent, child, other associations
- **hash**: unique content-derived identifier
- **additional_metadata**: optional extra data (docstrings, decorators, etc.)[^1_8]

#### Best Practice

RAG code chunking for Python (and generally) should extract at least:

- file name/path
- chunk type (class/function/code/import)
- start/end line
- function/class names if present
- includes or import statements
- local and parent relationships

Tree-sitter-based libraries (like **py-tree-sitter** and **rag-pychunk**) enable this by parsing the code to obtain the concrete syntax tree, traversing nodes for relevant labels, and mapping out chunk boundaries and metadata fields[^1_3][^1_8].

**These tools generally work across languages supported by Tree-sitter**, but you'll need to handle language-specific node types (e.g., dealing with "def" vs. "function" or bespoke class structures). Start with py-tree-sitter or rag-pychunk for Python; for multi-language support, consider asts or writing generic visitor patterns[^1_1][^1_3][^1_8].

<div style="text-align: center">⁂</div>

[^1_1]: <https://pypi.org/project/asts/>

[^1_3]: <https://github.com/tree-sitter/py-tree-sitter>

[^1_6]: <https://www.inf.uni-hamburg.de/en/inst/ab/lt/publications/2024-strichetal-aclsrw.pdf>

[^1_8]: <https://github.com/jimysancho/rag-pychunk>

[^1_9]: <https://dev.to/shreshthgoyal/understanding-code-structure-a-beginners-guide-to-tree-sitter-3bbc>

[^1_11]: <https://pypi.org/project/tree-sitter-type-provider/>
