<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# for a maturin python and rust project with rust bindings to multiple rust libs, what are the recommendations for code folder structure. should I separate rust from python code? what is best of I plan to distribute the artifact as wheel or sdist package? or to upload it to pypi? where should I place tests and examples and documentation?

For a **maturin** project combining Python and Rust, especially with Rust bindings to multiple Rust libraries, the best-practice folder structure and distribution setup should prioritize clarity, IDE compatibility, and clean separation of concerns. Here are the key recommendations:

### 1. **Folder Structure: Separation of Rust and Python Code**

For a project intended to be distributable as a **wheel** or **sdist** (and uploadable to PyPI), you should **separate Rust and Python code**. The most common and maintainable approach is:

```
my-project/
├── pyproject.toml
├── README.md
├── python/            # Python code (src layout)
│   └── cocoindex_code_mcp_server/
│       ├── __init__.py
│       └── bar.py
├── rust/           # Rust code
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── ...     # more Rust code/libs, if needed
└── tests/
    ├── test_foo.py
    └── test_bar.rs
```

- Place all **Python code in a dedicated directory** such as `python/cocoindex_code_mcp_server` to avoid import path confusion and ease distribution[^1_3][^1_6].
- Place **Rust code in a sibling directory**, such as `rust/`, with its own `Cargo.toml` and typical Rust src layout for multiple libraries (use [Cargo workspaces](https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html) if you have multiple crates)[^1_4][^1_5][^1_6].
- This separation allows the Rust libraries to remain idiomatic Rust code, reusable for other Rust projects, and keeps the Python bindings cleanly split[^1_2][^1_4].
- **If you use pyo3 and maturin, configure the Rust Python extension crate to depend on your “core” Rust libraries as normal Rust dependencies.**

### 2. **Configuring maturin and PyPI Packaging**

- Set `tool.maturin.python-source = "src"` in your `pyproject.toml` to tell maturin where your Python package lives[^1_3][^1_9].
- If you have a Rust native extension for Python, set the module name so it doesn't clash with your Python package, e.g. `module-name = "cocoindex_code_mcp_server._my_project"`[^1_3][^1_6].
  + In your Rust `#[pymodule]`, use the same name.
- This keeps IDEs and code completion working correctly.

### 3. **Tests, Examples, and Documentation**

- **Python tests**: Place inside `tests/` at the project root or within your Python package as needed, e.g. `tests/test_something.py`[^1_2].
- **Rust tests**: In the standard Cargo format (`rust/tests/` folder or inside `rust/src/lib.rs` as `#[cfg(test)]` modules)[^1_2][^1_4].
- **Examples**: Put Python examples in an `examples/` directory, and Rust examples in `rust/examples/`.
- **Documentation**: Place documentation in `README.md` and, if extensive, make a `docs/` folder.
  + Consider including usage instructions for both Python and Rust users.

### 4. **Additional Notes**

- For **multi-crate Rust workspaces**: Use `rust/` as a Cargo workspace root with multiple library crates under it, and have one crate provide the pyo3 binding crate for Python[^1_4][^1_5].
- Avoid placing your Rust `Cargo.toml` in the same directory as your Python package, as this can lead to confusion for both build tools and IDEs[^1_3][^1_6].
- When distributing as a wheel or sdist, maturin will correctly include the native extension from Rust in your Python package, provided your config matches the directory structure above[^1_3][^1_6][^1_9].

### **Summary Table**

| Component | Recommended Location |
| :-- | :-- |
| Python code | `python/cocoindex_code_mcp_server/` |
| Rust code | `rust/` (workspace root) |
| Rust extension lib | `rust/src/` (crate) |
| Tests (Python) | `tests/` |
| Tests (Rust) | `rust/tests/` |
| Examples | `examples/` (Python), `rust/examples/` (Rust) |
| Documentation | `README.md`, `docs/` |

This setup maximizes clarity, maintainability, and correct distribution with maturin, and is the "best practice" for publishing on PyPI and supporting both Python and Rust users[^1_3][^1_4][^1_6].

<div style="text-align: center">⁂</div>

[^1_2]: <https://stackoverflow.com/questions/78188760/how-to-separate-rust-library-and-exported-python-extensions-which-wrap-it>

[^1_3]: <https://www.maturin.rs/project_layout.html>

[^1_4]: <https://users.rust-lang.org/t/strategy-for-creating-python-bindings-to-a-rust-library/89403>

[^1_5]: <https://www.reddit.com/r/rust/comments/1j67lpg/idiomatic_rust_python_project_structure/>

[^1_6]: <https://www.maturin.rs/project_layout.html?highlight=stub>

[^1_9]: <https://docs.rs/maturin/latest/maturin/>

---

## Best Practices for Including and Loading `.lark` Grammar Files in a Python Package

### Where to Place `.lark` Files

- **Place all `.lark` files inside your Python package directory** (e.g., in `python/cocoindex_code_mcp_server/grammars/` if using a `src` layout).
- Treat `.lark` files as *package data* (non-Python resource files) that must be shipped with your package on PyPI, so they are always available both in development and after installation via wheel or sdist.
- The typical structure looks like this:

```
cocoindex_code_mcp_server/
├── python/
│   └── cocoindex_code_mcp_server/
│       ├── __init__.py
│       └── grammars/
│           └── my_grammar.lark
```

- Ensure these files are included as *package data* by:
  + Specifying them in `pyproject.toml` or `setup.cfg` (or `setup.py`) using `package_data` or similar.
  + Adding corresponding rules in a `MANIFEST.in`, e.g.:

```
recursive-include python/cocoindex_code_mcp_server/grammars/*.lark
```

This ensures distribution in both sdists (source) and wheels (binary) formats[^2_1][^2_2].

### Canonical Way to Load `.lark` at Runtime

#### The **Recommended** Approach

- Use **`importlib.resources`** (Python 3.7+) to access packaged resource files in a way that works across all installation methods.
- Example code to read a `.lark` grammar file from within your package:

```python
from importlib.resources import files

grammar_path = files('cocoindex_code_mcp_server.grammars').joinpath('my_grammar.lark')
with grammar_path.open('r', encoding='utf-8') as f:
    grammar = f.read()
```

- Then you can pass `grammar` to Lark, e.g.:

```python
from lark import Lark
parser = Lark(grammar, parser="lalr")
```

- This method:
  + Works during development (editable installs).
  + Works after installing as wheel or from sdist.
  + Avoids path issues across OSes and zip imports[^2_3][^2_2].

#### Alternative: Using `Lark.open_from_package`

- Lark provides `Lark.open_from_package` to load grammars as resources from a package:

```python
from lark import Lark

parser = Lark.open_from_package(
    'cocoindex_code_mcp_server.grammars',          # Package (as string)
    'my_grammar.lark',              # Filename
    parser="lalr"
)
```

- It uses resource loaders and is compatible with packaged, zipped, or installed distributions[^2_4].

### Summary Table

| Purpose | Recommended Path | How to Access |
| :-- | :-- | :-- |
| Package grammar files | `python/cocoindex_code_mcp_server/grammars/*.lark` | Use `importlib.resources` or `Lark.open_from_package` |
| Include in distributions | MANIFEST.in, `package_data` in config | Ensures file present in sdist/wheel |
| Runtime loading | `importlib.resources` (3.7+), fallback to pkg_resources for older Python | Universal for both development and installed environments |

**Key Points:**

- Always keep `.lark` resource/grammar files inside your package, not outside or in top-level project folders.
- Use standard resource loading tools like `importlib.resources` or Lark's high-level API for runtime access, ensuring your code works regardless of environment[^2_3][^2_2][^2_4].
- Confirm inclusion of your grammar files as package data in your Python build system configuration.

This approach is now widely regarded as best practice for any non-Python resource, including grammar files[^2_1][^2_2].

<div style="text-align: center">⁂</div>

[^2_1]: <https://www.reddit.com/r/learnpython/comments/rkvjiz/resource_files_best_practices/>

[^2_2]: <https://www.turing.com/kb/7-ways-to-include-non-python-files-into-python-package>

[^2_3]: <https://docs.python.org/3/library/importlib.resources.html>

[^2_4]: <https://lark-parser.readthedocs.io/en/latest/classes.html>

---

# The projects includes special support for a bunch of languages on the python site. So I think about having directories like "lang/java" and "lang/haskell" for sub modules. Is this the right approach in python?

Yes, **organizing your Python package with subdirectories like `lang/java/` and `lang/haskell/` as subpackages is the right and canonical approach** in Python for grouping related modules or features by language or domain**. This follows standard Python package structuring best practices** that use nested packages (subpackages) to logically separate concerns, avoid name collisions, and keep the codebase maintainable and navigable[^3_1][^3_3][^3_4].

### Why use such a structure?

- **Packages and subpackages create a hierarchical namespace.** For example, `cocoindex_code_mcp_server.lang.java` and `cocoindex_code_mcp_server.lang.haskell` become proper subpackages that can be imported explicitly.
- This avoids dumping all language support code into a single package or module, which would become unwieldy and harder to maintain.
- You get clean absolute or relative imports using dot notation, e.g.:

```python
from cocoindex_code_mcp_server.lang.java import parser
from cocoindex_code_mcp_server.lang.haskell import evaluator
```

- You can keep language-specific tests, data, and resources inside each language subpackage if needed.
- Allows defining or customizing `__init__.py` files in each subpackage to control imports or initialization logic.

### Recommended folder structure example

```
cocoindex_code_mcp_server/
├── python/
│   └── cocoindex_code_mcp_server/
│       ├── __init__.py
│       └── lang/
│           ├── __init__.py
│           ├── java/
│           │   ├── __init__.py
│           │   ├── parser.py
│           │   └── other_java_module.py
│           └── haskell/
│               ├── __init__.py
│               ├── evaluator.py
│               └── utils.py
├── tests/
│   ├── lang/
│   │   ├── test_java.py
│   │   └── test_haskell.py
│   └── ...
└── pyproject.toml
```

### Additional notes

- Use an `__init__.py` file (possibly empty) inside each directory to mark it as a package/subpackage for Python ≤3.8 or for explicit clarity.
- Avoid using underscores in package names; subpackage directories like `java` and `haskell` are preferred for simplicity.
- This structure works well with standard tools, IDEs, and Python import mechanisms, and supports building wheels, sdist, and PyPI distribution cleanly[^3_1][^3_3][^3_5].

**In summary:** Organizing language-specific submodules as subpackages under `lang/` is idiomatic Python and the recommended way to modularize your multi-language support code within your project.

<div style="text-align: center">⁂</div>

[^3_1]: <https://realpython.com/python-modules-packages/>

[^3_3]: <https://python.land/project-structure/python-packages>

[^3_4]: <https://docs.python.org/3/tutorial/modules.html>

[^3_5]: <https://packaging.python.org/tutorials/packaging-projects/>
