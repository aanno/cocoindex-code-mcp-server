# Distribution

## Maturin

### Maturin bindings

* [maturin bindings](https://www.maturin.rs/bindings.html)
  + [pyo3](https://www.maturin.rs/bindings.html#pyo3)
  + [cffi](https://www.maturin.rs/bindings.html#cffi)
  + [cpython](https://www.maturin.rs/bindings.html#cpython)
  + [rust-cpython](https://www.maturin.rs/bindings.html#rust-cpython)
  + [pyo3-ffi](https://www.maturin.rs/bindings.html#pyo3-ffi)
* [uniffi-rs](https://mozilla.github.io/uniffi-rs/latest/)
  unified bindings for XPCOM, Python, Ruby, Node.js, Swift, Kotlin/Java and C++

### Pyo3

* Py_LIMITED_API/abi3
    + Stable ABI for CPython 3.2 and later
    + Compile with `--features=abi3`
    + Set `pyo3/abi3-py<version>` feature to set minimum supported Python version
    + Produces a single wheel that works on all Python versions >= minimum version
    + Cannot use any non-abi3 features of the Python C API
    + Cannot use any third-party C extensions that are not also compiled with abi3
    + Cannot use any Cython extensions that are not also compiled with abi3
    + Cannot use any features of PyO3 that are not compatible with abi3 (e.g. `PyAny::is_instance`)
    + Cannot use `pyo3-ffi` crate directly
    + Cannot use `PyO3` features that require a specific Python version (e.g. `PyDict::from_keys`)
    + Cannot use `PyO3` features that require the GIL to be held (e.g. `Python::with_gil`)
    + Cannot use `PyO3` features that require the Python interpreter to be initialized (e.g. `Python::acquire_gil`)
    + Cannot use `PyO3` features that require the Python interpreter to be finalized (e.g. `Python::release_gil`)
    + Cannot use `PyO3` features that require the Python interpreter to be in a specific state (e.g. `Python::is_initialized`)
    + Cannot use `PyO3` features that require the Python interpreter to be in a specific thread (e.g. `Python::is_main_thread`)
    + Cannot use `PyO3` features that require the Python interpreter to be in a specific version (e.g. `Python::version_info`)
    + Cannot use `PyO3` features that require the Python interpreter to be in a specific implementation (e.g. `Python::implementation`)
    + Cannot use `PyO3` features that require the Python interpreter to be in a specific platform (e.g. `Python::platform`)
    + Cannot use `PyO3` features that require the Python interpreter to be in a specific locale (e.g. `Python::locale`)
    + Cannot use `PyO3` features that require the Python interpreter to be in a specific encoding (e.g. `Python::encoding`)
    + Cannot use `PyO3` features that require the Python interpreter to be in a specific filesystem encoding (e.g. `Python::filesystem_encoding`)
    + Cannot use `PyO3` features that require the Python interpreter to be in a specific byte order (e.g. `Python::byteorder`)
    + Cannot use `PyO3` features that require the Python interpreter to be in a specific endianness (e.g. `Python::endianness`)
    + Cannot use `PyO3` features that require the Python interpreter to be in a specific architecture (e.g. `Python::architecture`)
* PyO3 has feature flags abi3-py37, abi3-py38, abi3-py39 etc. to set the minimum required Python version

#### Pyo3 features

* [Pyo3 features](https://pyo3.rs/v0.26.0/features)
* [anyhow](https://docs.rs/anyhow/latest/anyhow/)
  error type for easy idiomatic error handling
* [pythonize](https://github.com/davidhewitt/pythonize)
  Convert Rust types to Python types and vice versa (Serde-based)
  + [Serde](https://github.com/serde-rs/serde)
    framework for serializing and deserializing Rust data structures efficiently and generically
* [pyo3_async_runtimes](https://docs.rs/pyo3-async-runtimes/latest/pyo3_async_runtimes/)
  interop between Python and Rust’s async/await models
  * [pyo3_async_runtimes on github](https://github.com/PyO3/pyo3-async-runtimes)

## cibuildwheel

* [cibuildwheel](https://github.com/pypa/cibuildwheel)
  builds and tests your wheels across all of your platforms
  + [cibuildwheel options](https://cibuildwheel.pypa.io/en/stable/options/#repair-wheel-command)
  + [cibuildwhee examples](https://github.com/pypa/cibuildwheel/tree/main/examples)
* [cibuildwheel tips and tricks](https://cibuildwheel.pypa.io/en/stable/faq/)
* [auditwheel](https://github.com/pypa/auditwheel)
  repairs wheels with external shared library dependencies to make them compliant with the manylinux standard
* [delocate](https://github.com/matthew-brett/delocate)
  macOS utilities to repair and examine Python wheels with external shared library dependencies
* [delvewheel](https://github.com/adang1345/delvewheel)
  wheel packages for Windows that have DLL dependencies that may not be present on the target system
* [abi3audit](https://github.com/pypa/abi3audit)
  check that a Python extension module is compatible with the stable ABI (abi3)

## act + GitHub Actions

### act - Run GitHub Actions locally

* [act user guide](https://nektosact.com/)
* [act on github](https://github.com/nektos/act)

## References

* [Maturin distribution](https://maturin.rs/distributing.html#abi3)
  + includes section 'Cross-compile to Windows'
* [Pyo3 building and distribution](https://pyo3.rs/v0.26.0/building-and-distribution#cross-compiling)
* [Python Packaging User Guide](https://packaging.python.org/en/latest/)
  PiPy instructions for building and distributing packages
