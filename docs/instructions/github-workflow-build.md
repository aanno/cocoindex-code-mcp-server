GitHub Copilot Chat Assistant — Root cause and fix:

Root causes (from the failing job logs, ref 72f7edbaaa2b0a9c2887632c0d48de864af9186f):
- cargo metadata failed because the step ran where there was no Cargo.toml: "error: could not find `Cargo.toml` in `/home/runner/work/cocoindex-code-mcp-server/cocoindex-code-mcp-server` or any parent directory".
- maturin failed because no virtualenv/conda env was found: "Couldn't find a virtualenv or conda environment… set VIRTUAL_ENV (through activate) … or have a virtualenv called .venv".

Why the other (recent) job succeeded
- The successful Build job ran in the correct workspace and produced a wheel from the rust subcrate (logs show builds at /__w/.../rust and wheel generation). The Test workflow didn’t set the working directory (or action inputs) consistently and didn’t create/activate a Python venv before calling maturin.

Concrete fixes to apply in .github/workflows/test.yml

1) Ensure cargo / rust actions run in the directory that contains Cargo.toml
If your Rust crate is in the repository root or in a subfolder (e.g., rust/), run cargo from that folder or set working-directory on the step.

Example: run the cargo-related step from the rust folder
- either add a cd before cargo commands:
  run: |
    cd rust
    cargo build --release

- or use the step-level working-directory:
  - name: Build Rust
    working-directory: rust
    run: cargo build --release

If you use an action that internally runs cargo metadata (e.g. a cache action), make sure its inputs or the step use the rust directory. For example, if the action expects an input like rust-src-dir or project-root, set it to rust (or correct subpath). The failing logs include a check for 'rust-src-dir' — set that to the actual path.

2) Create and activate a Python virtualenv in the same step that runs maturin
maturin requires either VIRTUAL_ENV, CONDA_PREFIX or a .venv folder plus activation. Because shell steps do not preserve activation across steps, create+activate+run maturin in one step (or ensure PATH/VIRTUAL_ENV are set).

Example step to build the wheel (bash shell):
- name: Build Python wheel with maturin
  run: |
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip maturin
    cd rust            # change to your rust crate folder if needed
    maturin build --release
  shell: bash

If you prefer separate steps, export VIRTUAL_ENV and PATH so subsequent steps see the venv:
- name: Create venv and install maturin
  run: |
    python -m venv .venv
    . .venv/bin/activate
    pip install --upgrade pip maturin
  env:
    PATH: ${{ github.workspace }}/.venv/bin:${{ env.PATH }}

3) Minimal snippet patch for test.yml (illustrative)
Replace/adjust the relevant steps in your test workflow to something like:

- uses: actions/checkout@v4

- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'

- name: Prepare venv and build wheel
  run: |
    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip maturin
    cd rust                # <- set to your crate path containing Cargo.toml
    maturin build --release
  shell: bash

- name: Cache/restore cargo (if you use a cache action that runs cargo metadata)
  # ensure the cache action or step runs from the same rust folder; either:
  working-directory: rust
  run: <cache or cargo commands>

Notes / checklist
- Confirm where Cargo.toml lives (repo root vs rust/). Update all steps that run cargo/metadata/maturin to operate in that directory.
- If a third-party action expects a path input like rust-src-dir, set that input to the correct subpath (e.g., rust).
- Use a combined step for venv creation + maturin invocation (activation must be in the same shell).
- Re-run the workflow; you should see cargo metadata succeed and maturin find .venv and build the wheel (similar to the successful Build job logs).

If you paste the test workflow (.github/workflows/test.yml) or indicate the crate layout (where Cargo.toml is), I will give a precise diff you can apply.
