# CocoIndex Version Updates

## 0.1.63 → 0.3.37

### Module restructuring

`functions.py`, `sources.py`, and `targets.py` were converted from flat modules into packages (directories with `__init__.py`). All public symbols are re-exported from the package root, so existing code using `cocoindex.functions.X`, `cocoindex.sources.X`, and `cocoindex.targets.X` continues to work without changes.

### New public API symbols

**`__init__.py` additions:**
- `cocoindex.__version__` — package version string
- `cocoindex.settings` — decorator to register a settings provider function
- `cocoindex.add_transient_auth_entry` — transient (in-memory) auth entry registration
- `cocoindex.open_flow` — new preferred alternative to the deprecated `add_flow_def`
- `cocoindex.add_flow_def`, `cocoindex.remove_flow` — marked **DEPRECATED**
- `cocoindex.FlowUpdaterStatusUpdates` — dataclass with `active_sources` and `updated_sources` fields
- `cocoindex.FtsIndexDef` — full-text search index definition
- `cocoindex.HnswVectorIndexMethod`, `cocoindex.IvfFlatVectorIndexMethod` — vector index method specs
- `cocoindex.GlobalExecutionOptions` — global execution tuning
- `cocoindex.QueryHandlerResultFields`, `cocoindex.QueryInfo`, `cocoindex.QueryOutput` — query handler support

**`flow.py` additions on `Flow`:**
- `Flow.close()` — explicit resource cleanup
- `Flow.add_query_handler()` / `Flow.query_handler()` decorator — attach query handlers to a flow
- `Flow._internal_flow()` — internal access to the engine flow object

**`FlowLiveUpdaterOptions` new fields:**
- `reexport_targets: bool = False` — force re-export even when source data is unchanged
- `full_reprocess: bool = False` — invalidate all caches and reprocess from scratch

**`FlowLiveUpdater` new methods:**
- `next_status_updates()` / `next_status_updates_async()` — poll for source activity since the last call

**`DataCollector.export()` additions:**
- `attachments: Sequence[op.TargetAttachmentSpec] = ()` — e.g. `PostgresSqlCommand` for custom DDL
- `fts_indexes: Sequence[index.FtsIndexDef] = ()` — attach FTS indexes at export time
- First positional parameter renamed `name` → `target_name` (positional-only, no call-site change needed)

**`DataSlice.row()` additions:**
- `max_inflight_rows: int | None = None`
- `max_inflight_bytes: int | None = None`

### `op.executor_class` — breaking change for `analyze()`

In 0.1.63 the framework passed input argument schemas to `analyze(self, arg1, arg2, ...)`. In 0.3.37 the framework calls `analyze(self)` with **no arguments**; type information is derived from the function signature of `__call__` instead. If `analyze()` still needs to return a dynamic type it can do so based on `self.spec` alone.

**Migration:** any `analyze` method that had required positional parameters must give them default values (e.g. `= None`) or remove them entirely.

```python
# Before (0.1.63)
def analyze(self, content: Any, language: Any = "Haskell") -> type:
    return list[HaskellChunkRow]

# After (0.3.37)
def analyze(self, content: Any = None, language: Any = "Haskell") -> type:
    return list[HaskellChunkRow]
```

`analyze()` is now **optional** — if omitted, the return type is inferred entirely from the `__call__` return annotation.

### `op.OpArgs` / `@op.executor_class` new kwargs

- `batching: bool = False` — enables batch execution; `__call__` receives `list[T]` and must return `list[R]`
- `max_batch_size: int | None = None` — cap on batch size (only with `batching=True`)
- `arg_relationship: tuple[ArgRelationship, str] | None = None` — declares semantic relationship between an input arg and the output (e.g. `(ArgRelationship.CHUNKS_BASE_TEXT, "content")`)
- `timeout: datetime.timedelta | None = None` — per-execution timeout
- New `ArgRelationship` enum: `EMBEDDING_ORIGIN_TEXT`, `CHUNKS_BASE_TEXT`, `RECTS_BASE_IMAGE`

### `targets.Postgres` new fields

- `schema: str | None = None` — target PostgreSQL schema (default: `public`)
- `column_options: dict[str, PostgresColumnOptions] | None = None` — per-column overrides (e.g. force `halfvec` type)

New helper: `targets.PostgresSqlCommand(op.TargetAttachmentSpec)` — attach custom setup/teardown SQL to a Postgres target.

### `sources` additions

- `sources.LocalFile` — new optional `max_file_size: int | None` field
- `sources.AmazonS3` — new `redis: RedisNotification | None` and `force_path_style: bool` fields
- `sources.AzureBlob` — new source for Azure Blob Storage
- `sources.Postgres` — new source for reading from a PostgreSQL table (with optional change-capture via `PostgresNotification`)

### `targets` additions

- `targets.FalkorDB` / `FalkorDBDeclaration` — graph storage via FalkorDB
- `targets.Ladybug` / `LadybugDeclaration` — replacement for the retired Kuzu target (`KuzuConnection`, `Kuzu`, `KuzuDeclaration` kept as backward-compatible aliases)
- `targets.LanceDB`, `targets.Pinecone`, `targets.Doris`, `targets.ChromaDB`, `targets.Turbopuffer` — new connectors

### Internal refactoring (no call-site impact)

- `convert.py` removed; replaced by `engine_object.py`, `engine_type.py`, `engine_value.py`
- `make_setup_bundle()` is now a sync wrapper around the new `make_setup_bundle_async()`
- `_engine.init_pyo3_runtime()` called automatically on import
- Version compatibility check (`_version_check`) runs on import
