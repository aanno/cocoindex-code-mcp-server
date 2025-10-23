<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# How does the cocoindex flow support incremental file updates? How to use that? Show me some code

**CocoIndex** supports **incremental file updates** by automatically tracking which parts of your source data have changed and only recomputing or updating the corresponding elements in your index, rather than reprocessing everything. This is handled internally via data lineage tracking and a computation cache[^3_1][^3_2][^3_4][^3_5].

### How incremental updates work

- **Detects updates**: CocoIndex monitors your data source for changes, either periodically (using a refresh interval) or through change notifications and recent-changes polling (for sources like Google Drive)[^3_3][^3_4].
- **Minimizes computation**: Only parts of the data pipeline affected by the changes are recomputed; cached results for unchanged data are reused.
- **Handles data lineage**: Maintains mapping from source data to index rows so that changed or deleted elements are managed automatically―no stale data left[^3_1][^3_4].

### How to use incremental updates

You do not need to implement low-level incremental logic yourself; **just define your ETL transformation as normal**. CocoIndex will automatically apply incremental processing on each run[^3_1][^3_2][^3_4][^3_5].

There are **two ways to run incremental updates**:

- **Batch mode**: One-time index update that efficiently processes only new/changed data.
- **Live update mode**: Runs a long-lived watcher for continuous real-time update, synchronizing your index to source changes with low latency.

### Example: Live update mode with code

Using the Python SDK, here's how you might set up an incremental live update from a Google Drive source:

```python
import cocoindex
import datetime

flow_builder = cocoindex.FlowBuilder()

# Add a source with change data capture enabled
data_scope = {}
data_scope["documents"] = flow_builder.add_source(
    cocoindex.sources.GoogleDrive(
        recent_changes_poll_interval=datetime.timedelta(seconds=10),
        refresh_interval=datetime.timedelta(minutes=1)
    )
)

# Build the flow as usual (e.g., transformations)
# flow_builder.add_transformation(...)

# Run the updater in live mode
updater = cocoindex.FlowLiveUpdater(flow_builder)
updater.run()
```

Or use the CLI:

```sh
cocoindex update -L
```

Where `-L` enables live incremental updates[^3_2][^3_3][^3_4].

**No custom incremental code is needed**; CocoIndex transparently manages the file updates, tracks lineage, and reuses cached computations, ensuring that reprocessing is minimal and efficient[^3_1][^3_2][^3_4].

For more real-world code, see the [official CocoIndex repository](https://github.com/cocoindex-io/cocoindex)[^3_5].

<div style="text-align: center">⁂</div>

[^3_1]: <https://hackernoon.com/why-recompute-everything-when-you-can-use-this-solution-to-keep-your-ai-index-fresh-automatically>

[^3_2]: <https://dev.to/badmonster0/cocoindex-changelog-2025-04-05-123i>

[^3_3]: <https://cocoindex.io/blogs/continuous-updates/>

[^3_4]: <https://cocoindex.io/blogs/incremental-processing>

[^3_5]: <https://github.com/cocoindex-io/cocoindex>

---
