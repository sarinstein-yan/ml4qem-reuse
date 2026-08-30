# Contributing

Contributions should keep the release small, hardware-disabled, and
provenance-complete.

1. Install with `uv sync --frozen`.
2. Run `uv run ruff check` and `uv run pytest -q`.
3. Add tests for changes to fetching, hashes, safety checks, serialization, or
   statistics.
4. Do not add credentials, remote quantum execution, publisher PDFs, or large
   prediction arrays to Git.
5. Update `DATA_REGISTRY.yaml`, `LICENSES.csv`, and third-party notices when a
   data source, transformation, or licence boundary changes.

Scientific changes should identify their evidence class and preserve grouped
split boundaries. A local simulation must not be presented as hardware data,
and archived measurements must not be presented as a current-device run.

