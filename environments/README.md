# Environment specifications

The maintained CPU environment is defined by the root `pyproject.toml` and
`uv.lock` and is installed with `uv sync --frozen`.

The optional paper-era recovery is isolated from the maintained environment:

- CPython 3.9 is selected by `code/bootstrap_legacy_environment.sh`.
- `legacy-bootstrap.txt` documents installation order and constraints.
- `legacy-pip-freeze.txt` records the complete resolved environment.
- `patches/legacy-rb-field.patch` is the only source repair and is applied to
  a copied DOI snapshot, never to the immutable cache extraction.

Neither environment contains quantum-service credentials or a public hardware
execution entry point.

