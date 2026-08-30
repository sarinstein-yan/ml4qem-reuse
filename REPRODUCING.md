# Public reproduction guide

## Prerequisites

- `uv` with access to CPython 3.12 or 3.13.
- Network access for the official publisher asset; `full` also needs the
  original Zenodo archive and downloads a CPython 3.9 environment.
- No IBM Quantum or other quantum-service account.

The maintained environment is fully specified by `pyproject.toml` and
`uv.lock`. The paper-era environment is separately specified by
`environments/legacy-bootstrap.txt` and
`environments/legacy-pip-freeze.txt`.

## Analysis profile

```bash
uv run ml4qem-reuse fetch --profile analysis
uv run ml4qem-reuse reproduce --profile analysis
```

The default output is `outputs/analysis/`:

- `results/frozen/frozen_summary.json`
- seven CSV tables under `results/frozen/`
- eight machine-readable LaTeX table views under `tables/latex/`
- eight transparent PDF and eight PNG figures under `figures/`
- `analysis_verification.json`

The verification record states whether the regenerated summary and CSV files
match the small Git references. Model fitting is not performed by this
profile.

## Benchmark profile

```bash
uv run ml4qem-reuse fetch --profile benchmark
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  uv run ml4qem-reuse reproduce --profile benchmark --threads 8
```

This is the full five-fold, 512-training-circuit fit and can take several
minutes. For a bounded one-fold check:

```bash
uv run ml4qem-reuse reproduce --profile benchmark --smoke \
  --threads 1 --bootstrap-draws 128
```

Both variants write a result JSON and prediction NPZ with
`hardware_jobs_submitted: 0`.

## Full legacy recovery

```bash
uv run ml4qem-reuse fetch --profile full
uv run ml4qem-reuse reproduce --profile full
```

The fetch verifies the original DOI archive at commit `9776e1b`, byte count
`659720205`, and SHA-256
`504b5df8ca32608cdecbb1a4882a983e44f7187488034145f48bc18b660d09e1`.
The tested release path is a source checkout on Linux x86_64 with Bash, GNU
`patch`, and at least 30 GiB of free disk space for the immutable cache, copied
working tree, and legacy environment. Allow at least 4 GiB of RAM; more is
recommended. `uv` may install CPython 3.9. In one clean reference run, fetch
took about five minutes and reproduction about six minutes, but network and CPU
speed can make the process take substantially longer. This profile is not
supported from a code-only wheel because it needs the bundled patch and
environment lock files.
The immutable cache extraction is copied before the one-line patch is applied.
The full workflow reconstructs the paper-era environment, runs the repaired
tests, re-exports stable numeric predictions, verifies their arrays, checks 80
legacy/current feature cases, and refits fixed random-forest designs in both
environments. This profile is CPU/network intensive and is not part of CI.

## Developer checks

```bash
uv sync --frozen
uv run ruff check
uv run pytest -q
```

The repository contains no hardware execution backend. Archived hardware
arrays are inputs to local NumPy/scikit-learn statistics only.
