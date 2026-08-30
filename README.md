# ML4QEM reuse

`ml4qem-reuse` is a maintained scientific-core implementation for recovering
the public ML4QEM artifact, checking its portable numerical interface, and
testing model reuse on a grouped local benchmark. The repository is designed
to work without quantum-service credentials. Every included execution path is
local; archived hardware-labelled arrays are analysed as historical data only.

The Git repository is intentionally small. Complete claim-bearing prediction
arrays and the generated benchmark are distributed in the separate
`ml4qem-reuse-data-v1.0.0` archive. The original 659.7 MB ML4QEM artifact and
publisher Source Data are never repackaged here.

## Quick start

Install the locked environment:

```bash
uv sync --frozen
```

The companion dataset is published at DOI
[`10.5281/zenodo.22168376`](https://doi.org/10.5281/zenodo.22168376). Fetch and
reproduce the analysis directly from the registered public archive:

```bash
uv run ml4qem-reuse fetch --profile analysis
uv run ml4qem-reuse reproduce --profile analysis
```

For an intentionally mirrored or local copy, `ML4QEM_REUSE_DATA_URL` may
override the registered URL with an absolute path or a `file://` URL.

## Profiles

```bash
uv run ml4qem-reuse fetch --profile analysis
uv run ml4qem-reuse reproduce --profile analysis

uv run ml4qem-reuse fetch --profile benchmark
uv run ml4qem-reuse reproduce --profile benchmark

uv run ml4qem-reuse fetch --profile full
uv run ml4qem-reuse reproduce --profile full
```

- `analysis` verifies and extracts the frozen data bundle, downloads the
  publisher Source Data workbook from its official URL, and rebuilds the
  public summary, every CSV table, and eight PDF/PNG figure pairs without
  refitting a model.
- `benchmark` uses the independently generated local benchmark and grouped
  split manifest to refit and evaluate the fixed model/ensemble specification.
  A controlled CI-friendly check is available as
  `reproduce --profile benchmark --smoke`.
- `full` additionally downloads the DOI-pinned ML4QEM artifact, reconstructs
  the Python 3.9 paper-era environment, applies the one-line repair to a copy,
  runs the recovered tests and exports, and performs feature and random-forest
  portability checks.

Downloads use a cache (default `.cache/ml4qem-reuse`), resume partial local or
HTTP transfers, retry transient HTTP failures, and fail before use if either
the registered byte count or SHA-256 differs. Tar and ZIP traversal is rejected
before extraction. Set `ML4QEM_REUSE_CACHE` or pass `--cache-dir` to move the
cache.

## Scientific scope

The local benchmark contains four four-qubit circuit families, three noise
mechanisms at three strengths, four shot budgets, three nested sampling
replicates, and four observables. All descendants of a base circuit stay in the
same outer partition. The repository preserves the distinction among
published-result recovery, fixed-array software portability, archived hardware
evidence, local simulation, and method-extension results.

The reference summary and small tables under `reference/` are included for
verification. Large NPZ files are absent from Git and are checked after the
companion data archive is extracted.

## Safety, provenance, and licences

- `ml4qem_reuse.safety` rejects real, remote, provider, service, and
  credential-bearing configurations.
- No command submits a QPU or other hardware job.
- `DATA_REGISTRY.yaml` records immutable URLs/DOIs, hashes, byte counts,
  licences, redistribution status, evidence class, and derived-from links.
- New code is Apache-2.0. New benchmark data and analyses are CC BY 4.0.
  Minimal files derived from ML4QEM retain Apache-2.0; see `LICENSES.csv` and
  `THIRD_PARTY_NOTICES.md`.
- Article files, Supplementary Information, and publisher PDFs are not
  redistributed. Publisher Source Data are fetched and verified, not bundled.

The maintainer and creator of the newly released material is Xianquan Yan
([ORCID 0009-0009-6952-421X](https://orcid.org/0009-0009-6952-421X)).

See `REPRODUCING.md` for output layout, expected costs, and verification
commands.
