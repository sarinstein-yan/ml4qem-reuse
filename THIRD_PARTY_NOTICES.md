# Third-party notices

## ML4QEM artifact

The optional `full` workflow acquires the Apache-2.0 ML4QEM artifact from
<https://doi.org/10.5281/zenodo.13769804>, archived commit `9776e1b`. The
artifact is not included in this Git repository. The one-line patch under
`patches/` is applied only to a copied source tree.

Minimal data extracts and transformations are distributed in the companion
data archive under Apache-2.0. Their file-level provenance and transformation
scripts are recorded in that archive's `DATA_REGISTRY.yaml`, `PROVENANCE.md`,
and `LICENSES.csv`.

## Publisher materials

The article DOI is <https://doi.org/10.1038/s42256-024-00927-2>. No article or
Supplementary Information PDF is distributed. The official Source Data
workbook is fetched from the publisher and verified but is not included in
either release archive.

Python dependencies retain their own licences; `uv.lock` records the exact
resolved versions. The Apache-2.0 grant in this repository does not relicense
third-party packages or publisher materials.
