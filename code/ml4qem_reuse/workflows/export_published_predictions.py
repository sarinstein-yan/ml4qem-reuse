#!/usr/bin/env python3
"""Convert distributed pandas pickles to a stable numeric prediction archive."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from ml4qem_reuse.workflows._paths import workspace_root


PROJECT = workspace_root()
SOURCE = PROJECT / "upstream/snapshots/qiskit-community-ml-qem-9776e1b/docs/paper_figures"
SPECS = {
    "random": ("random_circuits.pk", ("noisy", "zne", "ols_full", "rfr_list", "mlp", "gnn")),
    "ising_no_readout": (
        "no_readout_over_depths.pk",
        ("noisy", "zne", "ols_full", "rfr_list", "mlp", "gnn"),
    ),
    "ising_incoherent": (
        "incoherent_over_depths.pk",
        ("noisy", "zne", "ols_full", "rfr_list", "mlp", "gnn"),
    ),
    "ising_coherent": (
        "coherent_over_depths.pk",
        ("noisy", "zne", "ols_full", "rfr_list", "mlp", "gnn"),
    ),
    "ising_archived_hardware": (
        "hardware_over_depth.pk",
        ("noisy", "zne", "rfr_list"),
    ),
}


def _matrix(values: object) -> np.ndarray:
    return np.asarray([np.asarray(value, dtype=float) for value in values], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    arrays = {}
    manifest = {
        "schema_version": 1,
        "artifact": "10.5281/zenodo.13769804",
        "artifact_commit": "9776e1b",
        "provenance": "archived_hardware_or_simulation_as_labelled",
        "datasets": {},
        "hardware_jobs_submitted": 0,
    }
    for dataset, (filename, columns) in SPECS.items():
        with (SOURCE / filename).open("rb") as stream:
            frame = pickle.load(stream)["df"]
        arrays[f"{dataset}__target"] = _matrix(frame["ideal"])
        arrays[f"{dataset}__depth"] = np.asarray(frame["step"], dtype=int)
        arrays[f"{dataset}__base_id"] = np.asarray(
            [f"{dataset}:{index}" for index in range(len(frame))]
        )
        for column in columns:
            arrays[f"{dataset}__{column}"] = _matrix(frame[column])
        manifest["datasets"][dataset] = {
            "source_file": filename,
            "rows": int(len(frame)),
            "observables": int(arrays[f"{dataset}__target"].shape[1]),
            "depth_values": sorted(np.unique(arrays[f"{dataset}__depth"]).astype(int).tolist()),
            "available_predictions": list(columns),
            "evidence_class": (
                "archived_hardware" if dataset == "ising_archived_hardware" else "legacy_reproduction"
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
