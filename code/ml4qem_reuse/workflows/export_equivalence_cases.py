#!/usr/bin/env python3
"""Export paper-era circuits and features into a current-stack-neutral form."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np

from ml4qem_reuse.workflows._paths import workspace_root


PROJECT = workspace_root()
SNAPSHOT = PROJECT / "upstream/snapshots/qiskit-community-ml-qem-9776e1b"
DATA = SNAPSHOT / "docs/tutorials/data"
sys.path.insert(0, str(SNAPSHOT / "docs/tutorials"))

from mlp import encode_data_v2_ecr  # noqa: E402


SPECS = {
    "ising_incoherent": ("ising_init_from_qasm", (0, 7, 14)),
    "ising_coherent": ("ising_init_from_qasm_coherent", (0, 7, 14)),
    "ising_no_readout": ("ising_init_from_qasm_no_readout", (0, 7, 14)),
}


def _normalise_noisy(entry: Dict[str, object]) -> list[float]:
    noisy = entry["noisy_exp_values"]
    if len(noisy) == 1 and isinstance(noisy[0], (list, tuple, np.ndarray)):
        noisy = noisy[0]
    return np.asarray(noisy, dtype=float).tolist()


def cases(per_file: int) -> Iterator[Tuple[str, int, int, Dict[str, object]]]:
    for dataset, (directory, depths) in SPECS.items():
        for depth in depths:
            path = DATA / directory / "train" / f"step_{depth}.pk"
            with path.open("rb") as stream:
                entries = pickle.load(stream)
            for index, entry in enumerate(entries[:per_file]):
                yield dataset, depth, index, entry
    random_path = DATA / "extra_random_circuits/train/q12.pk"
    with random_path.open("rb") as stream:
        entries = pickle.load(stream)
    for index, entry in enumerate(entries[:per_file]):
        adapted = {
            "circuit": entry["circuit"],
            "noisy_exp_values": [np.asarray(entry["noisy_exp_val"], dtype=float).tolist()],
            "ideal_exp_value": np.asarray(entry["ideal_exp_val"], dtype=float).tolist(),
        }
        yield "random_12q", int(entry["circuit"].depth()), index, adapted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-file", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_file < 1:
        raise ValueError("--per-file must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(str(args.output), "wt", encoding="utf-8") as stream:
        for dataset, depth, index, entry in cases(args.per_file):
            circuit = entry["circuit"]
            noisy = _normalise_noisy(entry)
            target = np.asarray(entry["ideal_exp_value"], dtype=float).tolist()
            qasm = circuit.qasm()
            features, _ = encode_data_v2_ecr(
                [circuit],
                [target],
                [noisy],
                obs_size=len(noisy),
                two_q_gate="cx",
            )
            record = {
                "schema_version": 1,
                "case_id": f"{dataset}:depth={depth}:index={index}",
                "dataset": dataset,
                "source_depth": depth,
                "source_index": index,
                "qasm2": qasm,
                "qasm2_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
                "n_qubits": circuit.num_qubits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "count_ops": dict(circuit.count_ops()),
                "noisy": noisy,
                "target": target,
                "legacy_v2_features": np.asarray(features[0], dtype=float).tolist(),
                "feature_config": {
                    "two_qubit_gate": "cx",
                    "angle_bin_size_pi": 0.025,
                    "observable_count": len(noisy),
                },
                "provenance": "legacy_reproduction",
            }
            stream.write(json.dumps(record, sort_keys=True))
            stream.write("\n")
            count += 1
    print(f"exported_cases={count}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
