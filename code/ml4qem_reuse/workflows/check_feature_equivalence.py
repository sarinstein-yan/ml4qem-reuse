#!/usr/bin/env python3
"""Compare current-stack feature extraction with legacy exported values."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from qiskit import qasm2

from ml4qem_reuse.features import encode_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args()
    rows = []
    with gzip.open(args.input, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            circuit = qasm2.loads(
                record["qasm2"],
                custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            )
            config = record["feature_config"]
            current = encode_v2(
                [circuit],
                [record["noisy"]],
                observable_count=config["observable_count"],
                two_qubit_gate=config["two_qubit_gate"],
            )[0]
            legacy = np.asarray(record["legacy_v2_features"], dtype=float)
            difference = current.astype(float) - legacy
            rows.append(
                {
                    "case_id": record["case_id"],
                    "dataset": record["dataset"],
                    "feature_count": int(len(current)),
                    "max_absolute_difference": float(np.max(np.abs(difference))),
                    "mean_absolute_difference": float(np.mean(np.abs(difference))),
                    "nonzero_count": int(np.count_nonzero(difference)),
                    "equivalent_at_tolerance": bool(np.all(np.abs(difference) <= args.atol)),
                    "legacy_depth": record["depth"],
                    "current_depth": circuit.depth(),
                    "legacy_size": record["size"],
                    "current_size": circuit.size(),
                }
            )
    result = {
        "schema_version": 1,
        "input": str(args.input),
        "absolute_tolerance": args.atol,
        "n_cases": len(rows),
        "n_equivalent": sum(row["equivalent_at_tolerance"] for row in rows),
        "global_max_absolute_difference": max(row["max_absolute_difference"] for row in rows),
        "depth_mismatch_count": sum(row["legacy_depth"] != row["current_depth"] for row in rows),
        "size_mismatch_count": sum(row["legacy_size"] != row["current_size"] for row in rows),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in result if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
