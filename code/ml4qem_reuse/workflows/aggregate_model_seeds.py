#!/usr/bin/env python3
"""Aggregate repeated model-seed results without treating seeds as circuits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "sample_sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in args.inputs]
    expected = {record["model_seed"] for record in records}
    if len(expected) != len(records):
        raise ValueError("model seeds must be unique")
    invariant_keys = ("input_sha256", "split_manifest_sha256", "training_size_base_circuits", "folds")
    for key in invariant_keys:
        if any(record[key] != records[0][key] for record in records[1:]):
            raise ValueError(f"seed runs disagree on {key}")
    metrics = {}
    for variant_key in ("raw_metrics", "clipped_metrics"):
        metrics[variant_key] = {}
        for method in records[0][variant_key]:
            metrics[variant_key][method] = {
                metric: summarize([record[variant_key][method][metric] for record in records])
                for metric in records[0][variant_key][method]
            }
    effects = {}
    for reference in records[0]["raw_paired_circuit_mae_effects"]:
        effects[reference] = {}
        for method in records[0]["raw_paired_circuit_mae_effects"][reference]:
            effects[reference][method] = {
                field: summarize(
                    [
                        record["raw_paired_circuit_mae_effects"][reference][method][field]
                        for record in records
                    ]
                )
                for field in ("estimate", "ci_low", "ci_high")
            }
    output = {
        "schema_version": 1,
        "analysis": "model_seed_sensitivity",
        "model_seeds": sorted(expected),
        "seed_runs_are_sensitivity_repeats_not_independent_circuits": True,
        **{key: records[0][key] for key in invariant_keys},
        "failure_definition": records[0]["failure_definition"],
        "metrics": metrics,
        "paired_effects": effects,
        "hardware_jobs_submitted": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"model_seeds": output["model_seeds"]}, indent=2))
    for method in ("rf", "safe_simplex", "safe_ridge", "safe_cell_simplex"):
        mae = metrics["raw_metrics"][method]["mae"]
        print(
            f"{method:18s} MAE mean={mae['mean']:.6f} SD={mae['sample_sd']:.6f} "
            f"range={mae['minimum']:.6f}..{mae['maximum']:.6f}"
        )


if __name__ == "__main__":
    main()
