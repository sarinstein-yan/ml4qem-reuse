#!/usr/bin/env python3
"""Collect the frozen nested training-size curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml4qem_reuse.serialization import strict_json_dumps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in args.inputs]
    records.sort(key=lambda item: item["training_size_base_circuits"])
    for key in ("input_sha256", "split_manifest_sha256", "folds", "model_seed"):
        if any(record[key] != records[0][key] for record in records[1:]):
            raise ValueError(f"training-size runs disagree on {key}")
    methods = (
        "noisy_sampled",
        "rf",
        "simplex",
        "safe_simplex",
        "safe_ridge",
        "raw_cell_affine",
        "safe_cell_simplex",
    )
    metrics = (
        "mae",
        "rmse",
        "p95_absolute_error",
        "failure_rate",
        "worst_family_mae",
        "worst_noise_family_mae",
        "worst_strength_mae",
        "worst_shot_budget_mae",
        "interval_condition_circuit_joint_coverage",
        "finite_interval_available",
        "interval_mean_width",
    )
    cells = []
    for record in records:
        cells.append(
            {
                "training_size": record["training_size_base_circuits"],
                "methods": {
                    method: {metric: record["raw_metrics"][method][metric] for metric in metrics}
                    for method in methods
                },
                "paired_mae_effect_vs_noisy": {
                    method: record["raw_paired_circuit_mae_effects"]["noisy_sampled"][method]
                    for method in methods
                    if method not in {"noisy_sampled", "rf"}
                },
            }
        )
    output = {
        "schema_version": 1,
        "analysis": "nested_training_size_curve",
        "training_sizes": [cell["training_size"] for cell in cells],
        "input_sha256": records[0]["input_sha256"],
        "split_manifest_sha256": records[0]["split_manifest_sha256"],
        "folds": records[0]["folds"],
        "model_seed": records[0]["model_seed"],
        "failure_definition": records[0]["failure_definition"],
        "cells": cells,
        "hardware_jobs_submitted": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(strict_json_dumps(output, indent=2, sort_keys=True) + "\n")
    for cell in cells:
        noisy = cell["methods"]["noisy_sampled"]["mae"]
        safe = cell["methods"]["safe_simplex"]["mae"]
        interval = cell["paired_mae_effect_vs_noisy"]["safe_simplex"]
        print(
            f"n={cell['training_size']:3d} noisy={noisy:.6f} global-safe={safe:.6f} "
            f"difference={safe - noisy:+.6f} "
            f"CI={interval['ci_low']:+.6f}..{interval['ci_high']:+.6f}"
        )


if __name__ == "__main__":
    main()
