#!/usr/bin/env python3
"""Aggregate outer-fold shift predictions and compute circuit bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml4qem_reuse.ensemble import paired_bootstrap, paired_signflip_test
from ml4qem_reuse.workflows.train_local_within_domain import (
    LocalData,
    _aggregate_by_base,
    local_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=57721)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    output = {
        "schema_version": 1,
        "analysis": "aggregated_local_distribution_shifts",
        "input_sha256": result["input_sha256"],
        "split_manifest_sha256": result["split_manifest_sha256"],
        "model_seed": result["model_seed"],
        "bootstrap_draws": args.bootstrap_draws,
        "failure_definition": result["failure_definition"],
        "scenarios": {},
        "hardware_jobs_submitted": 0,
    }
    with np.load(args.data, allow_pickle=False) as archive, np.load(
        args.predictions, allow_pickle=False
    ) as predictions:
        data = LocalData(archive)
        row_keys = [key for key in predictions.files if key.endswith("__row_index")]
        for scenario_index, (scenario, labels) in enumerate(result["scenarios"].items()):
            output["scenarios"][scenario] = {}
            for label_index, label in enumerate(labels):
                prefixes = [
                    key[: -len("__row_index")]
                    for key in row_keys
                    if key.startswith(f"{scenario}__{label}__cell")
                ]
                if not prefixes:
                    raise ValueError(f"no predictions for {scenario}/{label}")
                rows = np.concatenate([predictions[f"{prefix}__row_index"] for prefix in prefixes])
                method_sets = [
                    {
                        key[len(prefix) + 2 : -len("__prediction")]
                        for key in predictions.files
                        if key.startswith(f"{prefix}__") and key.endswith("__prediction")
                    }
                    for prefix in prefixes
                ]
                methods = sorted(set.intersection(*method_sets))
                method_predictions = {
                    method: np.concatenate(
                        [predictions[f"{prefix}__{method}__prediction"] for prefix in prefixes]
                    )
                    for method in methods
                }
                method_radii = {
                    method: np.concatenate(
                        [predictions[f"{prefix}__{method}__radius"] for prefix in prefixes]
                    )
                    for method in methods
                }
                target = data.y(rows)
                sampled = data.row_sampled[rows]
                base = data.row_base[rows]
                metrics = {
                    method: local_metrics(
                        prediction, target, sampled, method_radii[method], rows, data
                    )
                    for method, prediction in method_predictions.items()
                }
                effects = {}
                for reference_index, reference in enumerate(("noisy_sampled", "rf")):
                    reference_error = _aggregate_by_base(
                        np.mean(np.abs(method_predictions[reference] - target), axis=1), base
                    )
                    effects[reference] = {}
                    for method_index, method in enumerate(methods):
                        if method == reference:
                            continue
                        error = _aggregate_by_base(
                            np.mean(np.abs(method_predictions[method] - target), axis=1), base
                        )
                        inference = paired_bootstrap(
                            error,
                            reference_error,
                            draws=args.bootstrap_draws,
                            seed=(
                                args.seed
                                + 10_000 * scenario_index
                                + 1000 * label_index
                                + 100 * reference_index
                                + method_index
                            ),
                        )
                        inference["signflip_p_value_two_sided"] = paired_signflip_test(
                            error,
                            reference_error,
                            draws=args.bootstrap_draws,
                            seed=(
                                args.seed
                                + 50_000
                                + 10_000 * scenario_index
                                + 1000 * label_index
                                + 100 * reference_index
                                + method_index
                            ),
                        )
                        effects[reference][method] = inference
                output["scenarios"][scenario][label] = {
                    "outer_parts": len(prefixes),
                    "base_circuits": int(len(np.unique(base))),
                    "rows": int(len(rows)),
                    "available_methods": methods,
                    "metrics": metrics,
                    "paired_circuit_mae_effects": effects,
                }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    for scenario, labels in output["scenarios"].items():
        print(f"\n{scenario}")
        for label, cell in labels.items():
            method = (
                "safe_cell_simplex"
                if "safe_cell_simplex" in cell["available_methods"]
                else "safe_simplex"
            )
            noisy = cell["metrics"]["noisy_sampled"]["mae"]
            safe = cell["metrics"][method]["mae"]
            effect = cell["paired_circuit_mae_effects"]["noisy_sampled"][method]
            print(
                f"  {label:24s} {method}: {safe:.6f} vs noisy {noisy:.6f}; "
                f"difference {effect['estimate']:+.6f} "
                f"[{effect['ci_low']:+.6f}, {effect['ci_high']:+.6f}]"
            )


if __name__ == "__main__":
    main()
