#!/usr/bin/env python3
"""Reanalyse the public ibm_algiers Fig. 4 result table without new QPU use."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ml4qem_reuse.ensemble import (
    FAILURE_MARGIN,
    conformal_radius,
    evaluate_predictions,
    fit_calibration_split,
    paired_bootstrap,
)
from ml4qem_reuse.workflows._paths import workspace_root


PROJECT = workspace_root()
METHODS = ("noisy", "zne", "rf")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_base(values: np.ndarray) -> np.ndarray:
    return np.mean(values, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=PROJECT / "data/derived/published_predictions.npz"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=141421)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        prefix = "ising_archived_hardware__"
        target = archive[prefix + "target"]
        depth = archive[prefix + "depth"]
        base_id = archive[prefix + "base_id"]
        source_predictions = {
            "noisy": archive[prefix + "noisy"],
            "zne": archive[prefix + "zne"],
            "rf": archive[prefix + "rfr_list"],
        }
    outer_fold = np.full(len(target), -1, dtype=np.int8)
    radii = {method: np.full(len(target), np.nan) for method in METHODS}
    fold_records = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(depth)), depth)):
        _, calibration = fit_calibration_split(
            train, depth[train], seed=args.seed + fold, calibration_fraction=0.25
        )
        for method in METHODS:
            radii[method][test] = conformal_radius(
                source_predictions[method][calibration], target[calibration]
            )
        outer_fold[test] = fold
        fold_records.append(
            {
                "fold": fold,
                "calibration_circuits": int(len(calibration)),
                "test_circuits": int(len(test)),
            }
        )
    metrics = {
        method: evaluate_predictions(
            prediction, target, source_predictions["noisy"], depth, radii[method]
        )
        for method, prediction in source_predictions.items()
    }
    effects = {}
    for reference_index, reference in enumerate(("noisy", "zne", "rf")):
        reference_error = aggregate_base(np.abs(source_predictions[reference] - target))
        effects[reference] = {}
        for method_index, method in enumerate(METHODS):
            if method == reference:
                continue
            error = aggregate_base(np.abs(source_predictions[method] - target))
            effects[reference][method] = paired_bootstrap(
                error,
                reference_error,
                draws=args.bootstrap_draws,
                seed=args.seed + 100 * reference_index + method_index,
            )
    result = {
        "schema_version": 1,
        "analysis": "archived_ibm_algiers_figure4_reanalysis",
        "evidence_class": "archived_hardware",
        "device_reported_by_source": "ibm_algiers",
        "input_sha256": sha256(args.input),
        "source_result_file": "hardware_over_depth.pk (singular; the plural file does not match Source Data)",
        "test_circuits": 2500,
        "training_circuits_reported_by_source": 500,
        "shots_per_circuit_reported_by_source": 10_000,
        "observables_per_circuit": 4,
        "depth_steps": 10,
        "failure_definition": {
            "reference": "archived unmitigated hardware expectation",
            "absolute_error_margin": FAILURE_MARGIN,
            "rule": "candidate absolute error > reference absolute error + margin",
        },
        "fold_records": fold_records,
        "metrics": metrics,
        "paired_circuit_mae_effects": effects,
        "bootstrap_draws": args.bootstrap_draws,
        "resource_accounting": {
            "ml_qem_total_circuit_executions_reported": 3000,
            "zne_total_circuit_executions_reported": 5000,
            "ml_qem_total_shots_equivalent": 30_000_000,
            "zne_total_shots_equivalent": 50_000_000,
            "source_reported_overall_reduction_fraction": 0.40,
            "source_reported_runtime_reduction_fraction": 0.50,
            "source_reported_training_qpu_hours_at_assumed_2khz": 0.7,
            "interpretation": "historical source accounting, not newly billed or submitted QPU time",
        },
        "hardware_jobs_submitted_in_reanalysis": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions,
        base_id=base_id,
        depth=depth,
        target=target,
        outer_fold=outer_fold,
        **{f"{method}__prediction": value for method, value in source_predictions.items()},
        **{f"{method}__radius": value for method, value in radii.items()},
    )
    for method in METHODS:
        metric = metrics[method]
        print(
            f"{method:5s} MAE={metric['mae']:.6f} RMSE={metric['rmse']:.6f} "
            f"failure={metric['failure_rate']:.4f} joint90={metric['interval_joint_coverage']:.4f}"
        )
    for reference, comparisons in effects.items():
        for method, effect in comparisons.items():
            print(
                f"{method}-{reference}: {effect['estimate']:+.6f} "
                f"[{effect['ci_low']:+.6f}, {effect['ci_high']:+.6f}]"
            )


if __name__ == "__main__":
    main()
