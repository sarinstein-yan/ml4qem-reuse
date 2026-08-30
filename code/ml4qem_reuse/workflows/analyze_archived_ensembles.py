#!/usr/bin/env python3
"""Cross-fitted ensemble analysis of the published ML4QEM predictions.

This does not rerun or retrain the four original base estimators. It treats
their distributed held-out predictions as a reusable second-stage dataset and
fits every combiner and uncertainty radius strictly inside each outer fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ml4qem_reuse.ensemble import (
    FAILURE_MARGIN,
    conformal_radius,
    evaluate_predictions,
    failure_mask,
    fit_calibration_split,
    fit_ridge_nested,
    fit_simplex,
    paired_bootstrap,
)
from ml4qem_reuse.workflows._paths import workspace_root


PROJECT = workspace_root()
DATASETS = ("random", "ising_no_readout", "ising_incoherent", "ising_coherent")
BASE_KEYS = ("ols_full", "rfr_list", "mlp", "gnn")
BASE_LABELS = ("ols", "rf", "mlp", "gnn")
ENSEMBLE_LABELS = ("mean", "median", "simplex", "ridge")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_predictions(
    base: np.ndarray, simplex: object, ridge: object
) -> dict[str, np.ndarray]:
    return {
        "mean": np.mean(base, axis=-1),
        "median": np.median(base, axis=-1),
        "simplex": simplex.predict(base),
        "ridge": ridge.predict(base),
    }


def _metric_effects(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    noisy: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    rf_error = np.mean(np.abs(predictions["rf"] - target), axis=1)
    rf_failure = np.mean(
        failure_mask(np.abs(predictions["rf"] - target), np.abs(noisy - target)), axis=1
    )
    output = {}
    for offset, method in enumerate(ENSEMBLE_LABELS):
        error = np.mean(np.abs(predictions[method] - target), axis=1)
        failure = np.mean(
            failure_mask(np.abs(predictions[method] - target), np.abs(noisy - target)), axis=1
        )
        output[method] = {
            "paired_circuit_mae_difference": paired_bootstrap(
                error, rf_error, draws=draws, seed=seed + offset
            ),
            "paired_failure_fraction_difference": paired_bootstrap(
                failure, rf_failure, draws=draws, seed=seed + 100 + offset
            ),
        }
    return output


def analyze_dataset(
    archive: np.lib.npyio.NpzFile,
    dataset: str,
    *,
    seed: int,
    draws: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    target = archive[f"{dataset}__target"]
    depth = archive[f"{dataset}__depth"]
    base_id = archive[f"{dataset}__base_id"]
    noisy = archive[f"{dataset}__noisy"]
    if len(np.unique(base_id)) != len(base_id):
        raise ValueError(f"{dataset}: repeated base identifiers require a group-aware row splitter")
    base = np.stack([archive[f"{dataset}__{key}"] for key in BASE_KEYS], axis=-1)
    direct = {
        "noisy": noisy,
        "zne": archive[f"{dataset}__zne"],
        **{label: base[..., index] for index, label in enumerate(BASE_LABELS)},
    }
    labels = tuple(direct) + ENSEMBLE_LABELS
    predictions = {label: np.full_like(target, np.nan) for label in labels}
    radii = {label: np.full(len(target), np.nan) for label in labels}
    clipped_radii = {label: np.full(len(target), np.nan) for label in labels}
    folds = np.full(len(target), -1, dtype=int)
    fit_role = np.full((len(target), 5), "test", dtype="<U11")
    fit_records: list[dict[str, object]] = []
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(outer.split(np.zeros(len(depth)), depth)):
        fit, calibration = fit_calibration_split(
            train, depth[train], seed=seed + 1000 + fold
        )
        simplex = fit_simplex(base[fit], target[fit])
        ridge = fit_ridge_nested(
            base[fit], target[fit], depth[fit], seed=seed + 2000 + fold
        )
        all_candidates = {**direct, **_candidate_predictions(base, simplex, ridge)}
        for label, values in all_candidates.items():
            predictions[label][test] = values[test]
            radii[label][test] = conformal_radius(values[calibration], target[calibration])
            clipped_radii[label][test] = conformal_radius(
                np.clip(values[calibration], -1.0, 1.0), target[calibration]
            )
        folds[test] = fold
        fit_role[fit, fold] = "fit"
        fit_role[calibration, fold] = "calibration"
        fit_records.append(
            {
                "fold": fold,
                "fit_circuits": int(len(fit)),
                "calibration_circuits": int(len(calibration)),
                "test_circuits": int(len(test)),
                "simplex_weights": simplex.weights.tolist(),
                "ridge_weights": ridge.weights.tolist(),
                "ridge_intercept": ridge.intercept,
                "ridge_alpha": ridge.alpha,
            }
        )
    if any(np.any(~np.isfinite(values)) for values in predictions.values()):
        raise RuntimeError(f"{dataset}: incomplete outer predictions")
    variants: dict[str, dict[str, object]] = {}
    saved_arrays: dict[str, np.ndarray] = {
        "fold": folds,
        "fit_role": fit_role,
        "base_id": base_id,
    }
    for variant in ("raw", "clipped"):
        variant_predictions = {
            label: values if variant == "raw" else np.clip(values, -1.0, 1.0)
            for label, values in predictions.items()
        }
        selected_radii = radii if variant == "raw" else clipped_radii
        for label, values in variant_predictions.items():
            saved_arrays[f"{variant}__{label}__prediction"] = values
            saved_arrays[f"{variant}__{label}__radius"] = selected_radii[label]
        variants[variant] = {
            "metrics": {
                label: evaluate_predictions(
                    values, target, noisy, depth, selected_radii[label]
                )
                for label, values in variant_predictions.items()
            },
            "effects_vs_rf": _metric_effects(
                variant_predictions,
                target,
                noisy,
                draws=draws,
                seed=seed + (0 if variant == "raw" else 10_000),
            ),
        }
    result = {
        "circuits": int(len(target)),
        "observables_per_circuit": int(target.shape[1]),
        "depths": np.unique(depth).astype(int).tolist(),
        "outer_folds": 5,
        "fold_fits": fit_records,
        "variants": variants,
    }
    return result, saved_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=PROJECT / "data/derived/published_predictions.npz"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    start = time.perf_counter()
    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    result: dict[str, object] = {
        "schema_version": 1,
        "analysis": "cross_fitted_combiners_on_distributed_held_out_predictions",
        "evidence_class": "method_extension_on_legacy_predictions",
        "input_sha256": sha256(args.input),
        "seed": args.seed,
        "bootstrap_draws": args.bootstrap_draws,
        "ensemble_base_order": list(BASE_LABELS),
        "uncertainty": "90% circuit-max split conformal; fit/calibration nested inside outer train",
        "failure_definition": {
            "reference": "unmitigated expectation",
            "absolute_error_margin": FAILURE_MARGIN,
            "rule": "candidate absolute error > reference absolute error + margin",
        },
        "hardware_jobs_submitted": 0,
        "datasets": {},
    }
    saved: dict[str, np.ndarray] = {}
    with np.load(args.input, allow_pickle=False) as archive:
        for offset, dataset in enumerate(DATASETS):
            analysis, arrays = analyze_dataset(
                archive,
                dataset,
                seed=args.seed + 10_000 * offset,
                draws=args.bootstrap_draws,
            )
            result["datasets"][dataset] = analysis
            saved.update({f"{dataset}__{key}": value for key, value in arrays.items()})
    end_usage = resource.getrusage(resource.RUSAGE_SELF)
    result["resources"] = {
        "wall_seconds": time.perf_counter() - start,
        "user_cpu_seconds": end_usage.ru_utime - start_usage.ru_utime,
        "system_cpu_seconds": end_usage.ru_stime - start_usage.ru_stime,
        "peak_rss_kib": int(end_usage.ru_maxrss),
        "logical_cpus_visible": os.cpu_count(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "gpus_used": 0,
        "qpu_jobs": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.predictions, **saved)
    print(json.dumps(result["resources"], indent=2, sort_keys=True))
    for dataset, analysis in result["datasets"].items():
        raw = analysis["variants"]["raw"]
        print(f"\n{dataset}")
        for method in ("rf",) + ENSEMBLE_LABELS:
            metric = raw["metrics"][method]
            print(
                f"  {method:8s} MAE={metric['mae']:.6f} "
                f"P95={metric['p95_absolute_error']:.6f} "
                f"failure={metric['failure_rate']:.4f} "
                f"joint90={metric['interval_joint_coverage']:.4f}"
            )


if __name__ == "__main__":
    main()
