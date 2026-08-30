#!/usr/bin/env python3
"""Test transfer from single-qubit to unseen two-qubit Z observables."""

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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml4qem_reuse.ensemble import (
    FAILURE_MARGIN,
    conformal_radius,
    failure_mask,
    fit_simplex,
    paired_bootstrap,
    paired_signflip_test,
)
from ml4qem_reuse.simulation import OBSERVABLE_LABELS, OBSERVABLE_MASKS
from ml4qem_reuse.workflows._paths import workspace_root
from ml4qem_reuse.workflows.train_local_within_domain import LocalData


PROJECT = workspace_root()
BASE_METHODS = ("linear", "rf", "hgb", "mlp")
METHODS = ("noisy_sampled",) + BASE_METHODS + (
    "simplex",
    "safe_simplex",
    "safe_cell_simplex",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_long_model(name: str, seed: int, threads: int) -> object:
    if name == "linear":
        return make_pipeline(StandardScaler(), LinearRegression(n_jobs=threads))
    if name == "rf":
        return RandomForestRegressor(n_estimators=100, n_jobs=threads, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1e-4,
            random_state=seed,
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                batch_size=512,
                learning_rate_init=1e-3,
                max_iter=100,
                tol=1e-5,
                random_state=seed,
            ),
        )
    raise ValueError(name)


def expand_long(
    data: LocalData, rows: np.ndarray, observable_indices: tuple[int, ...]
) -> dict[str, np.ndarray]:
    expanded_rows = np.repeat(rows, len(observable_indices))
    observables = np.tile(np.asarray(observable_indices, dtype=np.int8), len(rows))
    base = data.row_base[expanded_rows]
    config = data.row_config[expanded_rows]
    descriptor = np.zeros((len(expanded_rows), 5), dtype=np.float32)
    for index, observable in enumerate(observables):
        mask = int(OBSERVABLE_MASKS[observable])
        descriptor[index, :4] = [(mask >> qubit) & 1 for qubit in range(4)]
        descriptor[index, 4] = mask.bit_count() / 2.0
    metadata = np.column_stack(
        [
            np.log2(data.row_shots[expanded_rows]) / np.log2(10_000),
            data.config_strength[config] / 2.0,
        ]
    ).astype(np.float32)
    noisy = data.row_sampled[expanded_rows, observables][:, None]
    x = np.concatenate([data.static[base], noisy, descriptor, metadata], axis=1)
    return {
        "x": x,
        "y": data.target[base, observables],
        "noisy": noisy[:, 0],
        "rows": expanded_rows,
        "observable": observables,
        "base": base,
        "config": config,
    }


def strength_shot_keys(long: dict[str, np.ndarray], data: LocalData) -> np.ndarray:
    return np.asarray(
        [
            f"S{strength}:N{shots}"
            for strength, shots in zip(
                data.config_strength[long["config"]], data.row_shots[long["rows"]]
            )
        ]
    )


def fit_cell_simplex_long(
    predictions: np.ndarray, target: np.ndarray, keys: np.ndarray
) -> dict[str, object]:
    return {
        key: fit_simplex(predictions[keys == key, None, :], target[keys == key, None])
        for key in np.unique(keys)
    }


def predict_cell_simplex_long(
    predictions: np.ndarray, keys: np.ndarray, models: dict[str, object]
) -> np.ndarray:
    output = np.empty(len(predictions), dtype=float)
    for key in np.unique(keys):
        output[keys == key] = models[key].predict(predictions[keys == key, None, :])[:, 0]
    return output


def group_radius(prediction: np.ndarray, target: np.ndarray, base: np.ndarray) -> float:
    scores = np.asarray(
        [np.max(np.abs(prediction[base == group] - target[base == group])) for group in np.unique(base)]
    )
    return conformal_radius(scores[:, None], np.zeros((len(scores), 1)))


def worst_group(error: np.ndarray, labels: np.ndarray) -> float:
    return float(max(np.mean(error[labels == label]) for label in np.unique(labels)))


def metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    noisy: np.ndarray,
    radius: np.ndarray,
    long: dict[str, np.ndarray],
    data: LocalData,
) -> dict[str, float]:
    absolute = np.abs(prediction - target)
    _, base_inverse = np.unique(long["base"], return_inverse=True)
    joint_covered = np.ones(int(np.max(base_inverse)) + 1, dtype=bool)
    np.logical_and.at(joint_covered, base_inverse, absolute <= radius)
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(absolute)))),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "failure_rate": float(np.mean(failure_mask(absolute, np.abs(noisy - target)))),
        "worst_observable_mae": worst_group(absolute, long["observable"]),
        "worst_family_mae": worst_group(absolute, data.family[long["base"]]),
        "worst_noise_family_mae": worst_group(
            absolute, data.config_noise[long["config"]]
        ),
        "worst_strength_mae": worst_group(
            absolute, data.config_strength[long["config"]]
        ),
        "worst_shot_budget_mae": worst_group(absolute, data.row_shots[long["rows"]]),
        "physical_range_violation_rate": float(np.mean((prediction < -1.0) | (prediction > 1.0))),
        "interval_marginal_coverage": float(np.mean(absolute <= radius)),
        "interval_base_circuit_joint_coverage": float(np.mean(joint_covered)),
        "interval_mean_width": float(2.0 * np.mean(radius)),
    }


def aggregate_base(values: np.ndarray, base: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(base, return_inverse=True)
    return np.bincount(inverse, weights=values) / np.bincount(inverse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=PROJECT / "data/derived/local_benchmark_v1.npz"
    )
    parser.add_argument(
        "--splits", type=Path, default=PROJECT / "protocol/local_split_manifest_v1.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    if not set(args.folds) <= set(range(5)):
        raise ValueError("folds must be drawn from 0..4")
    split_manifest = json.loads(args.splits.read_text())
    start = time.perf_counter()
    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    collected = {method: [] for method in METHODS}
    collected_radii = {method: [] for method in METHODS}
    collected_long = []
    fold_records = []
    with np.load(args.input, allow_pickle=False) as archive:
        data = LocalData(archive)
        for fold in args.folds:
            split = split_manifest["within_domain_folds"][fold]
            cell = split["training_cells"]["512"]
            development = expand_long(
                data, data.rows_for_ids(cell["development_ids"]), (0, 1)
            )
            stacking = expand_long(data, data.rows_for_ids(cell["stacking_ids"]), (0, 1))
            calibration = expand_long(
                data, data.rows_for_ids(cell["calibration_ids"]), (0, 1)
            )
            test = expand_long(data, data.rows_for_ids(split["test_ids"]), (2, 3))
            models = {}
            fit_seconds = {}
            for offset, name in enumerate(BASE_METHODS):
                model = make_long_model(
                    name, args.model_seed + 1000 * fold + offset, args.threads
                )
                started = time.perf_counter()
                model.fit(development["x"], development["y"])
                fit_seconds[name] = time.perf_counter() - started
                models[name] = model
            stack_base = np.column_stack(
                [models[name].predict(stacking["x"]) for name in BASE_METHODS]
            )
            simplex = fit_simplex(stack_base[:, None, :], stacking["y"][:, None])
            safe_stack = np.column_stack([stacking["noisy"], stack_base])
            safe_simplex = fit_simplex(safe_stack[:, None, :], stacking["y"][:, None])
            stack_keys = strength_shot_keys(stacking, data)
            cell_models = fit_cell_simplex_long(safe_stack, stacking["y"], stack_keys)
            calibration_base = np.column_stack(
                [models[name].predict(calibration["x"]) for name in BASE_METHODS]
            )
            test_base = np.column_stack(
                [models[name].predict(test["x"]) for name in BASE_METHODS]
            )
            calibration_safe = np.column_stack([calibration["noisy"], calibration_base])
            test_safe = np.column_stack([test["noisy"], test_base])
            calibration_predictions = {
                "noisy_sampled": calibration["noisy"],
                **{
                    name: calibration_base[:, index]
                    for index, name in enumerate(BASE_METHODS)
                },
                "simplex": simplex.predict(calibration_base[:, None, :])[:, 0],
                "safe_simplex": safe_simplex.predict(calibration_safe[:, None, :])[:, 0],
                "safe_cell_simplex": predict_cell_simplex_long(
                    calibration_safe,
                    strength_shot_keys(calibration, data),
                    cell_models,
                ),
            }
            test_predictions = {
                "noisy_sampled": test["noisy"],
                **{name: test_base[:, index] for index, name in enumerate(BASE_METHODS)},
                "simplex": simplex.predict(test_base[:, None, :])[:, 0],
                "safe_simplex": safe_simplex.predict(test_safe[:, None, :])[:, 0],
                "safe_cell_simplex": predict_cell_simplex_long(
                    test_safe, strength_shot_keys(test, data), cell_models
                ),
            }
            for method in METHODS:
                radius = group_radius(
                    calibration_predictions[method], calibration["y"], calibration["base"]
                )
                collected[method].append(test_predictions[method])
                collected_radii[method].append(np.full(len(test["y"]), radius))
            collected_long.append(test)
            fold_records.append(
                {
                    "fold": fold,
                    "development_base_circuits": len(cell["development_ids"]),
                    "stacking_base_circuits": len(cell["stacking_ids"]),
                    "calibration_base_circuits": len(cell["calibration_ids"]),
                    "test_base_circuits": len(split["test_ids"]),
                    "fit_seconds": fit_seconds,
                    "simplex_weights": simplex.weights.tolist(),
                    "safe_simplex_weights": safe_simplex.weights.tolist(),
                    "safe_cell_weights": {
                        key: model.weights.tolist() for key, model in sorted(cell_models.items())
                    },
                }
            )
        predictions = {method: np.concatenate(value) for method, value in collected.items()}
        radii = {method: np.concatenate(value) for method, value in collected_radii.items()}
        long = {
            key: np.concatenate([part[key] for part in collected_long])
            for key in collected_long[0]
            if key != "x"
        }
        raw_metrics = {
            method: metrics(
                prediction,
                long["y"],
                long["noisy"],
                radii[method],
                long,
                data,
            )
            for method, prediction in predictions.items()
        }
        effects = {}
        for reference_index, reference in enumerate(("noisy_sampled", "rf")):
            reference_error = aggregate_base(
                np.abs(predictions[reference] - long["y"]), long["base"]
            )
            effects[reference] = {}
            for method_index, method in enumerate(METHODS):
                if method == reference:
                    continue
                error = aggregate_base(np.abs(predictions[method] - long["y"]), long["base"])
                inference = paired_bootstrap(
                    error,
                    reference_error,
                    draws=args.bootstrap_draws,
                    seed=88741 + 100 * reference_index + method_index,
                )
                inference["signflip_p_value_two_sided"] = paired_signflip_test(
                    error,
                    reference_error,
                    draws=args.bootstrap_draws,
                    seed=188741 + 100 * reference_index + method_index,
                )
                effects[reference][method] = inference
        saved = {
            "row_index": long["rows"],
            "base_index": long["base"],
            "observable_index": long["observable"],
            "target": long["y"],
            **{f"{method}__prediction": value for method, value in predictions.items()},
            **{f"{method}__radius": value for method, value in radii.items()},
        }
    end_usage = resource.getrusage(resource.RUSAGE_SELF)
    output = {
        "schema_version": 1,
        "analysis": "single_to_two_qubit_observable_shift",
        "evidence_class": "local_simulation_and_method_extension",
        "input_sha256": sha256(args.input),
        "split_manifest_sha256": sha256(args.splits),
        "folds": args.folds,
        "training_observables": list(OBSERVABLE_LABELS[:2]),
        "test_observables": list(OBSERVABLE_LABELS[2:]),
        "feature_descriptor": "four support bits, Pauli weight, sampled target-observable value, shots and strength",
        "model_seed": args.model_seed,
        "fold_records": fold_records,
        "raw_metrics": raw_metrics,
        "paired_circuit_mae_effects": effects,
        "bootstrap_draws": args.bootstrap_draws,
        "failure_definition": {
            "reference": "unmitigated sampled expectation",
            "absolute_error_margin": FAILURE_MARGIN,
            "rule": "candidate absolute error > reference absolute error + margin",
        },
        "hardware_jobs_submitted": 0,
        "resources": {
            "process_accounting_scope": "RUSAGE_SELF; external GNU time log includes child processes for clean resource run",
            "wall_seconds": time.perf_counter() - start,
            "user_cpu_seconds": end_usage.ru_utime - start_usage.ru_utime,
            "system_cpu_seconds": end_usage.ru_stime - start_usage.ru_stime,
            "peak_rss_kib": int(end_usage.ru_maxrss),
            "peak_rss_scope": "main_process_only",
            "threads_requested": args.threads,
            "logical_cpus_visible": os.cpu_count(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "gpus_used": 0,
            "qpu_jobs": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.predictions, **saved)
    print(json.dumps(output["resources"], indent=2, sort_keys=True))
    for method in ("noisy_sampled", "rf", "simplex", "safe_simplex", "safe_cell_simplex"):
        metric = raw_metrics[method]
        print(
            f"{method:18s} MAE={metric['mae']:.6f} "
            f"P95={metric['p95_absolute_error']:.6f} failure={metric['failure_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
