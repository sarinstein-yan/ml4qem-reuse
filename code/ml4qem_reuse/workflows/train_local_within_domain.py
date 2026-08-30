#!/usr/bin/env python3
"""Train leakage-safe baselines and ensembles on the local benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml4qem_reuse.ensemble import (
    FAILURE_MARGIN,
    conformal_radius,
    failure_mask,
    fit_ridge_grouped,
    fit_simplex,
    paired_bootstrap,
    paired_signflip_test,
)
from ml4qem_reuse.serialization import finite_interval_metric_fields, strict_json_dumps
from ml4qem_reuse.workflows._paths import workspace_root


PROJECT = workspace_root()
BASE_METHODS = ("linear", "rf", "hgb", "mlp")
ENSEMBLE_METHODS = (
    "mean",
    "median",
    "simplex",
    "ridge",
    "safe_simplex",
    "safe_ridge",
    "raw_cell_affine",
    "safe_cell_simplex",
)
ALL_METHODS = ("noisy_sampled", "noisy_exact") + BASE_METHODS + ENSEMBLE_METHODS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model(name: str, *, seed: int, threads: int) -> object:
    """Return a fixed, untuned base estimator."""

    if name == "linear":
        return make_pipeline(StandardScaler(), LinearRegression(n_jobs=threads))
    if name == "rf":
        # The released ML4QEM demos fit one scalar-target forest per
        # observable.  MultiOutputRegressor preserves that estimator topology:
        # four independently fitted forests rather than one forest whose tree
        # splits are shared across all targets.
        return MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=1,
                max_features=1.0,
                n_jobs=max(1, threads // 4),
                random_state=seed,
            ),
            n_jobs=min(threads, 4),
        )
    if name == "hgb":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1e-4,
                random_state=seed,
            ),
            n_jobs=min(threads, 4),
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                batch_size=512,
                learning_rate_init=1e-3,
                max_iter=100,
                tol=1e-5,
                early_stopping=False,
                random_state=seed,
            ),
        )
    raise ValueError(f"unknown model: {name}")


class LocalData:
    """Resolve normalized archive arrays into row-level views."""

    def __init__(self, archive: np.lib.npyio.NpzFile):
        self.base_id = archive["base_id"]
        self.family = archive["family"]
        self.layers = archive["layers"]
        self.depth = archive["depth"]
        self.static = archive["static_features"]
        self.target = archive["target"]
        self.config_base = archive["config_base_index"]
        self.config_noise = archive["config_noise_family"]
        self.config_strength = archive["config_strength_index"]
        self.config_exact = archive["config_exact_noisy"]
        self.row_config = archive["row_config_index"]
        self.row_sampled = archive["row_sampled_noisy"]
        self.row_shots = archive["row_shot_budget"]
        self.row_replicate = archive["row_shot_replicate"]
        self.row_base = self.config_base[self.row_config]

    def rows_for_ids(self, identifiers: list[str]) -> np.ndarray:
        base_mask = np.isin(self.base_id, identifiers)
        return np.flatnonzero(base_mask[self.row_base])

    def x(self, rows: np.ndarray) -> np.ndarray:
        return np.concatenate([self.static[self.row_base[rows]], self.row_sampled[rows]], axis=1)

    def y(self, rows: np.ndarray) -> np.ndarray:
        return self.target[self.row_base[rows]]

    def exact_noisy(self, rows: np.ndarray) -> np.ndarray:
        return self.config_exact[self.row_config[rows]]


def _ensemble_predictions(base: np.ndarray, simplex: object, ridge: object) -> dict[str, np.ndarray]:
    return {
        "mean": np.mean(base, axis=-1),
        "median": np.median(base, axis=-1),
        "simplex": simplex.predict(base),
        "ridge": ridge.predict(base),
    }


def _strength_shot_keys(data: LocalData, rows: np.ndarray) -> np.ndarray:
    config = data.row_config[rows]
    return np.asarray(
        [
            f"S{strength}:N{shots}"
            for strength, shots in zip(data.config_strength[config], data.row_shots[rows])
        ]
    )


def fit_cell_simplexes(
    predictions: np.ndarray, target: np.ndarray, rows: np.ndarray, data: LocalData
) -> dict[str, object]:
    keys = _strength_shot_keys(data, rows)
    return {
        key: fit_simplex(predictions[keys == key], target[keys == key])
        for key in np.unique(keys)
    }


def predict_cell_simplexes(
    predictions: np.ndarray, rows: np.ndarray, data: LocalData, models: dict[str, object]
) -> np.ndarray:
    keys = _strength_shot_keys(data, rows)
    missing = sorted(set(keys) - set(models))
    if missing:
        raise ValueError(f"missing cell simplex: {missing}")
    output = np.empty(predictions.shape[:2], dtype=float)
    for key in np.unique(keys):
        output[keys == key] = models[key].predict(predictions[keys == key])
    return output


def predict_cell_simplexes_assuming_strength(
    predictions: np.ndarray,
    rows: np.ndarray,
    data: LocalData,
    models: dict[str, object],
    assumed_strength: int,
) -> np.ndarray:
    """Apply cell weights using an assumed rather than simulator-known tier."""

    keys = np.asarray([f"S{assumed_strength}:N{shots}" for shots in data.row_shots[rows]])
    missing = sorted(set(keys) - set(models))
    if missing:
        raise ValueError(f"missing assumed-tier cell simplex: {missing}")
    output = np.empty(predictions.shape[:2], dtype=float)
    for key in np.unique(keys):
        output[keys == key] = models[key].predict(predictions[keys == key])
    return output


def fit_raw_cell_affines(
    sampled: np.ndarray, target: np.ndarray, rows: np.ndarray, data: LocalData
) -> dict[str, LinearRegression]:
    """Fit a raw-only affine calibration in each strength-by-shot cell."""

    keys = _strength_shot_keys(data, rows)
    models = {}
    for key in np.unique(keys):
        model = LinearRegression()
        model.fit(sampled[keys == key].reshape(-1, 1), target[keys == key].reshape(-1))
        models[key] = model
    return models


def predict_raw_cell_affines(
    sampled: np.ndarray, rows: np.ndarray, data: LocalData, models: dict[str, LinearRegression]
) -> np.ndarray:
    keys = _strength_shot_keys(data, rows)
    missing = sorted(set(keys) - set(models))
    if missing:
        raise ValueError(f"missing raw-cell affine: {missing}")
    output = np.empty_like(sampled, dtype=float)
    for key in np.unique(keys):
        selected = keys == key
        output[selected] = models[key].predict(sampled[selected].reshape(-1, 1)).reshape(
            output[selected].shape
        )
    return output


def _cell_keys(data: LocalData, rows: np.ndarray) -> np.ndarray:
    config = data.row_config[rows]
    return np.asarray(
        [
            f"{noise}:S{strength}:N{shots}"
            for noise, strength, shots in zip(
                data.config_noise[config], data.config_strength[config], data.row_shots[rows]
            )
        ]
    )


def calibrate_by_condition(
    prediction: np.ndarray, target: np.ndarray, rows: np.ndarray, data: LocalData
) -> dict[str, float]:
    """Calibrate simultaneous replicate-and-observable coverage per condition."""

    keys = _cell_keys(data, rows)
    output = {}
    for key in np.unique(keys):
        selected = np.flatnonzero(keys == key)
        groups = data.row_base[rows[selected]]
        scores = []
        for group in np.unique(groups):
            group_rows = selected[groups == group]
            scores.append(float(np.max(np.abs(prediction[group_rows] - target[group_rows]))))
        scores_array = np.asarray(scores)
        # Reuse the finite-sample rank implementation with one dummy observable.
        output[key] = conformal_radius(scores_array[:, None], np.zeros((len(scores), 1)))
    return output


def radii_for_rows(calibration: dict[str, float], rows: np.ndarray, data: LocalData) -> np.ndarray:
    keys = _cell_keys(data, rows)
    missing = sorted(set(keys) - set(calibration))
    if missing:
        raise ValueError(f"missing conformal cells: {missing[:3]}")
    return np.asarray([calibration[key] for key in keys], dtype=float)


def _worst_group(error: np.ndarray, labels: np.ndarray) -> float:
    return float(max(np.mean(error[labels == label]) for label in np.unique(labels)))


def local_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    sampled_noisy: np.ndarray,
    radii: np.ndarray,
    rows: np.ndarray,
    data: LocalData,
) -> dict[str, float | bool | None]:
    absolute = np.abs(prediction - target)
    noisy_absolute = np.abs(sampled_noisy - target)
    base = data.row_base[rows]
    config = data.row_config[rows]
    family = data.family[base]
    layers = data.layers[base]
    noise = data.config_noise[config]
    strength = data.config_strength[config]
    shots = data.row_shots[rows]
    radius_array = radii[:, None]
    condition_groups = np.asarray(
        [f"{b}:{n}:{s}:{q}" for b, n, s, q in zip(base, noise, strength, shots)]
    )
    _, condition_inverse = np.unique(condition_groups, return_inverse=True)
    n_condition_groups = int(np.max(condition_inverse)) + 1
    joint = np.ones(n_condition_groups, dtype=bool)
    np.logical_and.at(joint, condition_inverse, np.all(absolute <= radius_array, axis=1))
    failures = failure_mask(absolute, noisy_absolute)
    condition_failure = np.zeros(n_condition_groups, dtype=bool)
    np.logical_or.at(
        condition_failure,
        condition_inverse,
        np.any(failures, axis=1),
    )
    interval_fields = finite_interval_metric_fields(
        float(2.0 * np.mean(radii)),
        {
            "interval_marginal_coverage": float(np.mean(absolute <= radius_array)),
            "interval_row_joint_coverage": float(
                np.mean(np.all(absolute <= radius_array, axis=1))
            ),
            "interval_condition_circuit_joint_coverage": float(np.mean(joint)),
        },
    )
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(absolute)))),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "failure_rate": float(np.mean(failures)),
        "circuit_condition_failure_rate": float(np.mean(condition_failure)),
        "worst_family_mae": _worst_group(absolute, family),
        "worst_noise_family_mae": _worst_group(absolute, noise),
        "worst_strength_mae": _worst_group(absolute, strength),
        "worst_shot_budget_mae": _worst_group(absolute, shots),
        "worst_layer_mae": _worst_group(absolute, layers),
        "physical_range_violation_rate": float(np.mean((prediction < -1.0) | (prediction > 1.0))),
        **interval_fields,
    }


def _aggregate_by_base(values: np.ndarray, base: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(base, return_inverse=True)
    total = np.bincount(inverse, weights=values)
    count = np.bincount(inverse)
    return total / count


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
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--reuse-models", action="store_true")
    parser.add_argument("--training-size", type=int, default=512)
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    split_manifest = json.loads(args.splits.read_text())
    if args.training_size not in split_manifest["training_sizes"]:
        raise ValueError("training size is not in the frozen manifest")
    if not set(args.folds) <= set(range(5)):
        raise ValueError("folds must be drawn from 0..4")
    start = time.perf_counter()
    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    fold_records = []
    collected_rows = []
    collected_folds = []
    collected_predictions = {method: [] for method in ALL_METHODS}
    collected_radii = {method: [] for method in ALL_METHODS}
    collected_clipped_radii = {method: [] for method in ALL_METHODS}
    collected_assumed_tier_predictions = {tier: [] for tier in range(3)}
    with np.load(args.input, allow_pickle=False) as archive:
        data = LocalData(archive)
        for fold in args.folds:
            cell = split_manifest["within_domain_folds"][fold]["training_cells"][
                str(args.training_size)
            ]
            development_rows = data.rows_for_ids(cell["development_ids"])
            stacking_rows = data.rows_for_ids(cell["stacking_ids"])
            calibration_rows = data.rows_for_ids(cell["calibration_ids"])
            test_rows = data.rows_for_ids(split_manifest["within_domain_folds"][fold]["test_ids"])
            development_x = data.x(development_rows)
            development_y = data.y(development_rows)
            stacking_x = data.x(stacking_rows)
            stacking_y = data.y(stacking_rows)
            calibration_x = data.x(calibration_rows)
            calibration_y = data.y(calibration_rows)
            test_x = data.x(test_rows)
            models = {}
            fit_times = {}
            convergence_warnings = {}
            reused_models = {}
            for offset, name in enumerate(BASE_METHODS):
                model_path = (
                    args.model_dir / f"fold{fold}_{name}_seed{args.model_seed}.joblib"
                    if args.model_dir
                    else None
                )
                if args.reuse_models and model_path and model_path.exists():
                    model = joblib.load(model_path)
                    fit_times[name] = 0.0
                    convergence_warnings[name] = 0
                    reused_models[name] = True
                else:
                    model = make_model(
                        name, seed=args.model_seed + 1000 * fold + offset, threads=args.threads
                    )
                    fit_start = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(development_x, development_y)
                    fit_times[name] = time.perf_counter() - fit_start
                    convergence_warnings[name] = sum(
                        issubclass(item.category, ConvergenceWarning) for item in caught
                    )
                    reused_models[name] = False
                models[name] = model
                if model_path and not reused_models[name]:
                    args.model_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, model_path, compress=3)
            stack_base = np.stack([models[name].predict(stacking_x) for name in BASE_METHODS], axis=-1)
            simplex = fit_simplex(stack_base, stacking_y)
            stacking_groups = data.row_base[stacking_rows]
            ridge = fit_ridge_grouped(
                stack_base,
                stacking_y,
                stacking_groups,
                data.family[stacking_groups],
                seed=split_manifest["seed"] + fold,
            )
            safe_stack_base = np.concatenate(
                [data.row_sampled[stacking_rows, :, None], stack_base], axis=-1
            )
            safe_simplex = fit_simplex(safe_stack_base, stacking_y)
            safe_ridge = fit_ridge_grouped(
                safe_stack_base,
                stacking_y,
                stacking_groups,
                data.family[stacking_groups],
                seed=split_manifest["seed"] + 100 + fold,
            )
            safe_cell_simplexes = fit_cell_simplexes(
                safe_stack_base, stacking_y, stacking_rows, data
            )
            raw_cell_affines = fit_raw_cell_affines(
                data.row_sampled[stacking_rows], stacking_y, stacking_rows, data
            )
            calibration_base = np.stack(
                [models[name].predict(calibration_x) for name in BASE_METHODS], axis=-1
            )
            test_base = np.stack([models[name].predict(test_x) for name in BASE_METHODS], axis=-1)
            calibration_safe_base = np.concatenate(
                [data.row_sampled[calibration_rows, :, None], calibration_base], axis=-1
            )
            test_safe_base = np.concatenate(
                [data.row_sampled[test_rows, :, None], test_base], axis=-1
            )
            calibration_predictions = {
                "noisy_sampled": data.row_sampled[calibration_rows],
                "noisy_exact": data.exact_noisy(calibration_rows),
                **{
                    name: calibration_base[..., index]
                    for index, name in enumerate(BASE_METHODS)
                },
                **_ensemble_predictions(calibration_base, simplex, ridge),
                "safe_simplex": safe_simplex.predict(calibration_safe_base),
                "safe_ridge": safe_ridge.predict(calibration_safe_base),
                "raw_cell_affine": predict_raw_cell_affines(
                    data.row_sampled[calibration_rows],
                    calibration_rows,
                    data,
                    raw_cell_affines,
                ),
                "safe_cell_simplex": predict_cell_simplexes(
                    calibration_safe_base, calibration_rows, data, safe_cell_simplexes
                ),
            }
            test_predictions = {
                "noisy_sampled": data.row_sampled[test_rows],
                "noisy_exact": data.exact_noisy(test_rows),
                **{name: test_base[..., index] for index, name in enumerate(BASE_METHODS)},
                **_ensemble_predictions(test_base, simplex, ridge),
                "safe_simplex": safe_simplex.predict(test_safe_base),
                "safe_ridge": safe_ridge.predict(test_safe_base),
                "raw_cell_affine": predict_raw_cell_affines(
                    data.row_sampled[test_rows], test_rows, data, raw_cell_affines
                ),
                "safe_cell_simplex": predict_cell_simplexes(
                    test_safe_base, test_rows, data, safe_cell_simplexes
                ),
            }
            for assumed_tier in range(3):
                collected_assumed_tier_predictions[assumed_tier].append(
                    predict_cell_simplexes_assuming_strength(
                        test_safe_base,
                        test_rows,
                        data,
                        safe_cell_simplexes,
                        assumed_tier,
                    )
                )
            for method in ALL_METHODS:
                calibration = calibrate_by_condition(
                    calibration_predictions[method], calibration_y, calibration_rows, data
                )
                clipped_calibration = calibrate_by_condition(
                    np.clip(calibration_predictions[method], -1.0, 1.0),
                    calibration_y,
                    calibration_rows,
                    data,
                )
                collected_predictions[method].append(test_predictions[method])
                collected_radii[method].append(radii_for_rows(calibration, test_rows, data))
                collected_clipped_radii[method].append(
                    radii_for_rows(clipped_calibration, test_rows, data)
                )
            collected_rows.append(test_rows)
            collected_folds.append(np.full(len(test_rows), fold, dtype=np.int8))
            fold_records.append(
                {
                    "fold": fold,
                    "development_base_circuits": len(cell["development_ids"]),
                    "stacking_base_circuits": len(cell["stacking_ids"]),
                    "calibration_base_circuits": len(cell["calibration_ids"]),
                    "test_base_circuits": len(
                        split_manifest["within_domain_folds"][fold]["test_ids"]
                    ),
                    "development_rows": int(len(development_rows)),
                    "fit_seconds": fit_times,
                    "convergence_warnings": convergence_warnings,
                    "models_reused": reused_models,
                    "simplex_weights": simplex.weights.tolist(),
                    "ridge_weights": ridge.weights.tolist(),
                    "ridge_intercept": ridge.intercept,
                    "ridge_alpha": ridge.alpha,
                    "safe_simplex_weights": safe_simplex.weights.tolist(),
                    "safe_ridge_weights": safe_ridge.weights.tolist(),
                    "safe_ridge_intercept": safe_ridge.intercept,
                    "safe_ridge_alpha": safe_ridge.alpha,
                    "raw_cell_affine_parameters": {
                        key: {
                            "slope": float(model.coef_[0]),
                            "intercept": float(model.intercept_),
                        }
                        for key, model in sorted(raw_cell_affines.items())
                    },
                    "safe_cell_simplex_weights": {
                        key: model.weights.tolist()
                        for key, model in sorted(safe_cell_simplexes.items())
                    },
                }
            )
        rows = np.concatenate(collected_rows)
        folds = np.concatenate(collected_folds)
        predictions = {
            method: np.concatenate(values) for method, values in collected_predictions.items()
        }
        radii = {method: np.concatenate(values) for method, values in collected_radii.items()}
        clipped_radii = {
            method: np.concatenate(values) for method, values in collected_clipped_radii.items()
        }
        target = data.y(rows)
        sampled_noisy = data.row_sampled[rows]
        assumed_tier_predictions = {
            tier: np.concatenate(values)
            for tier, values in collected_assumed_tier_predictions.items()
        }
        raw_metrics = {
            method: local_metrics(
                prediction, target, sampled_noisy, radii[method], rows, data
            )
            for method, prediction in predictions.items()
        }
        clipped = {method: np.clip(value, -1.0, 1.0) for method, value in predictions.items()}
        clipped_metrics = {
            method: local_metrics(
                prediction, target, sampled_noisy, clipped_radii[method], rows, data
            )
            for method, prediction in clipped.items()
        }
        true_tier = data.config_strength[data.row_config[rows]]
        tier_metadata_sensitivity = {
            "description": (
                "MAE when the metadata-conditioned cell combiner is supplied each assumed "
                "strength tier; rows are stratified by simulator-known true tier"
            ),
            "unknown_tier_fallback_method": "safe_simplex",
            "unknown_tier_fallback_mae": raw_metrics["safe_simplex"]["mae"],
            "raw_only_matched_control_method": "raw_cell_affine",
            "raw_only_matched_control_mae": raw_metrics["raw_cell_affine"]["mae"],
            "mae_by_true_and_assumed_tier": {
                str(true): {
                    str(assumed): float(
                        np.mean(
                            np.abs(assumed_tier_predictions[assumed][true_tier == true]
                                   - target[true_tier == true])
                        )
                    )
                    for assumed in range(3)
                }
                for true in range(3)
            },
        }
        fold_effects = []
        for fold in args.folds:
            selected = folds == fold
            fold_effects.append(
                {
                    "fold": int(fold),
                    "base_circuits": int(len(np.unique(data.row_base[rows[selected]]))),
                    "mae": {
                        method: float(np.mean(np.abs(predictions[method][selected] - target[selected])))
                        for method in (
                            "noisy_sampled",
                            "rf",
                            "safe_simplex",
                            "raw_cell_affine",
                            "safe_cell_simplex",
                        )
                    },
                    "safe_simplex_effect_vs_noisy": float(
                        np.mean(np.abs(predictions["safe_simplex"][selected] - target[selected]))
                        - np.mean(np.abs(sampled_noisy[selected] - target[selected]))
                    ),
                    "safe_cell_effect_vs_noisy": float(
                        np.mean(np.abs(predictions["safe_cell_simplex"][selected] - target[selected]))
                        - np.mean(np.abs(sampled_noisy[selected] - target[selected]))
                    ),
                }
            )
        base = data.row_base[rows]
        effects = {}
        for reference_offset, reference in enumerate(("rf", "noisy_sampled")):
            reference_error = _aggregate_by_base(
                np.mean(np.abs(predictions[reference] - target), axis=1), base
            )
            effects[reference] = {}
            for offset, method in enumerate(ENSEMBLE_METHODS):
                error = _aggregate_by_base(
                    np.mean(np.abs(predictions[method] - target), axis=1), base
                )
                inference = paired_bootstrap(
                    error,
                    reference_error,
                    draws=args.bootstrap_draws,
                    seed=(
                        split_manifest["seed"]
                        + args.model_seed * 100
                        + reference_offset * 1000
                        + offset
                    ),
                )
                inference["signflip_p_value_two_sided"] = paired_signflip_test(
                    error,
                    reference_error,
                    draws=args.bootstrap_draws,
                    seed=(
                        split_manifest["seed"]
                        + 50_000
                        + args.model_seed * 100
                        + reference_offset * 1000
                        + offset
                    ),
                )
                effects[reference][method] = inference
        saved = {
            "row_index": rows,
            "outer_fold": folds,
            "target": target,
            **{f"raw__{method}__prediction": value for method, value in predictions.items()},
            **{f"raw__{method}__radius": value for method, value in radii.items()},
            **{f"clipped__{method}__prediction": value for method, value in clipped.items()},
            **{f"clipped__{method}__radius": value for method, value in clipped_radii.items()},
            **{
                f"diagnostic__safe_cell_assumed_tier{tier}__prediction": value
                for tier, value in assumed_tier_predictions.items()
            },
        }
    end_usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "schema_version": 1,
        "analysis": "local_within_domain_grouped",
        "evidence_class": "local_simulation_and_method_extension",
        "input_sha256": sha256(args.input),
        "split_manifest_sha256": sha256(args.splits),
        "training_size_base_circuits": args.training_size,
        "folds": args.folds,
        "model_seed": args.model_seed,
        "feature_set": "compact_v2_static_plus_four_sampled_noisy_expectations",
        "base_methods": list(BASE_METHODS),
        "base_hyperparameters": {
            "linear": "StandardScaler + unregularized LinearRegression",
            "rf": (
                "one independently fitted 100-tree forest per observable via "
                "MultiOutputRegressor; min_samples_leaf=1, max_features=1.0"
            ),
            "hgb": "four outputs; 120 iterations, learning_rate=0.08, 31 leaves, L2=1e-4",
            "mlp": "StandardScaler; hidden 128,64; batch 512; max_iter=100; no test-directed early stopping",
        },
        "ensemble_base_order": list(BASE_METHODS),
        "safety_ensemble_base_order": ["noisy_sampled", *BASE_METHODS],
        "fold_records": fold_records,
        "fold_effects": fold_effects,
        "tier_metadata_sensitivity": tier_metadata_sensitivity,
        "raw_metrics": raw_metrics,
        "clipped_metrics": clipped_metrics,
        "raw_paired_circuit_mae_effects": effects,
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
    args.output.write_text(strict_json_dumps(result, indent=2, sort_keys=True) + "\n")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.predictions, **saved)
    print(json.dumps(result["resources"], indent=2, sort_keys=True))
    for method in ("noisy_sampled", "rf") + ENSEMBLE_METHODS:
        metric = raw_metrics[method]
        coverage = metric["interval_condition_circuit_joint_coverage"]
        coverage_text = "unavailable" if coverage is None else f"{coverage:.4f}"
        print(
            f"{method:14s} MAE={metric['mae']:.6f} "
            f"P95={metric['p95_absolute_error']:.6f} "
            f"failure={metric['failure_rate']:.4f} "
            f"condition90={coverage_text}"
        )


if __name__ == "__main__":
    main()
