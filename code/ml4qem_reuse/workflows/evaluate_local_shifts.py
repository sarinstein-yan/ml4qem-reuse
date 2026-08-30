#!/usr/bin/env python3
"""Evaluate circuit and measurement distribution shifts on unseen base circuits."""

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

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from ml4qem_reuse.ensemble import FAILURE_MARGIN, conformal_radius, fit_simplex
from ml4qem_reuse.workflows._paths import workspace_root
from ml4qem_reuse.workflows.train_local_within_domain import (
    BASE_METHODS,
    LocalData,
    _cell_keys,
    calibrate_by_condition,
    fit_cell_simplexes,
    local_metrics,
    make_model,
    predict_cell_simplexes,
    radii_for_rows,
)


PROJECT = workspace_root()
METHODS = ("noisy_sampled",) + BASE_METHODS + (
    "simplex",
    "safe_simplex",
    "safe_cell_simplex",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filter_rows(data: LocalData, rows: np.ndarray, scenario: str, held: object, train: bool) -> np.ndarray:
    config = data.row_config[rows]
    if scenario == "noise_family":
        selected = data.config_noise[config] != held if train else data.config_noise[config] == held
    elif scenario == "shot_budget":
        selected = data.row_shots[rows] != held if train else data.row_shots[rows] == held
    elif scenario == "noise_strength":
        selected = (
            data.config_strength[config] != held if train else data.config_strength[config] == held
        )
    else:
        selected = np.ones(len(rows), dtype=bool)
    return rows[selected]


def global_group_radius(
    prediction: np.ndarray, target: np.ndarray, rows: np.ndarray, data: LocalData
) -> float:
    groups = data.row_base[rows]
    scores = np.asarray(
        [
            np.max(np.abs(prediction[groups == group] - target[groups == group]))
            for group in np.unique(groups)
        ]
    )
    return conformal_radius(scores[:, None], np.zeros((len(scores), 1)))


def fit_and_predict(
    data: LocalData,
    development_rows: np.ndarray,
    stacking_rows: np.ndarray,
    calibration_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    seed: int,
    threads: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    development_x = data.x(development_rows)
    development_y = data.y(development_rows)
    stacking_x = data.x(stacking_rows)
    stacking_y = data.y(stacking_rows)
    calibration_x = data.x(calibration_rows)
    test_x = data.x(test_rows)
    models = {}
    fit_seconds = {}
    convergence_warnings = {}
    for offset, name in enumerate(BASE_METHODS):
        model = make_model(name, seed=seed + offset, threads=threads)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(development_x, development_y)
        fit_seconds[name] = time.perf_counter() - started
        convergence_warnings[name] = sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        models[name] = model
    stack_base = np.stack([models[name].predict(stacking_x) for name in BASE_METHODS], axis=-1)
    simplex = fit_simplex(stack_base, stacking_y)
    safe_stack = np.concatenate([data.row_sampled[stacking_rows, :, None], stack_base], axis=-1)
    safe_simplex = fit_simplex(safe_stack, stacking_y)
    stack_keys = set(
        f"S{strength}:N{shots}"
        for strength, shots in zip(
            data.config_strength[data.row_config[stacking_rows]], data.row_shots[stacking_rows]
        )
    )
    test_keys = set(
        f"S{strength}:N{shots}"
        for strength, shots in zip(
            data.config_strength[data.row_config[test_rows]], data.row_shots[test_rows]
        )
    )
    cell_models = (
        fit_cell_simplexes(safe_stack, stacking_y, stacking_rows, data)
        if test_keys <= stack_keys
        else None
    )
    calibration_base = np.stack(
        [models[name].predict(calibration_x) for name in BASE_METHODS], axis=-1
    )
    test_base = np.stack([models[name].predict(test_x) for name in BASE_METHODS], axis=-1)
    calibration_safe = np.concatenate(
        [data.row_sampled[calibration_rows, :, None], calibration_base], axis=-1
    )
    test_safe = np.concatenate([data.row_sampled[test_rows, :, None], test_base], axis=-1)
    calibration_predictions = {
        "noisy_sampled": data.row_sampled[calibration_rows],
        **{name: calibration_base[..., index] for index, name in enumerate(BASE_METHODS)},
        "simplex": simplex.predict(calibration_base),
        "safe_simplex": safe_simplex.predict(calibration_safe),
    }
    test_predictions = {
        "noisy_sampled": data.row_sampled[test_rows],
        **{name: test_base[..., index] for index, name in enumerate(BASE_METHODS)},
        "simplex": simplex.predict(test_base),
        "safe_simplex": safe_simplex.predict(test_safe),
    }
    if cell_models is not None:
        calibration_predictions["safe_cell_simplex"] = predict_cell_simplexes(
            calibration_safe, calibration_rows, data, cell_models
        )
        test_predictions["safe_cell_simplex"] = predict_cell_simplexes(
            test_safe, test_rows, data, cell_models
        )
    record = {
        "development_base_circuits": int(len(np.unique(data.row_base[development_rows]))),
        "stacking_base_circuits": int(len(np.unique(data.row_base[stacking_rows]))),
        "calibration_base_circuits": int(len(np.unique(data.row_base[calibration_rows]))),
        "test_base_circuits": int(len(np.unique(data.row_base[test_rows]))),
        "development_rows": int(len(development_rows)),
        "test_rows": int(len(test_rows)),
        "fit_seconds": fit_seconds,
        "convergence_warnings": convergence_warnings,
        "simplex_weights": simplex.weights.tolist(),
        "safe_simplex_weights": safe_simplex.weights.tolist(),
        "safe_cell_available": cell_models is not None,
        "safe_cell_weights": (
            {key: model.weights.tolist() for key, model in sorted(cell_models.items())}
            if cell_models is not None
            else None
        ),
    }
    return calibration_predictions, test_predictions, record


def scenario_cells(manifest: dict[str, object], scenario: str) -> list[dict[str, object]]:
    if scenario == "circuit_family":
        return [
            {"label": family, **cell}
            for family, cell in sorted(manifest["family_holdouts"].items())
        ]
    if scenario == "deep_circuits":
        return [{"label": "layers_7_8", **manifest["deep_holdout"]}]
    if scenario == "noise_family":
        held_values = ("coherent_overrotation", "damping_dephasing", "depolarizing_readout")
    elif scenario == "shot_budget":
        held_values = (128,)
    elif scenario == "noise_strength":
        held_values = (2,)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    cells = []
    for held in held_values:
        for fold in range(5):
            split = manifest["within_domain_folds"][fold]
            training = split["training_cells"]["512"]
            cells.append(
                {
                    "label": str(held),
                    "fold": fold,
                    "held": held,
                    "development_ids": training["development_ids"],
                    "stacking_ids": training["stacking_ids"],
                    "calibration_ids": training["calibration_ids"],
                    "test_ids": split["test_ids"],
                }
            )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=PROJECT / "data/derived/local_benchmark_v1.npz"
    )
    parser.add_argument(
        "--splits", type=Path, default=PROJECT / "protocol/local_split_manifest_v1.json"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=("circuit_family", "deep_circuits", "noise_family", "shot_budget", "noise_strength"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--model-seed", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.splits.read_text())
    start = time.perf_counter()
    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    results = {}
    saved = {}
    with np.load(args.input, allow_pickle=False) as archive:
        data = LocalData(archive)
        for scenario_index, scenario in enumerate(args.scenarios):
            cell_outputs: dict[str, list[dict[str, object]]] = {}
            for cell_index, cell in enumerate(scenario_cells(manifest, scenario)):
                development_rows = filter_rows(
                    data, data.rows_for_ids(cell["development_ids"]), scenario, cell.get("held"), True
                )
                stacking_rows = filter_rows(
                    data, data.rows_for_ids(cell["stacking_ids"]), scenario, cell.get("held"), True
                )
                calibration_rows = filter_rows(
                    data, data.rows_for_ids(cell["calibration_ids"]), scenario, cell.get("held"), True
                )
                test_rows = filter_rows(
                    data, data.rows_for_ids(cell["test_ids"]), scenario, cell.get("held"), False
                )
                calibration_predictions, test_predictions, fit_record = fit_and_predict(
                    data,
                    development_rows,
                    stacking_rows,
                    calibration_rows,
                    test_rows,
                    seed=args.model_seed + 100_000 * scenario_index + 1000 * cell_index,
                    threads=args.threads,
                )
                condition_calibration = set(_cell_keys(data, test_rows)) <= set(
                    _cell_keys(data, calibration_rows)
                )
                target_calibration = data.y(calibration_rows)
                method_outputs = {}
                for method, prediction in test_predictions.items():
                    if condition_calibration:
                        calibration = calibrate_by_condition(
                            calibration_predictions[method],
                            target_calibration,
                            calibration_rows,
                            data,
                        )
                        radius = radii_for_rows(calibration, test_rows, data)
                        clipped_calibration = calibrate_by_condition(
                            np.clip(calibration_predictions[method], -1.0, 1.0),
                            target_calibration,
                            calibration_rows,
                            data,
                        )
                        clipped_radius = radii_for_rows(clipped_calibration, test_rows, data)
                        calibration_mode = "condition_specific"
                    else:
                        radius = np.full(
                            len(test_rows),
                            global_group_radius(
                                calibration_predictions[method],
                                target_calibration,
                                calibration_rows,
                                data,
                            ),
                        )
                        clipped_radius = np.full(
                            len(test_rows),
                            global_group_radius(
                                np.clip(calibration_predictions[method], -1.0, 1.0),
                                target_calibration,
                                calibration_rows,
                                data,
                            ),
                        )
                        calibration_mode = "global_seen_distribution"
                    target = data.y(test_rows)
                    sampled = data.row_sampled[test_rows]
                    method_outputs[method] = {
                        "raw_metrics": local_metrics(
                            prediction, target, sampled, radius, test_rows, data
                        ),
                        "clipped_metrics": local_metrics(
                            np.clip(prediction, -1.0, 1.0),
                            target,
                            sampled,
                            clipped_radius,
                            test_rows,
                            data,
                        ),
                        "calibration_mode": calibration_mode,
                    }
                    key = f"{scenario}__{cell['label']}__cell{cell_index}__{method}"
                    saved[f"{key}__prediction"] = prediction
                    saved[f"{key}__radius"] = radius
                saved[f"{scenario}__{cell['label']}__cell{cell_index}__row_index"] = test_rows
                cell_outputs.setdefault(cell["label"], []).append(
                    {"fold": cell.get("fold"), "fit": fit_record, "methods": method_outputs}
                )
            scenario_result = {}
            for label, parts in cell_outputs.items():
                scenario_result[label] = {"parts": parts}
                if len(parts) > 1:
                    # Metrics are recomputed from concatenated predictions in a separate
                    # aggregation script; part-level results preserve exact fold evidence.
                    scenario_result[label]["outer_parts"] = len(parts)
            results[scenario] = scenario_result
    end_usage = resource.getrusage(resource.RUSAGE_SELF)
    output = {
        "schema_version": 1,
        "analysis": "local_distribution_shifts",
        "evidence_class": "local_simulation_and_method_extension",
        "input_sha256": sha256(args.input),
        "split_manifest_sha256": sha256(args.splits),
        "scenarios": results,
        "model_seed": args.model_seed,
        "bootstrap_draws_reserved_for_aggregation": args.bootstrap_draws,
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
    for scenario, labels in results.items():
        print(f"\n{scenario}")
        for label, cell in labels.items():
            for part in cell["parts"]:
                suffix = "" if part["fold"] is None else f" fold={part['fold']}"
                values = part["methods"]
                safe = "safe_cell_simplex" if "safe_cell_simplex" in values else "safe_simplex"
                print(
                    f"  {label}{suffix}: noisy={values['noisy_sampled']['raw_metrics']['mae']:.5f} "
                    f"rf={values['rf']['raw_metrics']['mae']:.5f} "
                    f"{safe}={values[safe]['raw_metrics']['mae']:.5f}"
                )


if __name__ == "__main__":
    main()
