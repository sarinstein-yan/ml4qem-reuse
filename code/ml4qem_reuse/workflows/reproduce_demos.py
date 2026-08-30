#!/usr/bin/env python3
"""Repeat the two public ML4QEM demos using archived data only.

This script is intentionally compatible with the paper-era Python 3.9
environment. It never imports an IBM provider and contains no backend execution
path. The target in demo 1 is archived ZNE, whereas the target in demo 2 is the
archived ideal simulation distributed with the hardware results.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ml4qem_reuse.ensemble import FAILURE_MARGIN, failure_mask
from ml4qem_reuse.workflows._paths import workspace_root


PROJECT = workspace_root()
SNAPSHOT = PROJECT / "upstream/snapshots/qiskit-community-ml-qem-9776e1b"
TUTORIALS = SNAPSHOT / "docs/tutorials"
sys.path.insert(0, str(TUTORIALS))

from mlp import encode_data_v2_ecr  # noqa: E402


def _matches_index(name: str, stem: str, indices: Iterable[int], suffix: str) -> bool:
    return name.endswith(suffix) and any(f"{stem}_{index:02d}" in name for index in indices)


def _load_demo1() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    data_dir = TUTORIALS / "data/ising_zne_hardware/100q_brisbane"
    circuits: List[object] = []
    couplings: List[float] = []
    files = sorted(
        path
        for path in data_dir.iterdir()
        if _matches_index(path.name, "step", range(1, 11), ".pk")
    )
    for path in files:
        with path.open("rb") as stream:
            for entry in pickle.load(stream):
                circuits.append(entry["circuit"])
                couplings.append(entry["J"])

    zne_dir = TUTORIALS / "zne_mitigated/twirl_100q_brisbane"
    noise_factor_1: List[object] = []
    noise_factor_3: List[object] = []
    for step in range(1, 11):
        with (zne_dir / f"step{step:02d}.json").open() as stream:
            loaded = json.load(stream)
        noise_factor_1.extend(loaded["noise_factor_1"])
        noise_factor_3.extend(loaded["noise_factor_3"])

    nf1 = np.asarray(noise_factor_1, dtype=float)
    nf3 = np.asarray(noise_factor_3, dtype=float)
    nf1 = nf1.reshape(nf1.shape[0], 5, 5).mean(axis=-1)
    nf3 = nf3.reshape(nf3.shape[0], 5, 5).mean(axis=-1)
    zne = nf1 - (nf3 - nf1) / 2.0

    train_indices: List[int] = []
    test_indices: List[int] = []
    for start in range(0, len(circuits), 50):
        train_indices.extend(range(start, start + 10))
        test_indices.extend(range(start + 10, start + 50))

    train_circuits = [circuits[index] for index in train_indices]
    test_circuits = [circuits[index] for index in test_indices]
    x_train, y_train = encode_data_v2_ecr(
        train_circuits,
        zne[train_indices].tolist(),
        nf1[train_indices].tolist(),
        obs_size=5,
    )
    x_test, y_test = encode_data_v2_ecr(
        test_circuits,
        zne[test_indices].tolist(),
        nf1[test_indices].tolist(),
        obs_size=5,
    )
    metadata = {
        "target": "archived_zne",
        "provenance": "archived_hardware",
        "device": "ibm_brisbane",
        "n_circuits": len(circuits),
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "n_observables": 5,
        "source_files": len(files) + 10,
        "test_couplings": [float(couplings[index]) for index in test_indices],
    }
    return (
        np.asarray(x_train),
        np.asarray(y_train),
        np.asarray(x_test),
        np.asarray(y_test),
        {**metadata, "noisy_test": nf1[test_indices]},
    )


def _unshuffle(values: Sequence[object], order: Sequence[int]) -> List[object]:
    result: List[object] = [None] * len(values)
    for source_index, destination_index in enumerate(order):
        result[destination_index] = values[source_index]
    return result


def _load_demo2() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    data_dir = TUTORIALS / "data/ising_init_from_qasm_hardware/shuffled"
    batch_files = sorted(
        path
        for path in data_dir.iterdir()
        if _matches_index(path.name, "batch", range(60), ".pk")
    )
    circuit_batches = []
    for path in batch_files:
        with path.open("rb") as stream:
            circuit_batches.append(pickle.load(stream)["circuit_batch"])
    circuits = [circuit for batch in circuit_batches for circuit in batch]

    with (data_dir / "results.pk").open("rb") as stream:
        loaded = pickle.load(stream)
    noisy = np.asarray(loaded["noisy"], dtype=float).reshape(-1, 4).tolist()
    zne = np.asarray(loaded["zne_mitigated"], dtype=float).reshape(-1, 4).tolist()
    ideal = loaded["ideal"]
    with (data_dir / "index_order.json").open() as stream:
        order = json.load(stream)
    noisy = _unshuffle(noisy, order)
    zne = _unshuffle(zne, order)
    ideal = _unshuffle(ideal, order)
    circuits = _unshuffle(circuits, order)

    rows = []
    for step, start in enumerate(range(0, 2000, 200)):
        rows.extend(
            (step, ideal[index], noisy[index], zne[index], circuits[index])
            for index in range(start, start + 200)
        )
    for step, start in enumerate(range(2000, 3000, 100)):
        rows.extend(
            (step, ideal[index], noisy[index], zne[index], circuits[index])
            for index in range(start, start + 100)
        )
    rows.sort(key=lambda row: row[0])

    train_rows = []
    test_rows = []
    for step in range(10):
        step_rows = [row for row in rows if row[0] == step]
        train_rows.extend(step_rows[:50])
        test_rows.extend(step_rows[-250:])

    train_ideal = [row[1] for row in train_rows]
    train_noisy = [row[2] for row in train_rows]
    train_circuits = [row[4] for row in train_rows]
    test_ideal = [row[1] for row in test_rows]
    test_noisy = [row[2] for row in test_rows]
    test_circuits = [row[4] for row in test_rows]
    x_train, y_train = encode_data_v2_ecr(
        train_circuits,
        train_ideal,
        train_noisy,
        obs_size=4,
        two_q_gate="cx",
    )
    x_test, y_test = encode_data_v2_ecr(
        test_circuits,
        test_ideal,
        test_noisy,
        obs_size=4,
        two_q_gate="cx",
    )
    metadata = {
        "target": "archived_ideal_simulation",
        "provenance": "archived_hardware",
        "device": "ibm_hardware_as_distributed",
        "n_circuits": len(rows),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "n_observables": 4,
        "source_files": len(batch_files) + 3,
        "test_steps": [int(row[0]) for row in test_rows],
        "zne_test": np.asarray([row[3] for row in test_rows], dtype=float),
        "noisy_test": np.asarray(test_noisy, dtype=float),
    }
    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), metadata


def _metrics(target: np.ndarray, noisy: np.ndarray, prediction: np.ndarray) -> Dict[str, object]:
    error = prediction - target
    noisy_error = noisy - target
    absolute = np.abs(error)
    noisy_absolute = np.abs(noisy_error)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(absolute)),
        "per_observable_rmse": np.sqrt(np.mean(np.square(error), axis=0)).tolist(),
        "per_observable_mae": np.mean(absolute, axis=0).tolist(),
        "failure_rate": float(np.mean(failure_mask(absolute, noisy_absolute))),
        "tie_rate": float(np.mean(np.abs(absolute - noisy_absolute) <= 1e-6)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "max_absolute_error": float(np.max(absolute)),
        "physical_range_violation_rate": float(np.mean((prediction < -1.0) | (prediction > 1.0))),
        "mean_l2_per_circuit": float(np.mean(np.linalg.norm(error, axis=1))),
    }


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=("demo1", "demo2"), required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--design", type=Path)
    args = parser.parse_args()

    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    loader = _load_demo1 if args.demo == "demo1" else _load_demo2
    start = time.perf_counter()
    x_train, y_train, x_test, y_test, metadata = loader()
    noisy_test = np.asarray(metadata.pop("noisy_test"), dtype=float)
    seed_metrics = []
    predictions = []
    for seed in range(args.seeds):
        models = [
            RandomForestRegressor(
                n_estimators=args.trees,
                random_state=seed,
                n_jobs=args.n_jobs,
            ).fit(x_train, y_train[:, observable])
            for observable in range(y_train.shape[1])
        ]
        prediction = np.column_stack([model.predict(x_test) for model in models])
        predictions.append(prediction)
        seed_metrics.append({"seed": seed, **_metrics(y_test, noisy_test, prediction)})

    aggregate = {}
    for key in ("rmse", "mae", "failure_rate", "p95_absolute_error", "physical_range_violation_rate"):
        values = np.asarray([entry[key] for entry in seed_metrics], dtype=float)
        aggregate[key] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    result = {
        "schema_version": 1,
        "demo": args.demo,
        "artifact": "10.5281/zenodo.13769804",
        "artifact_commit": "9776e1b",
        "estimator": "independent_random_forests_per_observable",
        "n_trees": args.trees,
        "seeds": list(range(args.seeds)),
        "metadata": metadata,
        "unmitigated": _metrics(y_test, noisy_test, noisy_test),
        "runs": seed_metrics,
        "across_seed_summary": aggregate,
        "failure_definition": {
            "reference": "unmitigated expectation",
            "absolute_error_margin": FAILURE_MARGIN,
            "rule": "candidate absolute error > reference absolute error + margin",
        },
        "wall_seconds": time.perf_counter() - start,
        "hardware_jobs_submitted": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(_json_ready(result), stream, indent=2, sort_keys=True)
        stream.write("\n")
    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.predictions,
            target=y_test,
            noisy=noisy_test,
            prediction=np.asarray(predictions),
        )
    if args.design:
        args.design.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.design,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            noisy_test=noisy_test,
        )
    print(json.dumps(_json_ready(aggregate), indent=2, sort_keys=True))
    print(f"wall_seconds={result['wall_seconds']:.3f}")


if __name__ == "__main__":
    main()
