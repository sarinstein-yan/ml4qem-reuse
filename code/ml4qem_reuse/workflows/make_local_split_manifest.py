#!/usr/bin/env python3
"""Freeze base-circuit splits before fitting any local-benchmark model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from ml4qem_reuse.workflows._paths import workspace_root


PROJECT = workspace_root()
TRAINING_SIZES = (32, 64, 128, 256, 512)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interleaved_order(indices: np.ndarray, strata: np.ndarray, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    queues = []
    for stratum in sorted(np.unique(strata)):
        values = indices[strata == stratum].copy()
        generator.shuffle(values)
        queues.append(values)
    output = []
    for rank in range(max(map(len, queues))):
        output.extend(queue[rank] for queue in queues if rank < len(queue))
    return np.asarray(output, dtype=int)


def _internal_three_way(
    selected: np.ndarray, family: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    development_local, heldout_local = next(
        splitter.split(np.zeros(len(selected)), family[selected])
    )
    heldout = selected[heldout_local]
    second = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    stacking_local, calibration_local = next(
        second.split(np.zeros(len(heldout)), family[heldout])
    )
    return selected[development_local], heldout[stacking_local], heldout[calibration_local]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=PROJECT / "data/derived/local_benchmark_v1.npz"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=161803)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        base_id = archive["base_id"]
        family = archive["family"]
        layers = archive["layers"]
    composite = np.asarray([f"{name}:L{layer}" for name, layer in zip(family, layers)])
    folds = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(base_id)), composite)):
        ordered = _interleaved_order(train, composite[train], args.seed + 1000 * fold)
        cells = {}
        for size in TRAINING_SIZES:
            selected = ordered[:size]
            development, stacking, calibration = _internal_three_way(
                selected, family, seed=args.seed + 10_000 * fold + size
            )
            cells[str(size)] = {
                "selected_ids": base_id[selected].tolist(),
                "development_ids": base_id[development].tolist(),
                "stacking_ids": base_id[stacking].tolist(),
                "calibration_ids": base_id[calibration].tolist(),
            }
        folds.append(
            {
                "fold": fold,
                "test_ids": base_id[test].tolist(),
                "training_cells": cells,
            }
        )
    family_holdouts = {}
    for offset, held_family in enumerate(np.unique(family)):
        test = np.flatnonzero(family == held_family)
        available = np.flatnonzero(family != held_family)
        development, stacking, calibration = _internal_three_way(
            available, family, seed=args.seed + 50_000 + offset
        )
        family_holdouts[held_family] = {
            "test_ids": base_id[test].tolist(),
            "development_ids": base_id[development].tolist(),
            "stacking_ids": base_id[stacking].tolist(),
            "calibration_ids": base_id[calibration].tolist(),
        }
    shallow = np.flatnonzero(layers <= 6)
    deep = np.flatnonzero(layers >= 7)
    development, stacking, calibration = _internal_three_way(
        shallow, family, seed=args.seed + 60_000
    )
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "input_sha256": _sha256(args.input),
        "grouping_unit": "base_id; all noise, strength, shot, replicate and observable variants remain together",
        "training_sizes": list(TRAINING_SIZES),
        "within_domain_folds": folds,
        "family_holdouts": family_holdouts,
        "deep_holdout": {
            "training_layers": [1, 2, 3, 4, 5, 6],
            "test_layers": [7, 8],
            "test_ids": base_id[deep].tolist(),
            "development_ids": base_id[development].tolist(),
            "stacking_ids": base_id[stacking].tolist(),
            "calibration_ids": base_id[calibration].tolist(),
        },
        "row_filter_shifts": {
            "noise_family": "fit and calibrate on two families; test the third only on outer-test base circuits",
            "shot_budget": "fit and calibrate on 512, 2048 and 10000; test 128 only on outer-test base circuits",
            "noise_strength": "fit and calibrate on levels 0 and 1; test level 2 only on outer-test base circuits",
            "observable": "shared long-format estimator fits Z0 and Z1; tests Z0Z1 and Z2Z3 on outer-test base circuits",
        },
        "hardware_jobs_submitted": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "folds": len(folds),
                "test_circuits_per_fold": [len(fold["test_ids"]) for fold in folds],
                "family_holdout_test_circuits": {
                    key: len(value["test_ids"]) for key, value in family_holdouts.items()
                },
                "deep_holdout_test_circuits": len(deep),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
