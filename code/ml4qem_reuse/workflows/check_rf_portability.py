#!/usr/bin/env python3
"""Compare paper-era and current scikit-learn RF predictions on identical arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--legacy-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    with np.load(args.design, allow_pickle=False) as design:
        x_train = design["x_train"]
        y_train = design["y_train"]
        x_test = design["x_test"]
        y_test = design["y_test"]
        noisy = design["noisy_test"]
    with np.load(args.legacy_predictions, allow_pickle=False) as legacy:
        legacy_prediction = legacy["prediction"]
        if not np.array_equal(legacy["target"], y_test) or not np.array_equal(
            legacy["noisy"], noisy
        ):
            raise ValueError("legacy predictions do not match the exported design")
    current_predictions = []
    seed_records = []
    for seed in range(len(legacy_prediction)):
        models = [
            RandomForestRegressor(
                n_estimators=args.trees,
                random_state=seed,
                n_jobs=args.n_jobs,
            ).fit(x_train, y_train[:, observable])
            for observable in range(y_train.shape[1])
        ]
        current = np.column_stack([model.predict(x_test) for model in models])
        current_predictions.append(current)
        difference = current - legacy_prediction[seed]
        current_error = current - y_test
        legacy_error = legacy_prediction[seed] - y_test
        seed_records.append(
            {
                "seed": seed,
                "prediction_max_absolute_difference": float(np.max(np.abs(difference))),
                "prediction_mean_absolute_difference": float(np.mean(np.abs(difference))),
                "current_rmse": float(np.sqrt(np.mean(np.square(current_error)))),
                "legacy_rmse": float(np.sqrt(np.mean(np.square(legacy_error)))),
                "rmse_difference": float(
                    np.sqrt(np.mean(np.square(current_error)))
                    - np.sqrt(np.mean(np.square(legacy_error)))
                ),
                "current_mae": float(np.mean(np.abs(current_error))),
                "legacy_mae": float(np.mean(np.abs(legacy_error))),
            }
        )
    result = {
        "schema_version": 1,
        "analysis": "random_forest_current_stack_portability",
        "design_sha256": sha256(args.design),
        "legacy_predictions_sha256": sha256(args.legacy_predictions),
        "legacy_sklearn_version": "1.3.2",
        "current_sklearn_version": sklearn.__version__,
        "trees": args.trees,
        "seeds": len(seed_records),
        "global_prediction_max_absolute_difference": max(
            record["prediction_max_absolute_difference"] for record in seed_records
        ),
        "mean_prediction_mae_between_versions": float(
            np.mean([record["prediction_mean_absolute_difference"] for record in seed_records])
        ),
        "mean_rmse_difference": float(
            np.mean([record["rmse_difference"] for record in seed_records])
        ),
        "seed_records": seed_records,
        "hardware_jobs_submitted": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions,
        target=y_test,
        noisy=noisy,
        legacy_prediction=legacy_prediction,
        current_prediction=np.asarray(current_predictions),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "seed_records"}, indent=2))


if __name__ == "__main__":
    main()
