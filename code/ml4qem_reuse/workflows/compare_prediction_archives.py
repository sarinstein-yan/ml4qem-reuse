#!/usr/bin/env python3
"""Compare two compatible prediction archives without modifying either."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from ml4qem_reuse.serialization import strict_json_dumps


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_numeric_arrays(first: np.ndarray, second: np.ndarray) -> dict[str, object]:
    """Compare numeric arrays, treating matching signed infinities as equal."""

    first_float = first.astype(float)
    second_float = second.astype(float)
    if np.any(np.isnan(first_float)) or np.any(np.isnan(second_float)):
        raise ValueError("NaN values are not supported in prediction archives")
    first_finite = np.isfinite(first_float)
    second_finite = np.isfinite(second_float)
    if not np.array_equal(first_finite, second_finite):
        raise ValueError("finite and non-finite positions differ")
    nonfinite = ~first_finite
    if np.any(nonfinite):
        signed_infinities_match = np.array_equal(
            np.signbit(first_float[nonfinite]), np.signbit(second_float[nonfinite])
        )
        if not signed_infinities_match:
            raise ValueError("non-finite values differ in sign")
    difference = np.abs(first_float[first_finite] - second_float[first_finite])
    return {
        "array_equal": bool(np.array_equal(first, second)),
        "non_finite_values_match": True,
        "max_absolute_difference": float(np.max(difference)) if difference.size else 0.0,
        "mean_absolute_difference": float(np.mean(difference)) if difference.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("repeat", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.reference, allow_pickle=False) as reference, np.load(
        args.repeat, allow_pickle=False
    ) as repeat:
        if reference.files != repeat.files:
            raise ValueError("archive schemas differ")
        comparisons = {}
        for key in reference.files:
            first = reference[key]
            second = repeat[key]
            if first.shape != second.shape or first.dtype != second.dtype:
                raise ValueError(f"shape or dtype mismatch for {key}")
            if np.issubdtype(first.dtype, np.number):
                comparisons[key] = compare_numeric_arrays(first, second)
            else:
                comparisons[key] = {"array_equal": bool(np.array_equal(first, second))}
    output = {
        "schema_version": 1,
        "analysis": "clean_prediction_archive_repeat",
        "reference_sha256": sha256(args.reference),
        "repeat_sha256": sha256(args.repeat),
        "comparisons": comparisons,
        "global_max_absolute_difference": max(
            value.get("max_absolute_difference", 0.0) for value in comparisons.values()
        ),
        "hardware_jobs_submitted": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(strict_json_dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        strict_json_dumps(
            {"global_max_absolute_difference": output["global_max_absolute_difference"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
