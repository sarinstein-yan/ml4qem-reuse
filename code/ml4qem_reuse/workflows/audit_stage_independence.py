#!/usr/bin/env python3
"""Audit circuit-content overlap and inherited angle-range omissions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from qiskit import qasm2

from ml4qem_reuse.ensemble import paired_bootstrap
from ml4qem_reuse.workflows._paths import workspace_root


PROJECT = workspace_root()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_qasm(text: str) -> str:
    """Normalize only line-edge and blank-line whitespace for content hashing."""

    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def qasm_sha256(text: str) -> str:
    return hashlib.sha256(normalize_qasm(text).encode()).hexdigest()


def load_records(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def records_by_hash(records: list[dict[str, object]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[qasm_sha256(str(record["qasm2"]))].append(str(record["base_id"]))
    return dict(grouped)


def angle_range_audit(records: list[dict[str, object]]) -> dict[str, object]:
    omitted_by_family: dict[str, int] = defaultdict(int)
    affected_by_family: dict[str, int] = defaultdict(int)
    for record in records:
        circuit = qasm2.loads(
            str(record["qasm2"]), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
        )
        omitted = 0
        for instruction in circuit.data:
            operation = instruction.operation
            if operation.name in {"rx", "ry", "rz"} and len(operation.params) == 1:
                angle = float(operation.params[0])
                omitted += int(angle < -2.0 * np.pi or angle > 2.0 * np.pi)
        family = str(record["family"])
        omitted_by_family[family] += omitted
        affected_by_family[family] += int(omitted > 0)
    families = sorted({str(record["family"]) for record in records})
    return {
        "released_histogram_range": "[-2*pi, 2*pi] with 0.025*pi bins",
        "outside_range_rotation_entries": int(sum(omitted_by_family.values())),
        "affected_circuits": int(sum(affected_by_family.values())),
        "outside_range_entries_by_family": {
            family: int(omitted_by_family[family]) for family in families
        },
        "affected_circuits_by_family": {
            family: int(affected_by_family[family]) for family in families
        },
        "interpretation": (
            "These rotations remain in gate-count features but do not enter the inherited "
            "angle histogram; no periodic remapping was introduced."
        ),
    }


def aggregate_by_base(values: np.ndarray, base: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(base, return_inverse=True)
    return np.bincount(inverse, weights=values) / np.bincount(inverse)


def exclusion_sensitivity(
    data_path: Path,
    predictions_path: Path,
    excluded_base_ids: list[str],
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    with np.load(data_path, allow_pickle=False) as source:
        row_base_index = source["config_base_index"][source["row_config_index"]]
        source_base_ids = source["base_id"]
    with np.load(predictions_path, allow_pickle=False) as predictions:
        rows = predictions["row_index"]
        target = predictions["target"]
        unmitigated = predictions["raw__noisy_sampled__prediction"]
        candidate = predictions["raw__safe_simplex__prediction"]
    base_ids = source_base_ids[row_base_index[rows]]
    keep = ~np.isin(base_ids, np.asarray(excluded_base_ids))
    candidate_error = aggregate_by_base(
        np.mean(np.abs(candidate[keep] - target[keep]), axis=1), base_ids[keep]
    )
    reference_error = aggregate_by_base(
        np.mean(np.abs(unmitigated[keep] - target[keep]), axis=1), base_ids[keep]
    )
    effect = paired_bootstrap(
        candidate_error,
        reference_error,
        draws=draws,
        seed=seed,
    )
    return {
        "excluded_base_ids": excluded_base_ids,
        "excluded_base_circuits": len(excluded_base_ids),
        "retained_base_circuits": int(len(candidate_error)),
        "paired_global_anchor_mae_effect_vs_unmitigated": effect,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-circuits",
        type=Path,
        default=PROJECT / "data/derived/local_benchmark_v1_circuits.jsonl.gz",
    )
    parser.add_argument(
        "--confirmation-circuits",
        type=Path,
        default=PROJECT
        / "data/derived/local_benchmark_confirmation_v2_circuits.jsonl.gz",
    )
    parser.add_argument(
        "--confirmation-data",
        type=Path,
        default=PROJECT / "data/derived/local_benchmark_confirmation_v2.npz",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT
        / "results/confirmation/v2/local_within_n512_folds0-4_seed0_predictions.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results/confirmation/v2/stage_independence_audit.json",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=142_425)
    args = parser.parse_args()

    development = load_records(args.development_circuits)
    confirmation = load_records(args.confirmation_circuits)
    development_hashes = records_by_hash(development)
    confirmation_hashes = records_by_hash(confirmation)
    overlap_hashes = sorted(development_hashes.keys() & confirmation_hashes.keys())
    overlaps = [
        {
            "normalized_qasm_sha256": digest,
            "development_base_ids": development_hashes[digest],
            "confirmation_base_ids": confirmation_hashes[digest],
        }
        for digest in overlap_hashes
    ]
    excluded = sorted(
        base_id for overlap in overlaps for base_id in overlap["confirmation_base_ids"]
    )
    result = {
        "schema_version": 1,
        "evidence_class": "confirmation_sensitivity",
        "qasm_normalization": "strip line edges, remove blank lines, preserve instruction order",
        "inputs": {
            "development_circuits": {
                "path": str(args.development_circuits.relative_to(PROJECT)),
                "sha256": sha256(args.development_circuits),
            },
            "confirmation_circuits": {
                "path": str(args.confirmation_circuits.relative_to(PROJECT)),
                "sha256": sha256(args.confirmation_circuits),
            },
            "confirmation_data": {
                "path": str(args.confirmation_data.relative_to(PROJECT)),
                "sha256": sha256(args.confirmation_data),
            },
            "predictions": {
                "path": str(args.predictions.relative_to(PROJECT)),
                "sha256": sha256(args.predictions),
            },
        },
        "development": {
            "records": len(development),
            "unique_normalized_qasm": len(development_hashes),
            "within_stage_duplicate_groups": [
                {"normalized_qasm_sha256": digest, "base_ids": base_ids}
                for digest, base_ids in sorted(development_hashes.items())
                if len(base_ids) > 1
            ],
            "angle_range_audit": angle_range_audit(development),
        },
        "confirmation": {
            "records": len(confirmation),
            "unique_normalized_qasm": len(confirmation_hashes),
            "within_stage_duplicate_groups": [
                {"normalized_qasm_sha256": digest, "base_ids": base_ids}
                for digest, base_ids in sorted(confirmation_hashes.items())
                if len(base_ids) > 1
            ],
            "angle_range_audit": angle_range_audit(confirmation),
        },
        "cross_stage_overlap_groups": overlaps,
        "cross_stage_overlap_count": len(overlaps),
        "confirmation_exclusion_sensitivity": exclusion_sensitivity(
            args.confirmation_data,
            args.predictions,
            excluded,
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed,
        ),
        "hardware_jobs_submitted": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
