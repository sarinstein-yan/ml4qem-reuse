#!/usr/bin/env python3
"""Generate the prespecified four-qubit ML4QEM reuse benchmark locally."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
from qiskit import qasm2
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from ml4qem_reuse.features import circuit_summary, encode_v2
from ml4qem_reuse.safety import assert_hardware_disabled
from ml4qem_reuse.simulation import (
    NOISE_STRENGTHS,
    OBSERVABLE_LABELS,
    OBSERVABLE_MASKS,
    apply_symmetric_readout,
    build_noise_model,
    expectation_from_probabilities,
    make_circuit,
)
from ml4qem_reuse.workflows._paths import workspace_root


PROJECT = workspace_root()
FAMILIES = ("ising_trotter", "warm_start_qaoa", "hardware_efficient", "random_clifford")
SHOT_BUDGETS = (128, 512, 2048, 10_000)
SHOT_REPLICATES = 3
LAYERS = tuple(range(1, 9))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _simulate_probabilities(
    simulator: AerSimulator,
    circuits: list[object],
    *,
    noise_model: object,
    readout: float,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    output = np.empty((len(circuits), 16), dtype=np.float64)
    largest_roundoff_correction = 0.0
    for start in range(0, len(circuits), batch_size):
        stop = min(start + batch_size, len(circuits))
        batch = []
        for circuit in circuits[start:stop]:
            saved = circuit.copy()
            saved.save_probabilities()
            batch.append(saved)
        job = simulator.run(batch, noise_model=noise_model, seed_simulator=seed + start)
        result = job.result()
        if not result.success:
            raise RuntimeError(f"local Aer batch failed: {result.status}")
        for local_index in range(stop - start):
            probability = np.asarray(result.data(local_index)["probabilities"], dtype=float)
            if readout:
                probability = apply_symmetric_readout(probability, readout)
            minimum = float(np.min(probability))
            if minimum < -1e-10:
                raise RuntimeError(f"Aer returned a materially negative probability: {minimum}")
            largest_roundoff_correction = max(largest_roundoff_correction, max(0.0, -minimum))
            probability = np.clip(probability, 0.0, None)
            probability /= np.sum(probability)
            output[start + local_index] = probability
    return output, largest_roundoff_correction


def _sample_nested_counts(
    probability: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    expectations = []
    budgets = []
    replicates = []
    for replicate in range(SHOT_REPLICATES):
        cumulative = np.zeros(len(probability), dtype=np.int64)
        previous = 0
        for budget in SHOT_BUDGETS:
            cumulative += generator.multinomial(budget - previous, probability)
            expectations.append(expectation_from_probabilities(cumulative / budget))
            budgets.append(budget)
            replicates.append(replicate)
            previous = budget
    return (
        np.asarray(expectations, dtype=np.float32),
        np.asarray(budgets, dtype=np.int32),
        np.asarray(replicates, dtype=np.int8),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--circuits", type=Path, required=True)
    parser.add_argument("--circuits-per-family", type=int, default=240)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=120)
    args = parser.parse_args()
    if args.circuits_per_family % len(LAYERS):
        raise ValueError("circuits-per-family must be divisible by eight layer settings")
    if args.threads < 1:
        raise ValueError("threads must be positive")

    start = time.perf_counter()
    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    circuits = []
    base_ids = []
    families = []
    layers = []
    seeds = []
    per_layer = args.circuits_per_family // len(LAYERS)
    for family_index, family in enumerate(FAMILIES):
        for layer in LAYERS:
            for replicate in range(per_layer):
                circuit_seed = args.seed + family_index * 1_000_000 + layer * 10_000 + replicate
                circuits.append(make_circuit(family, layer, circuit_seed))
                base_ids.append(f"{family}:L{layer}:{replicate:03d}")
                families.append(family)
                layers.append(layer)
                seeds.append(circuit_seed)

    ideal_probabilities = np.asarray(
        [Statevector.from_instruction(circuit).probabilities() for circuit in circuits], dtype=float
    )
    targets = expectation_from_probabilities(ideal_probabilities).astype(np.float32)
    static_features = encode_v2(
        circuits,
        np.zeros((len(circuits), len(OBSERVABLE_MASKS))),
        observable_count=len(OBSERVABLE_MASKS),
        two_qubit_gate="cx",
    )[:, : -len(OBSERVABLE_MASKS)]
    summaries = [circuit_summary(circuit) for circuit in circuits]

    args.circuits.parent.mkdir(parents=True, exist_ok=True)
    with args.circuits.open("wb") as raw_stream:
        with gzip.GzipFile(filename="", fileobj=raw_stream, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                for base_id, family, layer, seed, circuit in zip(
                    base_ids, families, layers, seeds, circuits
                ):
                    stream.write(
                        json.dumps(
                            {
                                "base_id": base_id,
                                "family": family,
                                "layers": layer,
                                "seed": seed,
                                "qasm2": qasm2.dumps(circuit),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )

    simulator = AerSimulator(
        method="density_matrix",
        max_parallel_threads=args.threads,
        max_parallel_experiments=1,
    )
    assert_hardware_disabled({"mode": "local"}, simulator)
    config_base = []
    config_noise = []
    config_strength = []
    config_exact = []
    row_config = []
    row_sampled = []
    row_budget = []
    row_replicate = []
    config_index = 0
    largest_probability_correction = 0.0
    for noise_index, noise_family in enumerate(NOISE_STRENGTHS):
        for strength_index in range(len(NOISE_STRENGTHS[noise_family])):
            noise_model, readout = build_noise_model(noise_family, strength_index)
            probabilities, correction = _simulate_probabilities(
                simulator,
                circuits,
                noise_model=noise_model,
                readout=readout,
                batch_size=args.batch_size,
                seed=args.seed + 10_000_000 * noise_index + 100_000 * strength_index,
            )
            largest_probability_correction = max(largest_probability_correction, correction)
            exact = expectation_from_probabilities(probabilities).astype(np.float32)
            for base_index, probability in enumerate(probabilities):
                config_base.append(base_index)
                config_noise.append(noise_family)
                config_strength.append(strength_index)
                config_exact.append(exact[base_index])
                sampled, budgets, replicates = _sample_nested_counts(
                    probability,
                    seed=(
                        args.seed
                        + 100_000_000 * noise_index
                        + 1_000_000 * strength_index
                        + base_index
                    ),
                )
                row_config.extend([config_index] * len(sampled))
                row_sampled.extend(sampled)
                row_budget.extend(budgets)
                row_replicate.extend(replicates)
                config_index += 1

    arrays = {
        "base_id": np.asarray(base_ids),
        "family": np.asarray(families),
        "layers": np.asarray(layers, dtype=np.int8),
        "circuit_seed": np.asarray(seeds, dtype=np.int64),
        "depth": np.asarray([summary["depth"] for summary in summaries], dtype=np.int16),
        "size": np.asarray([summary["size"] for summary in summaries], dtype=np.int16),
        "n_single_qubit": np.asarray(
            [summary["n_single_qubit"] for summary in summaries], dtype=np.int16
        ),
        "n_two_qubit": np.asarray(
            [summary["n_two_qubit"] for summary in summaries], dtype=np.int16
        ),
        "static_features": static_features.astype(np.float32),
        "target": targets,
        "config_base_index": np.asarray(config_base, dtype=np.int32),
        "config_noise_family": np.asarray(config_noise),
        "config_strength_index": np.asarray(config_strength, dtype=np.int8),
        "config_exact_noisy": np.asarray(config_exact, dtype=np.float32),
        "row_config_index": np.asarray(row_config, dtype=np.int32),
        "row_sampled_noisy": np.asarray(row_sampled, dtype=np.float32),
        "row_shot_budget": np.asarray(row_budget, dtype=np.int32),
        "row_shot_replicate": np.asarray(row_replicate, dtype=np.int8),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    end_usage = resource.getrusage(resource.RUSAGE_SELF)
    manifest = {
        "schema_version": 1,
        "dataset_id": f"ml4qem-reuse-local-4q-v1-seed{args.seed}",
        "dataset_family_id": "ml4qem-reuse-local-4q-v1",
        "dataset_instance_id": f"ml4qem-reuse-local-4q-v1-seed{args.seed}",
        "evidence_class": "local_simulation",
        "seed": args.seed,
        "basis_gates": ["rz", "sx", "x", "cx"],
        "circuit_families": list(FAMILIES),
        "circuits_per_family": args.circuits_per_family,
        "base_circuits": len(circuits),
        "layers": list(LAYERS),
        "noise_strengths": NOISE_STRENGTHS,
        "exact_noisy_configurations": len(config_base),
        "shot_budgets": list(SHOT_BUDGETS),
        "shot_replicates": SHOT_REPLICATES,
        "sampled_rows": len(row_config),
        "observables": [
            {"label": label, "bit_mask": int(mask)}
            for label, mask in zip(OBSERVABLE_LABELS, OBSERVABLE_MASKS)
        ],
        "sampling": "nested cumulative multinomial counts within each shot replicate",
        "probability_roundoff_rule": "fail below -1e-10; otherwise clip at zero and renormalize",
        "largest_probability_roundoff_correction": largest_probability_correction,
        "ideal_solver": "Qiskit Statevector on four qubits",
        "noisy_solver": "local Aer density_matrix plus explicit classical readout channel",
        "hardware_jobs_submitted": 0,
        "remote_services_used": 0,
        "resources": {
            "wall_seconds": time.perf_counter() - start,
            "user_cpu_seconds": end_usage.ru_utime - start_usage.ru_utime,
            "system_cpu_seconds": end_usage.ru_stime - start_usage.ru_stime,
            "peak_rss_kib": int(end_usage.ru_maxrss),
            "threads_requested": args.threads,
            "logical_cpus_visible": os.cpu_count(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "gpus_used": 0,
            "qpu_jobs": 0,
        },
    }
    manifest["files"] = {
        "data": {"path": str(args.output), "sha256": sha256(args.output)},
        "circuits": {"path": str(args.circuits), "sha256": sha256(args.circuits)},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
