"""Maintained implementations of the compact ML4QEM circuit features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def rotation_angle_histogram(circuit: object, bin_size: float) -> np.ndarray:
    """Count Rx/Ry/Rz angles using the interval convention of the artifact."""

    angles = []
    for instruction in circuit.data:
        operation = instruction.operation if hasattr(instruction, "operation") else instruction[0]
        if operation.name in {"rx", "ry", "rz"} and len(operation.params) == 1:
            angles.append(float(operation.params[0]))
    edges = np.arange(-2.0 * np.pi, 2.0 * np.pi + bin_size, bin_size)
    return np.histogram(angles, bins=edges)[0]


def encode_v2(
    circuits: Sequence[object],
    noisy_expectations: Sequence[Sequence[float]],
    *,
    observable_count: int,
    two_qubit_gate: str = "ecr",
    measurement_bases: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Port the artifact's `encode_data_v2_ecr` without a torch dependency.

    The output order and 0.01 feature scaling intentionally match the public
    implementation. Targets are not accepted here so that feature generation
    cannot accidentally inspect held-out labels.
    """

    if len(circuits) != len(noisy_expectations):
        raise ValueError("circuits and noisy expectations must have equal length")
    if not circuits:
        return np.empty((0, 5 + 160 + observable_count), dtype=np.float32)
    gate_names = [two_qubit_gate, "sx", "x", "id", "rz"]
    bin_size = 0.025 * np.pi
    n_angle_bins = int(np.ceil(4 * np.pi / bin_size))
    basis_width = 0 if measurement_bases is None else len(measurement_bases[0])
    result = np.zeros(
        (len(circuits), len(gate_names) + n_angle_bins + observable_count + basis_width),
        dtype=np.float32,
    )
    gate_end = len(gate_names)
    angle_end = gate_end + n_angle_bins
    expectation_end = angle_end + observable_count

    for index, (circuit, noisy) in enumerate(zip(circuits, noisy_expectations)):
        if len(noisy) != observable_count:
            raise ValueError(f"row {index} has {len(noisy)} rather than {observable_count} observables")
        counts = circuit.count_ops()
        result[index, :gate_end] = [counts.get(name, 0) * 0.01 for name in gate_names]
        result[index, gate_end:angle_end] = rotation_angle_histogram(circuit, bin_size) * 0.01
        result[index, angle_end:expectation_end] = np.asarray(noisy, dtype=np.float32)
        if measurement_bases is not None:
            result[index, expectation_end:] = np.asarray(measurement_bases[index], dtype=np.float32)
    return result


def circuit_summary(circuit: object) -> dict[str, int | float]:
    """Return portable audit metadata without encoding a target value."""

    counts = circuit.count_ops()
    return {
        "n_qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "n_single_qubit": int(sum(value for name, value in counts.items() if name not in {"cx", "ecr", "cz", "swap"})),
        "n_two_qubit": int(sum(counts.get(name, 0) for name in ("cx", "ecr", "cz", "swap"))),
    }
