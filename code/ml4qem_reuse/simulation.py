"""Small-system circuit and noise definitions for the local benchmark."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_clifford
from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
    coherent_unitary_error,
    depolarizing_error,
    phase_damping_error,
)


BASIS_GATES = ("rz", "sx", "x", "cx")
OBSERVABLE_MASKS = np.asarray([0b0001, 0b0010, 0b0011, 0b1100], dtype=np.uint8)
OBSERVABLE_LABELS = ("Z0", "Z1", "Z0Z1", "Z2Z3")
NOISE_STRENGTHS = {
    "depolarizing_readout": (
        {"p1": 0.0005, "p2": 0.005, "readout": 0.005},
        {"p1": 0.002, "p2": 0.02, "readout": 0.02},
        {"p1": 0.008, "p2": 0.06, "readout": 0.06},
    ),
    "coherent_overrotation": (
        {"angle": 0.005},
        {"angle": 0.02},
        {"angle": 0.06},
    ),
    "damping_dephasing": (
        {"amplitude": 0.001, "phase": 0.001},
        {"amplitude": 0.005, "phase": 0.005},
        {"amplitude": 0.02, "phase": 0.02},
    ),
}


def expectation_from_probabilities(
    probabilities: np.ndarray, masks: np.ndarray = OBSERVABLE_MASKS
) -> np.ndarray:
    """Return Z-string expectations using Qiskit's little-endian state index."""

    probabilities = np.asarray(probabilities, dtype=float)
    n_states = probabilities.shape[-1]
    if n_states == 0 or n_states & (n_states - 1):
        raise ValueError("the probability axis must have power-of-two length")
    states = np.arange(n_states, dtype=np.uint64)
    signs = np.empty((len(masks), n_states), dtype=float)
    for index, mask in enumerate(masks):
        parity = np.fromiter(
            (int(state & int(mask)).bit_count() & 1 for state in states),
            dtype=np.int8,
            count=n_states,
        )
        signs[index] = 1.0 - 2.0 * parity
    return np.einsum("...s,os->...o", probabilities, signs)


def apply_symmetric_readout(probabilities: np.ndarray, probability: float) -> np.ndarray:
    """Apply independent symmetric classical bit flips to a full distribution."""

    probabilities = np.asarray(probabilities, dtype=float)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("readout probability must lie in [0, 1]")
    n_states = len(probabilities)
    n_qubits = int(np.log2(n_states))
    if 2**n_qubits != n_states:
        raise ValueError("the distribution length must be a power of two")
    output = np.zeros_like(probabilities)
    for true_state, mass in enumerate(probabilities):
        for measured_state in range(n_states):
            flips = (true_state ^ measured_state).bit_count()
            output[measured_state] += mass * probability**flips * (1.0 - probability) ** (
                n_qubits - flips
            )
    return output / np.sum(output)


def build_noise_model(family: str, strength_index: int) -> tuple[NoiseModel, float]:
    """Build one prespecified quantum channel and return its readout flip rate."""

    try:
        parameters = NOISE_STRENGTHS[family][strength_index]
    except (KeyError, IndexError) as error:
        raise ValueError(f"unknown noise cell: {family}/{strength_index}") from error
    model = NoiseModel()
    readout = 0.0
    if family == "depolarizing_readout":
        model.add_all_qubit_quantum_error(
            depolarizing_error(parameters["p1"], 1), ["sx", "x"]
        )
        model.add_all_qubit_quantum_error(depolarizing_error(parameters["p2"], 2), ["cx"])
        readout = parameters["readout"]
    elif family == "coherent_overrotation":
        angle = parameters["angle"]
        one_qubit = np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
        two_qubit = np.diag(
            [
                np.exp(-0.5j * angle),
                np.exp(0.5j * angle),
                np.exp(0.5j * angle),
                np.exp(-0.5j * angle),
            ]
        )
        model.add_all_qubit_quantum_error(coherent_unitary_error(one_qubit), ["sx", "x"])
        model.add_all_qubit_quantum_error(coherent_unitary_error(two_qubit), ["cx"])
    elif family == "damping_dephasing":
        one_qubit = amplitude_damping_error(parameters["amplitude"]).compose(
            phase_damping_error(parameters["phase"])
        )
        model.add_all_qubit_quantum_error(one_qubit, ["sx", "x"])
        model.add_all_qubit_quantum_error(one_qubit.tensor(one_qubit), ["cx"])
    else:  # pragma: no cover - guarded by the parameter lookup
        raise AssertionError(family)
    return model, float(readout)


def _ising_trotter(layers: int, generator: np.random.Generator) -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    for qubit in range(4):
        circuit.ry(generator.uniform(-np.pi, np.pi), qubit)
    coupling = generator.uniform(0.3, 1.2, size=3)
    field = generator.uniform(0.2, 1.0, size=4)
    step = generator.uniform(0.08, 0.25)
    for _ in range(layers):
        for qubit, value in enumerate(coupling):
            circuit.rzz(2.0 * step * value, qubit, qubit + 1)
        for qubit, value in enumerate(field):
            circuit.rx(2.0 * step * value, qubit)
    return circuit


def _warm_start_qaoa(layers: int, generator: np.random.Generator) -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    for qubit in range(4):
        circuit.ry(np.pi / 2 + generator.normal(0.0, 0.25), qubit)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    if generator.random() < 0.5:
        edges.append((0, 2))
    weights = generator.uniform(0.5, 1.5, size=len(edges))
    for _ in range(layers):
        gamma = generator.uniform(0.05, 0.8)
        beta = generator.uniform(0.05, 0.8)
        for (left, right), weight in zip(edges, weights):
            circuit.rzz(2.0 * gamma * weight, left, right)
        for qubit in range(4):
            circuit.rx(2.0 * beta, qubit)
    return circuit


def _hardware_efficient(layers: int, generator: np.random.Generator) -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    for layer in range(layers):
        for qubit in range(4):
            circuit.ry(generator.uniform(-np.pi, np.pi), qubit)
            circuit.rz(generator.uniform(-np.pi, np.pi), qubit)
        offset = layer % 2
        for qubit in range(offset, 3, 2):
            circuit.cx(qubit, qubit + 1)
        if layer % 2:
            circuit.cx(3, 0)
    return circuit


def _layered_clifford(layers: int, generator: np.random.Generator) -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    for layer in range(layers):
        local = random_clifford(1, seed=int(generator.integers(0, 2**32 - 1))).to_circuit()
        for qubit in range(4):
            circuit.compose(local, [qubit], inplace=True)
            if qubit < 3 and generator.random() < 0.5:
                local = random_clifford(
                    1, seed=int(generator.integers(0, 2**32 - 1))
                ).to_circuit()
        if layer % 2 == 0:
            circuit.cx(0, 1)
            circuit.cx(2, 3)
        else:
            circuit.cx(1, 2)
            circuit.cx(3, 0)
    return circuit


def make_circuit(family: str, layers: int, seed: int) -> QuantumCircuit:
    """Create and transpile one deterministic four-qubit benchmark circuit."""

    generator = np.random.default_rng(seed)
    builders = {
        "ising_trotter": _ising_trotter,
        "warm_start_qaoa": _warm_start_qaoa,
        "hardware_efficient": _hardware_efficient,
        "random_clifford": _layered_clifford,
    }
    try:
        circuit = builders[family](layers, generator)
    except KeyError as error:
        raise ValueError(f"unknown circuit family: {family}") from error
    return transpile(
        circuit,
        basis_gates=list(BASIS_GATES),
        optimization_level=1,
        seed_transpiler=seed,
    )
