import numpy as np
from qiskit import QuantumCircuit

from ml4qem_reuse.features import encode_v2, rotation_angle_histogram


def test_rotation_histogram_and_feature_layout() -> None:
    circuit = QuantumCircuit(2)
    circuit.rx(0.1, 0)
    circuit.rz(-0.2, 1)
    circuit.cx(0, 1)
    features = encode_v2([circuit], [[0.25, -0.5]], observable_count=2, two_qubit_gate="cx")
    assert features.shape == (1, 167)
    np.testing.assert_allclose(features[0, :5], [0.01, 0, 0, 0, 0.01])
    np.testing.assert_allclose(features[0, -2:], [0.25, -0.5])
    assert rotation_angle_histogram(circuit, 0.025 * np.pi).sum() == 2


def test_feature_encoding_does_not_accept_wrong_observable_width() -> None:
    circuit = QuantumCircuit(1)
    with np.testing.assert_raises(ValueError):
        encode_v2([circuit], [[0.1, 0.2]], observable_count=1)
