import numpy as np

from ml4qem_reuse.simulation import (
    apply_symmetric_readout,
    build_noise_model,
    expectation_from_probabilities,
    make_circuit,
)


def test_expectation_bit_order_and_strings() -> None:
    probability = np.zeros(16)
    probability[0b0101] = 1.0
    assert np.array_equal(expectation_from_probabilities(probability), [-1.0, 1.0, -1.0, -1.0])


def test_readout_channel_normalizes_and_uniform_limit() -> None:
    probability = np.zeros(16)
    probability[0] = 1.0
    assert np.isclose(np.sum(apply_symmetric_readout(probability, 0.07)), 1.0)
    assert np.allclose(apply_symmetric_readout(probability, 0.5), np.full(16, 1 / 16))


def test_noise_cells_and_circuit_determinism() -> None:
    for family in ("depolarizing_readout", "coherent_overrotation", "damping_dephasing"):
        model, readout = build_noise_model(family, 1)
        assert model is not None
        assert 0.0 <= readout <= 1.0
    first = make_circuit("ising_trotter", 3, 19)
    second = make_circuit("ising_trotter", 3, 19)
    assert first == second
    assert set(first.count_ops()) <= {"rz", "sx", "x", "cx"}
