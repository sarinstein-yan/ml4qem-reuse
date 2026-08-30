import pytest
from qiskit_aer import AerSimulator

from ml4qem_reuse.safety import HardwareExecutionDisabled, assert_hardware_disabled


def test_archived_data_and_aer_are_allowed() -> None:
    assert_hardware_disabled()
    assert_hardware_disabled({"mode": "simulation"}, AerSimulator())


@pytest.mark.parametrize("mode", ["real", "hardware", "qpu", "remote"])
def test_hardware_modes_are_rejected(mode: str) -> None:
    with pytest.raises(HardwareExecutionDisabled):
        assert_hardware_disabled({"mode": mode})


def test_unknown_backend_is_rejected() -> None:
    class Backend:
        name = "experimental_backend"

    with pytest.raises(HardwareExecutionDisabled):
        assert_hardware_disabled(backend=Backend())
