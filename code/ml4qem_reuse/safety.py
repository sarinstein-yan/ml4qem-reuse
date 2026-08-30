"""Hard safety boundary between local analysis and quantum hardware."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class HardwareExecutionDisabled(RuntimeError):
    """Raised when a configuration could address remote quantum hardware."""


_LOCAL_BACKEND_MARKERS = ("aer", "simulator", "statevector", "fake")
_REMOTE_MARKERS = ("ibm_", "ibmq_", "runtime", "quantum_inspire", "braket")


def assert_hardware_disabled(config: Mapping[str, Any] | None = None, backend: Any = None) -> None:
    """Reject real/remote execution modes and non-simulator backend objects.

    Project commands call this before circuit execution. Archived data analysis
    does not pass a backend and is permitted. The test is intentionally
    conservative: an unknown backend is rejected rather than guessed local.
    """

    config = config or {}
    modes = config.get("modes", ())
    if isinstance(modes, str):
        modes = (modes,)
    requested = {str(config.get("mode", "")).lower(), *(str(mode).lower() for mode in modes)}
    if requested & {"real", "hardware", "qpu", "remote"}:
        raise HardwareExecutionDisabled(f"hardware mode disabled: {sorted(requested)}")

    for key in ("service", "provider", "channel", "instance"):
        if config.get(key):
            raise HardwareExecutionDisabled(f"remote credential field disabled: {key}")

    if backend is None:
        return
    name_attribute = getattr(backend, "name", backend.__class__.__name__)
    name = name_attribute() if callable(name_attribute) else name_attribute
    identity = f"{backend.__class__.__module__}.{backend.__class__.__name__}:{name}".lower()
    if any(marker in identity for marker in _REMOTE_MARKERS):
        raise HardwareExecutionDisabled(f"remote backend disabled: {identity}")
    if not any(marker in identity for marker in _LOCAL_BACKEND_MARKERS):
        raise HardwareExecutionDisabled(f"backend is not explicitly local: {identity}")

