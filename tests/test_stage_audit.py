import numpy as np

from ml4qem_reuse.workflows.audit_stage_independence import (
    aggregate_by_base,
    normalize_qasm,
    qasm_sha256,
)


def test_qasm_normalization_ignores_only_blank_and_line_edge_whitespace() -> None:
    left = "OPENQASM 2.0;\n\n  x q[0];  \n"
    right = " OPENQASM 2.0; \nx q[0];"
    assert normalize_qasm(left) == normalize_qasm(right)
    assert qasm_sha256(left) == qasm_sha256(right)


def test_aggregate_by_base_keeps_circuit_as_the_unit() -> None:
    values = np.asarray([1.0, 3.0, 10.0, 14.0, 18.0])
    groups = np.asarray(["a", "a", "b", "b", "b"])
    np.testing.assert_allclose(aggregate_by_base(values, groups), [2.0, 14.0])
