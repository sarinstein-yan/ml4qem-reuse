import json

import numpy as np
import pytest

from ml4qem_reuse.workflows.compare_prediction_archives import compare_numeric_arrays
from ml4qem_reuse.serialization import (
    finite_interval_fields,
    finite_interval_metric_fields,
    migrate_finite_interval_fields,
    strict_json_dumps,
)


def test_finite_interval_abstention_is_strict_json() -> None:
    record = {
        "interval_mean_width": float("inf"),
        "interval_marginal_coverage": 1.0,
        "interval_row_joint_coverage": 1.0,
        "interval_condition_circuit_joint_coverage": 1.0,
    }
    assert migrate_finite_interval_fields(record) == 1
    assert record == {
        "finite_interval_available": False,
        "interval_mean_width": None,
        "interval_marginal_coverage": None,
        "interval_row_joint_coverage": None,
        "interval_condition_circuit_joint_coverage": None,
    }
    assert json.loads(strict_json_dumps(record)) == record
    assert finite_interval_fields(0.5) == {
        "finite_interval_available": True,
        "interval_mean_width": 0.5,
    }
    assert finite_interval_metric_fields(0.5, {"interval_marginal_coverage": 0.9}) == {
        "finite_interval_available": True,
        "interval_mean_width": 0.5,
        "interval_marginal_coverage": 0.9,
    }


def test_strict_json_rejects_unrelated_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        strict_json_dumps({"unexpected": float("nan")})


def test_prediction_comparison_handles_matching_infinities() -> None:
    first = np.asarray([1.0, np.inf, -np.inf])
    second = np.asarray([1.0 + 1e-8, np.inf, -np.inf])
    comparison = compare_numeric_arrays(first, second)
    assert comparison["non_finite_values_match"] is True
    assert comparison["max_absolute_difference"] == pytest.approx(1e-8)
    assert comparison["mean_absolute_difference"] == pytest.approx(1e-8)


def test_prediction_comparison_rejects_mismatched_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="positions differ"):
        compare_numeric_arrays(np.asarray([np.inf]), np.asarray([1.0]))
    with pytest.raises(ValueError, match="differ in sign"):
        compare_numeric_arrays(np.asarray([np.inf]), np.asarray([-np.inf]))
    with pytest.raises(ValueError, match="NaN"):
        compare_numeric_arrays(np.asarray([np.nan]), np.asarray([np.nan]))
