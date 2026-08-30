import numpy as np

from ml4qem_reuse.workflows.build_public_outputs import holm_adjust


def test_holm_adjustment_is_monotone_in_sorted_order_and_matches_reference() -> None:
    p_values = np.asarray([0.04, 0.01, 0.03, 0.002])
    adjusted = holm_adjust(p_values)

    np.testing.assert_allclose(adjusted, [0.06, 0.03, 0.06, 0.008])
    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= 0)


def test_holm_adjustment_clips_at_one() -> None:
    np.testing.assert_allclose(holm_adjust(np.asarray([0.8, 0.9])), [1.0, 1.0])
