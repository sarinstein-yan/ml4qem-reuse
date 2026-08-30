import numpy as np

from ml4qem_reuse.ensemble import (
    benjamini_hochberg,
    conformal_radius,
    evaluate_predictions,
    failure_mask,
    fit_calibration_split,
    fit_ridge_nested,
    fit_ridge_grouped,
    fit_simplex,
    paired_bootstrap,
    paired_signflip_test,
)


def _example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(7)
    target = generator.uniform(-0.8, 0.8, size=(60, 4))
    predictions = np.stack(
        [target + generator.normal(0.0, scale, target.shape) for scale in (0.2, 0.05, 0.1)],
        axis=-1,
    )
    strata = np.repeat(np.arange(3), 20)
    return predictions, target, strata


def test_simplex_constraints_and_preferred_model() -> None:
    predictions, target, _ = _example()
    model = fit_simplex(predictions, target)
    assert np.all(model.weights >= 0.0)
    assert np.isclose(np.sum(model.weights), 1.0)
    assert np.argmax(model.weights) == 1


def test_ridge_nested_is_circuit_shaped() -> None:
    predictions, target, strata = _example()
    model = fit_ridge_nested(predictions, target, strata, seed=11)
    assert model.predict(predictions).shape == target.shape
    assert model.alpha is not None


def test_ridge_grouped_accepts_repeated_circuit_rows() -> None:
    predictions, target, strata = _example()
    repeated_predictions = np.repeat(predictions, 2, axis=0)
    repeated_target = np.repeat(target, 2, axis=0)
    groups = np.repeat(np.arange(len(target)), 2)
    repeated_strata = np.repeat(strata, 2)
    model = fit_ridge_grouped(
        repeated_predictions, repeated_target, groups, repeated_strata, seed=13
    )
    assert model.predict(repeated_predictions).shape == repeated_target.shape


def test_fit_calibration_split_has_no_overlap() -> None:
    _, _, strata = _example()
    indices = np.arange(len(strata))
    fit, calibration = fit_calibration_split(indices, strata, seed=3)
    assert not set(fit) & set(calibration)
    assert set(fit) | set(calibration) == set(indices)
    assert set(strata[fit]) == set(strata[calibration]) == set(strata)


def test_group_conformal_and_metrics() -> None:
    predictions, target, strata = _example()
    candidate = predictions[..., 1]
    radius = conformal_radius(candidate[:30], target[:30])
    metrics = evaluate_predictions(
        candidate[30:], target[30:], predictions[30:, :, 0], strata[30:], radius
    )
    assert radius > 0
    assert 0.0 <= metrics["interval_joint_coverage"] <= 1.0
    assert metrics["interval_marginal_coverage"] >= metrics["interval_joint_coverage"]


def test_group_conformal_abstains_when_nominal_rank_is_unavailable() -> None:
    prediction = np.arange(5, dtype=float)[:, None]
    target = np.zeros_like(prediction)
    assert np.isinf(conformal_radius(prediction, target, alpha=0.1))


def test_paired_bootstrap_sign() -> None:
    effect = paired_bootstrap(np.zeros(40), np.ones(40), draws=200, seed=5)
    assert effect["estimate"] == -1.0
    assert effect["ci_high"] < 0.0


def test_signflip_and_bh_are_bounded_and_ordered() -> None:
    p_value = paired_signflip_test(np.zeros(40), np.ones(40), draws=999, seed=5)
    assert 0.0 < p_value <= 0.01
    adjusted = benjamini_hochberg(np.asarray([0.01, 0.04, 0.03, 0.8]))
    assert np.all((adjusted >= 0.0) & (adjusted <= 1.0))
    assert adjusted[0] <= adjusted[2] <= adjusted[1] <= adjusted[3]


def test_failure_mask_ignores_numerical_ties() -> None:
    candidate = np.asarray([0.1, 0.1000005, 0.100002, 0.09])
    reference = np.full(4, 0.1)
    assert failure_mask(candidate, reference).tolist() == [False, False, True, False]
