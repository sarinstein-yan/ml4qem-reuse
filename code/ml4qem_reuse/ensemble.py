"""Leakage-safe, low-complexity ensemble utilities.

The first array axis is always the indivisible circuit unit. Observables remain
on the second axis so a split can never place observables from one circuit in
different partitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, StratifiedKFold, StratifiedShuffleSplit


RIDGE_ALPHAS = np.asarray([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0])
# Absolute-error differences below this scale are numerically indistinguishable
# for the serialized float32 benchmark and the SLSQP simplex fits. A method is
# counted as a failure only when it is worse than the raw estimate by more than
# this margin. Strict zero-margin rates remain recoverable from saved predictions.
FAILURE_MARGIN = 1e-6


@dataclass(frozen=True)
class LinearCombiner:
    """A shared linear combiner across observables."""

    weights: np.ndarray
    intercept: float = 0.0
    alpha: float | None = None

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        return np.einsum("nom,m->no", predictions, self.weights) + self.intercept


def _validate_xy(predictions: np.ndarray, target: np.ndarray) -> None:
    if predictions.ndim != 3 or target.ndim != 2:
        raise ValueError("predictions must be [circuit, observable, model] and target [circuit, observable]")
    if predictions.shape[:2] != target.shape:
        raise ValueError("prediction and target axes do not match")
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(target)):
        raise ValueError("non-finite ensemble input")


def fit_simplex(predictions: np.ndarray, target: np.ndarray) -> LinearCombiner:
    """Fit a non-negative, sum-to-one least-squares blend."""

    _validate_xy(predictions, target)
    n_models = predictions.shape[-1]

    def objective(weights: np.ndarray) -> float:
        residual = np.einsum("nom,m->no", predictions, weights) - target
        return float(np.mean(np.square(residual)))

    result = minimize(
        objective,
        np.full(n_models, 1.0 / n_models),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"simplex optimization failed: {result.message}")
    weights = np.maximum(result.x, 0.0)
    weights /= np.sum(weights)
    return LinearCombiner(weights=weights)


def fit_ridge_nested(
    predictions: np.ndarray,
    target: np.ndarray,
    strata: np.ndarray,
    *,
    seed: int,
    alphas: np.ndarray = RIDGE_ALPHAS,
) -> LinearCombiner:
    """Select ridge strength by circuit-level inner CV, then refit."""

    _validate_xy(predictions, target)
    if len(strata) != len(target):
        raise ValueError("one stratum is required per circuit")
    counts = np.unique(strata, return_counts=True)[1]
    n_splits = min(5, int(np.min(counts)))
    if n_splits < 2:
        raise ValueError("at least two circuits per stratum are required")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    losses = np.zeros(len(alphas), dtype=float)
    for train, validation in splitter.split(np.zeros(len(strata)), strata):
        train_x = predictions[train].reshape(-1, predictions.shape[-1])
        train_y = target[train].reshape(-1)
        validation_x = predictions[validation].reshape(-1, predictions.shape[-1])
        validation_y = target[validation].reshape(-1)
        for index, alpha in enumerate(alphas):
            model = Ridge(alpha=float(alpha)).fit(train_x, train_y)
            losses[index] += float(np.mean(np.square(model.predict(validation_x) - validation_y)))
    selected = float(alphas[int(np.argmin(losses))])
    model = Ridge(alpha=selected).fit(
        predictions.reshape(-1, predictions.shape[-1]), target.reshape(-1)
    )
    return LinearCombiner(
        weights=np.asarray(model.coef_, dtype=float),
        intercept=float(model.intercept_),
        alpha=selected,
    )


def fit_ridge_grouped(
    predictions: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    strata: np.ndarray,
    *,
    seed: int,
    alphas: np.ndarray = RIDGE_ALPHAS,
) -> LinearCombiner:
    """Select ridge strength without splitting repeated rows from a base circuit."""

    _validate_xy(predictions, target)
    if len(groups) != len(target) or len(strata) != len(target):
        raise ValueError("one group and stratum are required per prediction row")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two circuit groups are required")
    n_splits = min(5, len(unique_groups))
    # GroupKFold remains well-defined when a small stacking set contains only
    # one circuit from some strata; StratifiedGroupKFold may emit an empty fold.
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    losses = np.zeros(len(alphas), dtype=float)
    for train, validation in splitter.split(np.zeros(len(strata)), groups=groups):
        train_x = predictions[train].reshape(-1, predictions.shape[-1])
        train_y = target[train].reshape(-1)
        validation_x = predictions[validation].reshape(-1, predictions.shape[-1])
        validation_y = target[validation].reshape(-1)
        for index, alpha in enumerate(alphas):
            model = Ridge(alpha=float(alpha)).fit(train_x, train_y)
            losses[index] += float(np.mean(np.square(model.predict(validation_x) - validation_y)))
    selected = float(alphas[int(np.argmin(losses))])
    model = Ridge(alpha=selected).fit(
        predictions.reshape(-1, predictions.shape[-1]), target.reshape(-1)
    )
    return LinearCombiner(
        weights=np.asarray(model.coef_, dtype=float),
        intercept=float(model.intercept_),
        alpha=selected,
    )


def fit_calibration_split(
    indices: np.ndarray,
    strata: np.ndarray,
    *,
    seed: int,
    calibration_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Split circuit indices while approximately preserving every stratum."""

    if len(indices) != len(strata):
        raise ValueError("indices and strata must have equal length")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=calibration_fraction, random_state=seed
    )
    fit_local, calibration_local = next(splitter.split(np.zeros(len(indices)), strata))
    return indices[fit_local], indices[calibration_local]


def conformal_radius(
    prediction: np.ndarray, target: np.ndarray, *, alpha: float = 0.1
) -> float:
    """Finite-sample split-conformal radius for simultaneous circuit coverage.

    Return an infinite radius when the requested order statistic is not
    available. This represents abstention from a finite interval at the
    requested nominal coverage rather than silently lowering that coverage.
    """

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share [circuit, observable] shape")
    circuit_scores = np.max(np.abs(prediction - target), axis=1)
    rank = int(np.ceil((len(circuit_scores) + 1) * (1.0 - alpha)))
    if rank > len(circuit_scores):
        return float("inf")
    rank = max(rank, 1)
    return float(np.partition(circuit_scores, rank - 1)[rank - 1])


def evaluate_predictions(
    prediction: np.ndarray,
    target: np.ndarray,
    noisy: np.ndarray,
    depth: np.ndarray,
    radius: np.ndarray | float,
) -> dict[str, float]:
    """Compute circuit-aware accuracy, reliability and coverage outcomes."""

    if prediction.shape != target.shape or noisy.shape != target.shape:
        raise ValueError("prediction, target and noisy arrays must match")
    absolute = np.abs(prediction - target)
    noisy_absolute = np.abs(noisy - target)
    radius_array = np.broadcast_to(np.asarray(radius, dtype=float).reshape(-1, 1), target.shape)
    depth_mae = [float(np.mean(absolute[depth == value])) for value in np.unique(depth)]
    joint_covered = np.all(absolute <= radius_array, axis=1)
    failures = failure_mask(absolute, noisy_absolute)
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(absolute)))),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "failure_rate": float(np.mean(failures)),
        "circuit_failure_rate": float(np.mean(np.any(failures, axis=1))),
        "worst_depth_mae": float(np.max(depth_mae)),
        "physical_range_violation_rate": float(np.mean((prediction < -1.0) | (prediction > 1.0))),
        "interval_joint_coverage": float(np.mean(joint_covered)),
        "interval_marginal_coverage": float(np.mean(absolute <= radius_array)),
        "interval_mean_width": float(2.0 * np.mean(radius_array)),
    }


def failure_mask(
    candidate_absolute_error: np.ndarray,
    reference_absolute_error: np.ndarray,
    *,
    margin: float = FAILURE_MARGIN,
) -> np.ndarray:
    """Return materially worse predictions using a declared numerical margin."""

    if candidate_absolute_error.shape != reference_absolute_error.shape:
        raise ValueError("candidate and reference errors must match")
    if margin < 0:
        raise ValueError("failure margin must be non-negative")
    return candidate_absolute_error > reference_absolute_error + margin


def paired_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Circuit bootstrap for a paired mean difference and percentile interval."""

    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("paired circuit-level vectors are required")
    difference = candidate - reference
    generator = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    chunk = 256
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        sampled = generator.integers(0, len(difference), size=(stop - start, len(difference)))
        estimates[start:stop] = np.mean(difference[sampled], axis=1)
    return {
        "estimate": float(np.mean(difference)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "draws": int(draws),
    }


def paired_signflip_test(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> float:
    """Two-sided Monte Carlo paired sign-flip test on circuit-level effects."""

    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("paired circuit-level vectors are required")
    if draws < 1:
        raise ValueError("draws must be positive")
    difference = candidate - reference
    observed = abs(float(np.mean(difference)))
    generator = np.random.default_rng(seed)
    exceedances = 0
    chunk = 1024
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        signs = generator.integers(0, 2, size=(stop - start, len(difference)), dtype=np.int8)
        signs = 2.0 * signs - 1.0
        null = np.abs(np.mean(signs * difference, axis=1))
        exceedances += int(np.sum(null >= observed))
    return float((exceedances + 1) / (draws + 1))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini--Hochberg adjusted q-values."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a finite one-dimensional vector in [0, 1]")
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output
