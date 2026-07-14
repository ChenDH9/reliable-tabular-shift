from __future__ import annotations

import numpy as np
import pytest

from reliable_shift.metrics.core import (
    calibration_intercept_slope,
    conformal_metrics,
    probability_metrics,
)


def test_metric_ranges() -> None:
    target = np.array([0, 0, 1, 1, 1, 0])
    probability = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.4])
    metrics = probability_metrics(target, probability)
    for name in ("auroc", "auprc", "brier", "ece_10_equal_width"):
        assert 0 <= metrics[name] <= 1
    sets = np.array([[1, 0], [1, 0], [0, 1], [1, 1], [0, 1], [0, 0]], dtype=bool)
    conformal = conformal_metrics(target, sets, 0.9)
    assert 0 <= conformal["coverage"] <= 1
    assert -0.9 <= conformal["coverage_gap"] <= 0.1


def test_calibration_slope_intercept_on_synthetic_data() -> None:
    rng = np.random.default_rng(42)
    probability = rng.uniform(0.05, 0.95, size=100_000)
    target = rng.binomial(1, probability)
    intercept, slope = calibration_intercept_slope(target, probability)
    assert abs(intercept) < 0.05
    assert abs(slope - 1) < 0.05


def test_extreme_probabilities_are_stable() -> None:
    metrics = probability_metrics(np.array([0, 0, 1, 1]), np.array([0.0, 1e-20, 1 - 1e-20, 1.0]))
    assert np.isfinite(list(metrics.values())).all()


def test_single_class_is_explicit_failure() -> None:
    with pytest.raises(ValueError, match="both classes"):
        calibration_intercept_slope(np.zeros(10), np.full(10, 0.1))
