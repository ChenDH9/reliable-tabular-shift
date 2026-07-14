from __future__ import annotations

import numpy as np
import pytest

from reliable_shift.conformal import conformal_quantile, fit_thresholds, prediction_sets


def test_conformal_quantile_calculation() -> None:
    assert conformal_quantile(np.array([0.1, 0.2, 0.3, 0.4]), alpha=0.2) == 0.4


def test_conformal_set_construction() -> None:
    thresholds = fit_thresholds(
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([0, 0, 1, 1]),
        method="standard",
        alpha=0.25,
    )
    sets = prediction_sets(np.array([0.05, 0.5, 0.95]), thresholds)
    assert sets.shape == (3, 2)
    assert sets[0, 0]
    assert sets[2, 1]


def test_mondrian_class_conditioning() -> None:
    probability = np.array([0.05, 0.1, 0.2, 0.3, 0.8, 0.85, 0.9, 0.95, 0.75, 0.7])
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    thresholds = fit_thresholds(
        probability, target, method="mondrian", alpha=0.2, min_class_count=4
    )
    assert thresholds.q_by_class is not None
    assert set(thresholds.q_by_class) == {0, 1}


def test_mondrian_tiny_class_fails() -> None:
    with pytest.raises(ValueError, match="minimum"):
        fit_thresholds(
            np.array([0.1, 0.2, 0.9]),
            np.array([0, 0, 1]),
            method="mondrian",
            alpha=0.1,
            min_class_count=2,
        )
