from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalThresholds:
    method: str
    q_global: float | None = None
    q_by_class: dict[int, float] | None = None
    calibration_counts: dict[int | str, int] | None = None


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Conformal scores must be a non-empty vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    order = int(np.ceil((values.size + 1) * (1 - alpha)))
    if order > values.size:
        return float("inf")
    return float(np.partition(values, order - 1)[order - 1])


def _true_label_scores(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    p1 = np.asarray(probability, dtype=float)
    y = np.asarray(target, dtype=int)
    true_probability = np.where(y == 1, p1, 1 - p1)
    return 1 - true_probability


def fit_thresholds(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    method: str,
    alpha: float,
    min_class_count: int = 5,
) -> ConformalThresholds:
    scores = _true_label_scores(probability, target)
    target = np.asarray(target, dtype=int)
    if method == "standard":
        return ConformalThresholds(
            method=method,
            q_global=conformal_quantile(scores, alpha),
            calibration_counts={"all": int(scores.size)},
        )
    if method == "mondrian":
        thresholds: dict[int, float] = {}
        for label in (0, 1):
            class_scores = scores[target == label]
            if class_scores.size < min_class_count:
                raise ValueError(
                    f"Mondrian class {label} has {class_scores.size} examples; "
                    f"minimum {min_class_count}"
                )
            thresholds[label] = conformal_quantile(class_scores, alpha)
        return ConformalThresholds(
            method=method,
            q_by_class=thresholds,
            calibration_counts={label: int((target == label).sum()) for label in (0, 1)},
        )
    raise ValueError(f"Unknown conformal method: {method}")


def prediction_sets(probability: np.ndarray, thresholds: ConformalThresholds) -> np.ndarray:
    p1 = np.asarray(probability, dtype=float)
    scores = np.column_stack((p1, 1 - p1))  # 1-p0=p1; 1-p1=1-p1
    if thresholds.method == "standard":
        if thresholds.q_global is None:
            raise ValueError("Missing global threshold")
        limits = np.array([thresholds.q_global, thresholds.q_global])
    elif thresholds.method == "mondrian":
        if thresholds.q_by_class is None:
            raise ValueError("Missing Mondrian thresholds")
        limits = np.array([thresholds.q_by_class[0], thresholds.q_by_class[1]])
    else:
        raise ValueError(f"Unknown threshold method: {thresholds.method}")
    return scores <= limits.reshape(1, 2)
