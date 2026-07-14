from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def calibration_intercept_slope(
    target: np.ndarray, probability: np.ndarray, epsilon: float = 1e-6
) -> tuple[float, float]:
    target = np.asarray(target, dtype=int)
    if np.unique(target).size < 2:
        raise ValueError("Calibration regression requires both classes")
    probability = np.clip(np.asarray(probability, dtype=float), epsilon, 1 - epsilon)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(logit, target)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def expected_calibration_error(
    target: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    target = np.asarray(target, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    indices = np.minimum(np.digitize(probability, edges[1:-1], right=False), bins - 1)
    value = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            value += float(mask.mean()) * abs(float(probability[mask].mean() - target[mask].mean()))
    return value


def probability_metrics(
    target: np.ndarray, probability: np.ndarray, bins: int = 10
) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    intercept, slope = calibration_intercept_slope(target, probability)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    prevalence = float(target.mean())
    brier = float(brier_score_loss(target, probability))
    model_log_loss = float(log_loss(target, clipped, labels=[0, 1]))
    null_brier = float(np.mean((target - prevalence) ** 2))
    null_probability = np.full(len(target), np.clip(prevalence, 1e-6, 1 - 1e-6))
    null_log_loss = float(log_loss(target, null_probability, labels=[0, 1]))
    return {
        "auroc": float(roc_auc_score(target, probability)),
        "auprc": float(average_precision_score(target, probability)),
        "brier": brier,
        "log_loss": model_log_loss,
        "evaluation_prevalence": prevalence,
        "null_brier": null_brier,
        "brier_skill_score": 1 - brier / null_brier,
        "null_log_loss": null_log_loss,
        "log_loss_skill_score": 1 - model_log_loss / null_log_loss,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece_10_equal_width": expected_calibration_error(target, probability, bins=bins),
    }


def conformal_metrics(
    target: np.ndarray,
    sets: np.ndarray,
    nominal_coverage: float,
    *,
    small_class_count: int = 30,
) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    sets = np.asarray(sets, dtype=bool)
    if sets.shape != (len(target), 2):
        raise ValueError("Binary prediction sets must have shape (n, 2)")
    covered = sets[np.arange(len(target)), target]
    coverage = float(covered.mean())
    sizes = sets.sum(axis=1)
    class_coverage: dict[int, float] = {}
    intervals: dict[int, tuple[float, float]] = {}
    counts: dict[int, int] = {}
    for label in (0, 1):
        mask = target == label
        counts[label] = int(mask.sum())
        if not mask.any():
            class_coverage[label] = float("nan")
            intervals[label] = (float("nan"), float("nan"))
            continue
        successes = int(covered[mask].sum())
        estimate = successes / counts[label]
        z = 1.959963984540054
        denominator = 1 + z**2 / counts[label]
        center = (estimate + z**2 / (2 * counts[label])) / denominator
        radius = (
            z
            * np.sqrt(estimate * (1 - estimate) / counts[label] + z**2 / (4 * counts[label] ** 2))
            / denominator
        )
        class_coverage[label] = float(estimate)
        intervals[label] = (float(max(0, center - radius)), float(min(1, center + radius)))
    gap = coverage - nominal_coverage
    result = {
        "coverage": coverage,
        "marginal_coverage": coverage,
        "coverage_gap": gap,
        "signed_coverage_gap": gap,
        "absolute_coverage_error": abs(gap),
        "undercoverage": max(0.0, -gap),
        "overcoverage": max(0.0, gap),
        "class0_coverage": class_coverage[0],
        "class1_coverage": class_coverage[1],
        "worst_class_coverage": float(np.nanmin([class_coverage[0], class_coverage[1]])),
        "class0_coverage_ci_low": intervals[0][0],
        "class0_coverage_ci_high": intervals[0][1],
        "class1_coverage_ci_low": intervals[1][0],
        "class1_coverage_ci_high": intervals[1][1],
        "class0_eval_count": float(counts[0]),
        "class1_eval_count": float(counts[1]),
        "class0_small_evaluation": float(counts[0] < small_class_count),
        "class1_small_evaluation": float(counts[1] < small_class_count),
        "mean_set_size": float(sizes.mean()),
        "empty_set_fraction": float((sizes == 0).mean()),
        "singleton_fraction": float((sizes == 1).mean()),
        "doubleton_fraction": float((sizes == 2).mean()),
    }
    return result
