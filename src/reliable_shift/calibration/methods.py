from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

FORMAL_CALIBRATION_METHODS = (
    "uncalibrated",
    "intercept_only_mle",
    "sigmoid",
    "isotonic",
    "jeffreys_smoothed_intercept_matching",
)

# Aliases are input compatibility only.  Formal result rows must always use the
# value returned by ``canonical_calibration_name``.
CALIBRATION_METHOD_ALIASES = {
    "intercept_only": "intercept_only_mle",
}

ISOTONIC_MIN_TOTAL = 25
ISOTONIC_MIN_PER_CLASS = 5
ISOTONIC_MIN_UNIQUE_INPUT_SCORES = 10


def canonical_calibration_name(name: str) -> str:
    """Return the frozen formal name for a calibration method."""

    canonical = CALIBRATION_METHOD_ALIASES.get(name, name)
    if canonical not in FORMAL_CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method: {name}")
    return canonical


def isotonic_preflight(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    min_total: int = ISOTONIC_MIN_TOTAL,
    min_per_class: int = ISOTONIC_MIN_PER_CLASS,
    min_unique_input_scores: int = ISOTONIC_MIN_UNIQUE_INPUT_SCORES,
) -> dict[str, int | bool | str]:
    """Evaluate the frozen, outcome-aware eligibility rule for isotonic fit.

    This function is intended to run only after a blind target sample has been
    selected.  It does not select observations and therefore cannot leak labels
    into acquisition.
    """

    probability = np.asarray(probability, dtype=float).ravel()
    target = np.asarray(target, dtype=int).ravel()
    if len(probability) != len(target):
        raise ValueError("Probability and target lengths differ")
    if min_total < 1 or min_per_class < 1 or min_unique_input_scores < 1:
        raise ValueError("Isotonic preflight thresholds must be positive")
    observed_labels = set(np.unique(target).tolist())
    if not observed_labels <= {0, 1}:
        raise ValueError("Isotonic calibration target must be binary")

    total = int(len(target))
    class0 = int((target == 0).sum())
    class1 = int((target == 1).sum())
    unique_scores = int(np.unique(probability).size)
    failures: list[str] = []
    if total < min_total:
        failures.append(f"n={total} < {min_total}")
    if class0 < min_per_class:
        failures.append(f"class0={class0} < {min_per_class}")
    if class1 < min_per_class:
        failures.append(f"class1={class1} < {min_per_class}")
    if unique_scores < min_unique_input_scores:
        failures.append(
            f"unique_input_scores={unique_scores} < {min_unique_input_scores}"
        )
    eligible = not failures
    return {
        "eligible": eligible,
        "status": "success" if eligible else "not_estimable",
        "status_reason": "" if eligible else "; ".join(failures),
        "total_count": total,
        "class0_count": class0,
        "class1_count": class1,
        "unique_input_scores": unique_scores,
        "min_total": int(min_total),
        "min_per_class": int(min_per_class),
        "min_unique_input_scores": int(min_unique_input_scores),
    }


def _clip(probability: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), epsilon, 1 - epsilon)


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = _clip(probability)
    return np.log(probability / (1 - probability)).reshape(-1, 1)


class Calibrator(Protocol):
    def fit(self, probability: np.ndarray, target: np.ndarray) -> Calibrator: ...

    def predict(self, probability: np.ndarray) -> np.ndarray: ...


class IdentityCalibrator:
    method_name = "uncalibrated"

    def fit(self, probability: np.ndarray, target: np.ndarray) -> IdentityCalibrator:
        del probability, target
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return np.asarray(probability, dtype=float)

    def diagnostics(self) -> dict[str, int | bool]:
        return {}


class SigmoidCalibrator:
    method_name = "sigmoid"

    def __init__(self) -> None:
        self.model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
        self.class_counts: dict[int, int] = {}

    def fit(self, probability: np.ndarray, target: np.ndarray) -> SigmoidCalibrator:
        target = np.asarray(target, dtype=int)
        if np.unique(target).size < 2:
            raise ValueError("Sigmoid calibration requires both classes")
        self.class_counts = {label: int((target == label).sum()) for label in (0, 1)}
        self.model.fit(_logit(probability), target)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_logit(probability))[:, 1]

    def diagnostics(self) -> dict[str, int | bool]:
        return {
            "fitted_calibrator_intercept": float(self.model.intercept_[0]),
            "fitted_calibrator_slope": float(self.model.coef_[0, 0]),
            "stable_support_5_per_class": min(self.class_counts.values(), default=0) >= 5,
            "single_class_calibration": len([n for n in self.class_counts.values() if n > 0]) < 2,
            **{f"class{label}_count": count for label, count in self.class_counts.items()},
        }


class InterceptOnlyCalibrator:
    """Logistic recalibration with the base logit as a unit-slope offset."""

    method_name = "intercept_only_mle"

    def __init__(self) -> None:
        self.intercept = 0.0
        self.class_counts: dict[int, int] = {}

    def fit(self, probability: np.ndarray, target: np.ndarray) -> InterceptOnlyCalibrator:
        from scipy.optimize import brentq

        target = np.asarray(target, dtype=int)
        if np.unique(target).size < 2:
            raise ValueError("Intercept-only calibration requires both classes")
        logits = _logit(probability).ravel()
        self.class_counts = {label: int((target == label).sum()) for label in (0, 1)}

        def score(offset: float) -> float:
            fitted = 1.0 / (1.0 + np.exp(-np.clip(logits + offset, -40, 40)))
            return float(np.sum(target - fitted))

        self.intercept = float(brentq(score, -40.0, 40.0))
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        values = _logit(probability).ravel() + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))

    def diagnostics(self) -> dict[str, int | bool | float]:
        return {
            "fitted_intercept": self.intercept,
            "fitted_calibrator_intercept": self.intercept,
            "fitted_calibrator_slope": 1.0,
            "single_class_calibration": len(
                [count for count in self.class_counts.values() if count > 0]
            )
            < 2,
            **{f"class{label}_count": count for label, count in self.class_counts.items()},
        }


class JeffreysSmoothedInterceptCalibrator(InterceptOnlyCalibrator):
    """Unit-slope intercept matching to (positive+0.5)/(n+1), including one-class samples."""

    method_name = "jeffreys_smoothed_intercept_matching"

    def fit(
        self, probability: np.ndarray, target: np.ndarray
    ) -> JeffreysSmoothedInterceptCalibrator:
        from scipy.optimize import brentq

        target = np.asarray(target, dtype=int)
        if len(target) == 0:
            raise ValueError("Smoothed intercept calibration requires at least one observation")
        logits = _logit(probability).ravel()
        desired = (float(target.sum()) + 0.5) / (len(target) + 1.0)
        self.class_counts = {label: int((target == label).sum()) for label in (0, 1)}

        def score(offset: float) -> float:
            fitted = 1.0 / (1.0 + np.exp(-np.clip(logits + offset, -40, 40)))
            return float(np.mean(fitted) - desired)

        self.intercept = float(brentq(score, -40.0, 40.0))
        return self


class IsotonicCalibrator:
    method_name = "isotonic"

    def __init__(self, *, enforce_preflight: bool = False) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._diagnostics: dict[str, int | bool] = {}
        self.enforce_preflight = enforce_preflight

    def fit(self, probability: np.ndarray, target: np.ndarray) -> IsotonicCalibrator:
        target = np.asarray(target, dtype=int)
        probability = np.asarray(probability, dtype=float)
        preflight = isotonic_preflight(probability, target)
        if self.enforce_preflight and not bool(preflight["eligible"]):
            raise ValueError(f"Isotonic calibration not estimable: {preflight['status_reason']}")
        if np.unique(target).size < 2:
            raise ValueError("Isotonic calibration requires both classes")
        self.model.fit(probability, target)
        fitted = np.asarray(self.model.predict(probability), dtype=float)
        self._diagnostics = {
            "total_count": int(len(target)),
            "class0_count": int((target == 0).sum()),
            "class1_count": int((target == 1).sum()),
            "distinct_input_probability_count": int(np.unique(probability).size),
            "distinct_output_probability_count": int(np.unique(fitted).size),
            "output_contains_zero": bool(np.any(fitted == 0)),
            "output_contains_one": bool(np.any(fitted == 1)),
            "unique_input_scores": int(np.unique(probability).size),
            "unique_output_scores": int(np.unique(fitted).size),
            "produced_exact_zero": bool(np.any(fitted == 0)),
            "produced_exact_one": bool(np.any(fitted == 1)),
            "single_class_calibration": False,
            "preflight_eligible": bool(preflight["eligible"]),
        }
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        predicted = self.model.predict(np.asarray(probability, dtype=float))
        return np.asarray(predicted, dtype=float)

    def diagnostics(self) -> dict[str, int | bool]:
        return dict(self._diagnostics)


def make_calibrator(name: str) -> Calibrator:
    name = canonical_calibration_name(name)
    if name == "uncalibrated":
        return IdentityCalibrator()
    if name == "sigmoid":
        return SigmoidCalibrator()
    if name == "intercept_only_mle":
        return InterceptOnlyCalibrator()
    if name == "jeffreys_smoothed_intercept_matching":
        return JeffreysSmoothedInterceptCalibrator()
    if name == "isotonic":
        return IsotonicCalibrator(enforce_preflight=True)
    raise AssertionError(f"Unhandled canonical calibration method: {name}")
