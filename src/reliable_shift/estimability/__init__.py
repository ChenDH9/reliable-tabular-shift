"""Exact finite-population class-support calculations for calibration samples."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, hypergeom


def _validate(pool_size: int, positive_count: int, budget: int) -> None:
    if pool_size <= 0 or not 0 <= positive_count <= pool_size:
        raise ValueError("invalid finite population")
    if not 0 <= budget <= pool_size:
        raise ValueError("budget must be between zero and pool size")


def _validate_min_counts(min_positive: int, min_negative: int) -> None:
    if min_positive < 0 or min_negative < 0:
        raise ValueError("minimum class counts must be nonnegative")


def hypergeometric_support_probability(
    pool_size: int, positive_count: int, budget: int, min_positive: int, min_negative: int
) -> float:
    """Exact P(min_positive <= K <= budget-min_negative) without replacement."""
    _validate(pool_size, positive_count, budget)
    _validate_min_counts(min_positive, min_negative)
    lower = int(min_positive)
    upper = min(budget - int(min_negative), positive_count, budget)
    if upper < lower:
        return 0.0
    # Difference of stable distribution tails. Select the representation that
    # avoids subtracting two values close to one.
    cdf_interval = hypergeom.cdf(upper, pool_size, positive_count, budget) - hypergeom.cdf(
        lower - 1, pool_size, positive_count, budget
    )
    sf_interval = hypergeom.sf(lower - 1, pool_size, positive_count, budget) - hypergeom.sf(
        upper, pool_size, positive_count, budget
    )
    return float(np.clip(max(cdf_interval, sf_interval), 0.0, 1.0))


def binomial_support_approximation(
    pool_size: int, positive_count: int, budget: int, min_positive: int, min_negative: int
) -> float:
    """Large-population with-replacement approximation, always reported separately."""
    _validate(pool_size, positive_count, budget)
    _validate_min_counts(min_positive, min_negative)
    lower = int(min_positive)
    upper = budget - int(min_negative)
    if upper < lower:
        return 0.0
    p = positive_count / pool_size
    return float(binom.cdf(upper, budget, p) - binom.cdf(lower - 1, budget, p))


def minimum_budget_for_support_probability(
    pool_size: int, positive_count: int, min_positive: int, min_negative: int,
    target_probability: float, *, maximum_budget: int | None = None,
) -> int | None:
    _validate(pool_size, positive_count, 0)
    _validate_min_counts(min_positive, min_negative)
    if not 0 <= target_probability <= 1:
        raise ValueError("target_probability must be in [0,1]")
    maximum = pool_size if maximum_budget is None else min(maximum_budget, pool_size)
    for budget in range(min_positive + min_negative, maximum + 1):
        if hypergeometric_support_probability(
            pool_size, positive_count, budget, min_positive, min_negative
        ) >= target_probability:
            return budget
    return None


def mondrian_finite_threshold_probability(
    pool_size: int, positive_count: int, budget: int, *, alpha: float = 0.10
) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    # ceil((n_class+1)*(1-alpha)) <= n_class; solve conservatively by enumeration.
    minimum = next(
        (n for n in range(1, pool_size + 1) if np.ceil((n + 1) * (1 - alpha)) <= n),
        None,
    )
    if minimum is None:
        return 0.0
    return hypergeometric_support_probability(
        pool_size, positive_count, budget, minimum, minimum
    )


def _counts_for_prevalence_interval(
    pool_size: int, prevalence_lower: float, prevalence_upper: float
) -> range:
    if not 0 <= prevalence_lower <= prevalence_upper <= 1:
        raise ValueError("prevalence bounds must satisfy 0 <= lower <= upper <= 1")
    lower_count = int(np.ceil(pool_size * prevalence_lower))
    upper_count = int(np.floor(pool_size * prevalence_upper))
    if upper_count < lower_count:
        raise ValueError("prevalence interval contains no attainable finite-population count")
    return range(lower_count, upper_count + 1)


def support_probability_over_prevalence_interval(
    pool_size: int,
    prevalence_lower: float,
    prevalence_upper: float,
    budget: int,
    min_positive: int,
    min_negative: int,
) -> float:
    """Worst-case exact support probability over an attainable prevalence interval.

    This is a planning calculation, not an oracle using the observed target labels.
    """
    _validate(pool_size, 0, budget)
    _validate_min_counts(min_positive, min_negative)
    probabilities = [
        hypergeometric_support_probability(
            pool_size, count, budget, min_positive, min_negative
        )
        for count in _counts_for_prevalence_interval(
            pool_size, prevalence_lower, prevalence_upper
        )
    ]
    return float(min(probabilities))


def minimum_budget_over_prevalence_interval(
    pool_size: int,
    prevalence_lower: float,
    prevalence_upper: float,
    min_positive: int,
    min_negative: int,
    target_probability: float,
    *,
    maximum_budget: int | None = None,
) -> int | None:
    """Minimum budget meeting a worst-case prevalence-range planning target."""
    _validate_min_counts(min_positive, min_negative)
    if not 0 <= target_probability <= 1:
        raise ValueError("target_probability must be in [0,1]")
    maximum = pool_size if maximum_budget is None else min(maximum_budget, pool_size)
    for budget in range(min_positive + min_negative, maximum + 1):
        if support_probability_over_prevalence_interval(
            pool_size,
            prevalence_lower,
            prevalence_upper,
            budget,
            min_positive,
            min_negative,
        ) >= target_probability:
            return budget
    return None


def class_support_probability_grid(
    pool_size: int,
    positive_counts: list[int] | np.ndarray,
    budgets: list[int] | np.ndarray,
    min_positive: int,
    min_negative: int,
) -> list[dict[str, int | float | str]]:
    """Return explicit true-composition oracle values for an audit-friendly grid."""
    rows: list[dict[str, int | float | str]] = []
    for positive_count in positive_counts:
        for budget in budgets:
            rows.append(
                {
                    "reference_type": "true_composition_reference",
                    "pool_size": int(pool_size),
                    "positive_count": int(positive_count),
                    "budget": int(budget),
                    "min_positive": int(min_positive),
                    "min_negative": int(min_negative),
                    "support_probability": hypergeometric_support_probability(
                        pool_size,
                        int(positive_count),
                        int(budget),
                        min_positive,
                        min_negative,
                    ),
                }
            )
    return rows


@dataclass(frozen=True)
class MonteCarloValidation:
    exact_probability: float
    empirical_probability: float
    absolute_error: float
    repetitions: int


def monte_carlo_validation(
    pool_size: int, positive_count: int, budget: int, min_positive: int, min_negative: int,
    *, repetitions: int = 100_000, seed: int = 0,
) -> MonteCarloValidation:
    _validate(pool_size, positive_count, budget)
    rng = np.random.default_rng(seed)
    counts = rng.hypergeometric(positive_count, pool_size - positive_count, budget, repetitions)
    empirical = float(((counts >= min_positive) & (counts <= budget - min_negative)).mean())
    exact = hypergeometric_support_probability(
        pool_size, positive_count, budget, min_positive, min_negative
    )
    return MonteCarloValidation(exact, empirical, abs(exact - empirical), repetitions)


__all__ = [
    "hypergeometric_support_probability", "binomial_support_approximation",
    "minimum_budget_for_support_probability", "mondrian_finite_threshold_probability",
    "support_probability_over_prevalence_interval",
    "minimum_budget_over_prevalence_interval", "class_support_probability_grid",
    "monte_carlo_validation", "MonteCarloValidation",
]
