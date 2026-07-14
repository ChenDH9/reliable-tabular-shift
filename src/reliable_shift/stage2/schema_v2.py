"""Canonical Stage 2 result schemas.

The schemas in this module are the interchange contract used by the formal
runner, shard merge, qualification checks, and lean handoff.  They deliberately
keep status rows in the same tables as successful rows: an unavailable budget
or an inestimable method is scientific evidence, not a missing result.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from reliable_shift.calibration import canonical_calibration_name

SCHEMA_VERSION = "stage2_v2"

ALLOWED_STATUSES = frozenset({"success", "not_available", "not_estimable", "failed"})

PROBABILITY_KEY = (
    "dataset",
    "source_family",
    "entity_unit",
    "split_seed",
    "base_model",
    "model_seed",
    "acquisition_strategy",
    "adaptation_seed",
    "budget",
    "fit_domain",
    "calibration_method",
    "evaluation_split",
)

CONFORMAL_KEY = PROBABILITY_KEY + ("conformal_method", "score_source")

# A threshold is fitted once and can then be evaluated on more than one split.
# Consequently ``evaluation_split`` is intentionally not part of this key.
THRESHOLD_KEY = tuple(column for column in PROBABILITY_KEY if column != "evaluation_split") + (
    "conformal_method",
    "score_source",
)

PROBABILITY_COLUMNS = PROBABILITY_KEY + (
    "schema_version",
    "status",
    "status_reason",
    "actual_entities",
    "positive_labels",
    "negative_labels",
    "n_eval",
    "positive_eval",
    "negative_eval",
    "evaluation_prevalence",
    "auroc",
    "auprc",
    "brier",
    "log_loss",
    "brier_skill_score",
    "log_loss_skill_score",
    "calibration_intercept",
    "calibration_slope",
    "abs_calibration_intercept",
    "abs_calibration_slope_error",
    "ece_15",
    "fitted_calibrator_intercept",
    "fitted_calibrator_slope",
    "unique_input_scores",
    "unique_output_scores",
    "produced_exact_zero",
    "produced_exact_one",
    "single_class_calibration",
    "stable_support_5_per_class",
    "runtime_seconds",
    "config_hash",
    "data_hash",
    "split_hash",
    "sample_hash",
    "commit",
    "timestamp_utc",
)

CONFORMAL_COLUMNS = CONFORMAL_KEY + (
    "schema_version",
    "status",
    "status_reason",
    "actual_entities",
    "positive_labels",
    "negative_labels",
    "n_eval",
    "positive_eval",
    "negative_eval",
    "evaluation_prevalence",
    "marginal_coverage",
    "class0_coverage",
    "class1_coverage",
    "worst_class_coverage",
    "worst_class_undercoverage",
    "absolute_coverage_error",
    "mean_set_size",
    "empty_set_fraction",
    "singleton_fraction",
    "doubleton_fraction",
    "estimable",
    "finite_threshold",
    "runtime_seconds",
    "config_hash",
    "data_hash",
    "split_hash",
    "sample_hash",
    "commit",
    "timestamp_utc",
)

THRESHOLD_COLUMNS = THRESHOLD_KEY + (
    "schema_version",
    "status",
    "status_reason",
    "standard_threshold",
    "class0_threshold",
    "class1_threshold",
    "standard_calibration_count",
    "class0_calibration_count",
    "class1_calibration_count",
    "standard_threshold_infinite",
    "class0_threshold_infinite",
    "class1_threshold_infinite",
    "all_thresholds_finite",
    "config_hash",
    "data_hash",
    "split_hash",
    "sample_hash",
    "commit",
    "timestamp_utc",
)

_COLUMNS_BY_KIND = {
    "probability": PROBABILITY_COLUMNS,
    "conformal": CONFORMAL_COLUMNS,
    "threshold": THRESHOLD_COLUMNS,
}
_KEY_BY_KIND = {
    "probability": PROBABILITY_KEY,
    "conformal": CONFORMAL_KEY,
    "threshold": THRESHOLD_KEY,
}


def _columns(kind: str) -> tuple[str, ...]:
    try:
        return _COLUMNS_BY_KIND[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown Stage 2 schema kind: {kind!r}") from exc


def empty_frame(kind: str) -> pd.DataFrame:
    """Return an empty frame carrying every required column for ``kind``."""

    return pd.DataFrame(columns=list(_columns(kind)))


# Explicit alias for callers that also use the legacy ``schemas.empty_frame``.
empty_frame_v2 = empty_frame


def validate_statuses(statuses: Iterable[object]) -> None:
    """Reject missing or non-canonical Stage 2 result statuses."""

    observed = set(statuses)
    invalid = {value for value in observed if pd.isna(value) or value not in ALLOWED_STATUSES}
    if invalid:
        raise ValueError(f"Invalid Stage 2 status values: {sorted(map(str, invalid))}")


def validate_frame(frame: pd.DataFrame, kind: str) -> None:
    """Validate required fields, primary-key uniqueness, version, and statuses.

    Additional columns are permitted so that runner-specific diagnostics can be
    retained without weakening the stable interchange schema.
    """

    required = set(_columns(kind))
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{kind} stage2_v2 schema missing columns: {sorted(missing)}")

    key = list(_KEY_BY_KIND[kind])
    if len(frame) == 0:
        return
    if frame[key].isna().any(axis=None):
        null_columns = frame[key].columns[frame[key].isna().any()].tolist()
        raise ValueError(f"{kind} contains null primary-key fields: {null_columns}")
    if frame.duplicated(key).any():
        raise ValueError(f"{kind} contains duplicate stage2_v2 primary keys")

    versions = set(frame["schema_version"].dropna().astype(str))
    if versions != {SCHEMA_VERSION} or frame["schema_version"].isna().any():
        raise ValueError(
            f"{kind} schema_version must be exactly {SCHEMA_VERSION!r}; observed {sorted(versions)}"
        )
    validate_statuses(frame["status"].tolist())

    non_success = frame["status"].ne("success")
    missing_reason = (
        frame["status_reason"].isna()
        | frame["status_reason"].astype(str).str.strip().eq("")
    )
    if (non_success & missing_reason).any():
        raise ValueError("Non-success rows require a non-empty status_reason")

    # Old aliases remain accepted by make_calibrator(), but never in formal rows.
    from reliable_shift.calibration import FORMAL_CALIBRATION_METHODS

    invalid_methods = set(frame["calibration_method"].dropna()) - set(
        FORMAL_CALIBRATION_METHODS
    )
    if invalid_methods:
        raise ValueError(
            "Formal results contain non-canonical calibration methods: "
            f"{sorted(map(str, invalid_methods))}"
        )


# Compatibility spelling used by earlier infrastructure modules.
validate_schema = validate_frame


__all__ = [
    "ALLOWED_STATUSES",
    "CONFORMAL_COLUMNS",
    "CONFORMAL_KEY",
    "PROBABILITY_COLUMNS",
    "PROBABILITY_KEY",
    "SCHEMA_VERSION",
    "THRESHOLD_COLUMNS",
    "THRESHOLD_KEY",
    "empty_frame",
    "empty_frame_v2",
    "canonical_calibration_name",
    "validate_frame",
    "validate_schema",
    "validate_statuses",
]
