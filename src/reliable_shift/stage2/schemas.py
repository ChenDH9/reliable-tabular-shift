from __future__ import annotations

import pandas as pd

PROBABILITY_KEY = ["dataset", "split_seed", "base_model", "model_seed", "adaptation_seed", "budget", "method"]
CONFORMAL_KEY = PROBABILITY_KEY + ["conformal_method", "score_source"]
THRESHOLD_KEY = ["dataset", "split_seed", "base_model", "model_seed", "adaptation_seed", "budget", "conformal_method", "score_source"]

PROBABILITY_COLUMNS = PROBABILITY_KEY + ["status", "error_message", "n_test", "auroc", "brier", "log_loss", "ece", "runtime_seconds", "config_hash", "data_hash", "commit"]
CONFORMAL_COLUMNS = CONFORMAL_KEY + ["status", "error_message", "n_test", "marginal_coverage", "class0_coverage", "class1_coverage", "mean_set_size", "singleton_rate", "empty_rate", "runtime_seconds", "config_hash", "data_hash", "commit"]
THRESHOLD_COLUMNS = THRESHOLD_KEY + ["standard_threshold", "class0_threshold", "class1_threshold", "standard_calibration_count", "class0_calibration_count", "class1_calibration_count", "standard_threshold_infinite", "class0_threshold_infinite", "class1_threshold_infinite", "all_thresholds_finite", "config_hash", "data_hash", "commit"]


def empty_frame(kind: str) -> pd.DataFrame:
    columns = {"probability": PROBABILITY_COLUMNS, "conformal": CONFORMAL_COLUMNS, "threshold": THRESHOLD_COLUMNS}[kind]
    return pd.DataFrame(columns=columns)


def validate_schema(frame: pd.DataFrame, kind: str) -> None:
    expected = set(empty_frame(kind).columns)
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{kind} schema missing columns: {sorted(missing)}")
    key = {"probability": PROBABILITY_KEY, "conformal": CONFORMAL_KEY, "threshold": THRESHOLD_KEY}[kind]
    if frame.duplicated(key).any():
        raise ValueError(f"{kind} contains duplicate primary keys")

