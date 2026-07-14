from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reliable_shift.calibration import (
    CALIBRATION_METHOD_ALIASES,
    FORMAL_CALIBRATION_METHODS,
    canonical_calibration_name,
    isotonic_preflight,
    make_calibrator,
)
from reliable_shift.stage2.schema_v2 import (
    ALLOWED_STATUSES,
    CONFORMAL_KEY,
    PROBABILITY_KEY,
    SCHEMA_VERSION,
    THRESHOLD_KEY,
    empty_frame,
    validate_frame,
)


def _valid_row(kind: str) -> dict[str, object]:
    row = {column: 0 for column in empty_frame(kind).columns}
    row.update(
        dataset="example",
        source_family="synthetic",
        entity_unit="entity_id",
        split_seed=0,
        base_model="logistic_regression",
        model_seed=0,
        acquisition_strategy="label_blind_random",
        adaptation_seed=-1,
        budget=0,
        fit_domain="source",
        calibration_method="sigmoid",
        schema_version=SCHEMA_VERSION,
        status="success",
        status_reason="",
        config_hash="config",
        data_hash="data",
        split_hash="split",
        sample_hash="sample",
        commit="a" * 40,
        timestamp_utc="2026-07-12T00:00:00Z",
    )
    if "evaluation_split" in row:
        row["evaluation_split"] = "target_final_test"
    if kind in {"conformal", "threshold"}:
        row.update(conformal_method="standard", score_source="base_uncalibrated_probability")
    return row


@pytest.mark.parametrize("kind", ["probability", "conformal", "threshold"])
def test_stage2_v2_schema_accepts_canonical_row(kind: str) -> None:
    validate_frame(pd.DataFrame([_valid_row(kind)]), kind)


def test_stage2_v2_primary_keys_cover_frozen_dimensions() -> None:
    expected = {
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
    }
    assert expected <= set(PROBABILITY_KEY)
    assert {"conformal_method", "score_source"} <= set(CONFORMAL_KEY)
    assert {"conformal_method", "score_source"} <= set(THRESHOLD_KEY)
    assert "stable_support_5_per_class" in empty_frame("probability")


def test_stage2_v2_rejects_duplicate_keys_and_old_method_alias() -> None:
    row = _valid_row("probability")
    with pytest.raises(ValueError, match="duplicate"):
        validate_frame(pd.DataFrame([row, row]), "probability")
    row["calibration_method"] = "intercept_only"
    with pytest.raises(ValueError, match="non-canonical"):
        validate_frame(pd.DataFrame([row]), "probability")


def test_stage2_v2_status_vocabulary_and_reason_are_enforced() -> None:
    assert ALLOWED_STATUSES == {"success", "not_available", "not_estimable", "failed"}
    row = _valid_row("probability")
    row.update(status="skipped", status_reason="legacy value")
    with pytest.raises(ValueError, match="Invalid Stage 2 status"):
        validate_frame(pd.DataFrame([row]), "probability")
    row.update(status="not_estimable", status_reason="")
    with pytest.raises(ValueError, match="status_reason"):
        validate_frame(pd.DataFrame([row]), "probability")


def test_formal_calibration_names_and_legacy_alias_mapping() -> None:
    assert set(FORMAL_CALIBRATION_METHODS) == {
        "uncalibrated",
        "intercept_only_mle",
        "sigmoid",
        "isotonic",
        "jeffreys_smoothed_intercept_matching",
    }
    assert CALIBRATION_METHOD_ALIASES == {"intercept_only": "intercept_only_mle"}
    assert canonical_calibration_name("intercept_only") == "intercept_only_mle"
    assert make_calibrator("intercept_only").method_name == "intercept_only_mle"
    assert make_calibrator("intercept_only_mle").method_name == "intercept_only_mle"


def test_sigmoid_reports_five_per_class_stability_diagnostic() -> None:
    probability = np.linspace(0.05, 0.95, 10)
    target = np.tile([0, 1], 5)
    fitted = make_calibrator("sigmoid").fit(probability, target)
    diagnostics = fitted.diagnostics()
    assert diagnostics["stable_support_5_per_class"] is True
    assert diagnostics["fitted_calibrator_slope"] is not None


def test_isotonic_preflight_frozen_boundaries() -> None:
    probability = np.linspace(0.01, 0.99, 25)
    target = np.tile([0, 1], 13)[:25]
    eligible = isotonic_preflight(probability, target)
    assert eligible["eligible"] is True
    assert eligible["status"] == "success"

    too_small = isotonic_preflight(probability[:24], target[:24])
    assert too_small["eligible"] is False
    assert "n=24 < 25" in str(too_small["status_reason"])

    rare_class = isotonic_preflight(probability, np.array([1] * 4 + [0] * 21))
    assert rare_class["eligible"] is False
    assert "class1=4 < 5" in str(rare_class["status_reason"])

    tied_scores = isotonic_preflight(np.arange(25) % 9, target)
    assert tied_scores["eligible"] is False
    assert "unique_input_scores=9 < 10" in str(tied_scores["status_reason"])


def test_make_isotonic_enforces_frozen_preflight() -> None:
    with pytest.raises(ValueError, match="not estimable"):
        make_calibrator("isotonic").fit(
            np.linspace(0.01, 0.99, 24), np.tile([0, 1], 12)
        )
    fitted = make_calibrator("isotonic").fit(
        np.linspace(0.01, 0.99, 25), np.tile([0, 1], 13)[:25]
    )
    assert fitted.diagnostics()["preflight_eligible"] is True


def test_smoothed_intercept_marks_single_class_sensitivity_fit() -> None:
    fitted = make_calibrator("jeffreys_smoothed_intercept_matching").fit(
        np.linspace(0.1, 0.9, 8), np.zeros(8, dtype=int)
    )
    assert fitted.diagnostics()["single_class_calibration"] is True
