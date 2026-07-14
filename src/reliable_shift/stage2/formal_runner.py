"""Production Stage 2 shard runner used for qualification and formal execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from reliable_shift.calibration import make_calibrator
from reliable_shift.conformal import fit_thresholds, prediction_sets
from reliable_shift.estimability import (
    hypergeometric_support_probability,
    mondrian_finite_threshold_probability,
)
from reliable_shift.metrics import conformal_metrics, probability_metrics
from reliable_shift.metrics.core import expected_calibration_error
from reliable_shift.models.factory import tune_model
from reliable_shift.preprocessing import prepare_features
from reliable_shift.stage2.acquisition import canonical_blind_budget_samples
from reliable_shift.stage2.loaders import load_registry
from reliable_shift.stage2.splits import capped_role_frames

SCHEMA_VERSION = "stage2_v2"
ENTITY_UNITS = {
    "acsincome_region": "person",
    "acsfoodstamps_household": "household",
    "brfss_diabetes": "respondent",
    "nhanes_lead": "respondent",
    "diabetes_readmission_patient_index": "patient",
    "college_scorecard_2018": "institution",
}
NUMERIC_COLUMNS = {
    "acsincome_region": ["AGEP", "WKHP", "WKW"],
    "acsfoodstamps_household": ["AGEP", "WKHP", "WKW"],
    "brfss_diabetes": ["PHYSHLTH", "MENTHLTH", "BMI5"],
    "nhanes_lead": ["RIDAGEYR"],
    "diabetes_readmission_patient_index": [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ],
    "college_scorecard_2018": [
        "NUMBRANCH",
        "ADM_RATE",
        "ADM_RATE_ALL",
        "SATVRMID",
        "SATMTMID",
        "ACTCMMID",
        "UGDS",
        "UGDS_WHITE",
        "UGDS_BLACK",
        "UGDS_HISP",
        "UGDS_ASIAN",
        "UGDS_AIAN",
        "UGDS_NHPI",
        "UGDS_2MOR",
        "UGDS_NRA",
        "UGDS_UNKN",
        "PPTUG_EF",
        "COSTT4_A",
        "COSTT4_P",
        "TUITIONFEE_IN",
        "TUITIONFEE_OUT",
        "TUITFTE",
        "INEXPFTE",
        "AVGFACSAL",
        "PFTFAC",
        "PCTPELL",
    ],
}
FORMAL_METHODS = (
    "intercept_only_mle",
    "sigmoid",
    "isotonic",
    "jeffreys_smoothed_intercept_matching",
)


class NotEstimable(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def shard_directory(
    root: Path, dataset: str, split_seed: int, base_model: str, model_seed: int
) -> Path:
    return root / dataset / f"split_{split_seed}" / base_model / f"model_seed_{model_seed}"


def _schema_module():
    from reliable_shift.stage2 import schema_v2

    return schema_v2


def _validate(frame: pd.DataFrame, kind: str) -> None:
    module = _schema_module()
    if hasattr(module, "validate_frame"):
        module.validate_frame(frame, kind)
    else:
        module.validate_schema(frame, kind)


def _columns(kind: str) -> list[str]:
    module = _schema_module()
    return list(getattr(module, f"{kind.upper()}_COLUMNS"))


def _canonical_method(name: str) -> str:
    module = _schema_module()
    return module.canonical_calibration_name(name)


def _base_key(
    *,
    dataset: str,
    source_family: str,
    split_seed: int,
    base_model: str,
    model_seed: int,
    adaptation_seed: int,
    budget: int,
    fit_domain: str,
    method: str,
    evaluation_split: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "source_family": source_family,
        "entity_unit": ENTITY_UNITS[dataset],
        "split_seed": int(split_seed),
        "base_model": base_model,
        "model_seed": int(model_seed),
        "acquisition_strategy": "blind_random",
        "adaptation_seed": int(adaptation_seed),
        "budget": int(budget),
        "fit_domain": fit_domain,
        "calibration_method": _canonical_method(method),
        "evaluation_split": evaluation_split,
    }


def _empty_payload(kind: str) -> dict[str, Any]:
    return {column: np.nan for column in _columns(kind)}


def _diagnostic_values(calibrator, scores: np.ndarray, calibrated: np.ndarray) -> dict[str, Any]:
    diagnostics = calibrator.diagnostics() if hasattr(calibrator, "diagnostics") else {}
    intercept = diagnostics.get("fitted_intercept", np.nan)
    slope = np.nan
    model = getattr(calibrator, "model", None)
    if model is not None and hasattr(model, "intercept_"):
        intercept = float(np.ravel(model.intercept_)[0])
        slope = float(np.ravel(model.coef_)[0])
    elif calibrator.__class__.__name__.endswith("InterceptCalibrator"):
        slope = 1.0
    return {
        "fitted_calibrator_intercept": intercept,
        "fitted_calibrator_slope": slope,
        "unique_input_scores": int(np.unique(scores).size),
        "unique_output_scores": int(np.unique(calibrated).size),
        "produced_exact_zero": bool(np.any(calibrated == 0)),
        "produced_exact_one": bool(np.any(calibrated == 1)),
        "stable_support_5_per_class": diagnostics.get("stable_support_5_per_class", np.nan),
    }


def _check_method_estimable(method: str, scores: np.ndarray, target: np.ndarray) -> None:
    counts = np.bincount(np.asarray(target, dtype=int), minlength=2)
    method = _canonical_method(method)
    if method in {"intercept_only_mle", "sigmoid"} and np.any(counts == 0):
        raise NotEstimable(f"{method} requires both classes")
    if method == "isotonic":
        if len(target) < 25:
            raise NotEstimable("isotonic requires n >= 25")
        if np.any(counts < 5):
            raise NotEstimable("isotonic requires at least 5 labels per class")
        if np.unique(scores).size < 10:
            raise NotEstimable("isotonic requires at least 10 unique input scores")


def _probability_row(
    key: dict[str, Any],
    *,
    fit_scores: np.ndarray,
    fit_y: np.ndarray,
    eval_scores: np.ndarray,
    eval_y: np.ndarray,
    method: str,
    hashes: dict[str, str],
    commit: str,
) -> dict[str, Any]:
    row = _empty_payload("probability") | key
    started = time.perf_counter()
    counts = np.bincount(np.asarray(fit_y, dtype=int), minlength=2)
    row.update(
        status="success",
        status_reason="",
        actual_entities=int(len(fit_y)),
        positive_labels=int(counts[1]),
        negative_labels=int(counts[0]),
        n_eval=int(len(eval_y)),
        positive_eval=int(np.sum(eval_y)),
        negative_eval=int(len(eval_y) - np.sum(eval_y)),
        evaluation_prevalence=float(np.mean(eval_y)),
        single_class_calibration=bool(np.unique(fit_y).size < 2) if len(fit_y) else False,
        **hashes,
        commit=commit,
        timestamp_utc=utc_now(),
    )
    try:
        if method == "uncalibrated":
            calibrated = np.asarray(eval_scores, dtype=float)
            diagnostics = {
                "fitted_calibrator_intercept": np.nan,
                "fitted_calibrator_slope": np.nan,
                "unique_input_scores": int(np.unique(eval_scores).size),
                "unique_output_scores": int(np.unique(eval_scores).size),
                "produced_exact_zero": bool(np.any(eval_scores == 0)),
                "produced_exact_one": bool(np.any(eval_scores == 1)),
                "stable_support_5_per_class": np.nan,
            }
        else:
            _check_method_estimable(method, fit_scores, fit_y)
            calibrator = make_calibrator(method).fit(fit_scores, fit_y)
            calibrated = calibrator.predict(eval_scores)
            diagnostics = _diagnostic_values(calibrator, fit_scores, calibrator.predict(fit_scores))
        metrics = probability_metrics(eval_y, calibrated, bins=15)
        row.update(
            auroc=metrics["auroc"],
            auprc=metrics["auprc"],
            brier=metrics["brier"],
            log_loss=metrics["log_loss"],
            brier_skill_score=metrics["brier_skill_score"],
            log_loss_skill_score=metrics["log_loss_skill_score"],
            calibration_intercept=metrics["calibration_intercept"],
            calibration_slope=metrics["calibration_slope"],
            abs_calibration_intercept=abs(metrics["calibration_intercept"]),
            abs_calibration_slope_error=abs(metrics["calibration_slope"] - 1),
            ece_15=expected_calibration_error(eval_y, calibrated, bins=15),
            **diagnostics,
        )
    except NotEstimable as exc:
        row.update(status="not_estimable", status_reason=str(exc))
    except Exception as exc:
        row.update(status="failed", status_reason=f"{type(exc).__name__}: {exc}")
    row["runtime_seconds"] = time.perf_counter() - started
    return row


def _status_probability_row(
    key: dict[str, Any],
    status: str,
    reason: str,
    actual: int,
    positive: int,
    hashes: dict[str, str],
    commit: str,
) -> dict[str, Any]:
    row = _empty_payload("probability") | key
    row.update(
        status=status,
        status_reason=reason,
        actual_entities=int(actual),
        positive_labels=int(positive),
        negative_labels=int(actual - positive),
        single_class_calibration=bool(actual and positive in {0, actual}),
        runtime_seconds=0.0,
        **hashes,
        commit=commit,
        timestamp_utc=utc_now(),
    )
    return row


def _threshold_values(fitted, fit_y: np.ndarray) -> dict[str, Any]:
    standard = fitted.q_global if fitted.q_global is not None else np.nan
    class0 = fitted.q_by_class.get(0, np.nan) if fitted.q_by_class else np.nan
    class1 = fitted.q_by_class.get(1, np.nan) if fitted.q_by_class else np.nan
    values = [value for value in (standard, class0, class1) if not pd.isna(value)]
    return {
        "standard_threshold": standard,
        "class0_threshold": class0,
        "class1_threshold": class1,
        "standard_calibration_count": int(len(fit_y)),
        "class0_calibration_count": int(np.sum(fit_y == 0)),
        "class1_calibration_count": int(np.sum(fit_y == 1)),
        "standard_threshold_infinite": bool(np.isinf(standard)),
        "class0_threshold_infinite": bool(np.isinf(class0)),
        "class1_threshold_infinite": bool(np.isinf(class1)),
        "all_thresholds_finite": bool(values and np.all(np.isfinite(values))),
    }


def _conformal_rows(
    key: dict[str, Any],
    *,
    fit_scores: np.ndarray,
    fit_y: np.ndarray,
    eval_scores: np.ndarray,
    eval_y: np.ndarray,
    conformal_method: str,
    score_source: str,
    hashes: dict[str, str],
    commit: str,
    alpha: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    extended = key | {"conformal_method": conformal_method, "score_source": score_source}
    row = _empty_payload("conformal") | extended
    threshold = _empty_payload("threshold") | extended
    started = time.perf_counter()
    counts = np.bincount(np.asarray(fit_y, dtype=int), minlength=2)
    common = {
        "status": "success",
        "status_reason": "",
        "actual_entities": int(len(fit_y)),
        "positive_labels": int(counts[1]),
        "negative_labels": int(counts[0]),
        **hashes,
        "commit": commit,
        "timestamp_utc": utc_now(),
    }
    row.update(
        common,
        n_eval=int(len(eval_y)),
        positive_eval=int(np.sum(eval_y)),
        negative_eval=int(len(eval_y) - np.sum(eval_y)),
        evaluation_prevalence=float(np.mean(eval_y)),
    )
    threshold.update(common)
    try:
        if conformal_method == "mondrian" and np.any(counts < 5):
            raise NotEstimable("Mondrian requires at least 5 labels per class")
        fitted = fit_thresholds(
            fit_scores, fit_y, method=conformal_method, alpha=alpha, min_class_count=5
        )
        sets = prediction_sets(eval_scores, fitted)
        metrics = conformal_metrics(eval_y, sets, 1 - alpha)
        thresholds = _threshold_values(fitted, fit_y)
        finite = bool(thresholds["all_thresholds_finite"])
        row.update(
            marginal_coverage=metrics["marginal_coverage"],
            class0_coverage=metrics["class0_coverage"],
            class1_coverage=metrics["class1_coverage"],
            worst_class_coverage=metrics["worst_class_coverage"],
            worst_class_undercoverage=max(0.0, (1 - alpha) - metrics["worst_class_coverage"]),
            absolute_coverage_error=metrics["absolute_coverage_error"],
            mean_set_size=metrics["mean_set_size"],
            empty_set_fraction=metrics["empty_set_fraction"],
            singleton_fraction=metrics["singleton_fraction"],
            doubleton_fraction=metrics["doubleton_fraction"],
            estimable=True,
            finite_threshold=finite,
        )
        threshold.update(thresholds, estimable=True)
    except NotEstimable as exc:
        row.update(
            status="not_estimable", status_reason=str(exc), estimable=False, finite_threshold=False
        )
        threshold.update(
            status="not_estimable",
            status_reason=str(exc),
            estimable=False,
            **_threshold_values_empty(fit_y),
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        row.update(status="failed", status_reason=reason, estimable=False, finite_threshold=False)
        threshold.update(
            status="failed", status_reason=reason, estimable=False, **_threshold_values_empty(fit_y)
        )
    row["runtime_seconds"] = time.perf_counter() - started
    threshold["runtime_seconds"] = row["runtime_seconds"]
    return row, threshold


def _threshold_values_empty(fit_y: np.ndarray) -> dict[str, Any]:
    return {
        "standard_threshold": np.nan,
        "class0_threshold": np.nan,
        "class1_threshold": np.nan,
        "standard_calibration_count": int(len(fit_y)),
        "class0_calibration_count": int(np.sum(fit_y == 0)),
        "class1_calibration_count": int(np.sum(fit_y == 1)),
        "standard_threshold_infinite": False,
        "class0_threshold_infinite": False,
        "class1_threshold_infinite": False,
        "all_thresholds_finite": False,
    }


def _status_conformal_rows(
    key: dict[str, Any],
    *,
    conformal_method: str,
    score_source: str,
    status: str,
    reason: str,
    actual: int,
    positive: int,
    hashes: dict[str, str],
    commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    extended = key | {"conformal_method": conformal_method, "score_source": score_source}
    common = {
        "status": status,
        "status_reason": reason,
        "actual_entities": int(actual),
        "positive_labels": int(positive),
        "negative_labels": int(actual - positive),
        "estimable": False,
        "runtime_seconds": 0.0,
        **hashes,
        "commit": commit,
        "timestamp_utc": utc_now(),
    }
    row = _empty_payload("conformal") | extended | common | {"finite_threshold": False}
    threshold = (
        _empty_payload("threshold")
        | extended
        | common
        | _threshold_values_empty(np.asarray([], dtype=int))
    )
    threshold.update(
        standard_calibration_count=int(actual),
        class0_calibration_count=int(actual - positive),
        class1_calibration_count=int(positive),
    )
    return row, threshold


def _predict(model, frame: pd.DataFrame, features: list[str], numeric: list[str]) -> np.ndarray:
    return np.asarray(
        model.predict_proba(prepare_features(frame, features, numeric))[:, 1], dtype=float
    )


def _load_inputs(
    config: dict[str, Any], dataset: str
) -> tuple[pd.Series, pd.DataFrame, list[str], list[str]]:
    registry = load_registry(Path(config["data_registry"]))
    matches = registry.loc[registry["dataset"].eq(dataset)]
    if len(matches) != 1:
        raise ValueError(f"registry must contain exactly one {dataset} row")
    row = matches.iloc[0]
    path = Path(row["audited_parquet_path"])
    actual_hash = sha256_file(path)
    if actual_hash != row["processed_checksum"]:
        raise ValueError(f"processed checksum mismatch for {dataset}")
    frame = pd.read_parquet(path)
    features = json.loads(row["feature_list"])
    numeric = [name for name in NUMERIC_COLUMNS[dataset] if name in features]
    required = {"entity_id", "target", "is_target_domain", *features}
    if not required.issubset(frame):
        raise ValueError(f"{dataset} missing columns: {sorted(required - set(frame))}")
    return row, frame, features, numeric


def _verify_frozen_sample(
    manifest: pd.DataFrame,
    *,
    dataset: str,
    split_seed: int,
    adaptation_seed: int,
    budget: int,
    sample: dict[str, Any],
) -> None:
    match = manifest.loc[
        manifest["dataset"].eq(dataset)
        & manifest["split_seed"].eq(split_seed)
        & manifest["adaptation_seed"].eq(adaptation_seed)
        & manifest["budget"].eq(budget)
    ]
    if len(match) != 1:
        raise ValueError("frozen budget manifest row missing or duplicated")
    row = match.iloc[0]
    if str(row["selected_entity_prefix_digest"]) != sample["sample_hash"]:
        raise ValueError("formal runner sample hash differs from frozen budget manifest")
    if str(row["adaptation_pool_digest"]) != sample["pool_hash"]:
        raise ValueError("formal runner pool hash differs from frozen budget manifest")
    if int(row["selected_count"]) != int(sample["selected_count"]):
        raise ValueError("formal runner selected count differs from frozen budget manifest")


def compute_shard(
    *,
    config: dict[str, Any],
    config_path: Path,
    dataset: str,
    split_seed: int,
    base_model: str,
    model_seed: int,
    commit: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    started = time.perf_counter()
    registry_row, frame, features, numeric = _load_inputs(config, dataset)
    data_hash = str(registry_row["processed_checksum"])
    roles, prevalence = capped_role_frames(frame, split_seed, config["role_caps"])
    prevalence.insert(0, "dataset", dataset)
    prevalence.insert(1, "source_family", registry_row["source_family"])
    if any(
        roles[role]["target"].nunique() < 2
        for role in (
            "source_probability_calibration",
            "source_conformal_calibration",
            "source_id_test",
        )
    ):
        raise ValueError("source calibration/test role lacks a class")
    split_hash = sha256_json(
        {
            role: audit
            for role, audit in zip(prevalence["role"], prevalence["role_hash"], strict=True)
        }
    )
    model_config = config["models"][base_model]
    model, tuning_records, tuning_seconds = tune_model(
        model_name=base_model,
        model_config=model_config,
        train=roles["source_train"],
        validation=roles["source_tune"],
        feature_columns=features,
        numeric_columns=numeric,
        seed=model_seed,
    )
    probabilities = {
        role: _predict(model, role_frame, features, numeric) for role, role_frame in roles.items()
    }
    labels = {
        role: role_frame["target"].astype(int).to_numpy() for role, role_frame in roles.items()
    }
    source_sigmoid = make_calibrator("sigmoid").fit(
        probabilities["source_probability_calibration"], labels["source_probability_calibration"]
    )
    source_sigmoid_probabilities = {
        role: source_sigmoid.predict(values) for role, values in probabilities.items()
    }
    config_hash = sha256_file(config_path)
    base_hashes = {
        "config_hash": config_hash,
        "data_hash": data_hash,
        "split_hash": split_hash,
        "sample_hash": hashlib.sha256(b"").hexdigest(),
    }
    source_family = str(registry_row["source_family"])
    probability_rows: list[dict[str, Any]] = []
    conformal_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    estimability_rows: list[dict[str, Any]] = []

    # Budget-0 base probability is stored once per evaluation split.
    for evaluation_split in ("source_id_test", "target_final_test"):
        key = _base_key(
            dataset=dataset,
            source_family=source_family,
            split_seed=split_seed,
            base_model=base_model,
            model_seed=model_seed,
            adaptation_seed=-1,
            budget=0,
            fit_domain="source",
            method="uncalibrated",
            evaluation_split=evaluation_split,
        )
        probability_rows.append(
            _probability_row(
                key,
                fit_scores=np.asarray([], dtype=float),
                fit_y=np.asarray([], dtype=int),
                eval_scores=probabilities[evaluation_split],
                eval_y=labels[evaluation_split],
                method="uncalibrated",
                hashes=base_hashes,
                commit=commit,
            )
        )

    # Every fitted method owns its source-domain budget-0 baseline.
    for method in FORMAL_METHODS:
        for evaluation_split in ("source_id_test", "target_final_test"):
            key = _base_key(
                dataset=dataset,
                source_family=source_family,
                split_seed=split_seed,
                base_model=base_model,
                model_seed=model_seed,
                adaptation_seed=-1,
                budget=0,
                fit_domain="source",
                method=method,
                evaluation_split=evaluation_split,
            )
            source_hashes = base_hashes | {
                "sample_hash": str(
                    prevalence.set_index("role").loc["source_probability_calibration", "role_hash"]
                )
            }
            probability_rows.append(
                _probability_row(
                    key,
                    fit_scores=probabilities["source_probability_calibration"],
                    fit_y=labels["source_probability_calibration"],
                    eval_scores=probabilities[evaluation_split],
                    eval_y=labels[evaluation_split],
                    method=method,
                    hashes=source_hashes,
                    commit=commit,
                )
            )

    # Budget-0 conformal thresholds are fitted only on the source conformal role.
    for score_source, score_map in (
        ("base_uncalibrated_probability", probabilities),
        ("source_sigmoid_probability", source_sigmoid_probabilities),
    ):
        for conformal_method in config["conformal_methods"]:
            for evaluation_split in ("source_id_test", "target_final_test"):
                key = _base_key(
                    dataset=dataset,
                    source_family=source_family,
                    split_seed=split_seed,
                    base_model=base_model,
                    model_seed=model_seed,
                    adaptation_seed=-1,
                    budget=0,
                    fit_domain="source",
                    method="uncalibrated",
                    evaluation_split=evaluation_split,
                )
                conformal_hashes = base_hashes | {
                    "sample_hash": str(
                        prevalence.set_index("role").loc[
                            "source_conformal_calibration", "role_hash"
                        ]
                    )
                }
                result, threshold = _conformal_rows(
                    key,
                    fit_scores=score_map["source_conformal_calibration"],
                    fit_y=labels["source_conformal_calibration"],
                    eval_scores=score_map[evaluation_split],
                    eval_y=labels[evaluation_split],
                    conformal_method=conformal_method,
                    score_source=score_source,
                    hashes=conformal_hashes,
                    commit=commit,
                    alpha=1 - float(config["nominal_coverage"]),
                )
                conformal_rows.append(result)
                if evaluation_split == "source_id_test":
                    threshold_rows.append(threshold)

    adaptation = roles["target_adaptation_pool"]
    frozen = pd.read_parquet(config["frozen_budget_manifest"])
    budgets = list(map(int, config["budgets"]))
    for adaptation_seed in range(int(config["adaptation_seeds"])):
        samples = canonical_blind_budget_samples(
            dataset=dataset,
            split_seed=split_seed,
            adaptation_seed=adaptation_seed,
            entity_ids=adaptation["entity_id"].astype(str),
            budgets=budgets,
        )
        by_hash = adaptation.set_index("entity_hash", drop=False)
        for budget in budgets:
            sample = samples[budget]
            _verify_frozen_sample(
                frozen,
                dataset=dataset,
                split_seed=split_seed,
                adaptation_seed=adaptation_seed,
                budget=budget,
                sample=sample,
            )
            if budget == 0:
                continue
            selected = (
                by_hash.loc[list(sample["selected_entity_hashes"])]
                if sample["selected_count"]
                else by_hash.iloc[:0]
            )
            selected_y = selected["target"].astype(int).to_numpy()
            # Scores are joined by entity hash so selection never depends on labels.
            base_score_series = pd.Series(
                probabilities["target_adaptation_pool"], index=adaptation["entity_hash"]
            )
            sigmoid_score_series = pd.Series(
                source_sigmoid_probabilities["target_adaptation_pool"],
                index=adaptation["entity_hash"],
            )
            selected_base = base_score_series.loc[list(sample["selected_entity_hashes"])].to_numpy()
            selected_sigmoid = sigmoid_score_series.loc[
                list(sample["selected_entity_hashes"])
            ].to_numpy()
            selected_positive = int(selected_y.sum())
            hashes = base_hashes | {"sample_hash": str(sample["sample_hash"])}
            available = bool(sample["available"])
            for method in FORMAL_METHODS:
                key = _base_key(
                    dataset=dataset,
                    source_family=source_family,
                    split_seed=split_seed,
                    base_model=base_model,
                    model_seed=model_seed,
                    adaptation_seed=adaptation_seed,
                    budget=budget,
                    fit_domain="target",
                    method=method,
                    evaluation_split="target_final_test",
                )
                if not available:
                    probability_rows.append(
                        _status_probability_row(
                            key,
                            "not_available",
                            "budget exceeds target adaptation pool",
                            int(sample["selected_count"]),
                            selected_positive,
                            hashes,
                            commit,
                        )
                    )
                else:
                    probability_rows.append(
                        _probability_row(
                            key,
                            fit_scores=selected_base,
                            fit_y=selected_y,
                            eval_scores=probabilities["target_final_test"],
                            eval_y=labels["target_final_test"],
                            method=method,
                            hashes=hashes,
                            commit=commit,
                        )
                    )
            for score_source, selected_scores, eval_scores in (
                (
                    "base_uncalibrated_probability",
                    selected_base,
                    probabilities["target_final_test"],
                ),
                (
                    "source_sigmoid_probability",
                    selected_sigmoid,
                    source_sigmoid_probabilities["target_final_test"],
                ),
            ):
                for conformal_method in config["conformal_methods"]:
                    key = _base_key(
                        dataset=dataset,
                        source_family=source_family,
                        split_seed=split_seed,
                        base_model=base_model,
                        model_seed=model_seed,
                        adaptation_seed=adaptation_seed,
                        budget=budget,
                        fit_domain="target",
                        method="uncalibrated",
                        evaluation_split="target_final_test",
                    )
                    if not available:
                        result, threshold = _status_conformal_rows(
                            key,
                            conformal_method=conformal_method,
                            score_source=score_source,
                            status="not_available",
                            reason="budget exceeds target adaptation pool",
                            actual=int(sample["selected_count"]),
                            positive=selected_positive,
                            hashes=hashes,
                            commit=commit,
                        )
                    else:
                        result, threshold = _conformal_rows(
                            key,
                            fit_scores=selected_scores,
                            fit_y=selected_y,
                            eval_scores=eval_scores,
                            eval_y=labels["target_final_test"],
                            conformal_method=conformal_method,
                            score_source=score_source,
                            hashes=hashes,
                            commit=commit,
                            alpha=1 - float(config["nominal_coverage"]),
                        )
                    conformal_rows.append(result)
                    threshold_rows.append(threshold)
            pool_size = len(adaptation)
            pool_positive = int(adaptation["target"].sum())
            estimability_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset": dataset,
                    "source_family": source_family,
                    "split_seed": split_seed,
                    "base_model": base_model,
                    "model_seed": model_seed,
                    "adaptation_seed": adaptation_seed,
                    "budget": budget,
                    "pool_size": pool_size,
                    "pool_positive_count": pool_positive,
                    "actual_entities": int(sample["selected_count"]),
                    "positive_labels": selected_positive,
                    "negative_labels": int(sample["selected_count"] - selected_positive),
                    "support_probability_min_1_1": hypergeometric_support_probability(
                        pool_size, pool_positive, min(budget, pool_size), 1, 1
                    ),
                    "support_probability_min_5_5": hypergeometric_support_probability(
                        pool_size, pool_positive, min(budget, pool_size), 5, 5
                    ),
                    "mondrian_finite_threshold_probability": mondrian_finite_threshold_probability(
                        pool_size,
                        pool_positive,
                        min(budget, pool_size),
                        alpha=1 - float(config["nominal_coverage"]),
                    ),
                    "reference_type": "true_composition_reference",
                    "sample_hash": sample["sample_hash"],
                    "pool_hash": sample["pool_hash"],
                    "config_hash": config_hash,
                    "data_hash": data_hash,
                    "split_hash": split_hash,
                    "commit": commit,
                    "timestamp_utc": utc_now(),
                }
            )

    probability = pd.DataFrame(probability_rows).reindex(columns=_columns("probability"))
    conformal = pd.DataFrame(conformal_rows).reindex(columns=_columns("conformal"))
    threshold = pd.DataFrame(threshold_rows).reindex(columns=_columns("threshold"))
    for kind, result in (
        ("probability", probability),
        ("conformal", conformal),
        ("threshold", threshold),
    ):
        _validate(result, kind)
    estimability = pd.DataFrame(estimability_rows)
    failures = pd.concat(
        [
            probability.loc[probability["status"].eq("failed")].assign(result_type="probability"),
            conformal.loc[conformal["status"].eq("failed")].assign(result_type="conformal"),
        ],
        ignore_index=True,
        sort=False,
    )
    not_estimable = pd.concat(
        [
            probability.loc[probability["status"].eq("not_estimable")].assign(
                result_type="probability"
            ),
            conformal.loc[conformal["status"].eq("not_estimable")].assign(result_type="conformal"),
        ],
        ignore_index=True,
        sort=False,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "source_family": source_family,
        "split_seed": split_seed,
        "base_model": base_model,
        "model_seed": model_seed,
        "config_hash": config_hash,
        "data_hash": data_hash,
        "split_hash": split_hash,
        "commit": commit,
        "tuning_records": tuning_records,
        "tuning_seconds": tuning_seconds,
        "runtime_seconds": time.perf_counter() - started,
        "timestamp_utc": utc_now(),
        "role_counts": {role: len(value) for role, value in roles.items()},
    }
    return {
        "probability": probability,
        "conformal": conformal,
        "threshold": threshold,
        "estimability": estimability,
        "failures": failures,
        "not_estimable": not_estimable,
        "split_prevalence_audit": prevalence,
    }, metadata


OUTPUT_FILES = {
    "probability": "probability.parquet",
    "conformal": "conformal.parquet",
    "threshold": "threshold.parquet",
    "estimability": "estimability.parquet",
    "failures": "failures.parquet",
    "not_estimable": "not_estimable.parquet",
    "split_prevalence_audit": "split_prevalence_audit.parquet",
}


@contextmanager
def _directory_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "shard.lock"
    descriptor = None
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, f"pid={os.getpid()} timestamp={utc_now()}\n".encode())
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
            path.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _marker_valid(directory: Path, expected: dict[str, Any]) -> tuple[bool, str]:
    marker_path = directory / "shard.complete.json"
    if not marker_path.is_file():
        return False, "complete marker missing"
    try:
        marker = _read_json(marker_path)
        for key, value in expected.items():
            if marker.get(key) != value:
                return False, f"marker {key} mismatch"
        if marker.get("schema_version") != SCHEMA_VERSION:
            return False, "schema version mismatch"
        for kind, filename in OUTPUT_FILES.items():
            path = directory / filename
            if not path.is_file():
                return False, f"{filename} missing"
            if sha256_file(path) != marker["files"][filename]["sha256"]:
                return False, f"{filename} checksum mismatch"
            frame = pd.read_parquet(path)
            if len(frame) != marker["files"][filename]["rows"]:
                return False, f"{filename} row-count mismatch"
            if kind in {"probability", "conformal", "threshold"}:
                _validate(frame, kind)
        return True, "valid"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _atomic_write_results(
    directory: Path, frames: dict[str, pd.DataFrame], metadata: dict[str, Any]
) -> dict[str, Any]:
    temporary_paths: list[Path] = []
    file_records: dict[str, dict[str, Any]] = {}
    try:
        for kind, filename in OUTPUT_FILES.items():
            destination = directory / filename
            with tempfile.NamedTemporaryFile(
                dir=directory, suffix=".parquet.tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
            temporary_paths.append(temporary)
            frames[kind].to_parquet(temporary, index=False)
            os.replace(temporary, destination)
            temporary_paths.remove(temporary)
            file_records[filename] = {
                "sha256": sha256_file(destination),
                "rows": int(len(frames[kind])),
                "bytes": destination.stat().st_size,
            }
        marker = metadata | {"files": file_records, "schema_version": SCHEMA_VERSION}
        marker_tmp = directory / "shard.complete.json.tmp"
        marker_tmp.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(marker_tmp, directory / "shard.complete.json")
        return marker
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def run_shard(
    *,
    config_path: Path,
    dataset: str,
    split_seed: int,
    base_model: str,
    model_seed: int,
    commit: str,
    output_root: Path | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = output_root or Path(config["output_root"])
    directory = shard_directory(root, dataset, split_seed, base_model, model_seed)
    registry = load_registry(Path(config["data_registry"]))
    registry_row = registry.loc[registry.dataset.eq(dataset)].iloc[0]
    expected = {
        "config_hash": sha256_file(config_path),
        "data_hash": str(registry_row.processed_checksum),
        "commit": commit,
    }
    valid, reason = _marker_valid(directory, expected)
    if valid:
        return {"status": "skipped", "reason": "valid completed shard", "directory": str(directory)}
    with _directory_lock(directory):
        # Recheck after taking the lock for concurrent callers.
        valid, reason = _marker_valid(directory, expected)
        if valid:
            return {
                "status": "skipped",
                "reason": "valid completed shard",
                "directory": str(directory),
            }
        marker_existed = (directory / "shard.complete.json").exists()
        output_existed = any((directory / name).exists() for name in OUTPUT_FILES.values())
        retry_path = directory / "retry_state.json"
        retry_state = _read_json(retry_path) if retry_path.exists() else {"technical_reruns": 0}
        corrupt_reason: str | None = None
        if marker_existed or output_existed:
            corrupt_reason = reason
            _append_jsonl(
                directory / "failure_history.jsonl",
                {
                    "timestamp_utc": utc_now(),
                    "event": "corrupt_or_incomplete_shard",
                    "reason": reason,
                    "preserved": True,
                },
            )
            if int(retry_state["technical_reruns"]) >= 1:
                raise RuntimeError("preregistered corrupt-shard technical rerun already consumed")
            retry_state["technical_reruns"] = int(retry_state["technical_reruns"]) + 1
            retry_path.write_text(json.dumps(retry_state, indent=2) + "\n", encoding="utf-8")
        frames, metadata = compute_shard(
            config=config,
            config_path=config_path,
            dataset=dataset,
            split_seed=split_seed,
            base_model=base_model,
            model_seed=model_seed,
            commit=commit,
        )
        if corrupt_reason is not None:
            technical_failure = pd.DataFrame(
                [
                    {
                        "result_type": "runner",
                        "dataset": dataset,
                        "split_seed": split_seed,
                        "base_model": base_model,
                        "model_seed": model_seed,
                        "status": "failed",
                        "status_reason": f"corrupt_or_incomplete_shard: {corrupt_reason}",
                        "failure_preserved": True,
                        "technical_rerun": int(retry_state["technical_reruns"]),
                        "commit": commit,
                        "timestamp_utc": utc_now(),
                    }
                ]
            )
            frames["failures"] = pd.concat(
                [frames["failures"], technical_failure], ignore_index=True, sort=False
            )
        metadata["technical_reruns"] = int(retry_state["technical_reruns"])
        marker = _atomic_write_results(directory, frames, metadata)
        return {
            "status": "success",
            "reason": "computed",
            "directory": str(directory),
            "technical_reruns": marker["technical_reruns"],
            "runtime_seconds": marker["runtime_seconds"],
            "rows": {name: record["rows"] for name, record in marker["files"].items()},
        }


def merge_shard_directories(directories: list[Path], output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    markers = [_read_json(directory / "shard.complete.json") for directory in directories]
    commits = {marker["commit"] for marker in markers}
    configs = {marker["config_hash"] for marker in markers}
    if len(commits) != 1 or len(configs) != 1:
        raise ValueError("qualification shards do not share commit/config")
    merged_records: dict[str, Any] = {}
    for kind, filename in OUTPUT_FILES.items():
        frames = [pd.read_parquet(directory / filename) for directory in directories]
        merged = pd.concat(frames, ignore_index=True, sort=False)
        expected_rows = sum(marker["files"][filename]["rows"] for marker in markers)
        if len(merged) != expected_rows:
            raise ValueError(f"{kind} merged row count mismatch")
        if kind in {"probability", "conformal", "threshold"}:
            _validate(merged, kind)
        destination = output_root / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        merged.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
        merged_records[filename] = {
            "rows": len(merged),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        }
    report = {
        "status": "pass",
        "schema_version": SCHEMA_VERSION,
        "shard_count": len(directories),
        "commit": next(iter(commits)),
        "config_hash": next(iter(configs)),
        "files": merged_records,
        "timestamp_utc": utc_now(),
    }
    (output_root / "merge_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m reliable_shift.stage2.formal_runner")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-shard")
    run.add_argument("--dataset", required=True)
    run.add_argument("--split-seed", type=int, required=True)
    run.add_argument("--base-model", required=True)
    run.add_argument("--model-seed", type=int, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--commit", required=True)
    run.add_argument("--output-root", type=Path)
    merge = sub.add_parser("merge")
    merge.add_argument("--shard-directory", type=Path, action="append", required=True)
    merge.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run-shard":
        result = run_shard(
            config_path=args.config,
            dataset=args.dataset,
            split_seed=args.split_seed,
            base_model=args.base_model,
            model_seed=args.model_seed,
            commit=args.commit,
            output_root=args.output_root,
        )
    else:
        result = merge_shard_directories(args.shard_directory, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
