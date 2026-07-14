from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from reliable_shift.preprocessing import build_preprocessor, prepare_features


def build_model_pipeline(
    *,
    kind: str,
    params: dict[str, Any],
    feature_columns: list[str],
    numeric_columns: list[str],
    seed: int,
) -> Pipeline:
    sparse = kind in {"logistic_regression", "xgboost"}
    preprocessor = build_preprocessor(
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        sparse_categorical=sparse,
    )
    if kind == "logistic_regression":
        logistic_params = {"solver": "liblinear", "random_state": seed, **params}
        model = LogisticRegression(**logistic_params)
    elif kind == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(random_state=seed, **params)
    elif kind == "xgboost":
        model = XGBClassifier(random_state=seed, **params)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def tune_model(
    *,
    model_name: str,
    model_config: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    seed: int,
) -> tuple[Pipeline, list[dict[str, Any]], float]:
    kind = str(model_config["kind"])
    fixed = dict(model_config.get("fixed", {}))
    records: list[dict[str, Any]] = []
    best_pipeline: Pipeline | None = None
    best_score = -float("inf")
    total_start = time.perf_counter()
    x_train = prepare_features(train, feature_columns, numeric_columns)
    x_validation = prepare_features(validation, feature_columns, numeric_columns)
    y_train = train["target"].astype(int).to_numpy()
    y_validation = validation["target"].astype(int).to_numpy()
    for candidate_id, candidate in enumerate(model_config["candidates"]):
        params = {**fixed, **candidate}
        started = time.perf_counter()
        record: dict[str, Any] = {
            "model": model_name,
            "candidate_id": candidate_id,
            "params": params,
            "status": "failed",
            "error_message": "",
        }
        try:
            pipeline = build_model_pipeline(
                kind=kind,
                params=params,
                feature_columns=feature_columns,
                numeric_columns=numeric_columns,
                seed=seed,
            )
            pipeline.fit(x_train, y_train)
            probability = pipeline.predict_proba(x_validation)[:, 1]
            score = float(roc_auc_score(y_validation, probability))
            record.update(status="success", validation_auroc=score)
            if score > best_score:
                best_score = score
                best_pipeline = pipeline
        except Exception as exc:  # failures are part of the research record
            record["error_message"] = f"{type(exc).__name__}: {exc}"
        record["runtime_seconds"] = time.perf_counter() - started
        records.append(record)
    if best_pipeline is None:
        raise RuntimeError(f"All configurations failed for {model_name}: {records}")
    return best_pipeline, records, time.perf_counter() - total_start
