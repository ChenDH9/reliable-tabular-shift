#!/usr/bin/env python3
"""Reproduce the frozen ACS Income representative shard in an isolated directory.

This release entrypoint performs only the preregistered Stage 2 representative
configuration.  It never downloads data, writes to the formal result tree, or
enumerates the full experiment.  The caller must provide the unchanged official
2018 ACS person PUMS ZIP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

TASK = "acsincome_region"
FORMAL_COMMIT = "d745e5412c1d530a2ae64e2eaa42c85c1f64e419"
FROZEN_FORMAL_RUNNER_SHA256 = (
    "16b310ce4c5e3f0a8b9579a6cfd83fb28bbe42c287a85dfc797426b2cee74314"
)
OFFICIAL_ZIP_SIZE = 562_224_936
OFFICIAL_ZIP_SHA256 = "73fba8988d84f6425966fa14ab26af6ab40ebc0f0e16156b0dada39322349243"
FROZEN_PROCESSED_SHA256 = "57d25909a06f8a044e9d4d68d15a5a83f9e4c354115085cbdfab74684c53967f"
FROZEN_SPLIT_HASH = "388280e28df93edaf3c7227106e42611ccb11fc45925a6ad0f0834bcadf5eee5"
FROZEN_ADAPTATION_POOL_DIGEST = (
    "be91f1f7e1abbf7ae1a7981cbfa641b51b47ff66a524a30771b7967352b98408"
)
FROZEN_TARGET_FINAL_TEST_DIGEST = (
    "3cd9344070bb0370a0523146751a3ad5b11ba159f3b306d9319c2ada1aef1e7e"
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
ATOL = 1e-10
RTOL = 1e-9

ACS_INCOME = (
    "COW",
    "ENG",
    "FER",
    "HINS1",
    "HINS2",
    "HINS3",
    "HINS4",
    "NWLA",
    "NWLK",
    "OCCP",
    "POBP",
    "RELP",
    "WKHP",
    "WKW",
    "WRK",
    "PINCP",
)
ACS_SHARED = (
    "AGEP",
    "SEX",
    "ST",
    "MAR",
    "CIT",
    "RAC1P",
    "SCHL",
    "DIVISION",
    "ACS_YEAR",
)
DOMAIN_IDENTIFIERS = frozenset({"DIVISION", "ST", "ACS_YEAR"})
ID_COLUMNS = frozenset(
    {"record_id", "group_id", "entity_id", "target", "is_target_domain"}
)

RESULT_FILES = {
    "probability": "probability.parquet",
    "conformal": "conformal.parquet",
    "threshold": "threshold.parquet",
    "estimability": "estimability.parquet",
    "failures": "failures.parquet",
    "not_estimable": "not_estimable.parquet",
    "split_prevalence_audit": "split_prevalence_audit.parquet",
}
EXPECTED_ROWS = {
    "probability": 2010,
    "conformal": 2008,
    "threshold": 2004,
    "estimability": 500,
    "failures": 0,
    "not_estimable": 6,
    "split_prevalence_audit": 7,
}
EXPECTED_STATUSES = {
    "probability": {"success": 2008, "not_estimable": 2},
    "conformal": {"success": 2004, "not_estimable": 4},
    "threshold": {"success": 2000, "not_estimable": 4},
}

MANIFEST_COLUMNS = (
    "dataset",
    "split_seed",
    "adaptation_seed",
    "budget",
    "available",
    "selected_count",
    "selected_entity_prefix_digest",
    "adaptation_pool_digest",
    "target_final_test_digest",
    "selection_reads_labels",
)
BUDGETS = (0, 25, 50, 100, 250, 500)
ADAPTATION_SEEDS = tuple(range(100))

EXCLUDED_COMPARISON_COLUMNS = frozenset(
    {"timestamp_utc", "runtime_seconds", "commit", "config_hash"}
)
BOOLEAN_COLUMNS = frozenset(
    {
        "produced_exact_zero",
        "produced_exact_one",
        "single_class_calibration",
        "stable_support_5_per_class",
        "estimable",
        "finite_threshold",
        "standard_threshold_infinite",
        "class0_threshold_infinite",
        "class1_threshold_infinite",
        "all_thresholds_finite",
        "cap_reads_labels",
    }
)
INTEGER_COLUMNS = frozenset(
    {
        "split_seed",
        "model_seed",
        "adaptation_seed",
        "budget",
        "actual_entities",
        "positive_labels",
        "negative_labels",
        "n_eval",
        "positive_eval",
        "negative_eval",
        "unique_input_scores",
        "unique_output_scores",
        "standard_calibration_count",
        "class0_calibration_count",
        "class1_calibration_count",
        "pool_size",
        "pool_positive_count",
        "rows",
        "entities",
        "positive_count",
        "negative_count",
    }
)
HEX64 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_lines(values: Any) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def resolve_from(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require_fresh_directory(path: Path, label: str) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise RuntimeError(f"{label} must be a new or empty directory: {path}")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_isolation(
    *,
    bundle_root: Path,
    output_root: Path,
    runtime_root: Path,
    reference_root: Path,
) -> None:
    protected = [
        bundle_root / "src",
        bundle_root / "scripts",
        bundle_root / "protocol",
        reference_root,
    ]
    for destination, label in ((output_root, "output root"), (runtime_root, "runtime root")):
        if destination == bundle_root or any(
            destination == item
            or is_relative_to(item, destination)
            or is_relative_to(destination, item)
            for item in protected
        ):
            raise RuntimeError(f"{label} overlaps a protected release path: {destination}")
        lowered_parts = tuple(part.lower() for part in destination.parts)
        if any(
            first == "results" and second == "stage2"
            for first, second in zip(lowered_parts, lowered_parts[1:], strict=False)
        ):
            raise RuntimeError(f"{label} resembles the protected formal result tree")
    if (
        output_root == runtime_root
        or is_relative_to(output_root, runtime_root)
        or is_relative_to(runtime_root, output_root)
    ):
        raise RuntimeError("output root and runtime root must be disjoint")
    require_fresh_directory(output_root, "output root")
    require_fresh_directory(runtime_root, "runtime root")


def verify_official_zip(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"official ACS person PUMS ZIP is missing: {path}")
    size = path.stat().st_size
    if size != OFFICIAL_ZIP_SIZE:
        raise ValueError(
            f"official ACS ZIP size mismatch: expected {OFFICIAL_ZIP_SIZE}, observed {size}"
        )
    observed = sha256_file(path)
    if observed != OFFICIAL_ZIP_SHA256:
        raise ValueError(
            f"official ACS ZIP SHA-256 mismatch: expected {OFFICIAL_ZIP_SHA256}, "
            f"observed {observed}"
        )
    with zipfile.ZipFile(path) as archive:
        members = [
            info
            for info in archive.infolist()
            if Path(info.filename).name.lower() in {"psam_pusa.csv", "psam_pusb.csv"}
        ]
        if len(members) != 2 or len({Path(info.filename).name.lower() for info in members}) != 2:
            raise ValueError(
                "official ACS ZIP must contain exactly psam_pusa.csv and psam_pusb.csv"
            )
        if any(info.flag_bits & 0x1 for info in members):
            raise ValueError("official ACS person CSV members must not be encrypted")
        member_record = [
            {"name": Path(info.filename).name, "uncompressed_bytes": int(info.file_size)}
            for info in sorted(members, key=lambda item: item.filename.lower())
        ]
    return {"size_bytes": size, "sha256": observed, "members": member_record}


def _folktables_symbols():
    try:
        from folktables import ACSIncome  # type: ignore[attr-defined]
    except ImportError:
        from folktables.acs import ACSIncome  # type: ignore[attr-defined]
    from folktables.acs import adult_filter

    return ACSIncome, adult_filter


def materialize_acsincome(
    archive_path: Path, destination: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the exact frozen ACS Income task definition to official person PUMS."""

    acs_income_problem, adult_filter = _folktables_symbols()
    del acs_income_problem  # Import is intentional provenance; the project uses its adult filter.

    features = [
        feature
        for feature in dict.fromkeys((*ACS_INCOME, *ACS_SHARED))
        if feature != "PINCP" and feature not in DOMAIN_IDENTIFIERS
    ]
    selected_columns = list(
        dict.fromkeys(
            (*features, "PINCP", "DIVISION", "ST", "ACS_YEAR", "SERIALNO", "SPORDER")
        )
    )
    needed = (set(selected_columns) - {"ACS_YEAR"}) | {"PWGTP"}
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(
            info.filename
            for info in archive.infolist()
            if Path(info.filename).name.lower() in {"psam_pusa.csv", "psam_pusb.csv"}
        )
        for name in names:
            with archive.open(name) as stream:
                frames.append(
                    pd.read_csv(
                        stream,
                        usecols=lambda column: column in needed,
                        low_memory=False,
                    )
                )
    raw = pd.concat(frames, ignore_index=True)
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"official ACS person archive is missing fields: {sorted(missing)}")
    raw["ACS_YEAR"] = 2018

    income = adult_filter(raw.copy())
    income = income.loc[:, selected_columns].copy()
    income["target"] = (income["PINCP"] >= 56000).astype("int8")
    income["is_target_domain"] = pd.to_numeric(
        income["DIVISION"], errors="coerce"
    ).eq(1)
    income["record_id"] = (
        income["ACS_YEAR"].astype(str)
        + ":"
        + income["ST"].astype(str)
        + ":"
        + income["SERIALNO"].astype(str)
        + ":"
        + income["SPORDER"].astype(str)
    )
    income["group_id"] = income["record_id"]
    income.drop(columns=["PINCP"], inplace=True)
    income["entity_id"] = income["record_id"].astype(str)

    if income.duplicated("entity_id").any():
        raise ValueError("materialized ACS Income entities are not unique")
    source = income.loc[~income["is_target_domain"]]
    target = income.loc[income["is_target_domain"]]
    observed_counts = {
        "rows": int(len(income)),
        "source_rows": int(len(source)),
        "target_rows": int(len(target)),
        "source_positive_count": int(source["target"].sum()),
        "target_positive_count": int(target["target"].sum()),
    }
    expected_counts = {
        "rows": 1_659_616,
        "source_rows": 1_575_270,
        "target_rows": 84_346,
        "source_positive_count": 517_992,
        "target_positive_count": 33_959,
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"materialized ACS Income counts differ from frozen counts: {observed_counts}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    income.to_parquet(destination, index=False)
    processed_hash = sha256_file(destination)
    if processed_hash != FROZEN_PROCESSED_SHA256:
        raise ValueError(
            "materialized ACS Income Parquet differs from the frozen processed checksum: "
            f"expected {FROZEN_PROCESSED_SHA256}, observed {processed_hash}"
        )
    audit = {
        **observed_counts,
        "feature_count": len(features),
        "features": features,
        "processed_sha256": processed_hash,
        "duplicate_entities": 0,
    }
    return income, audit


def write_runtime_registry(
    frame: pd.DataFrame, data_path: Path, destination: Path
) -> dict[str, Any]:
    features = [
        feature
        for feature in dict.fromkeys((*ACS_INCOME, *ACS_SHARED))
        if feature != "PINCP" and feature not in DOMAIN_IDENTIFIERS
    ]
    excluded = sorted(ID_COLUMNS | DOMAIN_IDENTIFIERS | {"SERIALNO", "SPORDER"})
    source = frame.loc[~frame["is_target_domain"]]
    target = frame.loc[frame["is_target_domain"]]
    row = {
        "dataset": TASK,
        "source_family": "acs",
        "audited_parquet_path": str(data_path),
        "raw_source_checksum": f"official_person={OFFICIAL_ZIP_SHA256}",
        "processed_checksum": FROZEN_PROCESSED_SHA256,
        "row_count": int(len(frame)),
        "unique_entity_count": int(frame["entity_id"].nunique()),
        "duplicate_entity_count": int(frame.duplicated("entity_id").sum()),
        "source_rows": int(len(source)),
        "target_rows": int(len(target)),
        "source_positive_count": int(source["target"].sum()),
        "target_positive_count": int(target["target"].sum()),
        "source_prevalence": float(source["target"].mean()),
        "target_prevalence": float(target["target"].mean()),
        "target_adaptation_capacity": int(len(target) * 0.4),
        "supports_budget_250": True,
        "supports_budget_500": True,
        "feature_list": json.dumps(features),
        "excluded_columns": json.dumps(excluded),
        "entity_rule": "person record",
        "prediction_time": "survey response",
        "license": "U.S. Census public-use terms",
        "leakage_audit": (
            "domain, outcome, identifiers, and post-outcome fields excluded; see task audit"
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(destination, index=False)
    return {"rows": 1, "sha256": sha256_file(destination), "features": features}


def parse_boolean(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{label} contains a non-boolean value: {value!r}")


def validate_and_write_manifest(
    source_path: Path,
    destination: Path,
    frame: pd.DataFrame,
    bundle_root: Path,
) -> dict[str, Any]:
    if not source_path.is_file():
        raise FileNotFoundError(f"packaged one-task budget manifest is missing: {source_path}")
    manifest = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    if tuple(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError(
            "one-task budget manifest must contain only the safe frozen fields in exact order: "
            + ", ".join(MANIFEST_COLUMNS)
        )
    if len(manifest) != 600:
        raise ValueError(
            f"one-task budget manifest must contain 600 rows, observed {len(manifest)}"
        )
    for column in ("split_seed", "adaptation_seed", "budget", "selected_count"):
        manifest[column] = pd.to_numeric(manifest[column], errors="raise").astype("int64")
    manifest["available"] = [
        parse_boolean(value, label="available") for value in manifest["available"]
    ]
    manifest["selection_reads_labels"] = [
        parse_boolean(value, label="selection_reads_labels")
        for value in manifest["selection_reads_labels"]
    ]
    for column in (
        "selected_entity_prefix_digest",
        "adaptation_pool_digest",
        "target_final_test_digest",
    ):
        if not manifest[column].map(lambda value: HEX64.fullmatch(str(value)) is not None).all():
            raise ValueError(f"one-task budget manifest has an invalid SHA-256 in {column}")

    if set(manifest["dataset"]) != {TASK} or set(manifest["split_seed"]) != {0}:
        raise ValueError("one-task budget manifest must contain only acsincome_region split 0")
    if set(manifest["adaptation_seed"]) != set(ADAPTATION_SEEDS):
        raise ValueError("one-task budget manifest adaptation seeds must be exactly 0..99")
    if set(manifest["budget"]) != set(BUDGETS):
        raise ValueError("one-task budget manifest budgets differ from the frozen six budgets")
    if manifest.duplicated(["dataset", "split_seed", "adaptation_seed", "budget"]).any():
        raise ValueError("one-task budget manifest contains duplicate primary keys")
    sizes = manifest.groupby("adaptation_seed", sort=False).size()
    if not sizes.eq(len(BUDGETS)).all():
        raise ValueError("each adaptation seed must contain exactly six budget rows")
    if not manifest["available"].all() or manifest["selection_reads_labels"].any():
        raise ValueError("ACS Income budgets must be available and label-blind")
    if not (manifest["selected_count"] == manifest["budget"]).all():
        raise ValueError("ACS Income selected counts must equal the requested budgets")
    if set(manifest["adaptation_pool_digest"]) != {FROZEN_ADAPTATION_POOL_DIGEST}:
        raise ValueError("one-task adaptation-pool digest differs from the frozen digest")
    if set(manifest["target_final_test_digest"]) != {FROZEN_TARGET_FINAL_TEST_DIGEST}:
        raise ValueError("one-task target-final-test digest differs from the frozen digest")
    budget_zero = manifest.loc[manifest["budget"].eq(0), "selected_entity_prefix_digest"]
    if set(budget_zero) != {EMPTY_SHA256}:
        raise ValueError("budget-zero selected-prefix digest is not the empty SHA-256")

    source_root = bundle_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from reliable_shift.stage2.acquisition import canonical_blind_budget_samples
    from reliable_shift.stage2.splits import assign_roles

    assigned = assign_roles(frame, 0)
    adaptation_mask = assigned["role"].eq("target_adaptation_pool").to_numpy()
    final_mask = assigned["role"].eq("target_final_test").to_numpy()
    adaptation_ids = frame.loc[adaptation_mask, "entity_id"].astype(str)
    target_final_hashes = sorted(assigned.loc[final_mask, "entity_hash"].astype(str))
    if digest_lines(target_final_hashes) != FROZEN_TARGET_FINAL_TEST_DIGEST:
        raise ValueError("rebuilt target-final-test entities differ from the frozen digest")

    indexed = manifest.set_index(["adaptation_seed", "budget"], verify_integrity=True)
    for adaptation_seed in ADAPTATION_SEEDS:
        samples = canonical_blind_budget_samples(
            dataset=TASK,
            split_seed=0,
            adaptation_seed=adaptation_seed,
            entity_ids=adaptation_ids,
            budgets=list(BUDGETS),
        )
        for budget in BUDGETS:
            record = indexed.loc[(adaptation_seed, budget)]
            sample = samples[budget]
            if str(record["selected_entity_prefix_digest"]) != sample["sample_hash"]:
                raise ValueError(
                    f"rebuilt sample digest differs at adaptation_seed={adaptation_seed}, "
                    f"budget={budget}"
                )
            if str(record["adaptation_pool_digest"]) != sample["pool_hash"]:
                raise ValueError("rebuilt adaptation-pool digest differs from the manifest")
            if int(record["selected_count"]) != int(sample["selected_count"]):
                raise ValueError("rebuilt selected count differs from the manifest")

    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(destination, index=False)
    return {
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "runtime_sha256": sha256_file(destination),
        "rows": int(len(manifest)),
        "safe_columns": list(MANIFEST_COLUMNS),
        "adaptation_pool_digest": FROZEN_ADAPTATION_POOL_DIGEST,
        "target_final_test_digest": FROZEN_TARGET_FINAL_TEST_DIGEST,
    }


def write_runtime_config(
    registry_path: Path,
    budget_manifest_path: Path,
    output_root: Path,
    destination: Path,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema_version": "stage2_v2",
        "data_registry": str(registry_path),
        "frozen_budget_manifest": str(budget_manifest_path),
        "output_root": str(output_root),
        "acquisition_strategy": "blind_random",
        "nominal_coverage": 0.90,
        "budgets": list(BUDGETS),
        "split_seeds": [0, 1, 2],
        "adaptation_seeds": 100,
        "probability_methods": [
            "uncalibrated",
            "intercept_only_mle",
            "sigmoid",
            "isotonic",
        ],
        "sensitivity_methods": ["jeffreys_smoothed_intercept_matching"],
        "conformal_methods": ["standard", "mondrian"],
        "score_sources": [
            "base_uncalibrated_probability",
            "source_sigmoid_probability",
        ],
        "role_caps": {
            "source_train": 100000,
            "source_tune": 20000,
            "source_probability_calibration": 20000,
            "source_conformal_calibration": 20000,
            "source_id_test": 30000,
            "target_final_test": 50000,
            "target_adaptation_pool": None,
        },
        "models": {
            "logistic_regression": {
                "kind": "logistic_regression",
                "fixed": {"solver": "liblinear", "max_iter": 1000, "class_weight": None},
                "candidates": [{"C": 0.1}, {"C": 1.0}],
            },
            "xgboost_cpu": {
                "kind": "xgboost",
                "fixed": {
                    "n_estimators": 120,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "tree_method": "hist",
                    "device": "cpu",
                    "n_jobs": 4,
                    "eval_metric": "logloss",
                },
                "candidates": [
                    {"min_child_weight": 1, "reg_lambda": 1.0},
                    {"min_child_weight": 5, "reg_lambda": 2.0},
                ],
            },
        },
    }
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {"path": str(destination), "sha256": sha256_file(destination)}


def run_formal_shard(
    *, bundle_root: Path, config_path: Path, output_root: Path
) -> tuple[dict[str, Any], list[str]]:
    command = [
        sys.executable,
        "-m",
        "reliable_shift.stage2.formal_runner",
        "run-shard",
        "--dataset",
        TASK,
        "--split-seed",
        "0",
        "--base-model",
        "logistic_regression",
        "--model-seed",
        "0",
        "--config",
        str(config_path),
        "--commit",
        FORMAL_COMMIT,
        "--output-root",
        str(output_root),
    ]
    environment = dict(os.environ)
    source_root = str(bundle_root / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else source_root + os.pathsep + environment["PYTHONPATH"]
    )
    environment["PYTHONHASHSEED"] = "0"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    completed = subprocess.run(
        command,
        cwd=bundle_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "formal runner returned nonzero: "
            f"returncode={completed.returncode}; stderr={completed.stderr[-2000:]}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("formal runner did not emit valid JSON") from exc
    if payload.get("status") != "success":
        raise RuntimeError(f"formal runner did not compute a fresh shard: {payload}")
    return payload, command


def stable_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in EXCLUDED_COMPARISON_COLUMNS]


def key_columns(kind: str) -> list[str]:
    probability = [
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
    ]
    if kind == "probability":
        return probability
    if kind == "conformal":
        return [*probability, "conformal_method", "score_source"]
    if kind == "threshold":
        return [
            *[column for column in probability if column != "evaluation_split"],
            "conformal_method",
            "score_source",
        ]
    if kind == "estimability":
        return [
            "dataset",
            "source_family",
            "split_seed",
            "base_model",
            "model_seed",
            "adaptation_seed",
            "budget",
        ]
    if kind == "split_prevalence_audit":
        return ["dataset", "source_family", "split_seed", "role"]
    return ["result_type", *probability, "conformal_method", "score_source"]


def canonical_scalar(value: Any) -> str:
    if pd.isna(value):
        return "__NA__"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def row_tokens(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame.loc[:, columns].apply(
        lambda row: "\x1f".join(canonical_scalar(value) for value in row), axis=1
    )


def _csv_missing(value: Any) -> bool:
    return str(value).strip().lower() in {"__na__", "<na>", "nan", "null"}


def coerce_csv_reference(raw: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    converted = pd.DataFrame(index=raw.index)
    for column in actual.columns:
        values = raw[column]
        if column in BOOLEAN_COLUMNS or pd.api.types.is_bool_dtype(actual[column].dtype):
            converted[column] = [
                pd.NA if _csv_missing(value) or value == "" else parse_boolean(value, label=column)
                for value in values
            ]
        elif column in INTEGER_COLUMNS or pd.api.types.is_numeric_dtype(actual[column].dtype):
            normalized = values.map(
                lambda value: np.nan if _csv_missing(value) or value == "" else value
            )
            converted[column] = pd.to_numeric(normalized, errors="raise")
        else:
            empty_is_missing = bool(pd.isna(actual[column]).any())
            converted[column] = values.map(
                lambda value: (
                    pd.NA
                    if _csv_missing(value) or (empty_is_missing and value == "")
                    else value
                )
            )
    return converted


def load_reference(
    reference_root: Path, kind: str, actual: pd.DataFrame
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    candidates = [
        reference_root / f"{kind}.csv",
        reference_root / f"{kind}.parquet",
    ]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise FileNotFoundError(
            f"reference {kind} must have exactly one CSV or Parquet file under {reference_root}"
        )
    path = present[0]
    actual_stable = stable_columns(actual)
    if path.suffix == ".parquet":
        reference = pd.read_parquet(path)
        reference_mode = "parquet"
    else:
        reference = pd.read_csv(path, dtype=str, keep_default_na=False)
        reference_mode = "canonical_csv"

    if list(reference.columns) == list(actual.columns):
        reference = reference.loc[:, actual_stable].copy()
    elif list(reference.columns) != actual_stable:
        raise ValueError(
            f"reference {kind} schema differs from the stable runtime schema: "
            f"expected {actual_stable}, observed {list(reference.columns)}"
        )
    actual_for_coercion = actual.loc[:, actual_stable]
    if path.suffix == ".csv":
        reference = coerce_csv_reference(reference, actual_for_coercion)
    else:
        actual_types = actual_for_coercion.dtypes.astype(str).tolist()
        reference_types = reference.dtypes.astype(str).tolist()
        if reference_types != actual_types:
            raise ValueError(
                f"reference {kind} Parquet dtype schema differs: "
                f"expected {actual_types}, observed {reference_types}"
            )
    return reference, reference_mode, {
        "file": path.name,
        "sha256": sha256_file(path),
    }


def compare_column(actual: pd.Series, reference: pd.Series, column: str) -> dict[str, Any]:
    actual_missing = pd.isna(actual).to_numpy()
    reference_missing = pd.isna(reference).to_numpy()
    if not np.array_equal(actual_missing, reference_missing):
        raise ValueError(f"{column}: missing-value mask differs")
    keep = ~actual_missing
    if not keep.any():
        return {"mode": "all_missing", "compared": 0}

    if column in BOOLEAN_COLUMNS or pd.api.types.is_bool_dtype(actual.dtype):
        left = np.asarray([bool(value) for value in actual[keep]], dtype=bool)
        right = np.asarray([bool(value) for value in reference[keep]], dtype=bool)
        if not np.array_equal(left, right):
            raise ValueError(f"{column}: boolean values differ")
        return {"mode": "exact_boolean", "compared": int(keep.sum())}

    if column in INTEGER_COLUMNS or pd.api.types.is_integer_dtype(actual.dtype):
        left = pd.to_numeric(actual[keep], errors="raise").to_numpy(dtype=np.int64)
        right_numeric = pd.to_numeric(reference[keep], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(right_numeric).all() or not np.equal(
            right_numeric, np.floor(right_numeric)
        ).all():
            raise ValueError(f"{column}: reference integer values are not integral")
        if not np.array_equal(left, right_numeric.astype(np.int64)):
            raise ValueError(f"{column}: integer values differ")
        return {"mode": "exact_integer", "compared": int(keep.sum())}

    if pd.api.types.is_numeric_dtype(actual.dtype):
        left = pd.to_numeric(actual[keep], errors="raise").to_numpy(dtype=float)
        right = pd.to_numeric(reference[keep], errors="raise").to_numpy(dtype=float)
        left_inf = np.isinf(left)
        right_inf = np.isinf(right)
        if not np.array_equal(left_inf, right_inf):
            raise ValueError(f"{column}: infinity mask differs")
        if left_inf.any() and not np.array_equal(
            np.signbit(left[left_inf]), np.signbit(right[right_inf])
        ):
            raise ValueError(f"{column}: infinity signs differ")
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any() and not np.isclose(
            left[finite], right[finite], atol=ATOL, rtol=RTOL, equal_nan=False
        ).all():
            differences = np.abs(left[finite] - right[finite])
            raise ValueError(
                f"{column}: numeric values exceed tolerance; max_abs={float(differences.max())}"
            )
        if finite.any():
            differences = np.abs(left[finite] - right[finite])
            denominator = np.maximum(np.abs(right[finite]), np.finfo(float).tiny)
            max_absolute = float(differences.max())
            max_relative = float((differences / denominator).max())
        else:
            max_absolute = 0.0
            max_relative = 0.0
        return {
            "mode": "numeric_tolerance",
            "compared": int(keep.sum()),
            "max_absolute_difference": max_absolute,
            "max_relative_difference": max_relative,
        }

    left = actual[keep].map(canonical_scalar).to_numpy(dtype=str)
    right = reference[keep].map(canonical_scalar).to_numpy(dtype=str)
    if not np.array_equal(left, right):
        raise ValueError(f"{column}: exact string/hash values differ")
    return {"mode": "exact_string", "compared": int(keep.sum())}


def compare_table(kind: str, actual_path: Path, reference_root: Path) -> dict[str, Any]:
    actual_full = pd.read_parquet(actual_path)
    if len(actual_full) != EXPECTED_ROWS[kind]:
        raise ValueError(
            f"{kind}: expected {EXPECTED_ROWS[kind]} rows, observed {len(actual_full)}"
        )
    if kind in EXPECTED_STATUSES:
        statuses = {
            str(key): int(value)
            for key, value in actual_full["status"].value_counts().to_dict().items()
        }
        if statuses != EXPECTED_STATUSES[kind]:
            raise ValueError(f"{kind}: status counts differ from frozen counts: {statuses}")

    stable = stable_columns(actual_full)
    actual = actual_full.loc[:, stable].copy()
    reference, reference_mode, reference_record = load_reference(
        reference_root, kind, actual_full
    )
    if len(reference) != len(actual):
        raise ValueError(
            f"{kind}: reference row count differs: expected {len(actual)}, "
            f"observed {len(reference)}"
        )
    if list(reference.columns) != list(actual.columns):
        raise ValueError(f"{kind}: comparable schemas differ after excluding volatile columns")

    keys = key_columns(kind)
    missing_keys = [column for column in keys if column not in actual.columns]
    if missing_keys:
        raise ValueError(f"{kind}: primary key columns missing: {missing_keys}")
    actual_tokens = row_tokens(actual, keys)
    reference_tokens = row_tokens(reference, keys)
    if actual_tokens.duplicated().any() or reference_tokens.duplicated().any():
        raise ValueError(f"{kind}: duplicate comparison primary keys")
    if sorted(actual_tokens.tolist()) != sorted(reference_tokens.tolist()):
        raise ValueError(f"{kind}: exact primary-key sets differ")
    actual = actual.assign(_comparison_key=actual_tokens).sort_values(
        "_comparison_key", kind="stable"
    )
    reference = reference.assign(_comparison_key=reference_tokens).sort_values(
        "_comparison_key", kind="stable"
    )
    actual.drop(columns="_comparison_key", inplace=True)
    reference.drop(columns="_comparison_key", inplace=True)
    actual.reset_index(drop=True, inplace=True)
    reference.reset_index(drop=True, inplace=True)

    column_reports = {
        column: compare_column(actual[column], reference[column], column)
        for column in actual.columns
    }
    max_absolute = max(
        (
            float(record.get("max_absolute_difference", 0.0))
            for record in column_reports.values()
        ),
        default=0.0,
    )
    max_relative = max(
        (
            float(record.get("max_relative_difference", 0.0))
            for record in column_reports.values()
        ),
        default=0.0,
    )
    return {
        "status": "PASS",
        "rows": int(len(actual)),
        "columns": len(actual.columns),
        "key_columns": keys,
        "reference_mode": reference_mode,
        "reference": reference_record,
        "excluded_volatile_columns": sorted(
            set(actual_full.columns) & EXCLUDED_COMPARISON_COLUMNS
        ),
        "actual_sha256": sha256_file(actual_path),
        "max_absolute_difference": max_absolute,
        "max_relative_difference": max_relative,
        "column_modes": {
            mode: sum(1 for record in column_reports.values() if record["mode"] == mode)
            for mode in sorted({record["mode"] for record in column_reports.values()})
        },
    }


def parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bundle-root", type=Path, default=default_root)
    result.add_argument(
        "--acs-zip",
        type=Path,
        default=Path("data/raw/acs/2018/1-Year/csv_pus.zip"),
    )
    result.add_argument(
        "--protocol-manifest",
        type=Path,
        default=Path("protocol/acsincome_region_budget_manifest.csv"),
    )
    result.add_argument("--reference-root", type=Path, default=Path("reference/one_task"))
    result.add_argument(
        "--runtime-root", type=Path, default=Path("outputs/runtime/acsincome_region")
    )
    result.add_argument("--output-root", type=Path, default=Path("outputs/one_task"))
    result.add_argument(
        "--report",
        type=Path,
        default=Path("validation/ONE_TASK_REPRODUCTION_REPORT.json"),
    )
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    bundle_root = args.bundle_root.resolve()
    archive_path = resolve_from(bundle_root, args.acs_zip)
    manifest_source = resolve_from(bundle_root, args.protocol_manifest)
    reference_root = resolve_from(bundle_root, args.reference_root)
    runtime_root = resolve_from(bundle_root, args.runtime_root)
    output_root = resolve_from(bundle_root, args.output_root)
    validate_isolation(
        bundle_root=bundle_root,
        output_root=output_root,
        runtime_root=runtime_root,
        reference_root=reference_root,
    )
    formal_runner_path = bundle_root / "src/reliable_shift/stage2/formal_runner.py"
    if not formal_runner_path.is_file():
        raise FileNotFoundError("release bundle is missing the formal runner source")
    formal_runner_hash = sha256_file(formal_runner_path)
    if formal_runner_hash != FROZEN_FORMAL_RUNNER_SHA256:
        raise ValueError(
            "formal runner source differs from the frozen qualified implementation: "
            f"expected {FROZEN_FORMAL_RUNNER_SHA256}, observed {formal_runner_hash}"
        )
    if not reference_root.is_dir():
        raise FileNotFoundError(f"one-task reference directory is missing: {reference_root}")

    runtime_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    official = verify_official_zip(archive_path)
    data_path = runtime_root / "acsincome_region_audited.parquet"
    frame, materialization = materialize_acsincome(archive_path, data_path)
    registry_path = runtime_root / "dataset_registry_one_task.csv"
    registry = write_runtime_registry(frame, data_path, registry_path)
    budget_manifest_path = runtime_root / "acsincome_region_budget_manifest.parquet"
    manifest = validate_and_write_manifest(
        manifest_source, budget_manifest_path, frame, bundle_root
    )
    del frame

    config_path = runtime_root / "stage2_formal_one_task_runtime.yml"
    runtime_config = write_runtime_config(
        registry_path, budget_manifest_path, output_root, config_path
    )
    runner_payload, command = run_formal_shard(
        bundle_root=bundle_root,
        config_path=config_path,
        output_root=output_root,
    )
    shard_root = output_root / TASK / "split_0" / "logistic_regression" / "model_seed_0"
    marker_path = shard_root / "shard.complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError("formal runner did not create shard.complete.json")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_marker = {
        "schema_version": "stage2_v2",
        "dataset": TASK,
        "source_family": "acs",
        "split_seed": 0,
        "base_model": "logistic_regression",
        "model_seed": 0,
        "config_hash": runtime_config["sha256"],
        "data_hash": FROZEN_PROCESSED_SHA256,
        "split_hash": FROZEN_SPLIT_HASH,
        "commit": FORMAL_COMMIT,
        "technical_reruns": 0,
    }
    marker_mismatches = {
        key: {"expected": expected, "observed": marker.get(key)}
        for key, expected in expected_marker.items()
        if marker.get(key) != expected
    }
    if marker_mismatches:
        raise ValueError(f"formal runner marker differs from the fixed task: {marker_mismatches}")

    comparisons: dict[str, Any] = {}
    for kind, filename in RESULT_FILES.items():
        result_path = shard_root / filename
        if not result_path.is_file():
            raise FileNotFoundError(f"formal runner output is missing: {result_path}")
        marker_record = marker.get("files", {}).get(filename, {})
        observed_hash = sha256_file(result_path)
        if marker_record.get("sha256") != observed_hash:
            raise ValueError(f"formal runner marker checksum differs for {filename}")
        if marker_record.get("rows") != EXPECTED_ROWS[kind]:
            raise ValueError(f"formal runner marker row count differs for {filename}")
        comparisons[kind] = compare_table(kind, result_path, reference_root)

    return {
        "schema_version": "stage3r-one-task-reproduction-v1",
        "status": "PASS",
        "task": TASK,
        "configuration": {
            "split_seed": 0,
            "base_model": "logistic_regression",
            "model_seed": 0,
            "formal_commit": FORMAL_COMMIT,
            "formal_runner_sha256": FROZEN_FORMAL_RUNNER_SHA256,
            "numeric_atol": ATOL,
            "numeric_rtol": RTOL,
        },
        "official_input": official,
        "materialization": materialization,
        "runtime_registry": registry,
        "budget_manifest": manifest,
        "runtime_config": runtime_config,
        "runner": {
            "status": runner_payload.get("status"),
            "reason": runner_payload.get("reason"),
            "rows": runner_payload.get("rows"),
            "command": command,
        },
        "comparison": comparisons,
        "excluded_comparison_columns": sorted(EXCLUDED_COMPARISON_COLUMNS),
        "runtime_seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    bundle_root = args.bundle_root.resolve()
    report_path = resolve_from(bundle_root, args.report)
    try:
        report = execute(args)
    except Exception as exc:
        report = {
            "schema_version": "stage3r-one-task-reproduction-v1",
            "status": "FAIL",
            "task": TASK,
            "formal_commit": FORMAL_COMMIT,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
