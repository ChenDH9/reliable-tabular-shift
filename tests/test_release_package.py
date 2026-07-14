
from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.package
def test_package_is_gitless_and_has_authorized_mit_license() -> None:
    assert not (ROOT / ".git").exists()
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Copyright (c) 2026 Dinghao Chen" in license_text
    assert not (ROOT / "LICENSE_SELECTION_REQUIRED.md").exists()
    assert (ROOT / "LICENSE_SCOPE.md").is_file()
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["authors"] == [
        {
            "family-names": "Chen",
            "given-names": "Dinghao",
            "affiliation": (
                "College of Big Data, Baoshan University, Baoshan, Yunnan 678000, China"
            ),
        }
    ]
    assert citation["version"] == "1.0.0"
    assert citation["license"] == "MIT"


@pytest.mark.package
def test_fixed_config_and_claims_authority() -> None:
    config = yaml.safe_load((ROOT / "configs/stage2_formal_release.yml").read_text())
    assert config["budgets"] == [0, 25, 50, 100, 250, 500]
    assert config["split_seeds"] == [0, 1, 2]
    assert config["adaptation_seeds"] == 100
    claims = ROOT / "protocol/FINAL_CLAIMS_REGISTRY.csv"
    assert hashlib.sha256(claims.read_bytes()).hexdigest() == "1d7dd53c0c43292265a0c53451d4851b8707be5cef0e5fe9380e1b0dd900dd9a"


@pytest.mark.package
def test_level_b_authorities_are_aggregate_only() -> None:
    reference = ROOT / "reference/one_task"
    assert len(list(reference.glob("*.csv"))) == 7
    manifest = ROOT / "protocol/acsincome_region_budget_manifest.csv"
    assert sum(1 for _ in manifest.open(encoding="utf-8")) == 601
    forbidden = {"entity_id", "record_id", "group_id", "SERIALNO", "SPORDER"}
    for path in reference.glob("*.csv"):
        header = set(path.open(encoding="utf-8").readline().strip().split(","))
        assert not (header & forbidden)


@pytest.mark.package
def test_one_task_entrypoints_match_fixed_cli() -> None:
    assert "--task" not in (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "--task" not in (ROOT / "reproduce_one_task.sh").read_text(encoding="utf-8")


@pytest.mark.package
def test_canonical_csv_empty_string_respects_actual_missing_semantics() -> None:
    namespace = runpy.run_path(
        str(ROOT / "scripts/reproduce_one_task.py"), run_name="package_reference_audit"
    )
    raw = pd.DataFrame({"score_source": [""], "status_reason": [""]})
    actual = pd.DataFrame(
        {
            "score_source": pd.Series([pd.NA], dtype="string"),
            "status_reason": pd.Series([""], dtype="string"),
        }
    )
    converted = namespace["coerce_csv_reference"](raw, actual)
    assert pd.isna(converted.loc[0, "score_source"])
    assert converted.loc[0, "status_reason"] == ""
