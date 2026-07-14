from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def load_registry(path: Path) -> pd.DataFrame:
    registry = pd.read_csv(path)
    required = {"dataset", "audited_parquet_path", "processed_checksum", "feature_list", "excluded_columns"}
    if not required.issubset(registry):
        raise ValueError(f"registry missing {sorted(required - set(registry))}")
    return registry


def load_task(row: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(row["audited_parquet_path"])
    features = json.loads(row["feature_list"])
    if set(features) & set(json.loads(row["excluded_columns"])):
        raise ValueError("features overlap exclusions")
    return frame, features

