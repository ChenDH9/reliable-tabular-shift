from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schemas import validate_schema


def valid_completed_shard(path: Path, kind: str, expected: dict) -> bool:
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.exists() or not marker.exists():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        validate_schema(pd.read_parquet(path), kind)
        return True
    except Exception:
        return False


def should_run(path: Path, kind: str, expected: dict) -> bool:
    return not valid_completed_shard(path, kind, expected)

