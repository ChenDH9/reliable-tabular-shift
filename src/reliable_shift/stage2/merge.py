from __future__ import annotations

from pathlib import Path
import pandas as pd

from .schemas import validate_schema


def merge_shards(paths: list[Path], output: Path, kind: str) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    validate_schema(merged, kind)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    merged.to_parquet(temporary, index=False)
    temporary.replace(output)
    return merged

