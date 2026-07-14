from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .schemas import validate_schema


@contextmanager
def shard_lock(path: Path):
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = None
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
            lock.unlink(missing_ok=True)


def atomic_write_shard(frame: pd.DataFrame, path: Path, kind: str, metadata: dict) -> None:
    validate_schema(frame, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    with shard_lock(path):
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
            temporary = Path(handle.name)
        marker_tmp = path.with_suffix(path.suffix + ".complete.tmp")
        try:
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, path)
            marker_tmp.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(marker_tmp, path.with_suffix(path.suffix + ".complete"))
        finally:
            temporary.unlink(missing_ok=True)
            marker_tmp.unlink(missing_ok=True)

