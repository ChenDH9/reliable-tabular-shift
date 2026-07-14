
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import psutil
import yaml

from reliable_shift.stage2.formal_orchestration import (
    build_shard_matrix,
    datasets_from_registry,
    dynamic_worker_limit,
    run_formal_matrix,
    select_workers,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Guarded optional frozen 54-shard reproduction")
    parser.add_argument("--data-registry", type=Path, required=True)
    parser.add_argument("--budget-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if os.environ.get("CONFIRM_OPTIONAL_FULL_REPRODUCTION") != "YES":
        raise SystemExit("explicit Level C confirmation is missing")
    if args.commit != "d745e5412c1d530a2ae64e2eaa42c85c1f64e419":
        raise SystemExit("formal execution commit does not match the frozen authority")
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/stage2_formal_release.yml").read_text())
    config["data_registry"] = str(args.data_registry.resolve())
    config["frozen_budget_manifest"] = str(args.budget_manifest.resolve())
    config["output_root"] = str(args.output_root.resolve())
    args.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        config_path = Path(handle.name)
    try:
        matrix = build_shard_matrix(config, datasets_from_registry(args.data_registry))
        if len(matrix) != 54:
            raise SystemExit(f"expected 54 frozen shards, observed {len(matrix)}")
        available = psutil.virtual_memory().available / 2**30
        limit = dynamic_worker_limit(available, 2.68)
        workers = select_workers(args.workers, limit)
        result = run_formal_matrix(
            config_path=config_path,
            commit=args.commit,
            output_root=args.output_root,
            matrix=matrix,
            selected_workers=workers,
            dynamic_limit=limit,
            requested_workers=args.workers,
            available_ram_gib=available,
            peak_worker_ram_gib=2.68,
            blas_threads=1,
        )
        print(result)
    finally:
        config_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
