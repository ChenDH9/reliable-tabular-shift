"""Stage 2B orchestration around the frozen production shard runner.

This module deliberately does not implement any scientific computation.  It
enumerates the preregistered shard matrix, invokes ``formal_runner`` in isolated
subprocesses, and publishes the merged runner outputs under the Stage 2 contract
names.  The production runner remains the sole owner of data loading, model
fitting, calibration, conformal inference, estimability, and resume semantics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "stage2_v2"
EXPECTED_DATASETS = (
    "acsincome_region",
    "acsfoodstamps_household",
    "brfss_diabetes",
    "nhanes_lead",
    "diabetes_readmission_patient_index",
    "college_scorecard_2018",
)
EXPECTED_SPLIT_SEEDS = (0, 1, 2)

# Source files emitted by merge_shard_directories and the formal contract names.
# Estimability rows are also the exact support-probability records produced by the
# runner, so that canonical table is published under both requested views.
MERGED_OUTPUT_DESTINATIONS: dict[str, tuple[str, ...]] = {
    "probability.parquet": ("probability_results.parquet",),
    "conformal.parquet": ("conformal_results.parquet",),
    "threshold.parquet": ("threshold_manifest.parquet",),
    "estimability.parquet": (
        "estimability_results.parquet",
        "support_probability_results.parquet",
    ),
    "failures.parquet": ("failures.parquet",),
    "not_estimable.parquet": ("not_estimable.parquet",),
    "split_prevalence_audit.parquet": ("split_prevalence_audit.parquet",),
}

RESULT_DESCRIPTIONS = {
    "probability_results.parquet": "formal probability calibration results",
    "conformal_results.parquet": "formal standard and Mondrian conformal results",
    "threshold_manifest.parquet": "class-level conformal threshold manifest",
    "estimability_results.parquet": "observed estimability and exact support records",
    "support_probability_results.parquet": (
        "exact hypergeometric support-probability view of estimability records"
    ),
    "failures.parquet": "preserved technical failure rows",
    "not_estimable.parquet": "preserved scientifically not-estimable rows",
    "split_prevalence_audit.parquet": "frozen role split prevalence audit",
    "simulation_probability_results.parquet": "formal probability simulation results",
    "simulation_conformal_results.parquet": "formal conformal simulation results",
    "task_level_summary.parquet": "task-equal formal result summary",
    "source_family_summary.parquet": "source-family-equal formal result summary",
    "leave_one_task_out.parquet": "leave-one-task-out sensitivity summary",
    "leave_one_family_out.parquet": "leave-one-source-family-out sensitivity summary",
    "manuscript_facts.json": "machine-readable manuscript numerical fact registry",
    "claims_registry.csv": "claim-to-fact numerical registry",
}

RESULT_SCHEMAS = {
    **{
        name: SCHEMA_VERSION
        for destinations in MERGED_OUTPUT_DESTINATIONS.values()
        for name in destinations
    },
    "simulation_probability_results.parquet": "stage2-simulation-v1",
    "simulation_conformal_results.parquet": "stage2-simulation-v1",
    "task_level_summary.parquet": "stage2-derived-v1",
    "source_family_summary.parquet": "stage2-derived-v1",
    "leave_one_task_out.parquet": "stage2-derived-v1",
    "leave_one_family_out.parquet": "stage2-derived-v1",
    "manuscript_facts.json": "json",
    "claims_registry.csv": "csv",
}

BLAS_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, order=True)
class ShardSpec:
    dataset: str
    split_seed: int
    base_model: str
    model_seed: int

    @property
    def identifier(self) -> str:
        return (
            f"{self.dataset}__split-{self.split_seed}__"
            f"{self.base_model}__model-seed-{self.model_seed}"
        )

    def directory(self, output_root: Path) -> Path:
        return (
            output_root
            / self.dataset
            / f"split_{self.split_seed}"
            / self.base_model
            / f"model_seed_{self.model_seed}"
        )


def load_formal_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("formal config must contain a mapping")
    required = {"schema_version", "data_registry", "output_root", "split_seeds", "models"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"formal config missing required keys: {missing}")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"formal config schema must be {SCHEMA_VERSION!r}, got {config['schema_version']!r}"
        )
    return config


def datasets_from_registry(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "dataset" not in reader.fieldnames:
            raise ValueError("data registry must contain a dataset column")
        observed = [str(row["dataset"]).strip() for row in reader if row.get("dataset")]
    if len(observed) != len(set(observed)):
        raise ValueError("data registry contains duplicate dataset rows")
    missing = sorted(set(EXPECTED_DATASETS) - set(observed))
    extra = sorted(set(observed) - set(EXPECTED_DATASETS))
    if missing or extra:
        raise ValueError(f"formal registry task set mismatch: missing={missing}, extra={extra}")
    return EXPECTED_DATASETS


def _model_seed_plan(config: Mapping[str, Any]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    plan: list[tuple[str, tuple[int, ...]]] = []
    kinds_seen: set[str] = set()
    models = config.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("formal config models must be a mapping")
    for model_name, value in models.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"model {model_name!r} must contain a mapping")
        kind = str(value.get("kind", ""))
        if kind == "logistic_regression":
            seeds = (0,)
        elif kind == "xgboost":
            seeds = (0, 1)
        else:
            raise ValueError(f"unexpected formal model kind: {kind!r}")
        if kind in kinds_seen:
            raise ValueError(f"formal config contains more than one {kind!r} model")
        kinds_seen.add(kind)
        plan.append((str(model_name), seeds))
    if kinds_seen != {"logistic_regression", "xgboost"}:
        raise ValueError("formal config must contain one logistic_regression and one xgboost model")
    return tuple(plan)


def build_shard_matrix(
    config: Mapping[str, Any], datasets: Sequence[str] = EXPECTED_DATASETS
) -> tuple[ShardSpec, ...]:
    """Enumerate the frozen 6 x 3 x (1 + 2) = 54 formal model shards."""

    if tuple(datasets) != EXPECTED_DATASETS:
        missing = sorted(set(EXPECTED_DATASETS) - set(datasets))
        extra = sorted(set(datasets) - set(EXPECTED_DATASETS))
        raise ValueError(f"formal dataset order/set mismatch: missing={missing}, extra={extra}")
    splits = tuple(int(seed) for seed in config.get("split_seeds", ()))
    if splits != EXPECTED_SPLIT_SEEDS:
        raise ValueError(f"formal split seeds must be {EXPECTED_SPLIT_SEEDS}, got {splits}")
    matrix = tuple(
        ShardSpec(dataset, split_seed, model_name, model_seed)
        for dataset in datasets
        for split_seed in splits
        for model_name, model_seeds in _model_seed_plan(config)
        for model_seed in model_seeds
    )
    if len(matrix) != 54 or len(set(matrix)) != 54:
        raise RuntimeError("formal shard matrix is not exactly 54 unique shards")
    return matrix


def dynamic_worker_limit(
    available_ram_gib: float,
    qualification_peak_worker_ram_gib: float | None,
) -> int:
    """Apply the preregistered RAM formula, falling back to one safe worker."""

    if available_ram_gib <= 0:
        raise ValueError("available RAM must be positive")
    if qualification_peak_worker_ram_gib is None:
        return 1
    if qualification_peak_worker_ram_gib <= 0:
        raise ValueError("qualification peak worker RAM must be positive")
    calculated = math.floor((available_ram_gib - 16.0) / (1.5 * qualification_peak_worker_ram_gib))
    return max(1, min(8, calculated))


def select_workers(requested: int | None, dynamic_limit: int) -> int:
    if dynamic_limit < 1:
        raise ValueError("dynamic worker limit must be at least one")
    if requested is None:
        return dynamic_limit
    if requested < 1:
        raise ValueError("requested workers must be at least one")
    return min(requested, dynamic_limit)


def worker_environment(blas_threads: int, base: Mapping[str, str] | None = None) -> dict[str, str]:
    if blas_threads < 1:
        raise ValueError("BLAS threads per worker must be at least one")
    environment = dict(os.environ if base is None else base)
    for name in BLAS_THREAD_VARIABLES:
        environment[name] = str(blas_threads)
    return environment


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _run_one_shard(
    spec: ShardSpec,
    *,
    config_path: Path,
    commit: str,
    output_root: Path,
    log_root: Path,
    blas_threads: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "reliable_shift.stage2.formal_runner",
        "run-shard",
        "--dataset",
        spec.dataset,
        "--split-seed",
        str(spec.split_seed),
        "--base-model",
        spec.base_model,
        "--model-seed",
        str(spec.model_seed),
        "--config",
        str(config_path),
        "--commit",
        commit,
        "--output-root",
        str(output_root),
    ]
    started = time.perf_counter()
    started_utc = utc_now()
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=worker_environment(blas_threads),
    )
    elapsed = time.perf_counter() - started
    log = (
        f"started_utc={started_utc}\n"
        f"finished_utc={utc_now()}\n"
        f"runtime_seconds={elapsed:.9f}\n"
        f"returncode={completed.returncode}\n"
        f"command={json.dumps(command)}\n"
        "--- stdout ---\n"
        f"{completed.stdout}"
        "\n--- stderr ---\n"
        f"{completed.stderr}"
    )
    _atomic_write_text(log_root / f"{spec.identifier}.log", log)
    record: dict[str, Any] = {
        **asdict(spec),
        "identifier": spec.identifier,
        "runtime_seconds": elapsed,
        "returncode": completed.returncode,
        "finished_utc": utc_now(),
    }
    if completed.returncode != 0:
        record.update(status="technical_failure", reason="formal runner CLI returned nonzero")
        return record
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        record.update(status="technical_failure", reason=f"invalid runner JSON: {exc}")
        return record
    record.update(
        status=str(payload.get("status", "technical_failure")),
        reason=str(payload.get("reason", "")),
        runner_result=payload,
    )
    return record


def _manifest_payload(
    *,
    status: str,
    config_path: Path,
    commit: str,
    output_root: Path,
    matrix: Sequence[ShardSpec],
    records: Mapping[str, Mapping[str, Any]],
    worker_limit: int,
    selected_workers: int,
    requested_workers: int | None,
    available_ram_gib: float,
    peak_worker_ram_gib: float | None,
    blas_threads: int,
    started_utc: str,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "status_reason": reason,
        "formal_execution": True,
        "formal_runner_entrypoint": "python -m reliable_shift.stage2.formal_runner run-shard",
        "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "commit": commit,
        "output_root": str(output_root),
        "started_utc": started_utc,
        "updated_utc": utc_now(),
        "matrix_size": len(matrix),
        "matrix": [asdict(spec) for spec in matrix],
        "completed_shards": len(records),
        "shard_records": [records[key] for key in sorted(records)],
        "resources": {
            "available_ram_gib": available_ram_gib,
            "qualification_peak_worker_ram_gib": peak_worker_ram_gib,
            "dynamic_worker_limit": worker_limit,
            "requested_workers": requested_workers,
            "selected_workers": selected_workers,
            "blas_threads_per_worker": blas_threads,
            "blas_environment_variables": list(BLAS_THREAD_VARIABLES),
        },
    }


def promote_merged_outputs(merge_root: Path, output_root: Path) -> tuple[Path, ...]:
    """Atomically publish production merge files under contract output names."""

    output_root.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    for source_name, destination_names in MERGED_OUTPUT_DESTINATIONS.items():
        source = merge_root / source_name
        if not source.is_file():
            raise FileNotFoundError(f"production merge output missing: {source}")
        for destination_name in destination_names:
            destination = output_root / destination_name
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=output_root
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            published.append(destination)
    merge_manifest = merge_root / "merge_manifest.json"
    if merge_manifest.is_file():
        destination = output_root / "merge_manifest.json"
        _atomic_write_text(destination, merge_manifest.read_text(encoding="utf-8"))
    return tuple(published)


def _parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _result_row_count(path: Path) -> int | str:
    if path.suffix == ".parquet":
        return _parquet_row_count(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    return "n/a"


def write_result_index(output_root: Path, paths: Iterable[Path]) -> Path:
    destination = output_root / "result_index.csv"
    rows = []
    for path in sorted(paths, key=lambda item: item.name):
        rows.append(
            {
                "server_relative_path": f"results/stage2/{path.name}",
                "row_count": _result_row_count(path),
                "file_size": path.stat().st_size,
                "sha256": sha256_file(path),
                "schema": RESULT_SCHEMAS[path.name],
                "description": RESULT_DESCRIPTIONS[path.name],
            }
        )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=output_root,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def refresh_result_index(output_root: Path) -> Path:
    """Rebuild the index after simulations or postprocessing add formal artifacts.

    Only the explicit formal allowlist is considered. Qualification,
    microbenchmark, cache, and arbitrary files can therefore never leak into the
    formal result index.
    """

    paths = [output_root / name for name in RESULT_SCHEMAS if (output_root / name).is_file()]
    if not paths:
        raise FileNotFoundError(f"no indexed formal results found under {output_root}")
    return write_result_index(output_root, paths)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def write_runtime_summary(
    output_root: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    wall_seconds: float,
    workers: int,
    blas_threads: int,
) -> Path:
    runtimes = [float(record["runtime_seconds"]) for record in records]
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    payload = {
        "schema_version": SCHEMA_VERSION,
        "shard_count": len(records),
        "wall_runtime_seconds": wall_seconds,
        "summed_subprocess_runtime_seconds": sum(runtimes),
        "shard_runtime_seconds": {
            "minimum": min(runtimes),
            "median": statistics.median(runtimes),
            "p95_nearest_rank": _percentile(runtimes, 0.95),
            "maximum": max(runtimes),
        },
        "status_counts": status_counts,
        "workers": workers,
        "blas_threads_per_worker": blas_threads,
        "timestamp_utc": utc_now(),
    }
    destination = output_root / "runtime_summary.json"
    atomic_write_json(destination, payload)
    return destination


def run_formal_matrix(
    *,
    config_path: Path,
    commit: str,
    output_root: Path,
    matrix: Sequence[ShardSpec],
    selected_workers: int,
    dynamic_limit: int,
    requested_workers: int | None,
    available_ram_gib: float,
    peak_worker_ram_gib: float | None,
    blas_threads: int,
) -> dict[str, Any]:
    """Run/resume all shards, merge through formal_runner, and publish outputs."""

    if len(matrix) != 54 or len(set(matrix)) != 54:
        raise ValueError("refusing formal execution without exactly 54 unique shards")
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "logs"
    started = time.perf_counter()
    started_utc = utc_now()
    records: dict[str, dict[str, Any]] = {}
    manifest_path = output_root / "run_manifest.json"

    def persist(status: str, reason: str = "") -> None:
        atomic_write_json(
            manifest_path,
            _manifest_payload(
                status=status,
                config_path=config_path,
                commit=commit,
                output_root=output_root,
                matrix=matrix,
                records=records,
                worker_limit=dynamic_limit,
                selected_workers=selected_workers,
                requested_workers=requested_workers,
                available_ram_gib=available_ram_gib,
                peak_worker_ram_gib=peak_worker_ram_gib,
                blas_threads=blas_threads,
                started_utc=started_utc,
                reason=reason,
            ),
        )

    persist("running")
    with ThreadPoolExecutor(max_workers=selected_workers) as pool:
        future_specs = {
            pool.submit(
                _run_one_shard,
                spec,
                config_path=config_path,
                commit=commit,
                output_root=output_root,
                log_root=log_root,
                blas_threads=blas_threads,
            ): spec
            for spec in matrix
        }
        for future in as_completed(future_specs):
            spec = future_specs[future]
            try:
                record = future.result()
            except Exception as exc:  # keep orchestration evidence before stopping
                record = {
                    **asdict(spec),
                    "identifier": spec.identifier,
                    "status": "technical_failure",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "returncode": None,
                    "runtime_seconds": 0.0,
                    "finished_utc": utc_now(),
                }
            records[spec.identifier] = record
            persist("running")

    failures = [record for record in records.values() if record["status"] == "technical_failure"]
    if failures:
        reason = f"{len(failures)} formal shard subprocesses failed"
        persist("failed", reason)
        write_runtime_summary(
            output_root,
            records=list(records.values()),
            wall_seconds=time.perf_counter() - started,
            workers=selected_workers,
            blas_threads=blas_threads,
        )
        raise RuntimeError(reason)
    acceptable = {"success", "skipped"}
    unexpected = [record for record in records.values() if record["status"] not in acceptable]
    if unexpected:
        reason = f"{len(unexpected)} formal shards returned an unexpected status"
        persist("failed", reason)
        raise RuntimeError(reason)

    directories = [spec.directory(output_root) for spec in matrix]
    merge_root = output_root / ".formal_merge"
    from reliable_shift.stage2.formal_runner import merge_shard_directories

    merge_report = merge_shard_directories(directories, merge_root)
    published = promote_merged_outputs(merge_root, output_root)
    write_result_index(output_root, published)
    runtime_path = write_runtime_summary(
        output_root,
        records=list(records.values()),
        wall_seconds=time.perf_counter() - started,
        workers=selected_workers,
        blas_threads=blas_threads,
    )
    persist("complete")
    return {
        "status": "complete",
        "shard_count": len(matrix),
        "workers": selected_workers,
        "merge": merge_report,
        "published_outputs": [str(path) for path in published],
        "run_manifest": str(manifest_path),
        "runtime_summary": str(runtime_path),
        "result_index": str(output_root / "result_index.csv"),
    }


__all__ = [
    "BLAS_THREAD_VARIABLES",
    "EXPECTED_DATASETS",
    "EXPECTED_SPLIT_SEEDS",
    "MERGED_OUTPUT_DESTINATIONS",
    "SCHEMA_VERSION",
    "ShardSpec",
    "build_shard_matrix",
    "datasets_from_registry",
    "dynamic_worker_limit",
    "load_formal_config",
    "promote_merged_outputs",
    "refresh_result_index",
    "run_formal_matrix",
    "select_workers",
    "worker_environment",
    "write_result_index",
]
