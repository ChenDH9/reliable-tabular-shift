#!/usr/bin/env python3
"""Read-only verifier for the Stage 3R release-preparation tree.

``package`` validates only the material below ``release/stage3r``.  It is
therefore safe to run after copying or extracting that tree to a machine with
neither Git history nor the private frozen-results tree.  ``full_repository``
adds the Stage 3Q Git ancestry, branch, Claims Registry, and frozen-tree
authority checks.

The verifier never extracts an archive and never invokes an experiment,
manuscript builder, downloader, or network client.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import subprocess
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

STAGE3Q_SOURCE_COMMIT = "6cb094913cc6eaa7b388c9477fa5b17e181210c0"
STAGE3Q_RECHECK_COMMIT = "77c557aa0db4ab765315817a0b90595a167c12bc"
STAGE3R_BRANCH = "stage3r-reproducibility-release-packaging"
CLAIMS_REGISTRY = "results/stage2dl/FINAL_CLAIMS_REGISTRY.csv"
CLAIMS_SHA256 = "1d7dd53c0c43292265a0c53451d4851b8707be5cef0e5fe9380e1b0dd900dd9a"

PUBLIC_STRICT = "PUBLIC_STRICT"
SUBMISSION_STRICT = "SUBMISSION_STRICT"

REPRO_DIR = "reproducibility_bundle"
GITHUB_DIR = "github_staging"
PEERJ_DIR = "peerj_submission_materials_pending_author"

GITHUB_MANIFEST_FIELDS = (
    "path",
    "size_bytes",
    "sha256",
    "purpose",
    "license_status",
    "safe_to_publish",
)

FIXED_ETHICS = """This study used only publicly available, de-identified secondary datasets.
No surveys, interviews, interventions, human-subject experiments, or direct
interactions with individuals were conducted, and the authors did not access
direct personal identifiers. The submitting author's institution confirmed
that, given the exclusive use of publicly available, de-identified secondary
data and the absence of interaction with human participants, separate
institutional ethics review was not required."""

FIXED_AI_DISCLOSURE = """OpenAI Codex was used as a technical assistant for portions of code
generation
and debugging, workflow automation, automated testing, execution of the
author-specified computational pipeline, generation of derived tables and
figures, and language drafting. The author independently formulated the
research questions, designed the experiments, specified and approved the
analytical protocol, evaluated and interpreted the results, audited the final
code and outputs, verified the references and scientific claims, and takes full
responsibility for the integrity and accuracy of the submitted work. The AI
system was not listed as an author and did not make autonomous authorship,
ethical, or publication decisions."""

REPRO_REQUIRED_FILES = (
    "README_REPRODUCE.md",
    "QUICKSTART.md",
    "REPRODUCIBILITY_LEVELS.md",
    "LICENSE",
    "LICENSE_SCOPE.md",
    "CITATION.cff",
    "pyproject.toml",
    "environment.lock.yml",
    "requirements-export.txt",
    "Makefile",
    "reproduce_artifacts.sh",
    "reproduce_one_task.sh",
    "reproduce_full_experiment.sh",
    "verify_release.sh",
    "data_registry/OFFICIAL_DATA_SOURCES.csv",
    "data_registry/DATA_REBUILD_MANIFEST.csv",
    "data_registry/LICENSE_BOUNDARIES.md",
)
REPRO_REQUIRED_DIRS = (
    "src",
    "scripts",
    "tests",
    "configs",
    "protocol",
    "data_registry",
    "manuscript_build",
    "figure_source_data",
)
REPRO_AUTHORITY_FILES = (
    "configs/stage2_formal_release.yml",
    "protocol/FINAL_CLAIMS_REGISTRY.csv",
    "protocol/acsincome_region_budget_manifest.csv",
    "reference/one_task/reference_metadata.json",
    "reference/one_task/probability.csv",
    "reference/one_task/conformal.csv",
    "reference/one_task/threshold.csv",
    "reference/one_task/estimability.csv",
    "reference/one_task/failures.csv",
    "reference/one_task/not_estimable.csv",
    "reference/one_task/split_prevalence_audit.csv",
    "manuscript_build/AUDITED_ARTIFACT_HASHES.csv",
    "figure_source_data/SOURCE_DATA_INDEX.csv",
    "scripts/verify_release.py",
    "tests/test_release_package.py",
)
LEVEL_B_BUDGET_FIELDS = (
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
LEVEL_B_REFERENCE_ROWS = {
    "probability": 2010,
    "conformal": 2008,
    "threshold": 2004,
    "estimability": 500,
    "failures": 0,
    "not_estimable": 6,
    "split_prevalence_audit": 7,
}
LEVEL_B_ADAPTATION_POOL_DIGEST = (
    "be91f1f7e1abbf7ae1a7981cbfa641b51b47ff66a524a30771b7967352b98408"
)
LEVEL_B_FINAL_TEST_DIGEST = (
    "3cd9344070bb0370a0523146751a3ad5b11ba159f3b306d9319c2ada1aef1e7e"
)

GITHUB_REQUIRED_FILES = (
    "README.md",
    "QUICKSTART.md",
    "REPRODUCIBILITY_LEVELS.md",
    "CONTRIBUTING.md",
    "CITATION.cff",
    "LICENSE",
    "LICENSE_SCOPE.md",
    ".gitignore",
    ".gitattributes",
    "pyproject.toml",
    "environment.lock.yml",
    "requirements-export.txt",
    "Makefile",
    ".github/workflows/tests.yml",
    "GITHUB_UPLOAD_MANIFEST.csv",
    "PUBLIC_RELEASE_CHECKLIST.md",
    "FILES_EXCLUDED_FROM_PUBLIC_RELEASE.csv",
)
GITHUB_REQUIRED_DIRS = (
    "src",
    "scripts",
    "tests",
    "configs",
    "protocol",
    "data_registry",
    "manuscript_build",
    "figure_source_data",
)

PEERJ_REQUIRED_FILES = (
    "PEERJ_MATERIALS_MANIFEST.csv",
    "main_manuscript/manuscript_peerj.md",
    "main_manuscript/manuscript_peerj.tex",
    "main_manuscript/manuscript_peerj.pdf",
    "supplement/supplement_peerj.md",
    "supplement/supplement_peerj.tex",
    "supplement/supplement_peerj.pdf",
    "metadata/peerj_metadata_form.md",
    "metadata/AUTHOR_INPUT_TEMPLATE.yml",
    "metadata/AUTHOR_ACTIONS_CHECKLIST.md",
    "metadata/fixed_ethics_statement.md",
    "metadata/fixed_ai_disclosure.md",
    "metadata/funding_template.md",
    "metadata/competing_interests_template.md",
    "metadata/CRediT_template.csv",
    "metadata/acknowledgments_template.md",
    "metadata/data_code_availability_template.md",
    "submission_checklists/peerj_submission_checklist.md",
    "submission_checklists/FINAL_SUBMISSION_BLOCKERS.md",
    "references/references.bib",
    "references/reference_audit.csv",
    "references/data_source_trace.csv",
    "source_data/SOURCE_DATA_INDEX.csv",
)
PEERJ_REQUIRED_DIRS = ("figures", "tables", "source_data")
AUTHOR_INPUT_FIELDS = (
    "authors",
    "affiliations",
    "orcid",
    "corresponding_author",
    "submission_email",
    "funding",
    "competing_interests",
    "credit_roles",
    "acknowledgments",
    "ethics_confirmation",
    "codex_product",
    "codex_model_or_version",
    "codex_use_dates",
    "ai_figure_rights_confirmed",
    "ai_prompts_and_versions_retained",
    "repository_url",
    "release_commit",
    "release_tag",
    "software_license",
    "archive_doi",
    "archive_sha256",
    "icasp_2026_fulltext_reviewed",
)

PACKAGE_SPECS = (
    (
        "packages/reliable_tabular_shift_reproducibility.tar.gz",
        "packages/reliable_tabular_shift_reproducibility.tar.gz.sha256",
        REPRO_DIR,
        "tar",
    ),
    (
        "packages/github_staging.zip",
        "packages/github_staging.zip.sha256",
        GITHUB_DIR,
        "zip",
    ),
    (
        "packages/peerj_submission_materials_pending_author.zip",
        "packages/peerj_submission_materials_pending_author.zip.sha256",
        PEERJ_DIR,
        "zip",
    ),
)

TEXT_SUFFIXES = {
    "",
    ".bib",
    ".bst",
    ".cff",
    ".cfg",
    ".cls",
    ".csv",
    ".gitignore",
    ".gitattributes",
    ".ini",
    ".json",
    ".jsonl",
    ".lock",
    ".lua",
    ".md",
    ".py",
    ".r",
    ".sh",
    ".sty",
    ".template",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
PUBLIC_ALLOWED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".png", ".svg"}
SUBMISSION_ALLOWED_SUFFIXES = TEXT_SUFFIXES | {".docx", ".pdf", ".png", ".svg", ".xlsx"}

FORBIDDEN_SUFFIXES = (
    ".7z",
    ".bundle",
    ".db",
    ".diff",
    ".env",
    ".feather",
    ".h5",
    ".hdf5",
    ".joblib",
    ".npy",
    ".npz",
    ".parquet",
    ".patch",
    ".pem",
    ".pickle",
    ".pkl",
    ".pyc",
    ".pyo",
    ".rar",
    ".sqlite",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
)
FORBIDDEN_COMPONENTS = {
    ".git",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "archive",
    "build",
    "dist",
    "formal_results",
    "handoff",
    "htmlcov",
    "node_modules",
    "outputs",
    "pip-cache",
    "raw",
    "raw_data",
    "raw_results",
    "result_shards",
    "results",
    "run_outputs",
    "shards",
    "site-packages",
    "venv",
}

MAX_PUBLIC_FILE_BYTES = 25_000_000
MAX_PUBLIC_TREE_BYTES = 100_000_000
MAX_SUBMISSION_FILE_BYTES = 50_000_000
MAX_SUBMISSION_TREE_BYTES = 250_000_000
MAX_ARCHIVE_MEMBERS = 20_000
MAX_ARCHIVE_MEMBER_BYTES = 100_000_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 500_000_000
MAX_COMPRESSION_RATIO = 250.0


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_prose(text: str) -> str:
    """Normalize line wrapping without changing wording or punctuation."""

    return re.sub(r"\s+", " ", text).strip()


def safe_archive_name(name: str) -> bool:
    """Return whether an archive member is a normalized POSIX relative path."""

    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        return False
    if re.match(r"^[A-Za-z]:", name):
        return False
    path = PurePosixPath(name)
    parts = path.parts
    return bool(parts) and all(part not in {"", ".", ".."} for part in parts)


def basic_pdf_integrity(path: Path) -> bool:
    """Perform dependency-free structural checks suitable for a release gate."""

    if not path.is_file() or path.stat().st_size < 100:
        return False
    data = path.read_bytes()
    return (
        data.startswith(b"%PDF-")
        and b"/Type" in data
        and b"%%EOF" in data[-4096:]
        and b"/Encrypt" not in data
    )


def basic_png_integrity(path: Path) -> bool:
    """Verify the PNG signature, IHDR marker, and positive dimensions."""

    if not path.is_file() or path.stat().st_size < 24:
        return False
    data = path.read_bytes()[:24]
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return False
    return int.from_bytes(data[16:20], "big") > 0 and int.from_bytes(data[20:24], "big") > 0


class Audit:
    """Small JSON-serializable check collector."""

    def __init__(self, profile: str, root: Path, stage3r_root: Path) -> None:
        self.report: dict[str, Any] = {
            "status": "FAIL",
            "profile": profile,
            "root": str(root.resolve()),
            "stage3r_root": str(stage3r_root.resolve()),
            "checks": {},
            "errors": [],
            "warnings": [],
        }

    def add(self, name: str, ok: bool, details: Any) -> None:
        self.report["checks"][name] = {"ok": bool(ok), "details": details}
        if not ok:
            self.report["errors"].append(name)

    def finish(self) -> dict[str, Any]:
        self.report["status"] = "PASS" if not self.report["errors"] else "FAIL"
        self.report["check_count"] = len(self.report["checks"])
        self.report["passed_check_count"] = sum(
            int(item["ok"]) for item in self.report["checks"].values()
        )
        return self.report


def _locate_stage3r_root(root: Path) -> Path:
    nested = root / "release/stage3r"
    if nested.is_dir():
        return nested
    return root


def _is_standalone_repro_bundle(root: Path) -> bool:
    identifying_entries = (
        (root / "README_REPRODUCE.md").is_file(),
        (root / "src").is_dir(),
        (root / "scripts").is_dir(),
    )
    sibling_release_trees = (root / GITHUB_DIR, root / PEERJ_DIR, root / REPRO_DIR)
    return all(identifying_entries) and not any(path.exists() for path in sibling_release_trees)


def _is_standalone_github_staging(root: Path) -> bool:
    identifying_entries = (
        (root / "README.md").is_file(),
        (root / "GITHUB_UPLOAD_MANIFEST.csv").is_file(),
        (root / "src").is_dir(),
        (root / "scripts").is_dir(),
    )
    sibling_release_trees = (root / GITHUB_DIR, root / PEERJ_DIR, root / REPRO_DIR)
    return all(identifying_entries) and not any(path.exists() for path in sibling_release_trees)


def _missing_contract_entries(
    root: Path, required_files: Iterable[str], required_dirs: Iterable[str]
) -> dict[str, list[str]]:
    return {
        "missing_files": [
            relative for relative in required_files if not (root / relative).is_file()
        ],
        "missing_dirs": [relative for relative in required_dirs if not (root / relative).is_dir()],
    }


def _suffix(path: Path) -> str:
    name = path.name.lower()
    if name in {".gitignore", ".gitattributes"}:
        return name
    return path.suffix.lower()


def _forbidden_filename(relative: PurePosixPath) -> str | None:
    lowered = relative.as_posix().lower()
    components = {part.lower() for part in relative.parts}
    if components & FORBIDDEN_COMPONENTS:
        return "forbidden_cache_history_or_results_component"
    if any(lowered.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        return "forbidden_extension"
    if any(part.endswith(".egg-info") for part in components):
        return "packaging_cache"
    credential_names = {
        ".env",
        ".env.local",
        ".netrc",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ed25519",
        "id_rsa",
        "known_hosts",
    }
    if components & credential_names:
        return "credential_or_ssh_filename"
    return None


def _text_findings(text: str) -> set[str]:
    """Find risky material without echoing the matched secret into the report."""

    findings: set[str] = set()
    # Common private/server absolute path families; system shebangs under /usr are allowed.
    path_roots = "|".join(("Users", "home", "root", "tmp", "var", "mnt", "srv", "opt", "Volumes"))
    if re.search(rf"(?m)(?:^|[\s\"'=:(])/(?:{path_roots})(?:/|\b)", text):
        findings.add("private_absolute_path")
    generic_absolute = re.compile(
        r"(?m)(?:^|[\s\"'=:(])(/(?!/)[A-Za-z0-9._~-]+(?:/[A-Za-z0-9._~+-]+)+)"
    )
    safe_system_paths = {"/dev/null", "/bin/bash", "/bin/sh", "/usr/bin/env"}
    absolute_paths = (
        match.group(1).rstrip(".,;:)") for match in generic_absolute.finditer(text)
    )
    if any(path not in safe_system_paths for path in absolute_paths):
        findings.add("private_absolute_path")
    if re.search(r"(?i)(?:^|[\s\"'=:(])[A-Z]:[\\/][^\s\"']+", text):
        findings.add("private_absolute_path")

    ssh_pattern = r"(?i)\bssh://|\bssh\b[^\r\n]{0,120}(?:@\S+|-p\s+\d+)"
    if re.search(ssh_pattern, text):
        findings.add("ssh_endpoint")
    if re.search(r"(?im)^\s*(?:HostName|IdentityFile|ProxyJump)\s+\S+", text):
        findings.add("ssh_endpoint")
    endpoint = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{2,5})?\b")
    conda_dependency = re.compile(
        r"^\s*-\s*[A-Za-z0-9_.+-]+="
        r"(?P<version>\d{1,3}(?:\.\d{1,3}){3})=[^\s#]+\s*(?:#.*)?$"
    )
    unsafe_endpoint = False
    for line in text.splitlines():
        conda_match = conda_dependency.fullmatch(line)
        conda_version = conda_match.group("version") if conda_match else None
        for match in endpoint.finditer(line):
            value = match.group(0)
            # Four-component package versions are common in explicit Conda locks.
            # A port is never a version and therefore never receives this exemption.
            if ":" not in value and value == conda_version:
                continue
            unsafe_endpoint = True
            break
        if unsafe_endpoint:
            break
    if unsafe_endpoint:
        findings.add("ip_or_server_endpoint")
    if re.search(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text):
        findings.add("email_address")

    pem_marker = "-----BEGIN " + "PRIVATE KEY-----"
    if pem_marker in text or re.search(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", text):
        findings.add("private_key")
    high_risk_tokens = (
        r"\bgh[pousr]_[A-Za-z0-9]{20,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{20,}\b",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bsk-[A-Za-z0-9_-]{20,}\b",
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}",
    )
    if any(re.search(pattern, text) for pattern in high_risk_tokens):
        findings.add("credential_token")

    assignment = re.compile(
        r"(?im)^\s*[\"']?(?:password|passwd|api[_-]?key|access[_-]?token|auth[_-]?token|secret)"
        r"[\"']?\s*[:=]\s*[\"']?([^\s\"'#]{6,})"
    )
    safe_values = {"placeholder", "redacted", "required", "unknown", "null", "none"}
    for match in assignment.finditer(text):
        value = match.group(1).strip().lower()
        if value not in safe_values and not value.startswith(("${", "<author", "<required")):
            findings.add("credential_assignment")
            break
    return findings


def _walk_tree(root: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Walk without following links and report every non-regular object."""

    files: list[Path] = []
    violations: list[dict[str, str]] = []
    if root.is_symlink():
        return files, [{"path": ".", "rule": "symbolic_link"}]
    if not root.is_dir():
        return files, [{"path": ".", "rule": "missing_directory"}]
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            relative = directory.relative_to(root).as_posix() or "."
            violations.append({"path": relative, "rule": "unreadable_directory"})
            continue
        for path in children:
            relative = path.relative_to(root).as_posix()
            try:
                metadata = path.lstat()
            except OSError:
                violations.append({"path": relative, "rule": "unreadable_path"})
                continue
            mode = metadata.st_mode
            if stat.S_ISLNK(mode):
                violations.append({"path": relative, "rule": "symbolic_link"})
            elif stat.S_ISDIR(mode):
                stack.append(path)
            elif stat.S_ISREG(mode):
                if metadata.st_nlink != 1:
                    violations.append({"path": relative, "rule": "hard_link"})
                files.append(path)
            else:
                violations.append({"path": relative, "rule": "special_file"})
    return sorted(files), violations


def scan_tree(root: Path, policy: str) -> dict[str, Any]:
    """Apply PUBLIC_STRICT or SUBMISSION_STRICT to an on-disk tree."""

    if policy not in {PUBLIC_STRICT, SUBMISSION_STRICT}:
        raise ValueError(f"unknown scan policy: {policy}")
    files, violations = _walk_tree(root)
    allowed = PUBLIC_ALLOWED_SUFFIXES if policy == PUBLIC_STRICT else SUBMISSION_ALLOWED_SUFFIXES
    max_file = MAX_PUBLIC_FILE_BYTES if policy == PUBLIC_STRICT else MAX_SUBMISSION_FILE_BYTES
    max_tree = MAX_PUBLIC_TREE_BYTES if policy == PUBLIC_STRICT else MAX_SUBMISSION_TREE_BYTES
    total = 0
    for path in files:
        relative = PurePosixPath(path.relative_to(root).as_posix())
        size = path.stat().st_size
        total += size
        forbidden = _forbidden_filename(relative)
        if forbidden:
            violations.append({"path": relative.as_posix(), "rule": forbidden})
        suffix = _suffix(path)
        if suffix not in allowed:
            violations.append({"path": relative.as_posix(), "rule": "unapproved_extension"})
        if size > max_file:
            violations.append({"path": relative.as_posix(), "rule": "oversized_file"})
        if suffix in TEXT_SUFFIXES:
            try:
                data = path.read_bytes()
                if b"\x00" in data:
                    raise UnicodeError
                text = data.decode("utf-8")
            except (OSError, UnicodeError):
                violations.append({"path": relative.as_posix(), "rule": "non_utf8_text"})
                continue
            for finding in sorted(_text_findings(text)):
                violations.append({"path": relative.as_posix(), "rule": finding})
    if total > max_tree:
        violations.append({"path": ".", "rule": "oversized_tree"})
    unique = sorted({(item["path"], item["rule"]) for item in violations})
    return {
        "policy": policy,
        "file_count": len(files),
        "size_bytes": total,
        "violations": [{"path": path, "rule": rule} for path, rule in unique],
    }


def expected_github_license_status(relative: str) -> str | None:
    """Return the publication-manifest license classification for a public file."""

    if relative == "LICENSE":
        return "MIT_license_text"
    if relative == "LICENSE_SCOPE.md":
        return "license_scope_notice"
    if relative == "CITATION.cff":
        return "citation_metadata_for_MIT_software"
    software_roots = {
        "Makefile",
        "pyproject.toml",
        "reproduce_artifacts.sh",
        "reproduce_full_experiment.sh",
        "reproduce_one_task.sh",
        "verify_release.sh",
    }
    if relative in software_roots or relative.startswith(
        (".github/", "configs/", "scripts/", "src/", "tests/")
    ):
        return "MIT"
    documentation_roots = {
        ".gitattributes",
        ".gitignore",
        "CONTRIBUTING.md",
        "DOCUMENT_TOOLCHAIN.lock.md",
        "FILES_EXCLUDED_FROM_PUBLIC_RELEASE.csv",
        "PUBLIC_RELEASE_CHECKLIST.md",
        "QUICKSTART.md",
        "README.md",
        "REPRODUCIBILITY_LEVELS.md",
        "environment.lock.yml",
        "requirements-export.txt",
    }
    if relative in documentation_roots:
        return "not_covered_by_MIT_repository_documentation_or_metadata"
    if relative in {
        "data_registry/OFFICIAL_DATA_SOURCES.csv",
        "data_registry/DATA_REBUILD_MANIFEST.csv",
    }:
        return "not_covered_by_MIT_upstream_data_terms_apply"
    if relative == "data_registry/LICENSE_BOUNDARIES.md":
        return "not_covered_by_MIT_license_boundary_documentation"
    if relative.startswith(("figure_source_data/", "reference/")):
        return "not_covered_by_MIT_research_data"
    if relative.startswith(("manuscript_build/", "protocol/")):
        return "not_covered_by_MIT_author_research_material"
    return None


def verify_github_manifest(root: Path) -> dict[str, Any]:
    """Require a strict manifest/file-tree bijection, excluding the manifest itself."""

    manifest = root / "GITHUB_UPLOAD_MANIFEST.csv"
    errors: list[str] = []
    rows: list[dict[str, str]] = []
    headers: list[str] = []
    if not manifest.is_file():
        return {"errors": ["manifest_missing"], "row_count": 0, "actual_file_count": 0}
    try:
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error):
        return {"errors": ["manifest_unreadable"], "row_count": 0, "actual_file_count": 0}
    if tuple(headers) != GITHUB_MANIFEST_FIELDS:
        errors.append("fields_not_exact_or_ordered")

    listed: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=2):
        if set(row) != set(GITHUB_MANIFEST_FIELDS) or None in row:
            errors.append(f"row_{index}_schema")
            continue
        relative = row["path"]
        if not safe_archive_name(relative) or relative.endswith("/"):
            errors.append(f"row_{index}_unsafe_path")
            continue
        normalized = PurePosixPath(relative).as_posix()
        if normalized != relative:
            errors.append(f"row_{index}_noncanonical_path")
        if relative == "GITHUB_UPLOAD_MANIFEST.csv":
            errors.append("manifest_must_not_list_itself")
        if relative in listed:
            errors.append(f"duplicate_path:{relative}")
        if relative.casefold() in {item.casefold() for item in listed}:
            errors.append(f"casefold_path_collision:{relative}")
        listed[relative] = row
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            errors.append(f"invalid_sha256:{relative}")
        if not row["size_bytes"].isdigit() or str(int(row["size_bytes"] or "0")) != row[
            "size_bytes"
        ]:
            errors.append(f"invalid_size:{relative}")
        if not row["purpose"].strip():
            errors.append(f"missing_purpose:{relative}")
        if not row["license_status"].strip():
            errors.append(f"missing_license_status:{relative}")
        expected_license = expected_github_license_status(relative)
        if expected_license is None:
            errors.append(f"unclassified_license_status:{relative}")
        elif row["license_status"] != expected_license:
            errors.append(f"license_status_mismatch:{relative}")
        if row["safe_to_publish"] != "true":
            errors.append(f"not_publishable:{relative}")

    ordered_paths = [row.get("path", "") for row in rows]
    if ordered_paths != sorted(ordered_paths):
        errors.append("rows_not_path_sorted")

    files, tree_violations = _walk_tree(root)
    errors.extend(f"tree_{item['rule']}:{item['path']}" for item in tree_violations)
    actual = {
        path.relative_to(root).as_posix(): path
        for path in files
        if path.relative_to(root).as_posix() != "GITHUB_UPLOAD_MANIFEST.csv"
    }
    if set(listed) != set(actual):
        missing = sorted(set(actual) - set(listed))
        extra = sorted(set(listed) - set(actual))
        errors.extend(f"unlisted_file:{item}" for item in missing)
        errors.extend(f"nonexistent_manifest_path:{item}" for item in extra)
    for relative in sorted(set(listed) & set(actual)):
        row = listed[relative]
        path = actual[relative]
        if row["size_bytes"].isdigit() and int(row["size_bytes"]) != path.stat().st_size:
            errors.append(f"size_mismatch:{relative}")
        if re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            if row["sha256"] != sha256_file(path):
                errors.append(f"hash_mismatch:{relative}")
    return {
        "fields": headers,
        "row_count": len(rows),
        "actual_file_count": len(actual),
        "errors": sorted(set(errors)),
    }


def _statement_body(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    lines = [line for line in text.splitlines() if not re.match(r"^\s*#{1,6}\s+", line)]
    return canonical_prose("\n".join(lines))


def _peerj_reference_integrity(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    cited_all: set[str] = set()
    defined_all: set[str] = set()
    fragment_count = 0
    for relative in (
        "main_manuscript/manuscript_peerj.md",
        "supplement/supplement_peerj.md",
    ):
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        cited = set(re.findall(r"\]\(#ref-([^)]+)\)", text))
        defined = set(re.findall(r'<div\s+id="ref-([^"]+)"', text))
        fragments = set(re.findall(r'href="#([^"]+)"', text))
        identifiers = set(re.findall(r'\bid="([^"]+)"', text))
        cited_all.update(cited)
        defined_all.update(defined)
        fragment_count += len(fragments)
        for key in sorted(cited - defined):
            errors.append(f"undefined_citation:{relative}:{key}")
        for target in sorted(fragments - identifiers):
            errors.append(f"undefined_markdown_crossref:{relative}:{target}")
        if re.search(r"\[@[A-Za-z0-9]", text) or re.search(
            r"(?<![A-Za-z0-9_])@(?:fig|tbl|sec|eq):[A-Za-z0-9_-]+", text
        ):
            errors.append(f"unresolved_pandoc_reference:{relative}")

    for relative in (
        "main_manuscript/manuscript_peerj.tex",
        "supplement/supplement_peerj.tex",
    ):
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        references = set(
            re.findall(r"\\(?:ref|pageref|autoref|eqref)\{([^}]+)\}", text)
        )
        labels = set(re.findall(r"\\label\{([^}]+)\}", text))
        for target in sorted(references - labels):
            errors.append(f"undefined_latex_crossref:{relative}:{target}")

    bibliography = root / "references/references.bib"
    bib_keys: set[str] = set()
    if bibliography.is_file():
        bib_keys = set(
            re.findall(
                r"(?m)^\s*@\w+\s*\{\s*([^,\s]+)",
                bibliography.read_text(encoding="utf-8"),
            )
        )
        for key in sorted(cited_all - bib_keys):
            errors.append(f"citation_missing_from_bibliography:{key}")
        if cited_all:
            for key in sorted(bib_keys - cited_all):
                errors.append(f"uncited_bibliography_entry:{key}")

    main = root / "main_manuscript/manuscript_peerj.md"
    image_sources: set[str] = set()
    if main.is_file():
        image_sources = set(
            re.findall(r'<img\s+src="(\.\./figures/figure[1-5]\.png)"', main.read_text(encoding="utf-8"))
        )
        expected = {f"../figures/figure{number}.png" for number in range(1, 6)}
        if image_sources != expected:
            errors.append("main_manuscript_figure_reference_set")
    return {
        "cited_key_count": len(cited_all),
        "defined_reference_count": len(defined_all),
        "bibliography_key_count": len(bib_keys),
        "fragment_reference_count": fragment_count,
        "figure_reference_count": len(image_sources),
        "errors": errors,
    }


def _peerj_source_data_integrity(
    root: Path, source_authority: Path | None
) -> dict[str, Any]:
    errors: list[str] = []
    source_root = root / "source_data"
    index_path = source_root / "SOURCE_DATA_INDEX.csv"
    indexed: dict[str, dict[str, str]] = {}
    fields: tuple[str, ...] = ()
    if index_path.is_file():
        try:
            with index_path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = tuple(reader.fieldnames or ())
                rows = list(reader)
        except (OSError, UnicodeError, csv.Error):
            rows = []
            errors.append("source_data_index_unreadable")
        if fields != ("file", "sha256", "bytes"):
            errors.append("source_data_index_fields")
        for row in rows:
            name = row.get("file", "")
            if not safe_archive_name(name) or "/" in name or name in indexed:
                errors.append(f"source_data_index_path:{name}")
                continue
            indexed[name] = row

    actual = {
        path.name: path
        for path in source_root.glob("*.csv")
        if path.name != "SOURCE_DATA_INDEX.csv"
    }
    if set(indexed) != set(actual):
        errors.append("source_data_index_file_set_mismatch")
    for name in sorted(set(indexed) & set(actual)):
        row = indexed[name]
        try:
            expected_size = int(row.get("bytes", ""))
        except ValueError:
            errors.append(f"source_data_index_size:{name}")
            continue
        path = actual[name]
        if (
            not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))
            or path.stat().st_size != expected_size
            or sha256_file(path) != row.get("sha256")
        ):
            errors.append(f"source_data_index_mismatch:{name}")
        if source_authority is not None:
            authority = source_authority / name
            if not authority.is_file() or sha256_file(authority) != sha256_file(path):
                errors.append(f"source_data_authority_mismatch:{name}")

    figure_mapping: dict[str, str] = {}
    for number in range(1, 6):
        matches = sorted(name for name in actual if name.startswith(f"figure{number}_"))
        if len(matches) != 1:
            errors.append(f"figure_source_mapping:{number}")
        else:
            figure_mapping[f"figure{number}"] = matches[0]
    for number in range(1, 5):
        matches = [
            name
            for name in actual
            if re.fullmatch(rf"table{number}_.+_source\.csv", name)
        ]
        if len(matches) != 1:
            errors.append(f"table_source_mapping:{number}")
    return {
        "index_fields": fields,
        "indexed_file_count": len(indexed),
        "actual_file_count": len(actual),
        "figure_mapping": figure_mapping,
        "errors": errors,
    }


def _peerj_manifest_integrity(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest = root / "PEERJ_MATERIALS_MANIFEST.csv"
    listed: dict[str, dict[str, str]] = {}
    fields: tuple[str, ...] = ()
    if manifest.is_file():
        try:
            with manifest.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = tuple(reader.fieldnames or ())
                rows = list(reader)
        except (OSError, UnicodeError, csv.Error):
            rows = []
            errors.append("peerj_manifest_unreadable")
        if fields != ("path", "size_bytes", "sha256", "status"):
            errors.append("peerj_manifest_fields")
        for row in rows:
            relative = row.get("path", "")
            if not safe_archive_name(relative) or relative in listed:
                errors.append(f"peerj_manifest_path:{relative}")
                continue
            listed[relative] = row
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    if set(listed) != set(actual):
        errors.append("peerj_manifest_file_set_mismatch")
    for relative in sorted(set(listed) & set(actual)):
        row = listed[relative]
        try:
            expected_size = int(row.get("size_bytes", ""))
        except ValueError:
            errors.append(f"peerj_manifest_size:{relative}")
            continue
        path = actual[relative]
        if (
            row.get("status") != "pending_author"
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))
            or path.stat().st_size != expected_size
            or sha256_file(path) != row.get("sha256")
        ):
            errors.append(f"peerj_manifest_mismatch:{relative}")
    return {
        "fields": fields,
        "row_count": len(listed),
        "actual_file_count": len(actual),
        "errors": errors,
    }


def verify_peerj_materials(
    root: Path, source_authority: Path | None = None
) -> dict[str, Any]:
    """Validate the required PeerJ pending-author structure and fixed text."""

    contract = _missing_contract_entries(root, PEERJ_REQUIRED_FILES, PEERJ_REQUIRED_DIRS)
    errors = [f"missing_file:{item}" for item in contract["missing_files"]]
    errors.extend(f"missing_dir:{item}" for item in contract["missing_dirs"])

    ethics = root / "metadata/fixed_ethics_statement.md"
    ai = root / "metadata/fixed_ai_disclosure.md"
    if ethics.is_file() and _statement_body(ethics.read_text(encoding="utf-8")) != canonical_prose(
        FIXED_ETHICS
    ):
        errors.append("fixed_ethics_statement_mismatch")
    if ai.is_file() and _statement_body(ai.read_text(encoding="utf-8")) != canonical_prose(
        FIXED_AI_DISCLOSURE
    ):
        errors.append("fixed_ai_disclosure_mismatch")

    author_template = root / "metadata/AUTHOR_INPUT_TEMPLATE.yml"
    author_fields: set[str] = set()
    if author_template.is_file():
        author_text = author_template.read_text(encoding="utf-8")
        author_fields = set(re.findall(r"(?m)^([a-z][a-z0-9_]*)\s*:", author_text))
        errors.extend(
            f"missing_author_input_field:{field}"
            for field in AUTHOR_INPUT_FIELDS
            if field not in author_fields
        )

    pdfs = [
        root / "main_manuscript/manuscript_peerj.pdf",
        root / "supplement/supplement_peerj.pdf",
    ]
    for number in range(1, 6):
        pdfs.append(root / f"figures/figure{number}.pdf")
        png = root / f"figures/figure{number}.png"
        if not basic_png_integrity(png):
            errors.append(f"invalid_or_missing_png:figures/figure{number}.png")
    for pdf in pdfs:
        if not basic_pdf_integrity(pdf):
            errors.append(f"invalid_or_missing_pdf:{pdf.relative_to(root).as_posix()}")

    table_root = root / "tables"
    editable_suffixes = {".csv", ".md", ".tex", ".tsv", ".xlsx"}
    table_files = list(table_root.iterdir()) if table_root.is_dir() else []
    for number in range(1, 5):
        pattern = re.compile(rf"(?i)^table0*{number}(?:[_.-]|$)")
        present = any(
            pattern.search(path.name) and path.suffix.lower() in editable_suffixes
            for path in table_files
        )
        if not present:
            errors.append(f"missing_editable_table:{number}")
    source_root = root / "source_data"
    source_files, source_violations = _walk_tree(source_root)
    if not source_files:
        errors.append("source_data_empty")
    errors.extend(f"source_data_{item['rule']}:{item['path']}" for item in source_violations)
    reference_report = _peerj_reference_integrity(root)
    source_report = _peerj_source_data_integrity(root, source_authority)
    manifest_report = _peerj_manifest_integrity(root)
    errors.extend(reference_report["errors"])
    errors.extend(source_report["errors"])
    errors.extend(manifest_report["errors"])
    return {
        "required_file_count": len(PEERJ_REQUIRED_FILES),
        "pdf_count": len(pdfs),
        "figure_png_count": 5,
        "source_data_file_count": len(source_files),
        "author_input_field_count": len(author_fields),
        "reference_integrity": reference_report,
        "source_data_integrity": source_report,
        "manifest_integrity": manifest_report,
        "errors": sorted(set(errors)),
    }


def _unsafe_archive_mode(mode: int, is_directory: bool) -> bool:
    if mode & 0o7000 or mode & 0o002:
        return True
    if is_directory:
        return not bool(mode & stat.S_IRUSR and mode & stat.S_IXUSR)
    return not bool(mode & stat.S_IRUSR)


def _archive_tree_comparison(
    archived_files: dict[str, tuple[int, str]], source_root: Path
) -> list[str]:
    files, violations = _walk_tree(source_root)
    errors = [f"source_{item['rule']}:{item['path']}" for item in violations]
    actual = {
        path.relative_to(source_root).as_posix(): (path.stat().st_size, sha256_file(path))
        for path in files
    }
    if set(archived_files) != set(actual):
        missing = sorted(set(actual) - set(archived_files))
        extra = sorted(set(archived_files) - set(actual))
        errors.extend(f"archive_missing_file:{item}" for item in missing)
        errors.extend(f"archive_extra_file:{item}" for item in extra)
    for relative in sorted(set(archived_files) & set(actual)):
        if archived_files[relative] != actual[relative]:
            errors.append(f"archive_content_mismatch:{relative}")
    return errors


def _validate_zip(path: Path, expected_root: str, source_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    archived: dict[str, tuple[int, str]] = {}
    total = 0
    member_count = 0
    roots: set[str] = set()
    seen: set[str] = set()
    seen_casefold: set[str] = set()
    if not path.is_file():
        return {"errors": ["archive_missing"], "member_count": 0, "uncompressed_bytes": 0}
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            member_count = len(members)
            if member_count > MAX_ARCHIVE_MEMBERS:
                errors.append("too_many_members")
            for info in members:
                name = info.filename
                if not safe_archive_name(name):
                    errors.append(f"unsafe_member_path:{name}")
                    continue
                canonical = PurePosixPath(name).as_posix()
                if canonical.rstrip("/") != name.rstrip("/"):
                    errors.append(f"noncanonical_member_path:{name}")
                roots.add(PurePosixPath(name).parts[0])
                if name in seen:
                    errors.append(f"duplicate_member:{name}")
                if name.casefold() in seen_casefold:
                    errors.append(f"casefold_member_collision:{name}")
                seen.add(name)
                seen_casefold.add(name.casefold())
                if info.flag_bits & 0x1:
                    errors.append(f"encrypted_member:{name}")
                mode = info.external_attr >> 16
                is_directory = info.is_dir()
                if not mode:
                    errors.append(f"missing_unix_mode:{name}")
                elif is_directory:
                    if not stat.S_ISDIR(mode) or _unsafe_archive_mode(mode, True):
                        errors.append(f"unsafe_directory_mode:{name}")
                elif not stat.S_ISREG(mode) or _unsafe_archive_mode(mode, False):
                    errors.append(f"link_special_or_unsafe_mode:{name}")
                if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    errors.append(f"oversized_member:{name}")
                total += info.file_size
                if info.file_size > 1_000_000:
                    ratio = info.file_size / max(info.compress_size, 1)
                    if ratio > MAX_COMPRESSION_RATIO:
                        errors.append(f"compression_bomb_member:{name}")
                if not is_directory and safe_archive_name(name):
                    try:
                        digest = hashlib.sha256()
                        size = 0
                        with archive.open(info) as handle:
                            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                                size += len(chunk)
                                digest.update(chunk)
                        relative = PurePosixPath(name).relative_to(expected_root).as_posix()
                        archived[relative] = (size, digest.hexdigest())
                    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
                        errors.append(f"member_read_or_crc_failure:{name}")
            try:
                bad_crc = archive.testzip()
                if bad_crc:
                    errors.append(f"crc_failure:{bad_crc}")
            except (OSError, RuntimeError, zipfile.BadZipFile):
                errors.append("crc_scan_failed")
    except (OSError, zipfile.BadZipFile):
        return {"errors": ["invalid_zip"], "member_count": 0, "uncompressed_bytes": 0}
    if roots != {expected_root}:
        errors.append("not_single_expected_root")
    if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        errors.append("archive_uncompressed_size_limit")
    if total > 1_000_000 and total / max(path.stat().st_size, 1) > MAX_COMPRESSION_RATIO:
        errors.append("archive_compression_bomb_ratio")
    errors.extend(_archive_tree_comparison(archived, source_root))
    return {
        "member_count": member_count,
        "uncompressed_bytes": total,
        "single_root": sorted(roots),
        "errors": sorted(set(errors)),
    }


def _validate_tar(path: Path, expected_root: str, source_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    archived: dict[str, tuple[int, str]] = {}
    total = 0
    member_count = 0
    roots: set[str] = set()
    seen: set[str] = set()
    seen_casefold: set[str] = set()
    if not path.is_file():
        return {"errors": ["archive_missing"], "member_count": 0, "uncompressed_bytes": 0}
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            member_count = len(members)
            if member_count > MAX_ARCHIVE_MEMBERS:
                errors.append("too_many_members")
            for member in members:
                name = member.name
                if not safe_archive_name(name):
                    errors.append(f"unsafe_member_path:{name}")
                    continue
                canonical = PurePosixPath(name).as_posix()
                if canonical != name:
                    errors.append(f"noncanonical_member_path:{name}")
                roots.add(PurePosixPath(name).parts[0])
                if name in seen:
                    errors.append(f"duplicate_member:{name}")
                if name.casefold() in seen_casefold:
                    errors.append(f"casefold_member_collision:{name}")
                seen.add(name)
                seen_casefold.add(name.casefold())
                is_directory = member.isdir()
                if not (is_directory or member.isreg()):
                    errors.append(f"link_or_special_member:{name}")
                    continue
                if _unsafe_archive_mode(member.mode, is_directory):
                    errors.append(f"unsafe_mode:{name}")
                if member.size > MAX_ARCHIVE_MEMBER_BYTES:
                    errors.append(f"oversized_member:{name}")
                total += member.size
                if member.isreg():
                    try:
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise OSError
                        digest = hashlib.sha256()
                        size = 0
                        with extracted:
                            for chunk in iter(
                                lambda handle=extracted: handle.read(1024 * 1024), b""
                            ):
                                size += len(chunk)
                                digest.update(chunk)
                        relative = PurePosixPath(name).relative_to(expected_root).as_posix()
                        archived[relative] = (size, digest.hexdigest())
                    except (OSError, tarfile.TarError, ValueError):
                        errors.append(f"member_read_or_gzip_crc_failure:{name}")
    except (OSError, EOFError, tarfile.TarError):
        return {"errors": ["invalid_tar_gzip"], "member_count": 0, "uncompressed_bytes": 0}
    if roots != {expected_root}:
        errors.append("not_single_expected_root")
    if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        errors.append("archive_uncompressed_size_limit")
    if total > 1_000_000 and total / max(path.stat().st_size, 1) > MAX_COMPRESSION_RATIO:
        errors.append("archive_compression_bomb_ratio")
    errors.extend(_archive_tree_comparison(archived, source_root))
    return {
        "member_count": member_count,
        "uncompressed_bytes": total,
        "single_root": sorted(roots),
        "errors": sorted(set(errors)),
    }


def _verify_sidecar(archive: Path, sidecar: Path) -> list[str]:
    if not archive.is_file():
        return ["archive_missing"]
    if not sidecar.is_file():
        return ["sidecar_missing"]
    try:
        fields = sidecar.read_text(encoding="utf-8").strip().split()
    except (OSError, UnicodeError):
        return ["sidecar_unreadable"]
    if not fields or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
        return ["sidecar_invalid_sha256"]
    errors = []
    if fields[0] != sha256_file(archive):
        errors.append("sidecar_hash_mismatch")
    if len(fields) > 1 and fields[1].lstrip("*") != archive.name:
        errors.append("sidecar_filename_mismatch")
    if len(fields) > 2:
        errors.append("sidecar_extra_fields")
    return errors


def verify_reproducibility_authorities(root: Path) -> dict[str, Any]:
    """Check the frozen Claims and aggregate-only Level-B reference authorities."""

    errors: list[str] = []
    missing = [relative for relative in REPRO_AUTHORITY_FILES if not (root / relative).is_file()]
    errors.extend(f"missing_authority:{relative}" for relative in missing)

    claims = root / "protocol/FINAL_CLAIMS_REGISTRY.csv"
    if claims.is_file() and sha256_file(claims) != CLAIMS_SHA256:
        errors.append("claims_registry_hash_mismatch")

    config = root / "configs/stage2_formal_release.yml"
    if config.is_file():
        try:
            config_text = config.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            config_text = ""
            errors.append("release_config_unreadable")
        expected_config_patterns = {
            "budgets": r"(?m)^budgets:\s*\[0,\s*25,\s*50,\s*100,\s*250,\s*500\]\s*$",
            "split_seeds": r"(?m)^split_seeds:\s*\[0,\s*1,\s*2\]\s*$",
            "adaptation_seeds": r"(?m)^adaptation_seeds:\s*100\s*$",
            "blind_acquisition": r"(?m)^acquisition_strategy:\s*blind_random\s*$",
        }
        errors.extend(
            f"release_config_mismatch:{label}"
            for label, pattern in expected_config_patterns.items()
            if not re.search(pattern, config_text)
        )

    budget_manifest = root / "protocol/acsincome_region_budget_manifest.csv"
    budget_rows: list[dict[str, str]] = []
    if budget_manifest.is_file():
        try:
            with budget_manifest.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                budget_fields = tuple(reader.fieldnames or ())
                budget_rows = list(reader)
        except (OSError, UnicodeError, csv.Error):
            budget_fields = ()
            errors.append("level_b_budget_manifest_unreadable")
        if budget_fields != LEVEL_B_BUDGET_FIELDS:
            errors.append("level_b_budget_manifest_fields")
        expected_budgets = {0, 25, 50, 100, 250, 500}
        triples: set[tuple[int, int]] = set()
        for row in budget_rows:
            try:
                budget = int(row["budget"])
                adaptation_seed = int(row["adaptation_seed"])
                selected_count = int(row["selected_count"])
            except (KeyError, TypeError, ValueError):
                errors.append("level_b_budget_manifest_value")
                continue
            triples.add((adaptation_seed, budget))
            if (
                row.get("dataset") != "acsincome_region"
                or row.get("split_seed") != "0"
                or row.get("available", "").lower() != "true"
                or row.get("selection_reads_labels", "").lower() != "false"
                or selected_count != budget
            ):
                errors.append("level_b_budget_manifest_protocol_mismatch")
            if row.get("adaptation_pool_digest") != LEVEL_B_ADAPTATION_POOL_DIGEST:
                errors.append("level_b_adaptation_pool_digest_mismatch")
            if row.get("target_final_test_digest") != LEVEL_B_FINAL_TEST_DIGEST:
                errors.append("level_b_final_test_digest_mismatch")
            if not re.fullmatch(r"[0-9a-f]{64}", row.get("selected_entity_prefix_digest", "")):
                errors.append("level_b_selected_prefix_digest_invalid")
        expected_triples = {
            (seed, budget) for seed in range(100) for budget in expected_budgets
        }
        if len(budget_rows) != 600 or triples != expected_triples:
            errors.append("level_b_budget_grid_mismatch")

    reference_root = root / "reference/one_task"
    expected_csv_names = {f"{stem}.csv" for stem in LEVEL_B_REFERENCE_ROWS}
    actual_csv_names = (
        {path.name for path in reference_root.glob("*.csv")} if reference_root.is_dir() else set()
    )
    if actual_csv_names != expected_csv_names:
        errors.append("level_b_reference_file_set_mismatch")
    forbidden_identifiers = {"entity_id", "record_id", "group_id", "serialno", "sporder"}
    observed_rows: dict[str, int] = {}
    for stem, expected_rows in LEVEL_B_REFERENCE_ROWS.items():
        path = reference_root / f"{stem}.csv"
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                row_count = sum(1 for _ in reader)
        except (OSError, UnicodeError, csv.Error, StopIteration):
            errors.append(f"level_b_reference_unreadable:{stem}")
            continue
        observed_rows[stem] = row_count
        if row_count != expected_rows:
            errors.append(f"level_b_reference_row_count:{stem}")
        if {column.lower() for column in header} & forbidden_identifiers:
            errors.append(f"level_b_reference_plain_identifier:{stem}")

    metadata = reference_root / "reference_metadata.json"
    if metadata.is_file():
        try:
            metadata_payload = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            metadata_payload = {}
            errors.append("level_b_reference_metadata_unreadable")
        expected_metadata_keys = {f"{stem}.parquet" for stem in LEVEL_B_REFERENCE_ROWS}
        if set(metadata_payload) != expected_metadata_keys:
            errors.append("level_b_reference_metadata_keys")
        for stem, expected_rows in LEVEL_B_REFERENCE_ROWS.items():
            record = metadata_payload.get(f"{stem}.parquet", {})
            if record.get("rows") != expected_rows or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("sha256", ""))
            ):
                errors.append(f"level_b_reference_metadata_record:{stem}")

    source_index = root / "figure_source_data/SOURCE_DATA_INDEX.csv"
    indexed_files: set[str] = set()
    if source_index.is_file():
        try:
            with source_index.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                source_fields = tuple(reader.fieldnames or ())
                source_rows = list(reader)
        except (OSError, UnicodeError, csv.Error):
            source_fields = ()
            source_rows = []
            errors.append("source_data_index_unreadable")
        if source_fields != ("file", "sha256", "bytes"):
            errors.append("source_data_index_fields")
        for row in source_rows:
            relative = row.get("file", "")
            if not safe_archive_name(relative) or "/" in relative:
                errors.append("source_data_index_unsafe_path")
                continue
            indexed_files.add(relative)
            candidate = root / "figure_source_data" / relative
            try:
                size = int(row.get("bytes", ""))
            except ValueError:
                errors.append(f"source_data_index_size:{relative}")
                continue
            if (
                not candidate.is_file()
                or candidate.stat().st_size != size
                or sha256_file(candidate) != row.get("sha256")
            ):
                errors.append(f"source_data_index_mismatch:{relative}")
        actual_source_files = {
            path.name
            for path in (root / "figure_source_data").glob("*.csv")
            if path.name != "SOURCE_DATA_INDEX.csv"
        }
        if indexed_files != actual_source_files:
            errors.append("source_data_index_file_set_mismatch")

    audited_hashes = root / "manuscript_build/AUDITED_ARTIFACT_HASHES.csv"
    if audited_hashes.is_file():
        try:
            with audited_hashes.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                audited_fields = tuple(reader.fieldnames or ())
                audited_rows = list(reader)
        except (OSError, UnicodeError, csv.Error):
            audited_fields = ()
            audited_rows = []
            errors.append("audited_artifact_hashes_unreadable")
        if audited_fields != ("path", "sha256") or not audited_rows:
            errors.append("audited_artifact_hashes_schema")
        for row in audited_rows:
            if not safe_archive_name(row.get("path", "")) or not re.fullmatch(
                r"[0-9a-f]{64}", row.get("sha256", "")
            ):
                errors.append("audited_artifact_hashes_record")

    if (root / ".git").exists():
        errors.append("standalone_bundle_contains_git")
    license_path = root / "LICENSE"
    if license_path.is_file():
        try:
            license_text = license_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            license_text = ""
            errors.append("license_unreadable")
        if not license_text.startswith("MIT License") or (
            "Copyright (c) 2026 Dinghao Chen" not in license_text
        ):
            errors.append("authorized_mit_license_mismatch")
    if (root / "LICENSE_SELECTION_REQUIRED.md").exists():
        errors.append("obsolete_license_selection_blocker_present")
    if (root / "CITATION_TEMPLATE.cff").exists() or (root / "CITATION.cff.template").exists():
        errors.append("obsolete_citation_template_present")
    citation = root / "CITATION.cff"
    if citation.is_file():
        try:
            citation_text = citation.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            citation_text = ""
            errors.append("citation_unreadable")
        if "[AUTHOR TO PROVIDE" in citation_text:
            errors.append("citation_contains_author_placeholder")
    return {
        "required_authority_file_count": len(REPRO_AUTHORITY_FILES),
        "budget_manifest_rows": len(budget_rows),
        "reference_rows": observed_rows,
        "source_data_index_rows": len(indexed_files),
        "errors": sorted(set(errors)),
    }


def _git(root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
    )


def _repository_root(requested_root: Path, stage3r_root: Path) -> Path:
    candidates = (requested_root, stage3r_root.parent.parent, stage3r_root.parent)
    return next(
        (candidate for candidate in candidates if (candidate / ".git").exists()),
        requested_root,
    )


def _verify_full_repository(audit: Audit, requested_root: Path, stage3r_root: Path) -> None:
    repository = _repository_root(requested_root, stage3r_root)
    git_present = (repository / ".git").exists()
    audit.add("git_repository_present", git_present, {"repository": str(repository.resolve())})
    if not git_present:
        return
    branch = _git(repository, ["branch", "--show-current"])
    audit.add(
        "stage3r_isolated_branch",
        branch.returncode == 0 and branch.stdout.strip() == STAGE3R_BRANCH,
        {"actual": branch.stdout.strip(), "expected": STAGE3R_BRANCH},
    )
    resolved: dict[str, str] = {}
    commits_ok = True
    for commit in (STAGE3Q_SOURCE_COMMIT, STAGE3Q_RECHECK_COMMIT):
        result = _git(repository, ["rev-parse", f"{commit}^{{commit}}"])
        resolved[commit] = result.stdout.strip()
        commits_ok &= result.returncode == 0 and result.stdout.strip() == commit
    audit.add("stage3q_commits_resolve", commits_ok, resolved)
    ancestry_results = []
    for ancestor, descendant in (
        (STAGE3Q_SOURCE_COMMIT, STAGE3Q_RECHECK_COMMIT),
        (STAGE3Q_RECHECK_COMMIT, "HEAD"),
    ):
        result = _git(repository, ["merge-base", "--is-ancestor", ancestor, descendant])
        ancestry_results.append(
            {"ancestor": ancestor, "descendant": descendant, "ok": result.returncode == 0}
        )
    audit.add("stage3q_ancestry", all(item["ok"] for item in ancestry_results), ancestry_results)

    claims = repository / CLAIMS_REGISTRY
    claims_ok = claims.is_file() and sha256_file(claims) == CLAIMS_SHA256
    audit.add(
        "claims_registry_frozen",
        claims_ok,
        {
            "path": CLAIMS_REGISTRY,
            "expected_sha256": CLAIMS_SHA256,
            "actual_sha256": sha256_file(claims) if claims.is_file() else None,
        },
    )
    frozen = ("results", "data", "protocol", "configs", "src")
    committed = _git(
        repository,
        ["diff", "--name-only", f"{STAGE3Q_RECHECK_COMMIT}..HEAD", "--", *frozen],
    )
    working = _git(repository, ["diff", "--name-only", "--", *frozen])
    staged = _git(repository, ["diff", "--cached", "--name-only", "--", *frozen])
    untracked = _git(
        repository, ["ls-files", "--others", "--exclude-standard", "--", *frozen]
    )
    changed = sorted(
        {
            line
            for result in (committed, working, staged, untracked)
            for line in result.stdout.splitlines()
            if line.strip()
        }
    )
    commands_ok = all(
        result.returncode == 0 for result in (committed, working, staged, untracked)
    )
    audit.add("frozen_scientific_trees_unchanged", commands_ok and not changed, changed)


def verify(root: Path, profile: str = "package") -> dict[str, Any]:
    """Verify Stage 3R without mutating or extracting anything."""

    if profile not in {"package", "full_repository"}:
        raise ValueError("profile must be 'package' or 'full_repository'")
    root = root.resolve()
    stage3r_root = _locate_stage3r_root(root)
    audit = Audit(profile, root, stage3r_root)
    standalone_repro = profile == "package" and _is_standalone_repro_bundle(root)
    standalone_github = profile == "package" and _is_standalone_github_staging(root)
    audit.report["layout_mode"] = (
        "standalone_repro_bundle"
        if standalone_repro
        else "standalone_github_staging"
        if standalone_github
        else "complete_stage3r"
    )

    if standalone_repro:
        repro_contract = _missing_contract_entries(
            root, REPRO_REQUIRED_FILES, REPRO_REQUIRED_DIRS
        )
        audit.add(
            "reproducibility_bundle_contract",
            not repro_contract["missing_files"] and not repro_contract["missing_dirs"],
            repro_contract,
        )
        repro_scan = scan_tree(root, PUBLIC_STRICT)
        audit.add(
            "public_strict_reproducibility_bundle",
            not repro_scan["violations"],
            repro_scan,
        )
        executable_errors = [
            relative
            for relative in (
                "reproduce_artifacts.sh",
                "reproduce_one_task.sh",
                "reproduce_full_experiment.sh",
                "verify_release.sh",
            )
            if not (root / relative).is_file() or not os.access(root / relative, os.X_OK)
        ]
        audit.add(
            "reproduction_entrypoints_executable",
            not executable_errors,
            executable_errors,
        )
        authorities = verify_reproducibility_authorities(root)
        audit.add(
            "reproducibility_frozen_authorities",
            not authorities["errors"],
            authorities,
        )
        return audit.finish()

    if standalone_github:
        github_contract = _missing_contract_entries(
            root, GITHUB_REQUIRED_FILES, GITHUB_REQUIRED_DIRS
        )
        audit.add(
            "github_staging_contract",
            not github_contract["missing_files"] and not github_contract["missing_dirs"],
            github_contract,
        )
        github_scan = scan_tree(root, PUBLIC_STRICT)
        audit.add(
            "public_strict_github_staging",
            not github_scan["violations"],
            github_scan,
        )
        manifest = verify_github_manifest(root)
        audit.add("github_manifest_tree_isomorphism", not manifest["errors"], manifest)
        executable_errors = [
            relative
            for relative in (
                "reproduce_artifacts.sh",
                "reproduce_one_task.sh",
                "reproduce_full_experiment.sh",
                "verify_release.sh",
            )
            if not (root / relative).is_file() or not os.access(root / relative, os.X_OK)
        ]
        audit.add(
            "reproduction_entrypoints_executable",
            not executable_errors,
            executable_errors,
        )
        authorities = verify_reproducibility_authorities(root)
        audit.add(
            "reproducibility_frozen_authorities",
            not authorities["errors"],
            authorities,
        )
        return audit.finish()

    top_level = _missing_contract_entries(
        stage3r_root,
        (),
        (REPRO_DIR, GITHUB_DIR, PEERJ_DIR, "validation", "manifests", "packages"),
    )
    audit.add("stage3r_directory_contract", not top_level["missing_dirs"], top_level)

    repro = stage3r_root / REPRO_DIR
    github = stage3r_root / GITHUB_DIR
    peerj = stage3r_root / PEERJ_DIR
    repro_contract = _missing_contract_entries(repro, REPRO_REQUIRED_FILES, REPRO_REQUIRED_DIRS)
    github_contract = _missing_contract_entries(github, GITHUB_REQUIRED_FILES, GITHUB_REQUIRED_DIRS)
    audit.add(
        "reproducibility_bundle_contract",
        not repro_contract["missing_files"] and not repro_contract["missing_dirs"],
        repro_contract,
    )
    audit.add(
        "github_staging_contract",
        not github_contract["missing_files"] and not github_contract["missing_dirs"],
        github_contract,
    )

    repro_scan = scan_tree(repro, PUBLIC_STRICT)
    github_scan = scan_tree(github, PUBLIC_STRICT)
    peerj_scan = scan_tree(peerj, SUBMISSION_STRICT)
    audit.add("public_strict_reproducibility_bundle", not repro_scan["violations"], repro_scan)
    audit.add("public_strict_github_staging", not github_scan["violations"], github_scan)
    audit.add("submission_strict_peerj_materials", not peerj_scan["violations"], peerj_scan)

    executable_errors = [
        relative
        for relative in (
            "reproduce_artifacts.sh",
            "reproduce_one_task.sh",
            "reproduce_full_experiment.sh",
            "verify_release.sh",
        )
        if not (repro / relative).is_file() or not os.access(repro / relative, os.X_OK)
    ]
    audit.add("reproduction_entrypoints_executable", not executable_errors, executable_errors)

    manifest = verify_github_manifest(github)
    audit.add("github_manifest_tree_isomorphism", not manifest["errors"], manifest)
    peerj_report = verify_peerj_materials(peerj, repro / "figure_source_data")
    audit.add("peerj_materials_contract", not peerj_report["errors"], peerj_report)

    for archive_relative, sidecar_relative, source_name, kind in PACKAGE_SPECS:
        archive = stage3r_root / archive_relative
        sidecar = stage3r_root / sidecar_relative
        source = stage3r_root / source_name
        archive_report = (
            _validate_tar(archive, source_name, source)
            if kind == "tar"
            else _validate_zip(archive, source_name, source)
        )
        archive_report["sidecar_errors"] = _verify_sidecar(archive, sidecar)
        audit.add(
            f"archive_{source_name}",
            not archive_report["errors"] and not archive_report["sidecar_errors"],
            archive_report,
        )

    if profile == "full_repository":
        _verify_full_repository(audit, root, stage3r_root)
    return audit.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--profile", choices=("package", "full_repository"), default="package"
    )
    parser.add_argument("--json", action="store_true", help="emit the complete JSON report")
    arguments = parser.parse_args()
    try:
        report = verify(arguments.root, arguments.profile)
    except Exception as error:  # pragma: no cover - defensive CLI boundary
        report = {
            "status": "FAIL",
            "profile": arguments.profile,
            "checks": {},
            "errors": ["verifier_exception"],
            "exception_type": type(error).__name__,
        }
    if arguments.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"Stage 3R verification: {report['status']} ({report['profile']})")
        for name in report.get("errors", []):
            print(f"FAIL: {name}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
