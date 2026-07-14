#!/usr/bin/env python3
"""Rebuild the frozen Stage 3R Level-A manuscript artifacts.

The script is designed to be copied into ``reproducibility_bundle/scripts``.  Its only
scientific inputs are CSV files below ``figure_source_data`` and document sources below
``manuscript_build``.  It never imports an experiment runner and never reads repository
history, raw data, or an external results tree.

Five figures are rebuilt as PNG/PDF, and four tables are rebuilt as CSV/Markdown/LaTeX.
When both Pandoc and Tectonic are available, the main manuscript and supplement are rebuilt
from their Markdown sources.  Missing manuscript tools are a hard, reported failure: the
script never substitutes a previously compiled PDF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

SCHEMA_VERSION = "stage3r-artifact-build-v1"
FIGURE_STEMS = (
    "figure1_study_design",
    "figure2_complexity_budget",
    "figure3_common5_heterogeneity",
    "figure4_standard_mondrian_tradeoff",
    "figure5_auroc_reliability",
)
TABLE_STEMS = (
    "table1_tasks",
    "table2_methods",
    "table3_h2_h3",
    "table4_h5",
)
METHOD_ORDER = ("Intercept MLE", "Jeffreys intercept", "Sigmoid", "Isotonic")
COLORS = {
    "Intercept MLE": "#0072B2",
    "Jeffreys intercept": "#56B4E9",
    "Sigmoid": "#E69F00",
    "Isotonic": "#D55E00",
}
MODEL_LABEL = {
    "logistic_regression": "Logistic regression",
    "xgboost_cpu": "XGBoost",
}
TASK_ORDER = (
    "ACS Income",
    "ACS Food Stamps",
    "BRFSS Diabetes",
    "NHANES Lead",
    "Diabetes Readmission",
)
PDF_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)
DOCUMENT_TOOL_VERSIONS = {
    "pandoc": "pandoc 3.10",
    "tectonic": "Tectonic 0.16.9",
}

TABLE_CAPTIONS = {
    "table1_tasks": (
        "Task definitions, natural domains, entity units, audited counts, and analysis roles.",
        "tab:tasks",
    ),
    "table2_methods": (
        "Evaluated methods, support rules, and reported outcomes.",
        "tab:methods",
    ),
    "table3_h2_h3": (
        "Common-five probability-update effects, material-benefit heterogeneity, "
        "and deletion stability.",
        "tab:h2h3",
    ),
    "table4_h5": (
        "Joint-success Mondrian-minus-Standard metric differences and available-attempt feasibility.",
        "tab:h5",
    ),
}

TABLE_MARKERS = {
    "{{TABLE1_STAGE3Q}}": "table1_tasks.md",
    "{{TABLE2_STAGE3Q}}": "table2_methods.md",
    "{{TABLE3_STAGE3Q}}": "table3_h2_h3.md",
    "{{TABLE4_STAGE3Q}}": "table4_h5.md",
}


class BuildError(RuntimeError):
    """A reproducibility-contract or artifact-build failure."""


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, root: Path) -> Path:
    """Require a regular input file that cannot escape its allowed root via a symlink."""

    resolved_root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise BuildError(f"input escapes allowed root: {path.name}")
    if not resolved.is_file():
        raise BuildError(f"required input is missing: {path.name}")
    return resolved


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise BuildError(f"{name} is missing columns: {missing}")


def read_csv(path: Path, root: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(require_file(path, root), **kwargs)


def read_text_table(path: Path, root: Path) -> pd.DataFrame:
    return read_csv(path, root, dtype=str, keep_default_na=False)


def assert_text_frames_equal(actual: pd.DataFrame, expected: pd.DataFrame, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        raise BuildError(f"{name} expected table is inconsistent with its frozen source") from exc


def validate_source_index(source_root: Path) -> dict[str, Any]:
    """Validate indexed source files when the optional frozen index is present."""

    index_path = source_root / "SOURCE_DATA_INDEX.csv"
    if not index_path.exists():
        return {"present": False, "checked_rows": 0, "status": "NOT_PROVIDED"}
    index = read_csv(index_path, source_root)
    require_columns(index, {"file", "sha256", "bytes"}, index_path.name)
    checked = 0
    for row in index.to_dict("records"):
        candidate = source_root / str(row["file"])
        if not candidate.exists():
            continue
        candidate = require_file(candidate, source_root)
        if candidate.stat().st_size != int(row["bytes"]):
            raise BuildError(f"source index byte mismatch: {candidate.name}")
        if sha256(candidate) != str(row["sha256"]):
            raise BuildError(f"source index SHA-256 mismatch: {candidate.name}")
        checked += 1
    return {"present": True, "checked_rows": checked, "status": "PASS"}


def load_figure_sources(source_root: Path) -> dict[str, pd.DataFrame]:
    frames = {stem: read_csv(source_root / f"{stem}.csv", source_root) for stem in FIGURE_STEMS}
    require_columns(
        frames[FIGURE_STEMS[0]], {"order", "role", "domain", "purpose"}, FIGURE_STEMS[0]
    )
    require_columns(
        frames[FIGURE_STEMS[1]],
        {
            "budget",
            "task_equal_delta_log_loss",
            "material_harm_rate",
            "method_label",
            "q25",
            "q75",
            "minimum_task_model",
            "maximum_task_model",
            "task_model_cells",
        },
        FIGURE_STEMS[1],
    )
    require_columns(
        frames[FIGURE_STEMS[2]],
        {"base_model", "method_label", "budget", "task_label", "median"},
        FIGURE_STEMS[2],
    )
    require_columns(
        frames[FIGURE_STEMS[3]],
        {
            "budget",
            "delta_worst_class_coverage",
            "finite_delta_worst_class_coverage",
            "delta_mean_set_size",
            "finite_delta_mean_set_size",
            "mondrian_estimable_rate",
            "mondrian_finite_threshold_rate",
        },
        FIGURE_STEMS[3],
    )
    require_columns(
        frames[FIGURE_STEMS[4]],
        {"task_label", "base_model", "delta_auroc", "delta_brier", "delta_log_loss"},
        FIGURE_STEMS[4],
    )

    roles = frames[FIGURE_STEMS[0]]
    expected_roles = {
        "source_train",
        "source_tune",
        "source_probability_calibration",
        "source_conformal_calibration",
        "source_id_test",
        "target_adaptation_pool",
        "target_final_test",
    }
    if len(roles) != 7 or set(roles["role"]) != expected_roles:
        raise BuildError("figure1 source does not contain the seven frozen entity roles")

    figure2 = frames[FIGURE_STEMS[1]]
    budgets = sorted(figure2["budget"].astype(int).unique().tolist())
    if len(budgets) != 5 or len(figure2) != len(METHOD_ORDER) * len(budgets):
        raise BuildError("figure2 source does not contain the frozen 4-method x 5-budget grid")
    if set(figure2["method_label"]) != set(METHOD_ORDER):
        raise BuildError("figure2 source method labels changed")

    figure3 = frames[FIGURE_STEMS[2]]
    expected_cells = len(TASK_ORDER) * len(MODEL_LABEL) * len(METHOD_ORDER) * len(budgets)
    if len(figure3) != expected_cells:
        raise BuildError("figure3 source does not contain the frozen task-model-method-budget grid")
    if set(figure3["task_label"]) != set(TASK_ORDER):
        raise BuildError("figure3 source task labels changed")
    if figure3.duplicated(["task_label", "base_model", "method_label", "budget"]).any():
        raise BuildError("figure3 source contains duplicate frozen cells")

    figure4 = frames[FIGURE_STEMS[3]]
    if sorted(figure4["budget"].astype(int).tolist()) != budgets:
        raise BuildError("figure4 source budgets do not match figure2")
    if frames[FIGURE_STEMS[4]].empty:
        raise BuildError("figure5 source is empty")
    return frames


def derive_table4(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "budget",
        "delta_worst_class_coverage",
        "finite_delta_worst_class_coverage",
        "delta_mean_set_size",
        "finite_delta_mean_set_size",
        "finite_task_model_cells",
        "mondrian_estimable_rate",
        "mondrian_finite_threshold_rate",
    }
    require_columns(source, required, "table4_h5_source")
    task_model_denominator = int(source["finite_task_model_cells"].max())
    return pd.DataFrame(
        {
            "Budget": source["budget"].astype(int).astype(str),
            "Joint-success Δ worst-class coverage": source["delta_worst_class_coverage"].map(
                lambda value: f"{float(value):+.4f}"
            ),
            "Joint-success, finite-threshold Δ worst-class coverage": source[
                "finite_delta_worst_class_coverage"
            ].map(
                lambda value: f"{float(value):+.4f}"
            ),
            "Joint-success Δ mean set size": source["delta_mean_set_size"].map(
                lambda value: f"{float(value):+.4f}"
            ),
            "Joint-success, finite-threshold Δ mean set size": source[
                "finite_delta_mean_set_size"
            ].map(
                lambda value: f"{float(value):+.4f}"
            ),
            "Finite task–model cells": source["finite_task_model_cells"].map(
                lambda value: f"{int(value)}/{task_model_denominator}"
            ),
            "Mondrian estimable (available attempts)": source["mondrian_estimable_rate"].map(
                lambda value: f"{100 * float(value):.1f}%"
            ),
            "Finite threshold (available attempts)": source["mondrian_finite_threshold_rate"].map(
                lambda value: f"{100 * float(value):.1f}%"
            ),
        }
    )


def load_and_validate_tables(source_root: Path) -> dict[str, pd.DataFrame]:
    """Load public-facing expected tables and prove they agree with frozen source CSVs."""

    outputs: dict[str, pd.DataFrame] = {}
    provenance_columns = {
        "table1_tasks": {"processed_checksum", "source_evidence"},
        "table2_methods": {"source_file"},
        "table3_h2_h3": {"scientific_source"},
    }
    for stem in TABLE_STEMS:
        source_path = source_root / f"{stem}_source.csv"
        expected_path = source_root / f"{stem}_expected.csv"
        expected = read_text_table(expected_path, source_root)
        if stem == "table4_h5":
            numeric_source = read_csv(source_path, source_root)
            derived = derive_table4(numeric_source)
        else:
            source = read_text_table(source_path, source_root)
            drop = provenance_columns[stem]
            missing = sorted(drop.difference(source.columns))
            if missing:
                raise BuildError(f"{source_path.name} is missing provenance columns: {missing}")
            derived = source.drop(columns=sorted(drop))
        assert_text_frames_equal(derived.astype(str), expected.astype(str), stem)
        if stem == "table3_h2_h3":
            integer_columns = (
                "Budget",
                "Material-positive task–model cells / 10",
                "LOTO expected direction retained / 10",
            )
            for column in integer_columns:
                expected[column] = expected[column].astype(int)
        elif stem == "table4_h5":
            expected["Budget"] = expected["Budget"].astype(int)
        outputs[stem] = expected
    return outputs


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, directory: Path, stem: str) -> None:
    metadata = {
        "Creator": "Stage 3R deterministic matplotlib artifact build",
        "CreationDate": PDF_TIMESTAMP,
        "ModDate": PDF_TIMESTAMP,
    }
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight", metadata=metadata)
    figure.savefig(
        directory / f"{stem}.png",
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def build_figure1(source: pd.DataFrame, directory: Path) -> None:
    relation = dict(
        zip(
            source["role"],
            source.get("stage3q_relation", source["purpose"]),
            strict=True,
        )
    )
    figure, axis = plt.subplots(figsize=(14.2, 7.1))
    axis.set_xlim(0, 14.2)
    axis.set_ylim(0, 7.1)
    axis.axis("off")
    axis.text(0.3, 6.7, "SOURCE DOMAIN", fontsize=13, weight="bold", color="#244B68")
    axis.text(0.3, 3.35, "TARGET DOMAIN", fontsize=13, weight="bold", color="#7A4700")
    axis.plot([0.25, 13.9], [3.75, 3.75], color="#B0B0B0", lw=1.2)

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        face: str,
        edge: str,
        fontsize: float = 9.0,
    ) -> None:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.05",
                fc=face,
                ec=edge,
                lw=1.2,
            )
        )
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)

    box(0.45, 5.0, 2.2, 0.95, "source_train\nsource_tune", "#DCEAF7", "#3B6C91")
    box(
        3.35,
        5.0,
        2.25,
        0.95,
        "Frozen base predictor\n(no target retraining)",
        "#D6EAF8",
        "#24557A",
    )
    box(
        6.35,
        5.0,
        2.65,
        0.95,
        "source_probability_calibration\nsource_conformal_calibration",
        "#DCEAF7",
        "#3B6C91",
        8.2,
    )
    box(
        9.7,
        5.0,
        2.05,
        0.95,
        "source_id_test\nsource-only evaluation",
        "#DCEAF7",
        "#3B6C91",
        8.5,
    )
    for start, end in (
        ((2.65, 5.48), (3.35, 5.48)),
        ((5.6, 5.48), (6.35, 5.48)),
        ((9.0, 5.48), (9.7, 5.48)),
    ):
        axis.add_patch(
            FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=13, color="#4C4C4C")
        )

    box(0.45, 1.65, 2.3, 1.0, "Target entities\nassigned once by hash", "#FCE7CC", "#A86100")
    box(
        3.65, 2.15, 2.55, 1.0, "target_adaptation_pool\nblind nested prefixes", "#FCE7CC", "#A86100"
    )
    box(3.65, 0.55, 2.55, 1.0, "target_final_test\nevaluation only", "#FCE7CC", "#A86100")
    box(
        7.15,
        2.15,
        2.7,
        1.0,
        "Fit target probability mapping\nor conformal threshold",
        "#F8D9B0",
        "#8A4F00",
    )
    box(
        10.7,
        0.95,
        2.75,
        1.25,
        "Apply the frozen predictor\nand selected mapping;\nreport held-out metrics",
        "#E8E8E8",
        "#666666",
    )

    arrow_specs = (
        ((2.75, 2.15), (3.65, 2.65), "#A86100", "-"),
        ((2.75, 2.15), (3.65, 1.05), "#A86100", "-"),
        ((6.2, 2.65), (7.15, 2.65), "#A86100", "-"),
        ((9.85, 2.65), (11.0, 2.05), "#777777", "--"),
        ((6.2, 1.05), (10.7, 1.4), "#A86100", "-"),
        ((4.48, 4.98), (4.9, 3.2), "#24557A", ":"),
        ((4.48, 4.98), (11.7, 2.2), "#24557A", ":"),
    )
    for start, end, color, linestyle in arrow_specs:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                mutation_scale=12 if linestyle == ":" else 13,
                color=color,
                linestyle=linestyle,
            )
        )
    axis.text(
        4.9, 3.45, "frozen scores applied to target roles", ha="center", fontsize=8, color="#24557A"
    )
    axis.text(
        4.95,
        0.15,
        "Mutually exclusive target roles; no adaptation entity enters final testing",
        ha="center",
        fontsize=9.2,
        weight="bold",
        color="#7A4700",
    )
    axis.text(
        13.0,
        5.45,
        "One entity\nOne row\nOne label",
        ha="center",
        va="center",
        fontsize=9.3,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#F3F3F3", "ec": "#777777"},
    )
    # Assert that the input—not any repository authority—supplies every frozen role relation.
    if any(not str(relation[role]).strip() for role in sorted(relation)):
        raise BuildError("figure1 contains an empty frozen role relation")
    save_figure(figure, directory, FIGURE_STEMS[0])


def build_figure2(source: pd.DataFrame, directory: Path) -> None:
    budgets = sorted(source["budget"].astype(int).unique().tolist())
    x_values = np.arange(len(budgets))
    figure, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), sharex=True)
    for method in METHOD_ORDER:
        group = source[source["method_label"] == method].set_index("budget").loc[budgets]
        axes[0].fill_between(
            x_values,
            group["q25"],
            group["q75"],
            color=COLORS[method],
            alpha=0.16,
            linewidth=0,
        )
        axes[0].plot(
            x_values,
            group["task_equal_delta_log_loss"],
            marker="o",
            lw=2.1,
            color=COLORS[method],
            label=method,
        )
        axes[0].vlines(
            x_values,
            group["minimum_task_model"],
            group["maximum_task_model"],
            color=COLORS[method],
            alpha=0.28,
            lw=0.8,
        )
        axes[1].plot(
            x_values,
            100 * group["material_harm_rate"],
            marker="o",
            lw=2.1,
            color=COLORS[method],
            label=method,
        )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].axhline(0.002, color="#555555", lw=0.9, ls="--", label="Material-benefit threshold")
    axes[0].set_yscale("symlog", linthresh=0.002, linscale=1.0)
    axes[0].set_ylabel("Log-loss benefit (positive = improvement)\nSymmetric-log scale")
    axes[0].set_title("A. Task-equal mean, task–model IQR and range")
    axes[1].set_ylabel("Material-harm rate among successful pairs (%)")
    axes[1].set_ylim(-3, 103)
    axes[1].set_title("B. Conditional material harm")
    for axis in axes:
        axis.set_xticks(x_values, budgets)
        axis.set_xlabel("Blind-random target-label budget")
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    figure.tight_layout()
    save_figure(figure, directory, FIGURE_STEMS[1])


def build_figure3(source: pd.DataFrame, directory: Path) -> None:
    budgets = sorted(source["budget"].astype(int).unique().tolist())
    columns = pd.MultiIndex.from_product([METHOD_ORDER, budgets], names=["method", "budget"])
    figure, axes = plt.subplots(2, 1, figsize=(18.0, 8.0), sharex=True)
    normalizer = SymLogNorm(linthresh=0.002, linscale=1.0, vmin=-0.55, vmax=0.055, base=10)
    image = None
    for axis, model in zip(axes, MODEL_LABEL, strict=True):
        matrix = source[source["base_model"] == model].pivot(
            index="task_label", columns=["method_label", "budget"], values="median"
        )
        matrix = matrix.reindex(index=TASK_ORDER, columns=columns)
        if matrix.isna().any().any():
            raise BuildError(f"figure3 has missing cells for {model}")
        image = axis.imshow(
            matrix.to_numpy(),
            aspect="auto",
            cmap="RdBu",
            norm=normalizer,
            interpolation="nearest",
        )
        axis.set_yticks(range(len(TASK_ORDER)), TASK_ORDER, fontsize=10)
        axis.set_title(MODEL_LABEL[model], loc="left", fontsize=12, weight="bold")
        for boundary in np.arange(len(budgets), len(columns), len(budgets)) - 0.5:
            axis.axvline(boundary, color="white", lw=2.5)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                if abs(matrix.iloc[row, column]) < 0.002:
                    axis.plot(
                        column,
                        row,
                        marker="o",
                        markersize=3.3,
                        markerfacecolor="none",
                        markeredgecolor="black",
                        markeredgewidth=0.7,
                    )
    axes[-1].set_xticks(
        range(len(columns)),
        [str(budget) for _ in METHOD_ORDER for budget in budgets],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    axes[-1].set_xlabel("Budget within each target-domain probability update", fontsize=11)
    for index, method in enumerate(METHOD_ORDER):
        axes[0].text(
            index * len(budgets) + (len(budgets) - 1) / 2,
            -1.05,
            method,
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
    color_axis = figure.add_axes([0.925, 0.18, 0.014, 0.60])
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("Median log-loss benefit (positive = improvement)")
    figure.text(
        0.5,
        0.015,
        "Open circles mark |effect| < 0.002; colors remain interpretable "
        "in grayscale by sign and marker.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.16, right=0.90, top=0.89, bottom=0.15, hspace=0.28)
    save_figure(figure, directory, FIGURE_STEMS[2])


def build_figure4(source: pd.DataFrame, directory: Path) -> None:
    data = source.sort_values("budget").reset_index(drop=True)
    x_values = np.arange(len(data))
    figure, axes = plt.subplots(1, 3, figsize=(15.4, 4.9))
    axes[0].plot(
        x_values,
        data["delta_worst_class_coverage"],
        marker="o",
        lw=2,
        color="#0072B2",
        label="Joint success",
    )
    axes[0].plot(
        x_values,
        data["finite_delta_worst_class_coverage"],
        marker="s",
        lw=2,
        color="#009E73",
        label="Joint success + finite threshold",
    )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Mondrian − Standard coverage")
    axes[0].set_title("A. Worst-class coverage")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(
        x_values,
        data["delta_mean_set_size"],
        marker="o",
        lw=2,
        color="#D55E00",
        label="Joint success",
    )
    axes[1].plot(
        x_values,
        data["finite_delta_mean_set_size"],
        marker="s",
        lw=2,
        color="#CC79A7",
        label="Joint success + finite threshold",
    )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Mondrian − Standard mean set size")
    axes[1].set_title("B. Set-size cost")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(
        x_values,
        data["mondrian_estimable_rate"],
        marker="o",
        lw=2,
        color="#0072B2",
        label="Estimable",
    )
    axes[2].plot(
        x_values,
        data["mondrian_finite_threshold_rate"],
        marker="s",
        lw=2,
        color="#D55E00",
        label="Finite threshold",
    )
    axes[2].set_ylim(-0.03, 1.05)
    axes[2].set_ylabel("Rate across available attempts")
    axes[2].set_title("C. Mondrian feasibility")
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    for axis in axes:
        axis.set_xticks(x_values, data["budget"].astype(int).astype(str))
        axis.set_xlabel("Target-label budget")
        axis.grid(axis="y", alpha=0.22)
    low_budget = data.iloc[0]
    axes[0].annotate(
        f"At budget {int(low_budget['budget'])}, joint-success differences\n"
        "mostly reflect infinite thresholds/full sets",
        xy=(0, low_budget["delta_worst_class_coverage"]),
        xytext=(0.6, 0.22),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=8,
    )
    figure.tight_layout()
    save_figure(figure, directory, FIGURE_STEMS[3])


def build_figure5(source: pd.DataFrame, directory: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), sharex=True)
    tasks = sorted(source["task_label"].unique())
    palette_colors = plt.get_cmap("tab10").colors[: len(tasks)]
    palette = dict(zip(tasks, palette_colors, strict=True))
    for task, group in source.groupby("task_label", sort=True):
        for model, default_marker in (("logistic_regression", "o"), ("xgboost_cpu", "s")):
            subset = group[group["base_model"] == model]
            marker = "*" if task == "College Scorecard" else default_marker
            size = 62 if marker == "*" else 35
            axes[0].scatter(
                subset["delta_auroc"],
                subset["delta_brier"],
                s=size,
                marker=marker,
                color=palette[task],
                alpha=0.82,
            )
            axes[1].scatter(
                subset["delta_auroc"],
                subset["delta_log_loss"],
                s=size,
                marker=marker,
                color=palette[task],
                alpha=0.82,
            )
    for axis in axes:
        axis.axvline(0, color="black", lw=0.7)
        axis.axvline(-0.01, color="#777777", lw=0.8, ls="--")
        axis.axvline(0.01, color="#777777", lw=0.8, ls="--")
        axis.axhline(0, color="black", lw=0.7)
        axis.set_xscale("symlog", linthresh=0.01, linscale=1.0)
        axis.set_xlabel("Target-final − source-ID AUROC (symmetric-log x-axis)")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Target-final − source-ID Brier score")
    axes[1].set_ylabel("Target-final − source-ID log loss")
    axes[0].set_title("A. Brier score")
    axes[1].set_title("B. Log loss")
    task_handles = [
        Line2D(
            [0],
            [0],
            marker="*" if task == "College Scorecard" else "o",
            linestyle="",
            color=palette[task],
            label=task,
            markersize=8 if task == "College Scorecard" else 6,
        )
        for task in tasks
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="#555555",
            label="Logistic regression",
            markersize=6,
        ),
        Line2D([0], [0], marker="s", linestyle="", color="#555555", label="XGBoost", markersize=6),
    ]
    handles = task_handles + model_handles
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        frameon=False,
        fontsize=7.6,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.tight_layout(rect=[0, 0.17, 1, 1])
    save_figure(figure, directory, FIGURE_STEMS[4])


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")

    header = "| " + " | ".join(cell(column) for column in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join((header, divider, *body)) + "\n"


def latex_table(frame: pd.DataFrame) -> str:
    """Render the small frozen tables without Pandas' optional Jinja2 dependency."""

    replacements = (
        ("\\", r"\textbackslash "),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde "),
        ("^", r"\textasciicircum "),
    )

    def cell(value: Any) -> str:
        rendered = str(value)
        for original, escaped in replacements:
            rendered = rendered.replace(original, escaped)
        return rendered

    alignment = "".join(
        "r" if pd.api.types.is_numeric_dtype(frame[column]) else "l"
        for column in frame.columns
    )
    rows = [
        " & ".join(cell(value) for value in frame.columns) + " \\\\",
        r"\midrule",
        *(
            " & ".join(cell(value) for value in row) + " \\\\"
            for row in frame.itertuples(index=False, name=None)
        ),
    ]
    return (
        f"\\begin{{tabular}}{{{alignment}}}\n\\toprule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def build_tables(tables: dict[str, pd.DataFrame], directory: Path) -> None:
    for stem in TABLE_STEMS:
        frame = tables[stem]
        frame.to_csv(directory / f"{stem}.csv", index=False)
        (directory / f"{stem}.md").write_text(markdown_table(frame), encoding="utf-8")
        caption, label = TABLE_CAPTIONS[stem]
        latex_body = latex_table(frame)
        (directory / f"{stem}.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n\\footnotesize\n"
            + latex_body
            + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table*}}\n",
            encoding="utf-8",
        )


def run_checked(command: list[str], cwd: Path) -> None:
    environment = os.environ.copy()
    environment.update({"SOURCE_DATE_EPOCH": "1767225600", "TZ": "UTC"})
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        tail = " | ".join(detail[-8:])
        raise BuildError(f"{Path(command[0]).name} failed with exit {completed.returncode}: {tail}")


def tool_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "--version"], check=False, capture_output=True, text=True
    )
    if completed.returncode:
        return "VERSION_QUERY_FAILED"
    return (completed.stdout or completed.stderr).strip().splitlines()[0]


def expand_document(source: str, tables_directory: Path) -> str:
    text = source
    for marker, filename in TABLE_MARKERS.items():
        if marker in text:
            table = (tables_directory / filename).read_text(encoding="utf-8").strip()
            text = text.replace(marker, table)
    remaining = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", text)))
    if remaining:
        raise BuildError(f"unexpanded manuscript markers: {remaining}")
    text = re.sub(r"(?m)^bibliography:.*$", "bibliography: references.bib", text)
    text = text.replace("figures_v2/", "../figures/")
    return text


def finalize_latex(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r",alt=\{.*?\}\]\{", "]{", text, flags=re.DOTALL)
    text = text.replace("\u00a0", "~")
    replacements = {
        "−": r"\ensuremath{-}",
        "≤": r"\ensuremath{\leq}",
        "≥": r"\ensuremath{\geq}",
        "Δ": r"\ensuremath{\Delta}",
        "∈": r"\ensuremath{\in}",
        "μ": r"\ensuremath{\mu}",
        "µ": r"\ensuremath{\mu}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    def break_hash(match: re.Match[str]) -> str:
        value = match.group(1)
        chunks = [value[index : index + 8] for index in range(0, len(value), 8)]
        return r"\texttt{" + r"\allowbreak{}".join(chunks) + "}"

    text = re.sub(r"\\texttt\{([0-9a-f]{40})\}", break_hash, text)

    def break_table_tokens(match: re.Match[str]) -> str:
        block = match.group(0)
        return block.replace(r"\_", r"\_\allowbreak{}").replace("/", r"/\allowbreak{}")

    text = re.sub(
        r"\\begin\{longtable\}.*?\\end\{longtable\}",
        break_table_tokens,
        text,
        flags=re.DOTALL,
    )
    text = text.replace(
        "\\begin{document}",
        "\\usepackage{pdflscape}\n\\setlength{\\emergencystretch}{3em}\n\\begin{document}",
        1,
    )
    text = text.replace(
        "{\\def\\LTcaptype{none} % do not increment counter\n\\begin{longtable}",
        "\\begin{landscape}\n\\begingroup\n\\scriptsize\n"
        "{\\def\\LTcaptype{none} % do not increment counter\n\\begin{longtable}",
    )
    text = text.replace(
        "\\end{longtable}\n}",
        "\\end{longtable}\n}\n\\endgroup\n\\end{landscape}",
    )
    path.write_text(text, encoding="utf-8")


def find_bibliography(manuscript_root: Path) -> Path:
    candidates = (
        manuscript_root / "references_stage3q.bib",
        manuscript_root / "references.bib",
    )
    bibliography = next((path for path in candidates if path.is_file()), None)
    if bibliography is None:
        raise BuildError("manuscript_build is missing references_stage3q.bib or references.bib")
    return require_file(bibliography, manuscript_root)


def build_documents(
    manuscript_root: Path,
    tables_directory: Path,
    output_directory: Path,
) -> dict[str, str]:
    pandoc = shutil.which("pandoc")
    tectonic = shutil.which("tectonic")
    missing = [
        name for name, executable in (("pandoc", pandoc), ("tectonic", tectonic)) if not executable
    ]
    if missing:
        raise BuildError(
            "required manuscript tool(s) unavailable; no frozen PDF was copied: "
            + ", ".join(missing)
        )

    detected_versions = {
        "pandoc": tool_version(pandoc),
        "tectonic": tool_version(tectonic),
    }
    mismatches = {
        name: {"expected": expected, "actual": detected_versions[name]}
        for name, expected in DOCUMENT_TOOL_VERSIONS.items()
        if detected_versions[name] != expected
    }
    if mismatches:
        raise BuildError(f"manuscript tool version mismatch: {mismatches}")

    main_path = require_file(manuscript_root / "manuscript_source.md", manuscript_root)
    supplement_path = require_file(manuscript_root / "supplement_source.md", manuscript_root)
    bibliography = find_bibliography(manuscript_root)
    shutil.copyfile(bibliography, output_directory / "references.bib")

    document_inputs = {
        "manuscript": main_path.read_text(encoding="utf-8"),
        "supplement": supplement_path.read_text(encoding="utf-8"),
    }
    for stem, source in document_inputs.items():
        expanded = expand_document(source, tables_directory)
        temporary = output_directory / f"_{stem}_expanded_source.md"
        temporary.write_text(expanded, encoding="utf-8")
        common = [
            "--from=markdown-smart",
            "--standalone",
            "--citeproc",
            "--bibliography=references.bib",
            "--resource-path=.:..",
            "--metadata=link-citations:true",
            "--metadata=lang:en-US",
        ]
        try:
            run_checked(
                [
                    pandoc,
                    temporary.name,
                    *common,
                    "--to=gfm+tex_math_dollars",
                    f"--output={stem}.md",
                ],
                output_directory,
            )
            run_checked(
                [
                    pandoc,
                    temporary.name,
                    *common,
                    "--to=latex",
                    "--shift-heading-level-by=-1",
                    f"--output={stem}.tex",
                ],
                output_directory,
            )
            finalize_latex(output_directory / f"{stem}.tex")
            run_checked(
                [tectonic, "-X", "compile", "--outdir", ".", f"{stem}.tex"],
                output_directory,
            )
        finally:
            temporary.unlink(missing_ok=True)
        pdf = output_directory / f"{stem}.pdf"
        if not pdf.is_file() or pdf.stat().st_size < 1000:
            raise BuildError(f"Tectonic did not create a valid nonempty {stem}.pdf")
        if not pdf.read_bytes().startswith(b"%PDF-"):
            raise BuildError(f"{stem}.pdf does not have a PDF signature")
    return detected_versions


def artifact_records(output_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(candidate for candidate in output_root.rglob("*") if candidate.is_file()):
        if path.name in {
            "ARTIFACT_BUILD_REPORT.json",
            "ARTIFACT_BUILD_REPORT.json.sha256",
            "ARTIFACT_SHA256SUMS.txt",
        }:
            continue
        records.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def sanitize_error(message: str, roots: tuple[Path, ...]) -> str:
    result = message
    for index, root in enumerate(roots, start=1):
        result = result.replace(str(root.resolve()), f"<RUNTIME_ROOT_{index}>")
    return result


def write_report(output_root: Path, report: dict[str, Any]) -> None:
    report_path = output_root / "ARTIFACT_BUILD_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256(report_path)
    (output_root / "ARTIFACT_BUILD_REPORT.json.sha256").write_text(
        f"{digest}  ARTIFACT_BUILD_REPORT.json\n", encoding="utf-8"
    )


def reset_output_directories(output_root: Path, input_roots: tuple[Path, ...]) -> dict[str, Path]:
    resolved_output = output_root.resolve()
    for input_root in input_roots:
        resolved_input = input_root.resolve()
        if resolved_output == resolved_input or resolved_output.is_relative_to(resolved_input):
            raise BuildError("output root must not be inside an input directory")
    output_root.mkdir(parents=True, exist_ok=True)
    directories = {
        "figures": output_root / "figures",
        "tables": output_root / "tables",
        "manuscript": output_root / "manuscript",
    }
    for directory in directories.values():
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    for filename in (
        "ARTIFACT_BUILD_REPORT.json",
        "ARTIFACT_BUILD_REPORT.json.sha256",
        "ARTIFACT_SHA256SUMS.txt",
    ):
        (output_root / filename).unlink(missing_ok=True)
    return directories


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="release-bundle root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("build/artifacts"),
        help="artifact output directory; relative paths are resolved below bundle root",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundle_root = args.bundle_root.resolve()
    source_root = bundle_root / "figure_source_data"
    manuscript_root = bundle_root / "manuscript_build"
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = bundle_root / output_root
    output_root = output_root.resolve()

    base_report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "reproducibility_level": "A_FROZEN_ARTIFACTS",
        "scientific_input_roots": ["figure_source_data", "manuscript_build"],
        "external_repository_authorities_read": False,
        "previously_compiled_pdfs_copied": False,
    }
    try:
        if not source_root.is_dir():
            raise BuildError("required input directory is missing: figure_source_data")
        if not manuscript_root.is_dir():
            raise BuildError("required input directory is missing: manuscript_build")
        directories = reset_output_directories(output_root, (source_root, manuscript_root))
        source_index = validate_source_index(source_root)
        figures = load_figure_sources(source_root)
        tables = load_and_validate_tables(source_root)
        configure_matplotlib()
        build_figure1(figures[FIGURE_STEMS[0]], directories["figures"])
        build_figure2(figures[FIGURE_STEMS[1]], directories["figures"])
        build_figure3(figures[FIGURE_STEMS[2]], directories["figures"])
        build_figure4(figures[FIGURE_STEMS[3]], directories["figures"])
        build_figure5(figures[FIGURE_STEMS[4]], directories["figures"])
        build_tables(tables, directories["tables"])
        manuscript_tools = build_documents(
            manuscript_root, directories["tables"], directories["manuscript"]
        )
        records = artifact_records(output_root)
        sums_path = output_root / "ARTIFACT_SHA256SUMS.txt"
        sums_path.write_text(
            "".join(f"{record['sha256']}  {record['path']}\n" for record in records),
            encoding="utf-8",
        )
        expected_paths = {
            *(
                f"figures/{stem}.{extension}"
                for stem in FIGURE_STEMS
                for extension in ("pdf", "png")
            ),
            *(
                f"tables/{stem}.{extension}"
                for stem in TABLE_STEMS
                for extension in ("csv", "md", "tex")
            ),
            "manuscript/manuscript.md",
            "manuscript/manuscript.tex",
            "manuscript/manuscript.pdf",
            "manuscript/supplement.md",
            "manuscript/supplement.tex",
            "manuscript/supplement.pdf",
            "manuscript/references.bib",
        }
        actual_paths = {record["path"] for record in records}
        missing_outputs = sorted(expected_paths.difference(actual_paths))
        if missing_outputs:
            raise BuildError(f"artifact build omitted required outputs: {missing_outputs}")
        report = {
            **base_report,
            "status": "PASS",
            "source_index": source_index,
            "source_files": [
                {
                    "path": path.relative_to(bundle_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in sorted(
                    [
                        *(source_root / f"{stem}.csv" for stem in FIGURE_STEMS),
                        *(source_root / f"{stem}_source.csv" for stem in TABLE_STEMS),
                        *(source_root / f"{stem}_expected.csv" for stem in TABLE_STEMS),
                        manuscript_root / "manuscript_source.md",
                        manuscript_root / "supplement_source.md",
                        find_bibliography(manuscript_root),
                    ]
                )
            ],
            "toolchain": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
                **manuscript_tools,
            },
            "validation": {
                "figure_sources_validated": len(FIGURE_STEMS),
                "table_source_expected_pairs_validated": len(TABLE_STEMS),
                "figure_files": 2 * len(FIGURE_STEMS),
                "table_files": 3 * len(TABLE_STEMS),
                "manuscript_pdfs_built_from_source": 2,
                "missing_required_outputs": [],
            },
            "artifacts": records,
            "sha256_manifest": {
                "path": "ARTIFACT_SHA256SUMS.txt",
                "sha256": sha256(sums_path),
            },
        }
        write_report(output_root, report)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "figures": 2 * len(FIGURE_STEMS),
                    "tables": 3 * len(TABLE_STEMS),
                    "manuscript_pdfs": 2,
                    "report": "ARTIFACT_BUILD_REPORT.json",
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        message = sanitize_error(str(exc), (bundle_root, output_root))
        failure = {
            **base_report,
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": message,
            "artifacts": artifact_records(output_root),
        }
        write_report(output_root, failure)
        print(json.dumps({"status": "FAIL", "error": message}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
