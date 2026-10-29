# 25_plot_all_models_grid_resolution_v5.py
#
# Final grid-resolution sensitivity plotting for the canonical v5 pipeline.
#
# Uses fresh 5-fold spatial-CV M>=4 model-comparison outputs at
# 15 km, 30 km and 60 km.
#
# Generates:
#   1. M>=4 positive-cell prevalence across grid resolutions
#   2. ROC-AUC across resolutions for all four models
#   3. PR-AUC across resolutions for all four models
#   4. F1 score across resolutions for all four models
#   5. MCC across resolutions for all four models
#
# Plot conventions:
#   - all visible text = 15 pt bold
#   - figure titles intentionally commented out
#   - PNG filename exactly matches intended figure title
#   - 600 DPI publication-quality output
#   - no manual colour specification: preserves existing matplotlib scheme

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PATHS / STYLE
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PLOTS_DIR = (
    SCRIPT_DIR
    / "plots_v5"
    / "grid_resolution_all_models"
)
PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FONT_SIZE = 15
DPI = 600

GRID_ORDER = [15, 30, 60]

MODEL_ORDER = [
    "extra_trees",
    "random_forest",
    "xgboost",
    "lightgbm",
]

MODEL_LABELS = {
    "extra_trees": "Extra Trees",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


# Fresh, validated v5 M>=4 target counts from model-dataset preflight.
TARGET_COUNTS = pd.DataFrame(
    [
        {
            "grid_km": 15,
            "total_cells": 2_260_416,
            "positive_cells": 91_387,
        },
        {
            "grid_km": 30,
            "total_cells": 565_104,
            "positive_cells": 42_213,
        },
        {
            "grid_km": 60,
            "total_cells": 141_276,
            "positive_cells": 18_330,
        },
    ]
)

TARGET_COUNTS["prevalence_percent"] = (
    TARGET_COUNTS["positive_cells"]
    / TARGET_COUNTS["total_cells"]
    * 100.0
)


METRICS = {
    "roc_auc": {
        "mean": "roc_auc_mean",
        "std": "roc_auc_std",
        "ylabel": "ROC-AUC",
        "title": "ROC-AUC Across Spatial Grid Resolutions",
    },
    "pr_auc": {
        "mean": "pr_auc_mean",
        "std": "pr_auc_std",
        "ylabel": "PR-AUC",
        "title": "PR-AUC Across Spatial Grid Resolutions",
    },
    "f1": {
        "mean": "f1_at_best_mean",
        "std": "f1_at_best_std",
        "ylabel": "F1 score",
        "title": "F1 Score Across Spatial Grid Resolutions",
    },
    "mcc": {
        "mean": "mcc_at_best_mean",
        "std": "mcc_at_best_std",
        "ylabel": "MCC",
        "title": "MCC Across Spatial Grid Resolutions",
    },
}


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate final v5 grid-resolution sensitivity figures "
            "from fresh 15/30/60-km model-comparison summaries."
        )
    )

    parser.add_argument(
        "--summary15",
        default=str(
            SCRIPT_DIR
            / "model_comparison_spatial_cv_v5_m4_18features_15km"
            / "model_comparison_summary.csv"
        ),
    )

    parser.add_argument(
        "--summary30",
        default=str(
            SCRIPT_DIR
            / "model_comparison_spatial_cv_v5_m4_18features_30km"
            / "model_comparison_summary.csv"
        ),
    )

    parser.add_argument(
        "--summary60",
        default=str(
            SCRIPT_DIR
            / "model_comparison_spatial_cv_v5_m4_18features_60km"
            / "model_comparison_summary.csv"
        ),
    )

    return parser.parse_args()


# =============================================================================
# STYLE
# =============================================================================

def apply_style():
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "font.weight": "bold",
            "axes.labelsize": FONT_SIZE,
            "axes.labelweight": "bold",
            "axes.titlesize": FONT_SIZE,
            "axes.titleweight": "bold",
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
        }
    )


def force_bold_text(ax):
    for label in (
        ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        label.set_fontsize(FONT_SIZE)
        label.set_fontweight("bold")

    ax.xaxis.label.set_fontsize(FONT_SIZE)
    ax.xaxis.label.set_fontweight("bold")

    ax.yaxis.label.set_fontsize(FONT_SIZE)
    ax.yaxis.label.set_fontweight("bold")

    legend = ax.get_legend()

    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(FONT_SIZE)
            text.set_fontweight("bold")


def save_figure(fig, title):
    output = PLOTS_DIR / f"{title}.png"

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.10,
    )

    plt.close(fig)

    print(
        f"Saved: {output.resolve()}"
    )


# =============================================================================
# LOAD / VALIDATE MODEL RESULTS
# =============================================================================

def load_one_summary(path, grid_km):
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {grid_km} km model-comparison summary:\n"
            f"{path}"
        )

    df = pd.read_csv(path)

    required = [
        "model",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "f1_at_best_mean",
        "f1_at_best_std",
        "mcc_at_best_mean",
        "mcc_at_best_std",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{grid_km} km summary is missing columns: "
            f"{missing}"
        )

    df = df[required].copy()

    df["model"] = (
        df["model"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    found = set(df["model"])
    expected = set(MODEL_ORDER)

    if found != expected:
        raise RuntimeError(
            f"{grid_km} km model set mismatch.\n"
            f"Expected: {sorted(expected)}\n"
            f"Found:    {sorted(found)}"
        )

    if df["model"].duplicated().any():
        raise RuntimeError(
            f"{grid_km} km summary contains duplicate models."
        )

    df["grid_km"] = int(grid_km)

    print(
        f"Loaded {grid_km} km:\n"
        f"  {path}"
    )

    return df


def load_all(args):
    combined = pd.concat(
        [
            load_one_summary(
                args.summary15,
                15,
            ),
            load_one_summary(
                args.summary30,
                30,
            ),
            load_one_summary(
                args.summary60,
                60,
            ),
        ],
        ignore_index=True,
    )

    combined.to_csv(
        PLOTS_DIR
        / "All Models 15 30 60 km Metrics.csv",
        index=False,
    )

    return combined


# =============================================================================
# PREVALENCE
# =============================================================================

def plot_prevalence():
    data = (
        TARGET_COUNTS
        .sort_values("grid_km")
        .reset_index(drop=True)
    )

    x = data[
        "grid_km"
    ].to_numpy(dtype=float)

    y = data[
        "prevalence_percent"
    ].to_numpy(dtype=float)

    title = (
        "M≥4 Positive-Cell Prevalence "
        "Across Spatial Grid Resolutions"
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.plot(
        x,
        y,
        marker="o",
        markersize=9,
        linewidth=2.4,
    )

    ax.set_xticks(
        GRID_ORDER
    )

    ax.set_xlim(
        12,
        63,
    )

    ax.set_ylim(
        0.0,
        float(np.max(y)) * 1.25,
    )

    ax.set_xlabel(
        "Grid resolution (km)"
    )

    ax.set_ylabel(
        "M≥4 positive-cell prevalence (%)"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    for xi, yi in zip(x, y):
        ax.annotate(
            f"{yi:.3f}%",
            (xi, yi),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
            fontweight="bold",
        )

    # Title intentionally omitted for manuscript figure.
    # ax.set_title(
    #     title,
    #     fontsize=FONT_SIZE,
    #     fontweight="bold",
    # )

    force_bold_text(ax)

    fig.tight_layout()

    save_figure(
        fig,
        title,
    )


# =============================================================================
# MODEL PERFORMANCE ACROSS RESOLUTIONS
# =============================================================================

def plot_metric(
    combined,
    spec,
):
    title = spec["title"]

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    all_low = []
    all_high = []

    for model in MODEL_ORDER:
        data = (
            combined[
                combined["model"] == model
            ]
            .sort_values("grid_km")
        )

        if (
            data["grid_km"].tolist()
            != GRID_ORDER
        ):
            raise RuntimeError(
                f"{model}: expected grids "
                f"{GRID_ORDER}, found "
                f"{data['grid_km'].tolist()}"
            )

        x = data[
            "grid_km"
        ].to_numpy(dtype=float)

        y = data[
            spec["mean"]
        ].to_numpy(dtype=float)

        yerr = data[
            spec["std"]
        ].to_numpy(dtype=float)

        all_low.extend(
            (y - yerr).tolist()
        )

        all_high.extend(
            (y + yerr).tolist()
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            markersize=8,
            linewidth=2.2,
            capsize=5,
            capthick=1.5,
            label=MODEL_LABELS[model],
        )

    ax.set_xticks(
        GRID_ORDER
    )

    ax.set_xlim(
        12,
        63,
    )

    ax.set_xlabel(
        "Grid resolution (km)"
    )

    ax.set_ylabel(
        spec["ylabel"]
    )

    lower = float(
        np.nanmin(all_low)
    )

    upper = float(
        np.nanmax(all_high)
    )

    span = max(
        upper - lower,
        0.01,
    )

    pad = max(
        0.015,
        0.12 * span,
    )

    ax.set_ylim(
        max(
            0.0,
            lower - pad,
        ),
        min(
            1.0,
            upper + pad,
        ),
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
        loc="best",
        prop={
            "size": FONT_SIZE,
            "weight": "bold",
        },
    )

    # Title intentionally omitted for manuscript figure.
    # ax.set_title(
    #     title,
    #     fontsize=FONT_SIZE,
    #     fontweight="bold",
    # )

    force_bold_text(ax)

    fig.tight_layout()

    save_figure(
        fig,
        title,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    apply_style()

    args = parse_args()

    combined = load_all(
        args
    )

    print(
        "\nFINAL V5 GRID-RESOLUTION "
        "MODEL RESULTS"
    )
    print("=" * 110)

    display = combined[
        [
            "grid_km",
            "model",
            "roc_auc_mean",
            "pr_auc_mean",
            "f1_at_best_mean",
            "mcc_at_best_mean",
        ]
    ].sort_values(
        [
            "grid_km",
            "model",
        ]
    )

    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\nTARGET PREVALENCE"
    )
    print("=" * 110)

    print(
        TARGET_COUNTS.to_string(
            index=False
        )
    )

    plot_prevalence()

    for spec in METRICS.values():
        plot_metric(
            combined,
            spec,
        )

    print(
        "\nAll final grid-resolution "
        "figures saved under:"
    )

    print(
        PLOTS_DIR.resolve()
    )


if __name__ == "__main__":
    main()