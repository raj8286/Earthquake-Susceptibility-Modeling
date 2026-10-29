# 26_plot_magnitude_threshold_sensitivity_v5.py
#
# Final magnitude-threshold sensitivity figures for the canonical v5
# 30-km Extra Trees model.
#
# Reads the fresh outputs from:
#   magnitude_threshold_sensitivity_v5_30km_18features
#
# Plot conventions:
#   - all visible text = 15 pt bold
#   - figure titles intentionally commented out
#   - PNG filename exactly matches intended title
#   - existing/default matplotlib colour scheme retained
#   - 600 DPI publication-quality output

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PATHS / STYLE
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = (
    SCRIPT_DIR
    / "magnitude_threshold_sensitivity_v5_30km_18features"
)

INPUT_FILE = (
    RESULTS_DIR
    / "magnitude_threshold_summary.csv"
)

PLOTS_DIR = (
    SCRIPT_DIR
    / "plots_v5"
    / "magnitude_threshold_sensitivity"
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FONT_SIZE = 15
DPI = 600

THRESHOLD_ORDER = [
    "M>=4.0",
    "M>=4.5",
    "M>=5.0",
    "M>=5.5",
    "M>=6.0",
]

DISPLAY_LABELS = [
    "M≥4.0",
    "M≥4.5",
    "M≥5.0",
    "M≥5.5",
    "M≥6.0",
]


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


def force_bold(ax):
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

    print(f"Saved: {output.resolve()}")


# =============================================================================
# LOAD / VALIDATE
# =============================================================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing fresh v5 magnitude-threshold summary:\n"
            f"{INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "magnitude_threshold",
        "positive_cells",
        "prevalence_percent",

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
            f"Missing required columns: {missing}"
        )

    found = set(
        df["magnitude_threshold"]
        .astype(str)
    )

    expected = set(
        THRESHOLD_ORDER
    )

    if found != expected:
        raise RuntimeError(
            "Magnitude-threshold mismatch.\n"
            f"Expected: {sorted(expected)}\n"
            f"Found:    {sorted(found)}"
        )

    rank = {
        name: index
        for index, name in enumerate(
            THRESHOLD_ORDER
        )
    }

    df = df.copy()

    df["_order"] = (
        df["magnitude_threshold"]
        .map(rank)
    )

    df = (
        df
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return df


# =============================================================================
# FIGURE 1 — POSITIVE-CELL PREVALENCE
# =============================================================================

def plot_prevalence(df):
    title = (
        "Positive-Cell Prevalence Across Earthquake Magnitude Thresholds"
    )

    x = np.arange(
        len(df)
    )

    y = df[
        "prevalence_percent"
    ].to_numpy(dtype=float)

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

    ax.set_xticks(x)

    ax.set_xticklabels(
        DISPLAY_LABELS
    )

    ax.set_xlabel(
        "Earthquake magnitude threshold"
    )

    ax.set_ylabel(
        "Positive-cell prevalence (%)"
    )

    ax.set_ylim(
        0.0,
        float(np.max(y)) * 1.25,
    )

    ax.grid(
        alpha=0.25,
    )

    for xi, yi in zip(x, y):
        ax.annotate(
            f"{yi:.3f}%",
            (xi, yi),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
            fontweight="bold",
        )

    # Title intentionally omitted.
    # ax.set_title(
    #     title,
    #     fontsize=FONT_SIZE,
    #     fontweight="bold",
    # )

    force_bold(ax)

    fig.tight_layout()

    save_figure(
        fig,
        title,
    )


# =============================================================================
# FIGURE 2 — MODEL PERFORMANCE
# =============================================================================

def plot_performance(df):
    title = (
        "Model Performance Across Earthquake Magnitude Thresholds"
    )

    x = np.arange(
        len(df)
    )

    metrics = [
        (
            "ROC-AUC",
            "roc_auc_mean",
            "roc_auc_std",
        ),
        (
            "PR-AUC",
            "pr_auc_mean",
            "pr_auc_std",
        ),
        (
            "F1",
            "f1_at_best_mean",
            "f1_at_best_std",
        ),
        (
            "MCC",
            "mcc_at_best_mean",
            "mcc_at_best_std",
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    all_low = []
    all_high = []

    for (
        label,
        mean_col,
        std_col,
    ) in metrics:

        y = df[
            mean_col
        ].to_numpy(dtype=float)

        yerr = df[
            std_col
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
            label=label,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        DISPLAY_LABELS
    )

    ax.set_xlabel(
        "Earthquake magnitude threshold"
    )

    ax.set_ylabel(
        "Mean spatial cross-validation score"
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
        0.025,
        0.08 * span,
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
        alpha=0.25,
    )

    ax.legend(
        frameon=True,
        loc="best",
        prop={
            "size": FONT_SIZE,
            "weight": "bold",
        },
    )

    # Title intentionally omitted.
    # ax.set_title(
    #     title,
    #     fontsize=FONT_SIZE,
    #     fontweight="bold",
    # )

    force_bold(ax)

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

    df = load_data()

    print(
        "FINAL V5 MAGNITUDE-THRESHOLD SENSITIVITY"
    )
    print("=" * 120)

    print(
        df[
            [
                "magnitude_threshold",
                "positive_cells",
                "prevalence_percent",
                "roc_auc_mean",
                "pr_auc_mean",
                "f1_at_best_mean",
                "mcc_at_best_mean",
            ]
        ].to_string(
            index=False
        )
    )

    plot_prevalence(
        df
    )

    plot_performance(
        df
    )

    print(
        "\nAll magnitude-threshold figures "
        "saved under:"
    )

    print(
        PLOTS_DIR.resolve()
    )


if __name__ == "__main__":
    main()