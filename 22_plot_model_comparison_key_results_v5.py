# 22_plot_model_comparison_key_results_v5.py
#
# Final model-comparison figures for the canonical v5 30-km M>=4 experiment.
#
# Plot conventions:
#   - fresh v5 fold-level results only
#   - all visible text = 15 pt bold
#   - figure titles intentionally commented out
#   - PNG filename exactly matches intended title
#   - existing/default matplotlib colour scheme preserved
#   - 600 DPI publication-quality output

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PATHS / CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_DIR = (
    SCRIPT_DIR
    / "model_comparison_spatial_cv_v5_m4_18features_30km"
)

INPUT_FILE = (
    INPUT_DIR
    / "model_comparison_fold_metrics.csv"
)

PLOTS_DIR = (
    SCRIPT_DIR
    / "plots_v5"
    / "model_comparison"
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FONT_SIZE = 15
DPI = 600

MODEL_ORDER = [
    "xgboost",
    "lightgbm",
    "random_forest",
    "extra_trees",
]

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
}

REQUIRED_COLUMNS = [
    "model",
    "fold",
    "roc_auc",
    "pr_auc",
    "precision_at_best",
    "recall_at_best",
    "f1_at_best",
    "mcc_at_best",
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
# DATA
# =============================================================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing fresh v5 model-comparison file:\n"
            f"{INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(INPUT_FILE)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.copy()

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
            f"Model mismatch.\n"
            f"Expected: {sorted(expected)}\n"
            f"Found:    {sorted(found)}"
        )

    for model in MODEL_ORDER:
        folds = sorted(
            df.loc[
                df["model"] == model,
                "fold",
            ]
            .astype(int)
            .tolist()
        )

        if folds != [1, 2, 3, 4, 5]:
            raise RuntimeError(
                f"{model}: expected folds 1-5, "
                f"found {folds}"
            )

    return df


def build_summary(df):
    summary = (
        df.groupby(
            "model",
            as_index=False,
        )
        .agg(
            roc_auc_mean=(
                "roc_auc",
                "mean",
            ),
            roc_auc_std=(
                "roc_auc",
                "std",
            ),
            pr_auc_mean=(
                "pr_auc",
                "mean",
            ),
            pr_auc_std=(
                "pr_auc",
                "std",
            ),
            precision_mean=(
                "precision_at_best",
                "mean",
            ),
            recall_mean=(
                "recall_at_best",
                "mean",
            ),
            f1_mean=(
                "f1_at_best",
                "mean",
            ),
            f1_std=(
                "f1_at_best",
                "std",
            ),
            mcc_mean=(
                "mcc_at_best",
                "mean",
            ),
            mcc_std=(
                "mcc_at_best",
                "std",
            ),
        )
    )

    order_map = {
        model: index
        for index, model in enumerate(
            MODEL_ORDER
        )
    }

    summary["order"] = (
        summary["model"]
        .map(order_map)
    )

    summary = (
        summary
        .sort_values("order")
        .drop(columns="order")
        .reset_index(drop=True)
    )

    summary["model_label"] = (
        summary["model"]
        .map(MODEL_LABELS)
    )

    return summary


# =============================================================================
# BAR PLOTS
# =============================================================================

def plot_metric_bar(
    summary,
    mean_col,
    std_col,
    ylabel,
    title,
):
    fig, ax = plt.subplots(
        figsize=(14, 9)
    )

    x = np.arange(
        len(summary)
    )

    y = summary[
        mean_col
    ].to_numpy(dtype=float)

    yerr = summary[
        std_col
    ].to_numpy(dtype=float)

    bars = ax.bar(
        x,
        y,
        yerr=yerr,
        capsize=6,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        summary["model_label"]
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        axis="y",
        alpha=0.30,
    )

    lower = float(
        np.min(y - yerr)
    )

    upper = float(
        np.max(y + yerr)
    )

    span = max(
        upper - lower,
        0.01,
    )

    pad = max(
        0.015,
        span * 0.20,
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

    for bar, value in zip(
        bars,
        y,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            bar.get_height()
            + pad * 0.15,
            f"{value:.3f}",
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
# PR-AUC BY FOLD
# =============================================================================

def plot_pr_auc_by_fold(df):
    title = (
        "PR-AUC Across Spatial Cross-Validation Folds"
    )

    fig, ax = plt.subplots(
        figsize=(14, 9)
    )

    for model in MODEL_ORDER:
        data = (
            df[
                df["model"] == model
            ]
            .sort_values("fold")
        )

        ax.plot(
            data["fold"],
            data["pr_auc"],
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=MODEL_LABELS[model],
        )

    ax.set_xlabel(
        "Spatial CV fold"
    )

    ax.set_ylabel(
        "PR-AUC"
    )

    ax.set_xticks(
        [1, 2, 3, 4, 5]
    )

    ax.grid(
        alpha=0.30,
    )

    ax.legend(
        loc="best",
        frameon=True,
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
# PRECISION / RECALL / F1
# =============================================================================

def plot_precision_recall_f1(summary):
    title = (
        "Precision Recall and F1 Score by Model"
    )

    fig, ax = plt.subplots(
        figsize=(14, 9)
    )

    x = np.arange(
        len(summary)
    )

    width = 0.24

    ax.bar(
        x - width,
        summary["precision_mean"],
        width,
        label="Precision",
    )

    ax.bar(
        x,
        summary["recall_mean"],
        width,
        label="Recall",
    )

    ax.bar(
        x + width,
        summary["f1_mean"],
        width,
        label="F1",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        summary["model_label"]
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.grid(
        axis="y",
        alpha=0.30,
    )

    ax.legend(
        loc="best",
        frameon=True,
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

    summary = build_summary(
        df
    )

    summary.to_csv(
        PLOTS_DIR
        / "Model Comparison Plot Summary.csv",
        index=False,
    )

    print(
        "FINAL V5 30-KM MODEL COMPARISON"
    )
    print("=" * 100)

    print(
        summary[
            [
                "model_label",
                "roc_auc_mean",
                "pr_auc_mean",
                "precision_mean",
                "recall_mean",
                "f1_mean",
                "mcc_mean",
            ]
        ].to_string(
            index=False
        )
    )

    plot_metric_bar(
        summary,
        "pr_auc_mean",
        "pr_auc_std",
        "PR-AUC",
        "Model Comparison PR-AUC",
    )

    plot_pr_auc_by_fold(
        df
    )

    plot_metric_bar(
        summary,
        "roc_auc_mean",
        "roc_auc_std",
        "ROC-AUC",
        "Model Comparison ROC-AUC",
    )

    plot_precision_recall_f1(
        summary
    )

    plot_metric_bar(
        summary,
        "mcc_mean",
        "mcc_std",
        "Matthews correlation coefficient (MCC)",
        "Model Comparison Matthews Correlation Coefficient",
    )

    print(
        "\nAll figures saved under:"
    )

    print(
        PLOTS_DIR.resolve()
    )


if __name__ == "__main__":
    main()