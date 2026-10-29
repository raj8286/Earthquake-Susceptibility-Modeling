# 24_plot_calibration_analysis_v5.py
#
# Final probability-calibration figures for the canonical v5
# 30-km M>=4 Extra Trees model.
#
# Plot conventions:
#   - fresh v5 calibration outputs only
#   - all visible text = 15 pt bold
#   - figure titles intentionally commented out
#   - PNG filename exactly matches intended title
#   - existing/default matplotlib colour scheme retained
#   - 600 DPI

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

CALIBRATION_DIR = (
    SCRIPT_DIR
    / "probability_calibration_v5_30km_m4_18features"
)

PLOTS_DIR = (
    SCRIPT_DIR
    / "plots_v5"
    / "calibration"
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FONT_SIZE = 15
DPI = 600

METHOD_ORDER = [
    "raw",
    "sigmoid",
    "isotonic",
]

METHOD_LABELS = {
    "raw": "Raw",
    "sigmoid": "Sigmoid",
    "isotonic": "Isotonic",
}


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

def load_inputs():
    fold_path = (
        CALIBRATION_DIR
        / "calibration_fold_metrics.csv"
    )

    summary_path = (
        CALIBRATION_DIR
        / "calibration_summary.csv"
    )

    pooled_path = (
        CALIBRATION_DIR
        / "calibration_pooled_metrics.csv"
    )

    reliability_path = (
        CALIBRATION_DIR
        / "calibration_reliability_bins.csv"
    )

    for path in [
        fold_path,
        summary_path,
        pooled_path,
        reliability_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing calibration output:\n"
                f"{path.resolve()}"
            )

    fold = pd.read_csv(fold_path)
    summary = pd.read_csv(summary_path)
    pooled = pd.read_csv(pooled_path)
    reliability = pd.read_csv(
        reliability_path
    )

    required_fold = {
        "fold",
        "method",
        "brier_score",
        "log_loss",
        "ece_10bin",
        "roc_auc",
        "pr_auc",
        "mean_predicted_probability",
        "observed_positive_fraction",
    }

    required_summary = {
        "method",
        "brier_score_mean",
        "brier_score_std",
        "log_loss_mean",
        "log_loss_std",
        "ece_10bin_mean",
        "ece_10bin_std",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
    }

    required_pooled = {
        "method",
        "mean_predicted_probability",
        "observed_positive_fraction",
    }

    required_reliability = {
        "method",
        "bin",
        "count",
        "mean_predicted_probability",
        "observed_positive_fraction",
    }

    for name, df, required in [
        (
            "fold metrics",
            fold,
            required_fold,
        ),
        (
            "summary",
            summary,
            required_summary,
        ),
        (
            "pooled metrics",
            pooled,
            required_pooled,
        ),
        (
            "reliability bins",
            reliability,
            required_reliability,
        ),
    ]:
        missing = sorted(
            required - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{name} missing columns: "
                f"{missing}"
            )

    for df in [
        fold,
        summary,
        pooled,
        reliability,
    ]:
        df["method"] = (
            df["method"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    expected = set(METHOD_ORDER)

    if set(summary["method"]) != expected:
        raise RuntimeError(
            "Calibration method mismatch in summary."
        )

    return (
        fold,
        summary,
        pooled,
        reliability,
    )


def order_methods(df):
    rank = {
        method: index
        for index, method in enumerate(
            METHOD_ORDER
        )
    }

    output = df.copy()

    output["_rank"] = (
        output["method"]
        .map(rank)
    )

    output = (
        output[
            output["_rank"].notna()
        ]
        .sort_values("_rank")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )

    return output


# =============================================================================
# GENERIC METRIC BAR
# =============================================================================

def plot_metric_bar(
    summary,
    mean_col,
    std_col,
    ylabel,
    title,
):
    data = order_methods(
        summary
    )

    x = np.arange(
        len(data)
    )

    means = data[
        mean_col
    ].to_numpy(dtype=float)

    stds = data[
        std_col
    ].to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(9.5, 6.8)
    )

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=7,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            METHOD_LABELS[method]
            for method in data["method"]
        ]
    )

    ax.set_xlabel(
        "Calibration method"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        axis="y",
        alpha=0.25,
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
# RELIABILITY DIAGRAM
# =============================================================================

def plot_reliability(reliability):
    title = (
        "Probability Calibration Reliability Diagram"
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 8.0)
    )

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=2.0,
        label="Perfect calibration",
    )

    for method in METHOD_ORDER:
        data = reliability[
            (
                reliability["method"]
                == method
            )
            & (
                reliability["count"] > 0
            )
        ].copy()

        data = data[
            np.isfinite(
                data[
                    "mean_predicted_probability"
                ]
            )
            & np.isfinite(
                data[
                    "observed_positive_fraction"
                ]
            )
        ].sort_values(
            "mean_predicted_probability"
        )

        ax.plot(
            data[
                "mean_predicted_probability"
            ],
            data[
                "observed_positive_fraction"
            ],
            marker="o",
            markersize=7,
            linewidth=2.2,
            label=METHOD_LABELS[method],
        )

    ax.set_xlabel(
        "Mean predicted probability"
    )

    ax.set_ylabel(
        "Observed positive fraction"
    )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
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
# ZOOMED RELIABILITY
# =============================================================================

def plot_reliability_zoomed(
    reliability,
):
    title = (
        "Probability Calibration Reliability Diagram Zoomed"
    )

    zoom_max = 0.40

    fig, ax = plt.subplots(
        figsize=(8.5, 8.0)
    )

    ax.plot(
        [0.0, zoom_max],
        [0.0, zoom_max],
        linestyle="--",
        linewidth=2.0,
        label="Perfect calibration",
    )

    for method in METHOD_ORDER:
        data = reliability[
            (
                reliability["method"]
                == method
            )
            & (
                reliability["count"] > 0
            )
        ].copy()

        data = data[
            np.isfinite(
                data[
                    "mean_predicted_probability"
                ]
            )
            & np.isfinite(
                data[
                    "observed_positive_fraction"
                ]
            )
            & (
                data[
                    "mean_predicted_probability"
                ]
                <= zoom_max
            )
            & (
                data[
                    "observed_positive_fraction"
                ]
                <= zoom_max
            )
        ].sort_values(
            "mean_predicted_probability"
        )

        ax.plot(
            data[
                "mean_predicted_probability"
            ],
            data[
                "observed_positive_fraction"
            ],
            marker="o",
            markersize=7,
            linewidth=2.2,
            label=METHOD_LABELS[method],
        )

    ax.set_xlabel(
        "Mean predicted probability"
    )

    ax.set_ylabel(
        "Observed positive fraction"
    )

    ax.set_xlim(
        0.0,
        zoom_max,
    )

    ax.set_ylim(
        0.0,
        zoom_max,
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
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
# CALIBRATION-IN-THE-LARGE
# =============================================================================

def plot_mean_predicted_vs_observed(
    pooled,
):
    title = (
        "Mean Predicted Probability versus Observed Positive Fraction"
    )

    data = order_methods(
        pooled
    )

    x = np.arange(
        len(data)
    )

    predicted = data[
        "mean_predicted_probability"
    ].to_numpy(dtype=float)

    observed = float(
        data[
            "observed_positive_fraction"
        ].mean()
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 6.8)
    )

    ax.bar(
        x,
        predicted,
        label="Mean predicted probability",
    )

    ax.axhline(
        observed,
        linestyle="--",
        linewidth=2.0,
        label=(
            "Observed positive fraction "
            f"({observed:.4f})"
        ),
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            METHOD_LABELS[method]
            for method in data["method"]
        ]
    )

    ax.set_xlabel(
        "Calibration method"
    )

    ax.set_ylabel(
        "Probability / observed fraction"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
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

    (
        fold,
        summary,
        pooled,
        reliability,
    ) = load_inputs()

    print(
        "FINAL V5 PROBABILITY CALIBRATION"
    )
    print("=" * 110)

    print(
        order_methods(
            summary
        )[
            [
                "method",
                "brier_score_mean",
                "log_loss_mean",
                "ece_10bin_mean",
                "roc_auc_mean",
                "pr_auc_mean",
            ]
        ].to_string(
            index=False
        )
    )

    # Main calibration figures.
    plot_reliability(
        reliability
    )

    plot_metric_bar(
        summary,
        "brier_score_mean",
        "brier_score_std",
        "Brier score",
        "Probability Calibration Brier Score",
    )

    plot_metric_bar(
        summary,
        "log_loss_mean",
        "log_loss_std",
        "Log loss",
        "Probability Calibration Log Loss",
    )

    plot_metric_bar(
        summary,
        "ece_10bin_mean",
        "ece_10bin_std",
        "Expected calibration error (ECE)",
        "Probability Calibration Expected Calibration Error",
    )

    # Supplementary / diagnostic figures.
    plot_reliability_zoomed(
        reliability
    )

    plot_metric_bar(
        summary,
        "roc_auc_mean",
        "roc_auc_std",
        "ROC-AUC",
        "Probability Calibration ROC-AUC",
    )

    plot_metric_bar(
        summary,
        "pr_auc_mean",
        "pr_auc_std",
        "PR-AUC",
        "Probability Calibration PR-AUC",
    )

    plot_mean_predicted_vs_observed(
        pooled
    )

    print(
        "\nAll calibration figures saved under:"
    )

    print(
        PLOTS_DIR.resolve()
    )


if __name__ == "__main__":
    main()