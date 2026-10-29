# 28_plot_coordinate_augmentation_v5.py
#
# Final coordinate-augmentation figure for the canonical v5
# 30-km M>=4 Extra Trees experiment.
#
# Plot conventions:
#   - fresh v5 results only
#   - all visible text = 15 pt bold
#   - title intentionally commented out
#   - PNG filename exactly matches intended title
#   - existing/default matplotlib colour scheme retained
#   - 600 DPI

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    SCRIPT_DIR
    / "coordinate_augmentation_v5_30km_m4_18features"
    / "coordinate_ablation_summary.csv"
)

PLOTS_DIR = (
    SCRIPT_DIR
    / "plots_v5"
    / "coordinate_augmentation"
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FONT_SIZE = 15
DPI = 600

TITLE = "Effect of Explicit Geographic Coordinate Augmentation"


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


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing fresh v5 coordinate summary:\n"
            f"{INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "experiment",
        "roc_auc_mean",
        "pr_auc_mean",
        "f1_mean",
        "mcc_mean",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    expected = {
        "physical_only",
        "physical_plus_xy",
    }

    found = set(
        df["experiment"].astype(str)
    )

    if found != expected:
        raise RuntimeError(
            f"Expected experiments {sorted(expected)}, "
            f"found {sorted(found)}"
        )

    return df


def main():
    apply_style()

    df = load_data()

    base = df[
        df["experiment"] == "physical_only"
    ].iloc[0]

    coord = df[
        df["experiment"] == "physical_plus_xy"
    ].iloc[0]

    labels = [
        "ROC-AUC",
        "PR-AUC",
        "F1",
        "MCC",
    ]

    base_values = [
        base["roc_auc_mean"],
        base["pr_auc_mean"],
        base["f1_mean"],
        base["mcc_mean"],
    ]

    coord_values = [
        coord["roc_auc_mean"],
        coord["pr_auc_mean"],
        coord["f1_mean"],
        coord["mcc_mean"],
    ]

    print("FINAL V5 COORDINATE-AUGMENTATION RESULTS")
    print("=" * 90)

    for label, b, c in zip(
        labels,
        base_values,
        coord_values,
    ):
        print(
            f"{label:<10} "
            f"physical={b:.6f}  "
            f"physical+x/y={c:.6f}  "
            f"delta={c-b:+.6f}"
        )

    x = np.arange(
        len(labels)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.bar(
        x - width / 2,
        base_values,
        width,
        label="Physical only",
    )

    ax.bar(
        x + width / 2,
        coord_values,
        width,
        label="Physical + x/y",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Mean 5-fold spatial-CV score"
    )

    ax.set_ylim(
        0.0,
        1.0,
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
    #     TITLE,
    #     fontsize=FONT_SIZE,
    #     fontweight="bold",
    # )

    force_bold(ax)

    fig.tight_layout()

    output = (
        PLOTS_DIR
        / f"{TITLE}.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.10,
    )

    plt.close(fig)

    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()