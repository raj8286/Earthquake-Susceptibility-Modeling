# 23_plot_feature_ablation_v5.py
#
# Final feature-family ablation figures for the canonical v5
# 30-km M>=4 Extra Trees model.
#
# IMPORTANT:
#   - Canonical v5 model contains 18 physical predictors.
#   - Five feature families are ablated.
#   - EMAG2/magnetic is NOT part of this ablation.
#   - Magnetic incremental screening is a separate experiment.
#
# Plot conventions:
#   - all visible text = 15 pt bold
#   - figure titles intentionally commented out
#   - PNG filename exactly matches intended title
#   - default/existing matplotlib colour scheme retained
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

RESULTS_DIR = (
    SCRIPT_DIR
    / "extra_trees_ablation_v5_m4_18features_30km"
)

SUMMARY_FILE = RESULTS_DIR / "ablation_summary.csv"
FOLD_FILE = RESULTS_DIR / "ablation_fold_metrics.csv"

PLOTS_DIR = (
    SCRIPT_DIR
    / "plots_v5"
    / "feature_ablation"
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FONT_SIZE = 15
DPI = 600


# =============================================================================
# FINAL V5 FEATURE FAMILIES
# =============================================================================

EXPERIMENT_LABELS = {
    "full": "Full model",
    "no_surface_morphology": "Surface morphology",
    "no_gravity_geoid": "Gravity/geoid",
    "no_crust_sediment": "Crust/sediment",
    "no_lithosphere_density": "Lithosphere/density",
    "no_faults": "Active faults",
}

REMOVAL_ORDER = [
    "no_surface_morphology",
    "no_gravity_geoid",
    "no_crust_sediment",
    "no_lithosphere_density",
    "no_faults",
]

FULL_ORDER = [
    "full",
    *REMOVAL_ORDER,
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
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{SUMMARY_FILE.resolve()}"
        )

    if not FOLD_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{FOLD_FILE.resolve()}"
        )

    summary = pd.read_csv(
        SUMMARY_FILE
    )

    folds = pd.read_csv(
        FOLD_FILE
    )

    required_summary = {
        "experiment",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "f1_at_best_mean",
        "f1_at_best_std",
        "mcc_at_best_mean",
        "mcc_at_best_std",
    }

    required_folds = {
        "experiment",
        "fold",
        "roc_auc",
        "pr_auc",
        "f1_at_best",
        "mcc_at_best",
    }

    missing_summary = (
        required_summary
        - set(summary.columns)
    )

    missing_folds = (
        required_folds
        - set(folds.columns)
    )

    if missing_summary:
        raise ValueError(
            f"Summary missing columns: "
            f"{sorted(missing_summary)}"
        )

    if missing_folds:
        raise ValueError(
            f"Fold metrics missing columns: "
            f"{sorted(missing_folds)}"
        )

    expected = set(FULL_ORDER)

    found_summary = set(
        summary["experiment"]
        .astype(str)
    )

    found_folds = set(
        folds["experiment"]
        .astype(str)
    )

    if found_summary != expected:
        raise RuntimeError(
            "Unexpected ablation experiments in summary.\n"
            f"Expected: {sorted(expected)}\n"
            f"Found:    {sorted(found_summary)}"
        )

    if found_folds != expected:
        raise RuntimeError(
            "Unexpected ablation experiments in fold metrics.\n"
            f"Expected: {sorted(expected)}\n"
            f"Found:    {sorted(found_folds)}"
        )

    return summary, folds


def order_dataframe(df, order):
    rank = {
        name: index
        for index, name in enumerate(order)
    }

    result = df.copy()

    result["_order"] = (
        result["experiment"]
        .map(rank)
    )

    return (
        result
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


# =============================================================================
# 1. PR-AUC DECREASE
# =============================================================================

def plot_pr_auc_decrease(summary):
    data = order_dataframe(
        summary[
            summary["experiment"]
            .isin(REMOVAL_ORDER)
        ],
        REMOVAL_ORDER,
    ).copy()

    full = summary[
        summary["experiment"] == "full"
    ].iloc[0]

    data["decrease"] = (
        float(full["pr_auc_mean"])
        - data["pr_auc_mean"]
    )

    title = (
        "PR-AUC Decrease by Removed Feature Family"
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    y = np.arange(
        len(data)
    )

    ax.barh(
        y,
        data["decrease"],
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        [
            EXPERIMENT_LABELS[name]
            for name in data["experiment"]
        ]
    )

    ax.set_xlabel(
        "Decrease in PR-AUC relative to full model"
    )

    ax.set_ylabel(
        "Feature family removed"
    )

    ax.grid(
        axis="x",
        alpha=0.30,
    )

    ax.axvline(
        0.0,
        linewidth=1.5,
    )

    ax.invert_yaxis()

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
# 2. MULTI-METRIC PERFORMANCE DECREASE
# =============================================================================

def plot_multi_metric_decrease(summary):
    data = order_dataframe(
        summary[
            summary["experiment"]
            .isin(REMOVAL_ORDER)
        ],
        REMOVAL_ORDER,
    ).copy()

    full = summary[
        summary["experiment"] == "full"
    ].iloc[0]

    data["PR-AUC"] = (
        float(full["pr_auc_mean"])
        - data["pr_auc_mean"]
    )

    data["F1"] = (
        float(full["f1_at_best_mean"])
        - data["f1_at_best_mean"]
    )

    data["MCC"] = (
        float(full["mcc_at_best_mean"])
        - data["mcc_at_best_mean"]
    )

    labels = [
        EXPERIMENT_LABELS[name]
        for name in data["experiment"]
    ]

    x = np.arange(
        len(labels),
        dtype=float,
    ) * 1.30

    width = 0.24

    title = (
        "Performance Decrease by Removed Feature Family"
    )

    fig, ax = plt.subplots(
        figsize=(13, 8)
    )

    ax.bar(
        x - width,
        data["PR-AUC"],
        width,
        label="PR-AUC",
    )

    ax.bar(
        x,
        data["F1"],
        width,
        label="F1",
    )

    ax.bar(
        x + width,
        data["MCC"],
        width,
        label="MCC",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels
    )

    ax.set_xlabel(
        "Feature family removed"
    )

    ax.set_ylabel(
        "Performance decrease relative to full model"
    )

    ax.axhline(
        0.0,
        linewidth=1.5,
    )

    ax.grid(
        axis="y",
        alpha=0.30,
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
# 3. ABSOLUTE PR-AUC
# =============================================================================

def plot_absolute_pr_auc(summary):
    data = order_dataframe(
        summary,
        FULL_ORDER,
    )

    title = (
        "Feature Ablation PR-AUC"
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    y = np.arange(
        len(data)
    )

    ax.barh(
        y,
        data["pr_auc_mean"],
        xerr=data["pr_auc_std"],
        capsize=6,
    )

    labels = []

    for experiment in data["experiment"]:
        if experiment == "full":
            labels.append(
                "Full model"
            )
        else:
            labels.append(
                "Remove "
                + EXPERIMENT_LABELS[experiment]
            )

    ax.set_yticks(y)

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "PR-AUC"
    )

    ax.set_ylabel(
        "Feature configuration"
    )

    ax.grid(
        axis="x",
        alpha=0.30,
    )

    ax.invert_yaxis()

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
# 4. FOLD-WISE PR-AUC
# =============================================================================

def plot_pr_auc_by_fold(folds):
    title = (
        "Feature Ablation PR-AUC Across Spatial Cross-Validation Folds"
    )

    fig, ax = plt.subplots(
        figsize=(13, 8)
    )

    for experiment in FULL_ORDER:
        data = (
            folds[
                folds["experiment"]
                == experiment
            ]
            .sort_values("fold")
        )

        if experiment == "full":
            label = "Full model"
        else:
            label = (
                "Remove "
                + EXPERIMENT_LABELS[
                    experiment
                ]
            )

        ax.plot(
            data["fold"],
            data["pr_auc"],
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=label,
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
        alpha=0.30
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

    summary, folds = load_data()

    print(
        "FINAL V5 FEATURE-FAMILY ABLATION"
    )
    print("=" * 110)

    display = (
        summary[
            [
                "experiment",
                "pr_auc_mean",
                "f1_at_best_mean",
                "mcc_at_best_mean",
            ]
        ]
        .copy()
    )

    full = summary[
        summary["experiment"] == "full"
    ].iloc[0]

    display["pr_auc_decrease"] = (
        float(full["pr_auc_mean"])
        - display["pr_auc_mean"]
    )

    print(
        order_dataframe(
            display,
            FULL_ORDER,
        ).to_string(
            index=False
        )
    )

    plot_pr_auc_decrease(
        summary
    )

    plot_multi_metric_decrease(
        summary
    )

    plot_absolute_pr_auc(
        summary
    )

    plot_pr_auc_by_fold(
        folds
    )

    print(
        "\nAll feature-ablation figures "
        "saved under:"
    )

    print(
        PLOTS_DIR.resolve()
    )


if __name__ == "__main__":
    main()




# # 23_plot_feature_ablation_v5.py
# #
# # Final feature-family ablation figures for the canonical v5
# # 30-km M>=4 Extra Trees model.
# #
# # IMPORTANT:
# #   - Canonical v5 model contains 18 physical predictors.
# #   - Five feature families are ablated.
# #   - EMAG2/magnetic is NOT part of this ablation.
# #   - Magnetic incremental screening is a separate experiment.
# #
# # Plot conventions:
# #   - all visible text = 15 pt bold
# #   - figure titles intentionally commented out
# #   - PNG filename exactly matches intended title
# #   - default/existing matplotlib colour scheme retained
# #   - 600 DPI

# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt


# # =============================================================================
# # PATHS / STYLE
# # =============================================================================

# SCRIPT_DIR = Path(__file__).resolve().parent

# RESULTS_DIR = (
#     SCRIPT_DIR
#     / "extra_trees_ablation_v5_m4_18features_30km"
# )

# SUMMARY_FILE = RESULTS_DIR / "ablation_summary.csv"
# FOLD_FILE = RESULTS_DIR / "ablation_fold_metrics.csv"

# PLOTS_DIR = (
#     SCRIPT_DIR
#     / "plots_v5"
#     / "feature_ablation"
# )

# PLOTS_DIR.mkdir(
#     parents=True,
#     exist_ok=True,
# )

# FONT_SIZE = 15
# DPI = 600


# # =============================================================================
# # FINAL V5 FEATURE FAMILIES
# # =============================================================================

# EXPERIMENT_LABELS = {
#     "full": "Full model",
#     "no_surface_morphology": "Surface morphology",
#     "no_gravity_geoid": "Gravity / geoid",
#     "no_crust_sediment": "Crust / sediment",
#     "no_lithosphere_density": "Lithosphere / density",
#     "no_faults": "Active faults",
# }

# REMOVAL_ORDER = [
#     "no_surface_morphology",
#     "no_gravity_geoid",
#     "no_crust_sediment",
#     "no_lithosphere_density",
#     "no_faults",
# ]

# FULL_ORDER = [
#     "full",
#     *REMOVAL_ORDER,
# ]


# # =============================================================================
# # STYLE
# # =============================================================================

# def apply_style():
#     plt.rcParams.update(
#         {
#             "font.size": FONT_SIZE,
#             "font.weight": "bold",
#             "axes.labelsize": FONT_SIZE,
#             "axes.labelweight": "bold",
#             "axes.titlesize": FONT_SIZE,
#             "axes.titleweight": "bold",
#             "xtick.labelsize": FONT_SIZE,
#             "ytick.labelsize": FONT_SIZE,
#             "legend.fontsize": FONT_SIZE,
#         }
#     )


# def force_bold(ax):
#     for label in (
#         ax.get_xticklabels()
#         + ax.get_yticklabels()
#     ):
#         label.set_fontsize(FONT_SIZE)
#         label.set_fontweight("bold")

#     ax.xaxis.label.set_fontsize(FONT_SIZE)
#     ax.xaxis.label.set_fontweight("bold")

#     ax.yaxis.label.set_fontsize(FONT_SIZE)
#     ax.yaxis.label.set_fontweight("bold")

#     legend = ax.get_legend()

#     if legend is not None:
#         for text in legend.get_texts():
#             text.set_fontsize(FONT_SIZE)
#             text.set_fontweight("bold")


# def save_figure(fig, title):
#     output = PLOTS_DIR / f"{title}.png"

#     fig.savefig(
#         output,
#         dpi=DPI,
#         bbox_inches="tight",
#         facecolor="white",
#         pad_inches=0.10,
#     )

#     plt.close(fig)

#     print(f"Saved: {output.resolve()}")


# # =============================================================================
# # LOAD / VALIDATE
# # =============================================================================

# def load_data():
#     if not SUMMARY_FILE.exists():
#         raise FileNotFoundError(
#             f"Missing:\n{SUMMARY_FILE.resolve()}"
#         )

#     if not FOLD_FILE.exists():
#         raise FileNotFoundError(
#             f"Missing:\n{FOLD_FILE.resolve()}"
#         )

#     summary = pd.read_csv(
#         SUMMARY_FILE
#     )

#     folds = pd.read_csv(
#         FOLD_FILE
#     )

#     required_summary = {
#         "experiment",
#         "roc_auc_mean",
#         "roc_auc_std",
#         "pr_auc_mean",
#         "pr_auc_std",
#         "f1_at_best_mean",
#         "f1_at_best_std",
#         "mcc_at_best_mean",
#         "mcc_at_best_std",
#     }

#     required_folds = {
#         "experiment",
#         "fold",
#         "roc_auc",
#         "pr_auc",
#         "f1_at_best",
#         "mcc_at_best",
#     }

#     missing_summary = (
#         required_summary
#         - set(summary.columns)
#     )

#     missing_folds = (
#         required_folds
#         - set(folds.columns)
#     )

#     if missing_summary:
#         raise ValueError(
#             f"Summary missing columns: "
#             f"{sorted(missing_summary)}"
#         )

#     if missing_folds:
#         raise ValueError(
#             f"Fold metrics missing columns: "
#             f"{sorted(missing_folds)}"
#         )

#     expected = set(FULL_ORDER)

#     found_summary = set(
#         summary["experiment"]
#         .astype(str)
#     )

#     found_folds = set(
#         folds["experiment"]
#         .astype(str)
#     )

#     if found_summary != expected:
#         raise RuntimeError(
#             "Unexpected ablation experiments in summary.\n"
#             f"Expected: {sorted(expected)}\n"
#             f"Found:    {sorted(found_summary)}"
#         )

#     if found_folds != expected:
#         raise RuntimeError(
#             "Unexpected ablation experiments in fold metrics.\n"
#             f"Expected: {sorted(expected)}\n"
#             f"Found:    {sorted(found_folds)}"
#         )

#     return summary, folds


# def order_dataframe(df, order):
#     rank = {
#         name: index
#         for index, name in enumerate(order)
#     }

#     result = df.copy()

#     result["_order"] = (
#         result["experiment"]
#         .map(rank)
#     )

#     return (
#         result
#         .sort_values("_order")
#         .drop(columns="_order")
#         .reset_index(drop=True)
#     )


# # =============================================================================
# # 1. PR-AUC DECREASE
# # =============================================================================

# def plot_pr_auc_decrease(summary):
#     data = order_dataframe(
#         summary[
#             summary["experiment"]
#             .isin(REMOVAL_ORDER)
#         ],
#         REMOVAL_ORDER,
#     ).copy()

#     full = summary[
#         summary["experiment"] == "full"
#     ].iloc[0]

#     data["decrease"] = (
#         float(full["pr_auc_mean"])
#         - data["pr_auc_mean"]
#     )

#     title = (
#         "PR-AUC Decrease by Removed Feature Family"
#     )

#     fig, ax = plt.subplots(
#         figsize=(11, 7)
#     )

#     y = np.arange(
#         len(data)
#     )

#     ax.barh(
#         y,
#         data["decrease"],
#     )

#     ax.set_yticks(y)

#     ax.set_yticklabels(
#         [
#             EXPERIMENT_LABELS[name]
#             for name in data["experiment"]
#         ]
#     )

#     ax.set_xlabel(
#         "Decrease in PR-AUC relative to full model"
#     )

#     ax.set_ylabel(
#         "Feature family removed"
#     )

#     ax.grid(
#         axis="x",
#         alpha=0.30,
#     )

#     ax.axvline(
#         0.0,
#         linewidth=1.5,
#     )

#     ax.invert_yaxis()

#     # Title intentionally omitted.
#     # ax.set_title(
#     #     title,
#     #     fontsize=FONT_SIZE,
#     #     fontweight="bold",
#     # )

#     force_bold(ax)

#     fig.tight_layout()

#     save_figure(
#         fig,
#         title,
#     )


# # =============================================================================
# # 2. MULTI-METRIC PERFORMANCE DECREASE
# # =============================================================================

# def plot_multi_metric_decrease(summary):
#     data = order_dataframe(
#         summary[
#             summary["experiment"]
#             .isin(REMOVAL_ORDER)
#         ],
#         REMOVAL_ORDER,
#     ).copy()

#     full = summary[
#         summary["experiment"] == "full"
#     ].iloc[0]

#     data["PR-AUC"] = (
#         float(full["pr_auc_mean"])
#         - data["pr_auc_mean"]
#     )

#     data["F1"] = (
#         float(full["f1_at_best_mean"])
#         - data["f1_at_best_mean"]
#     )

#     data["MCC"] = (
#         float(full["mcc_at_best_mean"])
#         - data["mcc_at_best_mean"]
#     )

#     labels = [
#         EXPERIMENT_LABELS[name]
#         for name in data["experiment"]
#     ]

#     x = np.arange(
#         len(labels)
#     )

#     width = 0.24

#     title = (
#         "Performance Decrease by Removed Feature Family"
#     )

#     fig, ax = plt.subplots(
#         figsize=(13, 8)
#     )

#     ax.bar(
#         x - width,
#         data["PR-AUC"],
#         width,
#         label="PR-AUC",
#     )

#     ax.bar(
#         x,
#         data["F1"],
#         width,
#         label="F1",
#     )

#     ax.bar(
#         x + width,
#         data["MCC"],
#         width,
#         label="MCC",
#     )

#     ax.set_xticks(x)

#     ax.set_xticklabels(
#         labels
#     )

#     ax.set_xlabel(
#         "Feature family removed"
#     )

#     ax.set_ylabel(
#         "Performance decrease relative to full model"
#     )

#     ax.axhline(
#         0.0,
#         linewidth=1.5,
#     )

#     ax.grid(
#         axis="y",
#         alpha=0.30,
#     )

#     ax.legend(
#         frameon=True,
#         prop={
#             "size": FONT_SIZE,
#             "weight": "bold",
#         },
#     )

#     # Title intentionally omitted.
#     # ax.set_title(
#     #     title,
#     #     fontsize=FONT_SIZE,
#     #     fontweight="bold",
#     # )

#     force_bold(ax)

#     fig.tight_layout()

#     save_figure(
#         fig,
#         title,
#     )


# # =============================================================================
# # 3. ABSOLUTE PR-AUC
# # =============================================================================

# def plot_absolute_pr_auc(summary):
#     data = order_dataframe(
#         summary,
#         FULL_ORDER,
#     )

#     title = (
#         "Feature Ablation PR-AUC"
#     )

#     fig, ax = plt.subplots(
#         figsize=(11, 7)
#     )

#     y = np.arange(
#         len(data)
#     )

#     ax.barh(
#         y,
#         data["pr_auc_mean"],
#         xerr=data["pr_auc_std"],
#         capsize=6,
#     )

#     labels = []

#     for experiment in data["experiment"]:
#         if experiment == "full":
#             labels.append(
#                 "Full model"
#             )
#         else:
#             labels.append(
#                 "Remove "
#                 + EXPERIMENT_LABELS[experiment]
#             )

#     ax.set_yticks(y)

#     ax.set_yticklabels(
#         labels
#     )

#     ax.set_xlabel(
#         "PR-AUC"
#     )

#     ax.set_ylabel(
#         "Feature configuration"
#     )

#     ax.grid(
#         axis="x",
#         alpha=0.30,
#     )

#     ax.invert_yaxis()

#     # Title intentionally omitted.
#     # ax.set_title(
#     #     title,
#     #     fontsize=FONT_SIZE,
#     #     fontweight="bold",
#     # )

#     force_bold(ax)

#     fig.tight_layout()

#     save_figure(
#         fig,
#         title,
#     )


# # =============================================================================
# # 4. FOLD-WISE PR-AUC
# # =============================================================================

# def plot_pr_auc_by_fold(folds):
#     title = (
#         "Feature Ablation PR-AUC Across Spatial Cross-Validation Folds"
#     )

#     fig, ax = plt.subplots(
#         figsize=(13, 8)
#     )

#     for experiment in FULL_ORDER:
#         data = (
#             folds[
#                 folds["experiment"]
#                 == experiment
#             ]
#             .sort_values("fold")
#         )

#         if experiment == "full":
#             label = "Full model"
#         else:
#             label = (
#                 "Remove "
#                 + EXPERIMENT_LABELS[
#                     experiment
#                 ]
#             )

#         ax.plot(
#             data["fold"],
#             data["pr_auc"],
#             marker="o",
#             linewidth=2.2,
#             markersize=7,
#             label=label,
#         )

#     ax.set_xlabel(
#         "Spatial CV fold"
#     )

#     ax.set_ylabel(
#         "PR-AUC"
#     )

#     ax.set_xticks(
#         [1, 2, 3, 4, 5]
#     )

#     ax.grid(
#         alpha=0.30
#     )

#     ax.legend(
#         loc="best",
#         frameon=True,
#         prop={
#             "size": FONT_SIZE,
#             "weight": "bold",
#         },
#     )

#     # Title intentionally omitted.
#     # ax.set_title(
#     #     title,
#     #     fontsize=FONT_SIZE,
#     #     fontweight="bold",
#     # )

#     force_bold(ax)

#     fig.tight_layout()

#     save_figure(
#         fig,
#         title,
#     )


# # =============================================================================
# # MAIN
# # =============================================================================

# def main():
#     apply_style()

#     summary, folds = load_data()

#     print(
#         "FINAL V5 FEATURE-FAMILY ABLATION"
#     )
#     print("=" * 110)

#     display = (
#         summary[
#             [
#                 "experiment",
#                 "pr_auc_mean",
#                 "f1_at_best_mean",
#                 "mcc_at_best_mean",
#             ]
#         ]
#         .copy()
#     )

#     full = summary[
#         summary["experiment"] == "full"
#     ].iloc[0]

#     display["pr_auc_decrease"] = (
#         float(full["pr_auc_mean"])
#         - display["pr_auc_mean"]
#     )

#     print(
#         order_dataframe(
#             display,
#             FULL_ORDER,
#         ).to_string(
#             index=False
#         )
#     )

#     plot_pr_auc_decrease(
#         summary
#     )

#     plot_multi_metric_decrease(
#         summary
#     )

#     plot_absolute_pr_auc(
#         summary
#     )

#     plot_pr_auc_by_fold(
#         folds
#     )

#     print(
#         "\nAll feature-ablation figures "
#         "saved under:"
#     )

#     print(
#         PLOTS_DIR.resolve()
#     )


# if __name__ == "__main__":
#     main()