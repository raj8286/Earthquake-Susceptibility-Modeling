# 20_plot_india_actual_vs_predicted_m4_v5.py
#
# India historical-positive vs predicted-positive comparison for the canonical
# v5 30 km M>=4 Extra Trees model using spatially OOF predictions.
#
# This is intentionally a minimal v5 update of the earlier v3 plotting script:
# same marker styling, colours, map extent, shapefile overlays, legend layout,
# and figure proportions; only the v5 input/output paths and current plotting
# conventions are updated.
#
# All visible text is 15 pt bold.
# Figure title is intentionally commented out.
# Output PNG filename is the same as the intended title.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.ticker import FixedLocator
import geopandas as gpd


# ============================================================
# PATHS / COLUMNS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "plots_v5"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT = (
    SCRIPT_DIR
    / "probability_calibration_v5_30km_m4_18features"
    / "calibration_oof_predictions.csv"
)

INDIA_BOUNDARY_SHP = SCRIPT_DIR / "IndiaBoundary.shp"
INDIA_STATES_SHP = SCRIPT_DIR / "India-States.shp"

TARGET = "target_m4_0plus"
PREDICTED = "predicted_m4_positive"

INDIA_EXTENT = [67.0, 98.5, 5.0, 38.5]
LONGITUDE_TICKS = [70, 80, 90]
LATITUDE_TICKS = [10, 20, 30]

TP_COLOR = "#762a83"
FN_COLOR = "#d73027"
FP_COLOR = "#2b83ba"

TITLE = "India Historical-Positive versus Predicted-Positive Cells"


# ============================================================
# STYLE
# ============================================================

FONT_SIZE = 15

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "font.weight": "bold",
        "axes.titlesize": FONT_SIZE,
        "axes.titleweight": "bold",
        "axes.labelsize": FONT_SIZE,
        "axes.labelweight": "bold",
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
    }
)


# ============================================================
# LOAD DATA
# ============================================================

def load_predictions():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT.resolve()}"
        )

    print(f"Loading OOF predictions:\n{INPUT.resolve()}")
    df = pd.read_csv(INPUT)

    required = [
        "cell_id",
        "lon",
        "lat",
        TARGET,
        PREDICTED,
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

    for column in [
        "lon",
        "lat",
        TARGET,
        PREDICTED,
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    if df["cell_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate cell_id values detected."
        )

    for column in [TARGET, PREDICTED]:
        values = set(
            df[column]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if not values.issubset({0, 1}):
            raise RuntimeError(
                f"{column} is not binary: {sorted(values)}"
            )

    return df


def load_india_boundaries():
    if not INDIA_BOUNDARY_SHP.exists():
        raise FileNotFoundError(
            f"Missing file:\n{INDIA_BOUNDARY_SHP.resolve()}"
        )

    if not INDIA_STATES_SHP.exists():
        raise FileNotFoundError(
            f"Missing file:\n{INDIA_STATES_SHP.resolve()}"
        )

    boundary = gpd.read_file(
        INDIA_BOUNDARY_SHP
    )
    states = gpd.read_file(
        INDIA_STATES_SHP
    )

    if boundary.crs is None or states.crs is None:
        raise RuntimeError(
            "India boundary/state shapefile is missing CRS metadata."
        )

    boundary = boundary.to_crs(
        epsg=4326
    )
    states = states.to_crs(
        epsg=4326
    )

    boundary = boundary[
        boundary.geometry.notna()
        & ~boundary.geometry.is_empty
    ].copy()

    states = states[
        states.geometry.notna()
        & ~states.geometry.is_empty
    ].copy()

    if boundary.empty:
        raise RuntimeError(
            "IndiaBoundary.shp contains no usable geometries."
        )

    if states.empty:
        raise RuntimeError(
            "India-States.shp contains no usable geometries."
        )

    return boundary, states


# ============================================================
# MAP
# ============================================================

def make_india_comparison(
    df,
    boundary,
    states,
):
    india = df[
        df["lon"].between(
            INDIA_EXTENT[0],
            INDIA_EXTENT[1],
        )
        & df["lat"].between(
            INDIA_EXTENT[2],
            INDIA_EXTENT[3],
        )
    ].copy()

    if india.empty:
        raise RuntimeError(
            "No cells found inside the India plotting extent."
        )

    actual = (
        india[TARGET].astype(int) == 1
    )
    predicted = (
        india[PREDICTED].astype(int) == 1
    )

    true_positive = actual & predicted
    false_negative = actual & ~predicted
    false_positive = ~actual & predicted
    true_negative = ~actual & ~predicted

    tp = int(true_positive.sum())
    fn = int(false_negative.sum())
    fp = int(false_positive.sum())
    tn = int(true_negative.sum())

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else np.nan
    )
    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else np.nan
    )
    f1 = (
        2 * precision * recall
        / (precision + recall)
        if np.isfinite(precision)
        and np.isfinite(recall)
        and (precision + recall) > 0
        else np.nan
    )

    print(
        "INDIA ACTUAL VS PREDICTED — V5 M>=4 OOF"
    )
    print("=" * 70)
    print(
        f"Cells in plotting extent: {len(india):,}"
    )
    print(
        f"Historical-positive cells: {int(actual.sum()):,}"
    )
    print(
        f"Predicted-positive cells:  {int(predicted.sum()):,}"
    )
    print(f"TP: {tp:,}")
    print(f"FN: {fn:,}")
    print(f"FP: {fp:,}")
    print(f"TN: {tn:,}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1:        {f1:.6f}")

    # --------------------------------------------------------
    # FIGURE / MAP
    # Retains the original figure proportions.
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(9, 11)
    )

    ax = fig.add_axes(
        [0.08, 0.24, 0.84, 0.70],
        projection=ccrs.PlateCarree(),
    )

    ax.set_extent(
        INDIA_EXTENT,
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="#eaf2f7",
        zorder=0,
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="#f4f4f1",
        zorder=0,
    )

    # --------------------------------------------------------
    # CELL-CENTRE MARKERS
    # These are 30-km model-cell centres, not earthquake points.
    # --------------------------------------------------------

    # Predicted positive only / false positives.
    ax.scatter(
        india.loc[
            false_positive,
            "lon",
        ],
        india.loc[
            false_positive,
            "lat",
        ],
        s=18,
        c=FP_COLOR,
        alpha=0.7,
        marker="o",
        edgecolors="none",
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    # Historical positive only / false negatives.
    ax.scatter(
        india.loc[
            false_negative,
            "lon",
        ],
        india.loc[
            false_negative,
            "lat",
        ],
        s=20,
        c=FN_COLOR,
        alpha=0.7,
        marker="o",
        edgecolors="none",
        transform=ccrs.PlateCarree(),
        zorder=11,
    )

    # True-positive overlap.
    ax.scatter(
        india.loc[
            true_positive,
            "lon",
        ],
        india.loc[
            true_positive,
            "lat",
        ],
        s=25,
        c=TP_COLOR,
        alpha=0.7,
        marker="o",
        edgecolors="white",
        linewidths=0.45,
        transform=ccrs.PlateCarree(),
        zorder=12,
    )

    # --------------------------------------------------------
    # INDIA STATE / OUTER BOUNDARIES
    # Retains original white-halo treatment.
    # --------------------------------------------------------

    state_geoms = list(
        states.boundary.geometry
    )
    outer_geoms = list(
        boundary.boundary.geometry
    )

    ax.add_geometries(
        state_geoms,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="white",
        linewidth=1.35,
        zorder=3,
    )

    ax.add_geometries(
        state_geoms,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="#444444",
        linewidth=0.60,
        zorder=4,
    )

    ax.add_geometries(
        outer_geoms,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="white",
        linewidth=2.5,
        zorder=5,
    )

    ax.add_geometries(
        outer_geoms,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="black",
        linewidth=1.25,
        zorder=6,
    )

    # --------------------------------------------------------
    # GRIDLINES
    # 10-degree coordinate intervals retained.
    # --------------------------------------------------------

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.65,
        alpha=0.40,
        linestyle="--",
        color="#555555",
        zorder=13,
    )

    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False

    gl.xlocator = FixedLocator(
        LONGITUDE_TICKS
    )
    gl.ylocator = FixedLocator(
        LATITUDE_TICKS
    )

    gl.xformatter = (
        LONGITUDE_FORMATTER
    )
    gl.yformatter = (
        LATITUDE_FORMATTER
    )

    gl.xlabel_style = {
        "size": FONT_SIZE,
        "weight": "bold",
    }
    gl.ylabel_style = {
        "size": FONT_SIZE,
        "weight": "bold",
    }

    gl.xpadding = 8
    gl.ypadding = 8

    # --------------------------------------------------------
    # TITLE
    # Intentionally omitted for the manuscript figure.
    # --------------------------------------------------------

    # ax.set_title(
    #     TITLE,
    #     fontsize=FONT_SIZE,
    #     fontweight="bold",
    # )

    # --------------------------------------------------------
    # LEGEND
    # Retains the original vertical, lower-centre legend.
    # --------------------------------------------------------

    legend_handles = [
        mpatches.Patch(
            facecolor=TP_COLOR,
            label=(
                f"True-positive overlap (TP), "
                f"n={tp:,}"
            ),
        ),
        mpatches.Patch(
            facecolor=FN_COLOR,
            label=(
                f"Historical positive only (FN), "
                f"n={fn:,}"
            ),
        ),
        mpatches.Patch(
            facecolor=FP_COLOR,
            label=(
                f"Predicted positive only (FP), "
                f"n={fp:,}"
            ),
        ),
    ]

    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=1,
        fontsize=FONT_SIZE,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#777777",
        borderpad=0.9,
        handlelength=1.8,
    )

    for text in legend.get_texts():
        text.set_fontsize(
            FONT_SIZE
        )
        text.set_fontweight(
            "bold"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output = (
        PLOTS_DIR
        / f"{TITLE}.png"
    )

    fig.savefig(
        output,
        dpi=600,
        facecolor="white",
        pad_inches=0.08,
    )

    plt.close(
        fig
    )

    print(
        f"\nSaved:\n"
        f"{output.resolve()}"
    )


def main():
    df = load_predictions()

    boundary, states = (
        load_india_boundaries()
    )

    make_india_comparison(
        df,
        boundary,
        states,
    )

    print(
        f"\nAll figures saved under:\n"
        f"{PLOTS_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
