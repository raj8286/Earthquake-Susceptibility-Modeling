# 19_plot_calibrated_probability_maps_v5.py
#
# Final spatially out-of-fold isotonic-calibrated class-probability maps for
# the canonical v5 30 km M>=4 model.
#
# IMPORTANT INTERPRETATION:
#   This is the calibrated probability that a 30 km cell belongs to the
#   historically M>=4-positive class under the 2000-2025 catalogue definition.
#   It is NOT the probability of a future earthquake within a time period.
#
# IMPORTANT:
#   This intentionally reads the v5 calibration OOF prediction table. Each
#   cell's calibrated probability was generated for its held-out outer fold,
#   using a calibration transform fitted on that fold's spatial validation data.
#
# India outer/state boundaries are drawn ONLY from:
#   IndiaBoundary.shp
#   India-States.shp
#
# Plot conventions:
#   - all visible text is 15 pt bold
#   - figure title line is intentionally commented out
#   - output PNG filename exactly matches the intended title
#   - world gridlines are linewidth 0.65, with 180W..180E and 60S..60N labels
#   - colorbar keeps the original compact rectangular style, without arrows

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FixedLocator, FormatStrFormatter

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas as gpd


# ============================================================
# PATHS
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

VALUE_COL = "isotonic_calibrated_probability"

NX = 1158
NY = 488

INDIA_EXTENT = [67.0, 98.5, 5.0, 38.5]
INDIA_X_TICKS = [70, 80, 90]
INDIA_Y_TICKS = [10, 20, 30]

GLOBAL_TITLE = (
    "Global Spatially Out-of-Fold Isotonic-Calibrated "
    "Earthquake Susceptibility Probability"
)

INDIA_TITLE = (
    "India Spatially Out-of-Fold Isotonic-Calibrated "
    "Earthquake Susceptibility Probability"
)


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


def force_gridliner_style(gl):
    gl.xlabel_style = {
        "size": FONT_SIZE,
        "weight": "bold",
    }

    gl.ylabel_style = {
        "size": FONT_SIZE,
        "weight": "bold",
    }


def style_colorbar(cbar):
    cbar.ax.tick_params(
        labelsize=FONT_SIZE
    )

    for label in (
        cbar.ax.get_xticklabels()
        + cbar.ax.get_yticklabels()
    ):
        label.set_fontsize(
            FONT_SIZE
        )
        label.set_fontweight(
            "bold"
        )

    cbar.ax.xaxis.label.set_fontsize(
        FONT_SIZE
    )

    cbar.ax.xaxis.label.set_fontweight(
        "bold"
    )


# ============================================================
# LOAD / VALIDATE
# ============================================================

def load_predictions():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT.resolve()}"
        )

    print(
        f"Loading OOF predictions:\n"
        f"{INPUT.resolve()}"
    )

    df = pd.read_csv(
        INPUT
    )

    required = [
        "cell_id",
        "ix",
        "iy",
        "lon",
        "lat",
        VALUE_COL,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{missing}"
        )

    for column in [
        "ix",
        "iy",
        "lon",
        "lat",
        VALUE_COL,
    ]:
        df[column] = (
            pd.to_numeric(
                df[column],
                errors="raise",
            )
        )

    if df[
        "cell_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate cell_id values detected."
        )

    if len(df) != NX * NY:
        raise RuntimeError(
            f"Expected "
            f"{NX * NY:,} "
            f"30 km cells, found "
            f"{len(df):,}."
        )

    if (
        (df["ix"] < 0).any()
        or (df["ix"] >= NX).any()
        or (df["iy"] < 0).any()
        or (df["iy"] >= NY).any()
    ):
        raise RuntimeError(
            "Some ix/iy values lie outside the finalized 30 km grid."
        )

    probabilities = (
        df[VALUE_COL]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if probabilities.empty:
        raise RuntimeError(
            "No finite isotonic-calibrated probabilities found."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise RuntimeError(
            "Calibrated probabilities fall outside [0, 1]."
        )

    print(
        "\nISOTONIC-CALIBRATED OOF CLASS-PROBABILITY SUMMARY"
    )

    print("=" * 70)

    print(
        f"Rows:   "
        f"{len(df):,}"
    )

    print(
        f"Min:    "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Mean:   "
        f"{probabilities.mean():.6f}"
    )

    print(
        f"Median: "
        f"{probabilities.median():.6f}"
    )

    print(
        f"P90:    "
        f"{probabilities.quantile(0.90):.6f}"
    )

    print(
        f"P95:    "
        f"{probabilities.quantile(0.95):.6f}"
    )

    print(
        f"P99:    "
        f"{probabilities.quantile(0.99):.6f}"
    )

    print(
        f"Max:    "
        f"{probabilities.max():.6f}"
    )

    return df


# ============================================================
# RECONSTRUCT STORED LON/LAT CELL-CENTRE GRID
# ============================================================

def build_lonlat_grids(df):
    value_grid = np.full(
        (NY, NX),
        np.nan,
        dtype=np.float64,
    )

    lon_grid = np.full(
        (NY, NX),
        np.nan,
        dtype=np.float64,
    )

    lat_grid = np.full(
        (NY, NX),
        np.nan,
        dtype=np.float64,
    )

    ix = df["ix"].to_numpy(
        dtype=np.int64
    )

    iy = df["iy"].to_numpy(
        dtype=np.int64
    )

    value_grid[iy, ix] = (
        df[VALUE_COL].to_numpy(
            dtype=np.float64
        )
    )

    lon_grid[iy, ix] = (
        df["lon"].to_numpy(
            dtype=np.float64
        )
    )

    lat_grid[iy, ix] = (
        df["lat"].to_numpy(
            dtype=np.float64
        )
    )

    if (
        np.isfinite(
            value_grid
        ).sum()
        != len(df)
    ):
        raise RuntimeError(
            "Reconstructed probability grid is incomplete."
        )

    if (
        np.isfinite(
            lon_grid
        ).sum()
        != len(df)
        or np.isfinite(
            lat_grid
        ).sum()
        != len(df)
    ):
        raise RuntimeError(
            "Reconstructed lon/lat grid is incomplete."
        )

    print(
        "\nSTRUCTURED GRID"
    )

    print("=" * 70)

    print(
        f"Grid shape:   "
        f"{value_grid.shape}"
    )

    print(
        f"Finite cells: "
        f"{np.isfinite(value_grid).sum():,}"
    )

    return (
        lon_grid,
        lat_grid,
        value_grid,
    )


# ============================================================
# PROBABILITY DISPLAY STYLE
# ============================================================

def build_probability_style():
    norm = Normalize(
        vmin=0.0,
        vmax=1.0,
        clip=True,
    )

    anchors = [
        0.00,
        0.01,
        0.03,
        0.06,
        0.10,
        0.20,
        0.35,
        0.50,
        0.70,
        1.00,
    ]

    colors = [
        "#eef7fb",
        "#d8eef5",
        "#aadbdc",
        "#73c7b4",
        "#52ae73",
        "#9dce4b",
        "#e5da42",
        "#f6ac3e",
        "#ed663d",
        "#b7252d",
    ]

    cmap = (
        LinearSegmentedColormap.from_list(
            "isotonic_m4_probability_oof_v5",
            list(
                zip(
                    anchors,
                    colors,
                )
            ),
            N=256,
        )
    )

    ticks = np.arange(
        0.0,
        1.01,
        0.1,
    )

    return cmap, norm, ticks


# ============================================================
# INDIA BOUNDARIES
# ============================================================

def load_india_boundaries():
    if not INDIA_BOUNDARY_SHP.exists():
        raise FileNotFoundError(
            "Missing India boundary shapefile:\n"
            f"{INDIA_BOUNDARY_SHP.resolve()}"
        )

    if not INDIA_STATES_SHP.exists():
        raise FileNotFoundError(
            "Missing India states shapefile:\n"
            f"{INDIA_STATES_SHP.resolve()}"
        )

    print(
        "\nLOADING PROVIDED INDIA SHAPEFILES"
    )

    print("=" * 70)

    print(
        f"Outer boundary: "
        f"{INDIA_BOUNDARY_SHP.resolve()}"
    )

    print(
        f"States/UTs:     "
        f"{INDIA_STATES_SHP.resolve()}"
    )

    boundary = (
        gpd.read_file(
            INDIA_BOUNDARY_SHP
        )
    )

    states = (
        gpd.read_file(
            INDIA_STATES_SHP
        )
    )

    if boundary.crs is None:
        raise RuntimeError(
            "IndiaBoundary.shp has no CRS metadata."
        )

    if states.crs is None:
        raise RuntimeError(
            "India-States.shp has no CRS metadata."
        )

    boundary = (
        boundary.to_crs(
            epsg=4326
        )
    )

    states = (
        states.to_crs(
            epsg=4326
        )
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

    print(
        f"Outer-boundary features: "
        f"{len(boundary):,}"
    )

    print(
        f"State/UT features:       "
        f"{len(states):,}"
    )

    return boundary, states


def add_india_boundaries(
    ax,
    boundary,
    states,
):
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
        linewidth=1.10,
        alpha=1.0,
        zorder=6,
    )

    ax.add_geometries(
        state_geoms,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="#444444",
        linewidth=0.50,
        alpha=1.0,
        zorder=7,
    )

    ax.add_geometries(
        outer_geoms,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="white",
        linewidth=2.20,
        alpha=1.0,
        zorder=8,
    )

    ax.add_geometries(
        outer_geoms,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="black",
        linewidth=1.10,
        alpha=1.0,
        zorder=9,
    )


# ============================================================
# COLORBAR
# ============================================================

def add_colorbar(
    fig,
    cax,
    mesh,
    ticks,
):
    cbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
        ticks=ticks,
    )

    cbar.ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    cbar.set_label(
        "Isotonic-calibrated M≥4 historical-positive class probability",
        fontsize=FONT_SIZE,
        fontweight="bold",
        labelpad=8,
    )

    style_colorbar(
        cbar
    )


# ============================================================
# GLOBAL MAP
# ============================================================

def make_global_map(
    lon_grid,
    lat_grid,
    value_grid,
    cmap,
    norm,
    ticks,
):
    print(
        "\nCreating GLOBAL calibrated OOF probability map..."
    )

    fig = plt.figure(
        figsize=(16, 10)
    )

    ax = fig.add_axes(
        [0.035, 0.15, 0.93, 0.80],
        projection=ccrs.Robinson(central_longitude=0),
    )

    cax = fig.add_axes([0.24, 0.065, 0.52, 0.035])

    ax.set_global()

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="#dbeeff",
        zorder=0,
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="#f9f9f9",
        zorder=0,
    )

    mesh = ax.pcolormesh(
        lon_grid,
        lat_grid,
        np.ma.masked_invalid(
            value_grid
        ),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
        zorder=1,
    )

    ax.coastlines(
        resolution="110m",
        linewidth=1.0,
        color="white",
        zorder=4,
    )

    ax.coastlines(
        resolution="110m",
        linewidth=0.45,
        color="#2b2b2b",
        zorder=5,
    )

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.28,
        edgecolor="#444444",
        alpha=0.70,
        zorder=5,
    )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.65,
        alpha=0.30,
        linestyle="--",
        color="#444444",
        zorder=6,
    )

    gl.top_labels = True
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = True

    gl.xlocator = FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = FixedLocator(np.arange(-60, 61, 30))

    gl.xformatter = (
        LONGITUDE_FORMATTER
    )

    gl.yformatter = (
        LATITUDE_FORMATTER
    )

    force_gridliner_style(
        gl
    )

    # ax.set_title(
    #     GLOBAL_TITLE,
    #     fontsize=FONT_SIZE,
    #     fontweight="bold",
    # )

    add_colorbar(
        fig,
        cax,
        mesh,
        ticks,
    )

    output = PLOTS_DIR / f"{GLOBAL_TITLE}.png"

    fig.savefig(
        output,
        dpi=600,
        facecolor="white",
        pad_inches=0.05,
    )

    plt.close(
        fig
    )

    print(
        f"Saved: "
        f"{output.resolve()}"
    )


# ============================================================
# INDIA MAP
# ============================================================

def get_india_subgrid(
    df,
    lon_grid,
    lat_grid,
    value_grid,
):
    mask = (
        df["lon"].between(
            INDIA_EXTENT[0],
            INDIA_EXTENT[1],
        )
        & df["lat"].between(
            INDIA_EXTENT[2],
            INDIA_EXTENT[3],
        )
    )

    india_extent_cells = (
        df.loc[mask]
    )

    if india_extent_cells.empty:
        raise RuntimeError(
            "No model cells found inside the India plotting extent."
        )

    ix_min = max(
        0,
        int(
            india_extent_cells[
                "ix"
            ].min()
        )
        - 1,
    )

    ix_max = min(
        NX - 1,
        int(
            india_extent_cells[
                "ix"
            ].max()
        )
        + 1,
    )

    iy_min = max(
        0,
        int(
            india_extent_cells[
                "iy"
            ].min()
        )
        - 1,
    )

    iy_max = min(
        NY - 1,
        int(
            india_extent_cells[
                "iy"
            ].max()
        )
        + 1,
    )

    lon_sub = lon_grid[
        iy_min : iy_max + 1,
        ix_min : ix_max + 1,
    ]

    lat_sub = lat_grid[
        iy_min : iy_max + 1,
        ix_min : ix_max + 1,
    ]

    value_sub = value_grid[
        iy_min : iy_max + 1,
        ix_min : ix_max + 1,
    ]

    print(
        "\nINDIA SUBGRID"
    )

    print(
        "=" * 70
    )

    print(
        f"Cells with centres in plotting extent: "
        f"{len(india_extent_cells):,}"
    )

    print(
        f"Subgrid shape: "
        f"{value_sub.shape}"
    )

    print(
        f"Finite subgrid cells: "
        f"{np.isfinite(value_sub).sum():,}"
    )

    print(
        f"OOF probability range: "
        f"{np.nanmin(value_sub):.6f} "
        f"to "
        f"{np.nanmax(value_sub):.6f}"
    )

    return (
        lon_sub,
        lat_sub,
        value_sub,
    )


def make_india_map(
    df,
    lon_grid,
    lat_grid,
    value_grid,
    cmap,
    norm,
    ticks,
    boundary,
    states,
):
    print(
        "\nCreating INDIA calibrated OOF probability map..."
    )

    (
        lon_sub,
        lat_sub,
        value_sub,
    ) = get_india_subgrid(
        df,
        lon_grid,
        lat_grid,
        value_grid,
    )

    fig = plt.figure(
        figsize=(9, 11)
    )

    ax = fig.add_axes(
        [0.08, 0.16, 0.84, 0.78],
        projection=ccrs.PlateCarree(),
    )

    cax = fig.add_axes([0.18, 0.065, 0.64, 0.035])

    ax.set_extent(
        INDIA_EXTENT,
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="#dbeeff",
        zorder=0,
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="#f9f9f9",
        zorder=0,
    )

    mesh = ax.pcolormesh(
        lon_sub,
        lat_sub,
        np.ma.masked_invalid(
            value_sub
        ),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
        zorder=1,
    )

    ax.coastlines(
        resolution="10m",
        linewidth=1.2,
        color="white",
        zorder=4,
    )

    ax.coastlines(
        resolution="10m",
        linewidth=0.55,
        color="#2b2b2b",
        zorder=5,
    )

    add_india_boundaries(
        ax,
        boundary,
        states,
    )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.65,
        alpha=0.28,
        linestyle="--",
        color="#444444",
        zorder=10,
    )

    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False

    gl.xlocator = (
        FixedLocator(
            INDIA_X_TICKS
        )
    )

    gl.ylocator = (
        FixedLocator(
            INDIA_Y_TICKS
        )
    )

    gl.xformatter = (
        LONGITUDE_FORMATTER
    )

    gl.yformatter = (
        LATITUDE_FORMATTER
    )

    force_gridliner_style(
        gl
    )

    # ax.set_title(
    #     INDIA_TITLE,
    #     fontsize=FONT_SIZE,
    #     fontweight="bold",
    # )

    add_colorbar(
        fig,
        cax,
        mesh,
        ticks,
    )

    output = PLOTS_DIR / f"{INDIA_TITLE}.png"

    fig.savefig(
        output,
        dpi=600,
        facecolor="white",
        pad_inches=0.05,
    )

    plt.close(
        fig
    )

    print(
        f"Saved: "
        f"{output.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_predictions()

    (
        lon_grid,
        lat_grid,
        value_grid,
    ) = build_lonlat_grids(
        df
    )

    (
        cmap,
        norm,
        ticks,
    ) = build_probability_style()

    boundary, states = (
        load_india_boundaries()
    )

    make_global_map(
        lon_grid,
        lat_grid,
        value_grid,
        cmap,
        norm,
        ticks,
    )

    make_india_map(
        df,
        lon_grid,
        lat_grid,
        value_grid,
        cmap,
        norm,
        ticks,
        boundary,
        states,
    )

    print("\n" + "=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(
        f"All figures saved under:\n"
        f"{PLOTS_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
