# 01_prepare_usgs_earthquakes_grid.py

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from pyproj import Transformer

from grid_config import (
    ensure_grid_directories,
    get_grid_config,
)


# ============================================================
# SHARED RAW EARTHQUAKE INPUT
#
# The 312 monthly USGS files are shared by every grid resolution.
# They are NOT copied or modified.
# ============================================================

USGS_RAW_DIR = Path(
    "data/usgs_raw_m4_2000_2025"
)

MIN_MAG = 4.0

THRESHOLDS = [
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
]

TO_EA = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:6933",
    always_xy=True,
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the cleaned USGS catalog and "
            "cell-level earthquake labels for a selected "
            "equal-area grid resolution."
        )
    )

    parser.add_argument(
        "--cell-km",
        type=int,
        required=True,
        choices=[15, 30, 60],
        help="Grid cell size in kilometres.",
    )

    return parser.parse_args()


# ============================================================
# LOAD RAW MONTHLY FILES
# ============================================================

def load_usgs_files():
    paths = sorted(
        USGS_RAW_DIR.glob(
            "usgs_m4_*.csv"
        )
    )

    print(
        f"USGS raw directory: "
        f"{USGS_RAW_DIR}"
    )

    print(
        f"Files found: "
        f"{len(paths)}"
    )

    if len(paths) != 312:
        raise RuntimeError(
            f"Expected 312 monthly USGS files, "
            f"found {len(paths)}."
        )

    required_cols = [
        "time",
        "latitude",
        "longitude",
        "depth",
        "mag",
        "id",
        "updated",
        "place",
        "type",
        "status",
        "net",
        "magType",
    ]

    optional_cols = [
        "nst",
        "gap",
        "dmin",
        "rms",
        "horizontalError",
        "depthError",
        "magError",
        "locationSource",
        "magSource",
    ]

    frames = []
    raw_rows = 0

    for path in paths:
        df = pd.read_csv(
            path
        )

        raw_rows += len(df)

        missing_required = [
            col
            for col in required_cols
            if col not in df.columns
        ]

        if missing_required:
            raise RuntimeError(
                f"{path.name} is missing required columns: "
                f"{missing_required}"
            )

        keep_cols = (
            required_cols
            + [
                col
                for col in optional_cols
                if col in df.columns
            ]
        )

        df = df[
            keep_cols
        ].copy()

        df[
            "source_file"
        ] = path.name

        frames.append(
            df
        )

    print(
        f"Raw rows: "
        f"{raw_rows:,}"
    )

    eq = pd.concat(
        frames,
        ignore_index=True,
    )

    return eq, raw_rows


# ============================================================
# CLEAN CATALOG
# ============================================================

def clean_catalog(eq):
    raw_count = len(eq)

    eq["time"] = pd.to_datetime(
        eq["time"],
        errors="coerce",
        utc=True,
    )

    eq["updated"] = pd.to_datetime(
        eq["updated"],
        errors="coerce",
        utc=True,
    )

    numeric_cols = [
        "latitude",
        "longitude",
        "depth",
        "mag",
        "nst",
        "gap",
        "dmin",
        "rms",
        "horizontalError",
        "depthError",
        "magError",
    ]

    for column in numeric_cols:
        if column in eq.columns:
            eq[column] = pd.to_numeric(
                eq[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Valid event ID
    # --------------------------------------------------------

    eq["id"] = (
        eq["id"]
        .astype("string")
        .str.strip()
    )

    valid_id = (
        eq["id"].notna()
        & (eq["id"] != "")
    )

    removed_missing_id = int(
        (~valid_id).sum()
    )

    eq = eq[
        valid_id
    ].copy()

    # --------------------------------------------------------
    # Valid time/location/magnitude
    # --------------------------------------------------------

    valid_basic = (
        eq["time"].notna()
        & eq["latitude"].between(
            -90,
            90,
        )
        & eq["longitude"].between(
            -180,
            180,
        )
        & eq["mag"].notna()
    )

    removed_invalid_basic = int(
        (~valid_basic).sum()
    )

    eq = eq[
        valid_basic
    ].copy()

    # --------------------------------------------------------
    # Earthquakes only
    # --------------------------------------------------------

    event_type = (
        eq["type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    earthquake_mask = (
        event_type
        == "earthquake"
    )

    removed_non_earthquake = int(
        (~earthquake_mask).sum()
    )

    eq = eq[
        earthquake_mask
    ].copy()

    # --------------------------------------------------------
    # Normalize magnitude-type text
    # --------------------------------------------------------

    eq["magType"] = (
        eq["magType"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Deduplicate first:
    # retain the latest available USGS version of each event.
    # --------------------------------------------------------

    print(
        f"Valid earthquake rows before deduplication: "
        f"{len(eq):,}"
    )

    eq = eq.sort_values(
        [
            "id",
            "updated",
        ],
        na_position="first",
    )

    before_dedup = len(eq)

    eq = (
        eq.drop_duplicates(
            subset=["id"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    duplicate_count = (
        before_dedup
        - len(eq)
    )

    print(
        f"Duplicate event IDs removed: "
        f"{duplicate_count:,}"
    )

    print(
        f"Unique earthquakes after deduplication: "
        f"{len(eq):,}"
    )

    # --------------------------------------------------------
    # Enforce final acquisition threshold after deduplication.
    # --------------------------------------------------------

    below_threshold = (
        eq["mag"]
        < MIN_MAG
    )

    removed_below_threshold = int(
        below_threshold.sum()
    )

    eq = eq[
        ~below_threshold
    ].copy()

    eq = eq.reset_index(
        drop=True
    )

    print(
        f"Rows removed below M{MIN_MAG:.1f}: "
        f"{removed_below_threshold:,}"
    )

    print(
        f"Final earthquakes after M{MIN_MAG:.1f} filter: "
        f"{len(eq):,}"
    )

    total_removed = (
        raw_count
        - len(eq)
    )

    print(
        f"Total rows removed from raw catalog: "
        f"{total_removed:,}"
    )

    cleaning_stats = {
        "raw_rows":
            raw_count,

        "removed_missing_id":
            removed_missing_id,

        "removed_invalid_basic":
            removed_invalid_basic,

        "removed_non_earthquake":
            removed_non_earthquake,

        "duplicates_removed":
            duplicate_count,

        "removed_below_min_mag":
            removed_below_threshold,

        "final_clean_rows":
            len(eq),
    }

    return eq, cleaning_stats


# ============================================================
# ASSIGN SELECTED GRID
# ============================================================

def assign_grid_cells(
    eq,
    config,
):
    x, y = TO_EA.transform(
        eq[
            "longitude"
        ].to_numpy(
            dtype=float
        ),
        eq[
            "latitude"
        ].to_numpy(
            dtype=float
        ),
    )

    eq = eq.copy()

    eq[
        "x_6933_m"
    ] = np.asarray(
        x,
        dtype=np.float64,
    )

    eq[
        "y_6933_m"
    ] = np.asarray(
        y,
        dtype=np.float64,
    )

    projected_valid = (
        np.isfinite(
            eq["x_6933_m"]
        )
        & np.isfinite(
            eq["y_6933_m"]
        )
    )

    rejected_projection = int(
        (~projected_valid).sum()
    )

    eq = eq[
        projected_valid
    ].copy()

    eq["ix"] = np.floor(
        (
            eq["x_6933_m"]
            - config.x_min
        )
        / config.cell_m
    ).astype(
        np.int64
    )

    eq["iy"] = np.floor(
        (
            eq["y_6933_m"]
            - config.y_min
        )
        / config.cell_m
    ).astype(
        np.int64
    )

    inside = (
        eq["ix"].between(
            0,
            config.nx - 1,
        )
        & eq["iy"].between(
            0,
            config.ny - 1,
        )
    )

    rejected_grid = int(
        (~inside).sum()
    )

    eq = eq[
        inside
    ].copy()

    eq["cell_id"] = (
        eq["iy"].astype(str)
        + "_"
        + eq["ix"].astype(str)
    )

    eq = eq.reset_index(
        drop=True
    )

    print(
        f"Invalid projected coordinates rejected: "
        f"{rejected_projection:,}"
    )

    print(
        f"Events outside {config.cell_km} km "
        f"study grid rejected: "
        f"{rejected_grid:,}"
    )

    return (
        eq,
        rejected_projection,
        rejected_grid,
    )


# ============================================================
# CELL LABELS
# ============================================================

def threshold_suffix(
    threshold,
):
    return str(
        threshold
    ).replace(
        ".",
        "_",
    )


def build_cell_labels(
    eq,
):
    work = eq.copy()

    for threshold in THRESHOLDS:
        suffix = threshold_suffix(
            threshold
        )

        work[
            f"is_m{suffix}plus"
        ] = (
            work["mag"]
            >= threshold
        ).astype(
            np.int8
        )

    aggregation = {
        "max_mag":
            (
                "mag",
                "max",
            ),

        "mean_mag":
            (
                "mag",
                "mean",
            ),

        "mean_depth_km":
            (
                "depth",
                "mean",
            ),

        "median_depth_km":
            (
                "depth",
                "median",
            ),

        "first_event_time":
            (
                "time",
                "min",
            ),

        "last_event_time":
            (
                "time",
                "max",
            ),
    }

    for threshold in THRESHOLDS:
        suffix = threshold_suffix(
            threshold
        )

        aggregation[
            f"event_count_m{suffix}plus"
        ] = (
            f"is_m{suffix}plus",
            "sum",
        )

    labels = (
        work.groupby(
            "cell_id",
            as_index=False,
        )
        .agg(
            **aggregation
        )
    )

    for threshold in THRESHOLDS:
        suffix = threshold_suffix(
            threshold
        )

        count_col = (
            f"event_count_m{suffix}plus"
        )

        target_col = (
            f"target_m{suffix}plus"
        )

        labels[
            target_col
        ] = (
            labels[
                count_col
            ] > 0
        ).astype(
            np.int8
        )

    return labels


# ============================================================
# QC REPORT
# ============================================================

def build_qc_lines(
    eq,
    labels,
    config,
    cleaning_stats,
    rejected_projection,
    rejected_grid,
):
    lines = []

    lines.append(
        "USGS EARTHQUAKE CATALOG QC"
    )
    lines.append(
        "=" * 72
    )

    lines.append(
        f"Grid resolution: "
        f"{config.cell_km} km"
    )

    lines.append(
        f"Grid shape: "
        f"{config.nx} x {config.ny} "
        f"= {config.nx * config.ny:,} cells"
    )

    lines.append(
        f"Projected X extent: "
        f"{config.x_min:.2f} to "
        f"{config.x_max:.2f}"
    )

    lines.append(
        f"Projected Y extent: "
        f"{config.y_min:.2f} to "
        f"{config.y_max:.2f}"
    )

    lines.append("")

    lines.append(
        f"Raw downloaded rows: "
        f"{cleaning_stats['raw_rows']:,}"
    )

    lines.append(
        f"Clean events before grid clipping: "
        f"{cleaning_stats['final_clean_rows']:,}"
    )

    lines.append(
        f"Invalid projected coordinates rejected: "
        f"{rejected_projection:,}"
    )

    lines.append(
        f"Events outside study grid rejected: "
        f"{rejected_grid:,}"
    )

    lines.append(
        f"Final events assigned to grid: "
        f"{len(eq):,}"
    )

    lines.append(
        f"Unique IDs: "
        f"{eq['id'].nunique():,}"
    )

    lines.append(
        f"Date range: "
        f"{eq['time'].min()} "
        f"to "
        f"{eq['time'].max()}"
    )

    lines.append(
        f"Magnitude range: "
        f"{eq['mag'].min():.2f} "
        f"to "
        f"{eq['mag'].max():.2f}"
    )

    lines.append("")
    lines.append(
        "Magnitude threshold counts"
    )

    for threshold in THRESHOLDS:
        suffix = threshold_suffix(
            threshold
        )

        event_count = int(
            (
                eq["mag"]
                >= threshold
            ).sum()
        )

        occupied_cells = int(
            (
                labels[
                    f"event_count_m{suffix}plus"
                ] > 0
            ).sum()
        )

        lines.append(
            f"M >= {threshold}: "
            f"{event_count:,} events; "
            f"{occupied_cells:,} occupied cells"
        )

    lines.append("")
    lines.append(
        "Review status"
    )

    lines.append(
        eq["status"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    lines.append("")
    lines.append(
        "Magnitude types"
    )

    lines.append(
        eq["magType"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    lines.append("")
    lines.append(
        "Events by year"
    )

    lines.append(
        eq.groupby(
            eq["time"].dt.year
        )
        .size()
        .to_string()
    )

    return lines


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    config = get_grid_config(
        args.cell_km
    )

    ensure_grid_directories(
        config
    )

    print(
        "=" * 72
    )

    print(
        f"EARTHQUAKE GRID PREPARATION: "
        f"{config.cell_km} km"
    )

    print(
        "=" * 72
    )

    print(
        f"Grid: "
        f"{config.nx} x {config.ny} "
        f"= {config.nx * config.ny:,} cells"
    )

    eq, raw_rows = (
        load_usgs_files()
    )

    eq, cleaning_stats = (
        clean_catalog(
            eq
        )
    )

    (
        eq,
        rejected_projection,
        rejected_grid,
    ) = assign_grid_cells(
        eq,
        config,
    )

    labels = build_cell_labels(
        eq
    )

    clean_output = (
        config.earthquake_dir
        / (
            "usgs_earthquakes_"
            "m4_2000_2025_clean.csv"
        )
    )

    label_output = (
        config.earthquake_dir
        / (
            f"earthquake_cell_labels_"
            f"{config.cell_km}km.csv"
        )
    )

    qc_output = (
        config.earthquake_dir
        / (
            f"usgs_earthquake_catalog_qc_"
            f"{config.cell_km}km.txt"
        )
    )

    eq.to_csv(
        clean_output,
        index=False,
    )

    labels.to_csv(
        label_output,
        index=False,
    )

    qc_lines = build_qc_lines(
        eq,
        labels,
        config,
        cleaning_stats,
        rejected_projection,
        rejected_grid,
    )

    qc_output.write_text(
        "\n".join(
            qc_lines
        ),
        encoding="utf-8",
    )

    print("\n")
    print(
        "\n".join(
            qc_lines[:20]
        )
    )

    print("\nSaved:")
    print(
        clean_output
    )
    print(
        label_output
    )
    print(
        qc_output
    )


if __name__ == "__main__":
    main()