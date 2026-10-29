# 03_build_model_dataset_grid_v5.py
# Builds the canonical v5 model-ready dataset for 15/30/60-km grids by
# merging the v5 18-feature geophysical table with cell-level earthquake labels.

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from grid_config_v5 import (
    ensure_grid_directories,
    get_grid_config,
)


# ============================================================
# CANONICAL V5 PHYSICAL FEATURE SCHEMA
# ============================================================

PHYSICAL_FEATURES = [
    "elevation_m",
    "freeair_mgal",
    "bouguer_mgal",
    "isostatic_mgal",
    "geoid_umg3",
    "sedthk_km",
    "crsthk_km",
    "lithoref_moho_depth_km",
    "lithoref_lab_depth_km",
    "lithoref_rho_c",
    "lithoref_rho_l",
    "lithoref_rho_sl",
    "lithoref_geoid_m",
    "lithoref_fa_mgal",
    "gem_fault_presence",
    "gem_fault_dist_km",
    "gem_fault_length_density_km_per_1000km2",
    "terrain_slope_deg",
]

assert len(PHYSICAL_FEATURES) == 18

SPATIAL_METADATA_COLS = [
    "cell_id",
    "ix",
    "iy",
    "x_6933_m",
    "y_6933_m",
    "lon",
    "lat",
]

LABEL_METADATA_COLS = [
    "max_mag",
    "mean_mag",
    "mean_depth_km",
    "median_depth_km",
    "first_event_time",
    "last_event_time",
]

EXPECTED_TARGET_COLS = [
    "target_m4_0plus",
    "target_m4_5plus",
    "target_m5_0plus",
    "target_m5_5plus",
    "target_m6_0plus",
]

EXPECTED_COUNT_COLS = [
    "event_count_m4_0plus",
    "event_count_m4_5plus",
    "event_count_m5_0plus",
    "event_count_m5_5plus",
    "event_count_m6_0plus",
]


# ============================================================
# ARGUMENTS
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical v5 model-ready earthquake-susceptibility "
            "dataset for a selected equal-area grid resolution."
        )
    )

    parser.add_argument(
        "--cell-km",
        type=int,
        required=True,
        choices=[15, 30, 60],
        help="Nominal projected grid-cell size in kilometres.",
    )

    parser.add_argument(
        "--label-file",
        type=str,
        default=None,
        help=(
            "Optional explicit earthquake-label CSV. If omitted, the script "
            "uses a v5 label file when present, otherwise the existing "
            "resolution-specific label file."
        ),
    )

    return parser.parse_args()


# ============================================================
# PATHS
# ============================================================


def get_feature_file(config) -> Path:
    return (
        config.feature_dir
        / f"global_features_{config.cell_km}km_v5.csv"
    )


def get_default_label_file(config) -> Path:
    """
    Prefer a v5-named label table if one exists. The earthquake labels are
    independent of the geodesic feature correction, so the existing label
    table is also valid when a v5 copy has not yet been generated.
    """
    v5 = (
        config.earthquake_dir
        / f"earthquake_cell_labels_{config.cell_km}km_v5.csv"
    )

    legacy = (
        config.earthquake_dir
        / f"earthquake_cell_labels_{config.cell_km}km.csv"
    )

    if v5.exists():
        return v5

    return legacy


def get_output_file(config) -> Path:
    return (
        config.model_dir
        / f"model_dataset_cell_{config.cell_km}km_v5.csv"
    )


# ============================================================
# VALIDATION HELPERS
# ============================================================


def validate_feature_table(features: pd.DataFrame, config) -> None:
    required = set(SPATIAL_METADATA_COLS + PHYSICAL_FEATURES)
    missing = sorted(required - set(features.columns))

    if missing:
        raise ValueError(
            "Feature table is missing required v5 columns: "
            f"{missing}"
        )

    unexpected_physical_names = {
        "terrain_slope_deg_grid",
        "terrain_slope_deg_30km",
        "emag2_nT",
        "depthtomoho_km",
    } & set(features.columns)

    if unexpected_physical_names:
        raise ValueError(
            "Feature table contains legacy/non-canonical predictors: "
            f"{sorted(unexpected_physical_names)}"
        )

    expected_rows = config.nx * config.ny
    if len(features) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows:,} feature rows for "
            f"{config.cell_km} km grid, found {len(features):,}."
        )

    if features["cell_id"].duplicated().any():
        raise ValueError("Feature table contains duplicate cell_id values.")

    expected_ix = set(range(config.nx))
    expected_iy = set(range(config.ny))

    actual_ix = set(pd.to_numeric(features["ix"], errors="raise").astype(int).unique())
    actual_iy = set(pd.to_numeric(features["iy"], errors="raise").astype(int).unique())

    if actual_ix != expected_ix:
        raise RuntimeError("Feature table ix coverage does not match the configured grid.")

    if actual_iy != expected_iy:
        raise RuntimeError("Feature table iy coverage does not match the configured grid.")

    feature_missing = features[PHYSICAL_FEATURES].isna().sum()
    all_missing = feature_missing[feature_missing == len(features)]

    if not all_missing.empty:
        raise RuntimeError(
            "One or more physical features are entirely missing: "
            f"{all_missing.index.tolist()}"
        )


def validate_label_table(labels: pd.DataFrame) -> None:
    required = {
        "cell_id",
        *EXPECTED_TARGET_COLS,
        *EXPECTED_COUNT_COLS,
    }

    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(
            "Earthquake label table is missing required columns: "
            f"{missing}"
        )

    if labels["cell_id"].duplicated().any():
        raise ValueError(
            "Earthquake label table contains duplicate cell_id values."
        )


# ============================================================
# MAIN
# ============================================================


def main():
    args = parse_args()

    config = get_grid_config(args.cell_km)
    ensure_grid_directories(config)

    feature_file = get_feature_file(config)

    if args.label_file is None:
        label_file = get_default_label_file(config)
    else:
        label_file = Path(args.label_file)
        if not label_file.is_absolute():
            label_file = Path.cwd() / label_file

    output_file = get_output_file(config)

    print("=" * 88)
    print(
        f"MODEL DATASET BUILD V5 — {config.cell_km} KM — "
        "18 CANONICAL PHYSICAL FEATURES"
    )
    print("=" * 88)
    print(f"Features: {feature_file}")
    print(f"Labels:   {label_file}")
    print(f"Output:   {output_file}")

    if not feature_file.exists():
        raise FileNotFoundError(
            f"V5 feature file not found: {feature_file}"
        )

    if not label_file.exists():
        raise FileNotFoundError(
            f"Earthquake label file not found: {label_file}"
        )

    print("\nLoading v5 geophysical features...")
    features = pd.read_csv(feature_file)
    print(f"Feature rows: {len(features):,}")

    print("\nLoading earthquake labels...")
    labels = pd.read_csv(label_file)
    print(f"Label rows: {len(labels):,}")

    validate_feature_table(features, config)
    validate_label_table(labels)

    label_cells_not_in_grid = (
        set(labels["cell_id"])
        - set(features["cell_id"])
    )

    if label_cells_not_in_grid:
        raise RuntimeError(
            f"{len(label_cells_not_in_grid):,} earthquake-label cells "
            "do not exist in the configured feature grid."
        )

    print("\nMerging features and earthquake labels...")

    model = features.merge(
        labels,
        on="cell_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )

    if len(model) != len(features):
        raise RuntimeError("Row count changed during feature/label merge.")

    # Cells absent from the occupied-cell label table have no qualifying
    # earthquakes, so event counts and binary targets are zero. Descriptive
    # earthquake metadata remains NaN for those cells by design.
    count_cols = [
        col
        for col in model.columns
        if col.startswith("event_count_")
    ]

    target_cols = [
        col
        for col in model.columns
        if col.startswith("target_")
    ]

    for col in count_cols:
        model[col] = (
            pd.to_numeric(model[col], errors="coerce")
            .fillna(0)
            .astype(np.int32)
        )

    for col in target_cols:
        model[col] = (
            pd.to_numeric(model[col], errors="coerce")
            .fillna(0)
            .astype(np.int8)
        )

    # Re-check the exact canonical v5 physical schema after the merge.
    missing_physical = [
        col for col in PHYSICAL_FEATURES if col not in model.columns
    ]
    if missing_physical:
        raise RuntimeError(
            f"Merged dataset lost physical predictors: {missing_physical}"
        )

    # Preserve a predictable column order: spatial metadata, 18 predictors,
    # then every label/descriptive column in the order supplied by the label file.
    label_output_cols = [
        col for col in labels.columns if col != "cell_id"
    ]

    ordered_cols = (
        SPATIAL_METADATA_COLS
        + PHYSICAL_FEATURES
        + label_output_cols
    )

    missing_ordered = [
        col for col in ordered_cols if col not in model.columns
    ]
    if missing_ordered:
        raise RuntimeError(
            "Expected output columns are missing after merge: "
            f"{missing_ordered}"
        )

    model = model[ordered_cols].copy()

    # --------------------------------------------------------
    # FINAL QC
    # --------------------------------------------------------

    expected_rows = config.nx * config.ny

    if len(model) != expected_rows:
        raise RuntimeError(
            f"Expected final dataset to contain {expected_rows:,} cells, "
            f"got {len(model):,}."
        )

    if model["cell_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate cell_id values found in final model dataset."
        )

    print("\n")
    print("=" * 88)
    print(f"FINAL MODEL DATASET QC V5 — {config.cell_km} KM")
    print("=" * 88)
    print(f"Total cells:              {len(model):,}")
    print(f"Total columns:            {len(model.columns)}")
    print(f"Physical feature columns: {len(PHYSICAL_FEATURES)}")
    print("Slope feature:             terrain_slope_deg")

    print("\nEarthquake target summary")
    for col in EXPECTED_TARGET_COLS:
        positive = int(model[col].sum())
        rate = positive / len(model) * 100.0
        print(
            f"{col:<28} "
            f"positive={positive:>8,} "
            f"({rate:7.3f}%)"
        )

    print("\nPhysical feature missingness")
    for col in PHYSICAL_FEATURES:
        missing = int(model[col].isna().sum())
        missing_pct = missing / len(model) * 100.0
        print(
            f"{col:<50} "
            f"missing={missing:>9,} "
            f"({missing_pct:6.2f}%)"
        )

    # Basic target nesting check: higher-magnitude positives must be subsets
    # of lower-magnitude positives for the same cell.
    nested_pairs = list(zip(EXPECTED_TARGET_COLS[:-1], EXPECTED_TARGET_COLS[1:]))
    for lower_col, higher_col in nested_pairs:
        violations = int(
            ((model[higher_col] == 1) & (model[lower_col] == 0)).sum()
        )
        if violations:
            raise RuntimeError(
                f"Target nesting violation: {violations:,} cells are positive "
                f"for {higher_col} but negative for {lower_col}."
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    model.to_csv(output_file, index=False)

    print("\n")
    print("=" * 88)
    print("SAVED V5 MODEL-READY DATASET")
    print("=" * 88)
    print(f"Output: {output_file}")
    print(f"Rows: {len(model):,}")
    print(f"Columns: {len(model.columns)}")
    print(f"Physical features: {len(PHYSICAL_FEATURES)}")


if __name__ == "__main__":
    main()
