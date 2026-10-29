# 26_location_susceptibility_lookup_v5.py

from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
import geopandas as gpd

from pyproj import Transformer, Geod

GEOD = Geod(ellps="WGS84")
from shapely.geometry import Point


# =============================================================================
# PATHS / CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

OOF_FILE = (
    SCRIPT_DIR
    / "probability_calibration_v5_30km_m4_18features"
    / "calibration_oof_predictions.csv"
)

MODEL_DATASET = (
    SCRIPT_DIR
    / "data"
    / "grid_30km"
    / "model"
    / "model_dataset_cell_30km_v5.csv"
)

TARGET_COL = "target_m4_0plus"
RAW_COL = "raw_susceptibility_score"
CALIBRATED_COL = "isotonic_calibrated_probability"
PREDICTED_COL = "predicted_m4_positive"

PROJECTED_CRS = "EPSG:6933"
GEOGRAPHIC_CRS = "EPSG:4326"

PHYSICAL_CONTEXT = [
    ("elevation_m", "Elevation", "m"),
    ("crsthk_km", "Crust thickness", "km"),
    ("lithoref_moho_depth_km", "LithoRef Moho depth", "km"),
    ("lithoref_lab_depth_km", "LithoRef LAB depth", "km"),
    ("terrain_slope_deg", "Terrain slope", "deg"),
    (
        "gem_fault_length_density_km_per_1000km2",
        "Mapped active-fault length density",
        "km / 1000 km²",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Look up final 30 km M>=4 earthquake susceptibility "
            "for a latitude/longitude."
        )
    )
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument(
        "--faults",
        type=str,
        default=None,
        help=(
            "Optional explicit path to GEM active-fault GeoJSON. "
            "If omitted, the project is searched automatically."
        ),
    )
    return parser.parse_args()


def find_fault_file(explicit_path=None):
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = SCRIPT_DIR / path
        if not path.exists():
            raise FileNotFoundError(
                f"Fault file not found:\n{path.resolve()}"
            )
        return path.resolve()

    preferred_names = [
        "gem_active_faults_harmonized.geojson",
        "gem_active_faults.geojson",
    ]

    for name in preferred_names:
        direct = SCRIPT_DIR / name
        if direct.exists():
            return direct.resolve()

    for name in preferred_names:
        matches = sorted(SCRIPT_DIR.rglob(name))
        if matches:
            return matches[0].resolve()

    raise FileNotFoundError(
        "Could not find a GEM active-fault geometry file.\n"
        "Expected gem_active_faults_harmonized.geojson or "
        "gem_active_faults.geojson under the project directory, "
        "or pass --faults <path>."
    )


def load_lookup_table():
    if not OOF_FILE.exists():
        raise FileNotFoundError(
            f"Missing final OOF calibration table:\n{OOF_FILE.resolve()}"
        )

    if not MODEL_DATASET.exists():
        raise FileNotFoundError(
            f"Missing 30 km model dataset:\n{MODEL_DATASET.resolve()}"
        )

    print("Loading final OOF predictions...")
    oof = pd.read_csv(OOF_FILE)

    required_oof = [
        "cell_id",
        "ix",
        "iy",
        "x_6933_m",
        "y_6933_m",
        "lon",
        "lat",
        TARGET_COL,
        RAW_COL,
        CALIBRATED_COL,
        "fold_operating_threshold",
        PREDICTED_COL,
        "oof_fold",
    ]

    missing = [c for c in required_oof if c not in oof.columns]
    if missing:
        raise ValueError(
            f"OOF table is missing required columns: {missing}"
        )

    header = pd.read_csv(MODEL_DATASET, nrows=0)

    desired_model_cols = [
        "cell_id",
        "event_count_m4_0plus",
        "gem_fault_dist_km",
    ] + [name for name, _, _ in PHYSICAL_CONTEXT]

    available_model_cols = [
        c for c in desired_model_cols
        if c in header.columns
    ]

    if "cell_id" not in available_model_cols:
        raise ValueError(
            "30 km model dataset is missing required column: cell_id"
        )

    print("Loading grid metadata / physical context...")
    model = pd.read_csv(
        MODEL_DATASET,
        usecols=available_model_cols,
    )

    if oof["cell_id"].duplicated().any():
        raise RuntimeError("Duplicate cell_id values in OOF predictions.")

    if model["cell_id"].duplicated().any():
        raise RuntimeError("Duplicate cell_id values in model dataset.")

    lookup = oof.merge(
        model,
        on="cell_id",
        how="left",
        validate="one_to_one",
    )

    if lookup[["x_6933_m", "y_6933_m"]].isna().any().any():
        raise RuntimeError(
            "Some OOF cells did not match the 30 km model dataset."
        )

    grid_x = pd.to_numeric(
        lookup["x_6933_m"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    grid_y = pd.to_numeric(
        lookup["y_6933_m"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    raw_scores = pd.to_numeric(
        lookup[RAW_COL],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    sorted_raw_scores = np.sort(
        raw_scores[np.isfinite(raw_scores)]
    )

    print(f"Lookup grid ready: {len(lookup):,} cells")

    return lookup, grid_x, grid_y, sorted_raw_scores


def load_faults(fault_path):
    print(f"Loading mapped active faults:\n{fault_path}")

    faults = gpd.read_file(fault_path)

    if faults.crs is None:
        raise RuntimeError("Fault geometry file has no CRS.")

    faults = faults[
        faults.geometry.notna()
        & ~faults.geometry.is_empty
    ].copy()

    if faults.empty:
        raise RuntimeError(
            "Fault geometry file contains no usable geometries."
        )

    faults_projected = faults.to_crs(PROJECTED_CRS)

    print(f"Mapped fault features ready: {len(faults_projected):,}")

    return faults_projected


def _nonempty_value(row, columns_by_lowercase, candidates):
    for candidate in candidates:
        actual = columns_by_lowercase.get(candidate.lower())
        if actual is None:
            continue

        value = row.get(actual)
        if pd.isna(value):
            continue

        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return actual, text

    return None, None


def describe_fault(row):
    columns_by_lowercase = {
        str(column).lower(): column
        for column in row.index
    }

    _, name = _nonempty_value(
        row,
        columns_by_lowercase,
        ["name", "fault_name", "faultname"],
    )

    _, identifier = _nonempty_value(
        row,
        columns_by_lowercase,
        ["catalog_id", "objectid", "id", "fid"],
    )

    _, slip_type = _nonempty_value(
        row,
        columns_by_lowercase,
        ["slip_type", "sliptype"],
    )

    if name is None:
        if identifier is not None:
            name = f"Unnamed mapped active fault (ID: {identifier})"
        else:
            name = f"Unnamed mapped active fault (feature index: {row.name})"

    return name, slip_type


def global_percentile(sorted_scores, score):
    if not np.isfinite(score):
        return np.nan

    position = np.searchsorted(
        sorted_scores,
        score,
        side="right",
    )

    return 100.0 * position / len(sorted_scores)


def nearest_grid_cell(
    latitude,
    longitude,
    lookup,
    grid_x,
    grid_y,
    transformer,
):
    query_x, query_y = transformer.transform(
        longitude,
        latitude,
    )

    squared_distance = (
        (grid_x - query_x) ** 2
        + (grid_y - query_y) ** 2
    )

    index = int(np.argmin(squared_distance))

    # Get the selected cell first.
    row = lookup.iloc[index]

    # Report the actual geographic distance to that cell centre
    # using WGS84 geodesic distance.
    _, _, distance_m = GEOD.inv(
        longitude,
        latitude,
        float(row["lon"]),
        float(row["lat"]),
    )

    distance_km = abs(float(distance_m)) / 1000.0

    return row, query_x, query_y, distance_km


def nearest_fault(
    query_x,
    query_y,
    faults_projected,
):
    query_point = Point(query_x, query_y)

    distances_m = faults_projected.geometry.distance(
        query_point
    )

    if distances_m.empty:
        raise RuntimeError(
            "No mapped active faults available for lookup."
        )

    nearest_position = int(
        np.nanargmin(
            distances_m.to_numpy(dtype=np.float64)
        )
    )

    fault_row = faults_projected.iloc[nearest_position]

    fault_distance_km = (
        float(distances_m.iloc[nearest_position])
        / 1000.0
    )

    fault_name, slip_type = describe_fault(
        fault_row
    )

    return fault_name, fault_distance_km, slip_type


def yes_no(value):
    return "Yes" if int(value) == 1 else "No"


def format_optional_number(
    row,
    column,
    decimals=2,
):
    if column not in row.index:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return f"{value:.{decimals}f}"


def run_lookup(
    latitude,
    longitude,
    lookup,
    grid_x,
    grid_y,
    sorted_raw_scores,
    transformer,
):
    if not (-90.0 <= latitude <= 90.0):
        raise ValueError("Latitude must be between -90 and 90.")

    if not (-180.0 <= longitude <= 180.0):
        raise ValueError("Longitude must be between -180 and 180.")

    (
        cell,
        query_x,
        query_y,
        cell_center_distance_km,
    ) = nearest_grid_cell(
        latitude,
        longitude,
        lookup,
        grid_x,
        grid_y,
        transformer,
    )

    raw_score = float(cell[RAW_COL])
    calibrated_probability = float(cell[CALIBRATED_COL])

    percentile = global_percentile(
        sorted_raw_scores,
        raw_score,
    )

    print("\n" + "=" * 88)
    print("30 KM EARTHQUAKE SUSCEPTIBILITY LOOKUP")
    print("=" * 88)

    print("\nLOCATION")
    print("-" * 88)
    print(
        f"Requested coordinates:        "
        f"{latitude:.5f}°, {longitude:.5f}°"
    )
    print(
        f"Nearest grid-cell centre:     "
        f"{float(cell['lat']):.5f}°, "
        f"{float(cell['lon']):.5f}°"
    )
    print(
        f"Distance to grid-cell centre: "
        f"{cell_center_distance_km:.2f} km"
    )
    print(
        f"30 km cell ID:                "
        f"{cell['cell_id']}"
    )

    print("\nSUSCEPTIBILITY")
    print("-" * 88)
    print(
        f"Raw susceptibility score:     "
        f"{raw_score:.6f}"
    )
    print(
        f"Global susceptibility percentile: "
        f"{percentile:.2f}%"
    )
    print(
        f"Isotonic calibrated class probability: "
        f"{calibrated_probability:.6f} "
        f"({100.0 * calibrated_probability:.2f}%)"
    )
    print(
        f"OOF raw-score predicted M>=4-positive cell: "
        f"{yes_no(cell[PREDICTED_COL])}"
    )
    print(
        f"OOF raw-score fold threshold:     "
        f"{float(cell['fold_operating_threshold']):.6f}"
    )

    print("\nHISTORICAL CATALOGUE CONTEXT")
    print("-" * 88)
    print(
        f"Historically M>=4-positive cell: "
        f"{yes_no(cell[TARGET_COL])}"
    )

    if (
        "event_count_m4_0plus" in cell.index
        and pd.notna(cell["event_count_m4_0plus"])
    ):
        print(
            f"Catalogued M>=4 event count:  "
            f"{int(cell['event_count_m4_0plus'])}"
        )

    print("\nMAPPED ACTIVE-FAULT CONTEXT")
    print("-" * 88)

    model_fault_distance = format_optional_number(
        cell,
        "gem_fault_dist_km",
        decimals=2,
    )

    if model_fault_distance is not None:
        print(
            f"Distance from cell centre to nearest mapped active fault: "
            f"{model_fault_distance} km"
        )
    else:
        print(
            "Distance from cell centre to nearest mapped active fault: unavailable"
        )

    print("\nMAJOR PHYSICAL FEATURE VALUES — SELECTED 30 KM CELL")
    print("-" * 88)

    printed_any = False

    for (
        column,
        label,
        unit,
    ) in PHYSICAL_CONTEXT:
        value = format_optional_number(
            cell,
            column,
            decimals=2,
        )

        if value is None:
            continue

        printed_any = True
        print(
            f"{label + ':':34s}"
            f"{value} {unit}"
        )

    if not printed_any:
        print(
            "Physical-context columns were not "
            "available in the model dataset."
        )

    # print("\nINTERPRETATION")
    # print("-" * 88)
    # print(
    #     "This is static spatial susceptibility. "
    #     "The calibrated value is the probability that the 30 km cell "
    #     "belongs to the historically M>=4-positive class under the "
    #     "2000-2025 catalogue definition; it is NOT the probability of "
    #     "a future earthquake occurring within a specified time period."
    # )
    # print(
    #     "Fault distance is a projected point-to-line distance computed "
    #     "from the same mapped active-fault source in EPSG:6933."
    # )

    # print("=" * 88)


def interactive_loop(
    lookup,
    grid_x,
    grid_y,
    sorted_raw_scores,
    transformer,
):
    print("\nInteractive mode. Enter q to quit.")

    while True:
        lat_text = input("\nLatitude: ").strip()

        if lat_text.lower() in {
            "q",
            "quit",
            "exit",
        }:
            break

        lon_text = input("Longitude: ").strip()

        if lon_text.lower() in {
            "q",
            "quit",
            "exit",
        }:
            break

        try:
            latitude = float(lat_text)
            longitude = float(lon_text)

            run_lookup(
                latitude,
                longitude,
                lookup,
                grid_x,
                grid_y,
                sorted_raw_scores,
                transformer,
            )

        except Exception as exc:
            print(f"\nLookup failed: {exc}")


def main():
    args = parse_args()

    if (args.lat is None) != (args.lon is None):
        raise ValueError(
            "Provide both --lat and --lon, or neither."
        )

    (
        lookup,
        grid_x,
        grid_y,
        sorted_raw_scores,
    ) = load_lookup_table()

    transformer = Transformer.from_crs(
        GEOGRAPHIC_CRS,
        PROJECTED_CRS,
        always_xy=True,
    )

    if (
        args.lat is not None
        and args.lon is not None
    ):
        run_lookup(
            args.lat,
            args.lon,
            lookup,
            grid_x,
            grid_y,
            sorted_raw_scores,
            transformer,
        )   
    else:
        interactive_loop(
            lookup,
            grid_x,
            grid_y,
            sorted_raw_scores,
            transformer,
        )


if __name__ == "__main__":
    main()
