# 02_prepare_global_features_grid_v5.py
# Builds the v5 geodesically corrected 18-feature global grid for 15/30/60-km resolutions.

from __future__ import annotations

from pathlib import Path
import argparse
import math
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from pyproj import Geod, Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, LineString, MultiLineString, box
from shapely.ops import transform as shapely_transform

from grid_config_v5 import (
    PROJECT_DIR,
    ensure_grid_directories,
    get_grid_config,
)


# ============================================================
# SHARED RAW SOURCE FILES
# ============================================================

GEBCO_FILE = PROJECT_DIR / "GEBCO_2026.nc"

WGM_FREEAIR = PROJECT_DIR / "WGM2012_Freeair_ponc_2min.grd"
WGM_BOUGUER = PROJECT_DIR / "WGM2012_Bouguer_ponc_2min.grd"
WGM_ISOSTATIC = PROJECT_DIR / "WGM2012_Isostatic_ponc_2min.grd"

GEOID_UMG3 = PROJECT_DIR / "Geoid_UMG3.xyz"
SEDTHK = PROJECT_DIR / "sedthk.xyz"
CRSTHK = PROJECT_DIR / "crsthk.xyz"
LITHOREF = PROJECT_DIR / "LithoRef_model" / "LithoRef18.xyz"

# Retained only as source references; not used in the final 18-feature model.
DEPTH_TO_MOHO = PROJECT_DIR / "depthtomoho.xyz"
EMAG2 = PROJECT_DIR / "EMAG2_V3_20170530.csv"

FAULT_FILE = PROJECT_DIR / "gem_active_faults_harmonized.geojson"


# ============================================================
# COORDINATE / GEODESIC HELPERS
# ============================================================

TO_EA = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:6933",
    always_xy=True,
)

FROM_EA = Transformer.from_crs(
    "EPSG:6933",
    "EPSG:4326",
    always_xy=True,
)

GEOD = Geod(ellps="WGS84")

# The nearest-fault distance is approximated by densifying each mapped fault
# trace geodesically. At <=2-km spacing, the sampling error is small relative
# to the 15/30/60-km model grids while keeping the computation manageable.
DEFAULT_FAULT_SAMPLE_SPACING_KM = 2.0
DEFAULT_NEAREST_CANDIDATES = 8


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare geodesically corrected global geophysical features "
            "for a selected EPSG:6933 equal-area grid resolution."
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
        "--fault-sample-spacing-km",
        type=float,
        default=DEFAULT_FAULT_SAMPLE_SPACING_KM,
        help="Maximum geodesic spacing used to sample fault traces for nearest-distance lookup.",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_NEAREST_CANDIDATES,
        help="Number of nearest sampled fault points refined with exact WGS84 geodesic distance.",
    )

    return parser.parse_args()


# ============================================================
# BASIC HELPERS
# ============================================================


def expected_feature_names():
    """Return the single canonical v5 18-feature schema for every grid size."""
    return {
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

    }


def output_feature_file(config):
    """Versioned v5 output path; older feature tables are never overwritten."""
    return config.feature_dir / f"global_features_{config.cell_km}km_v5.csv"


def empty_accumulators(config):
    return (
        np.zeros((config.ny, config.nx), dtype=np.float64),
        np.zeros((config.ny, config.nx), dtype=np.int32),
    )


def finalize_feature(sums, counts):
    result = np.full(sums.shape, np.nan, dtype=np.float32)
    valid = counts > 0
    result[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return result


def add_points(lon, lat, values, sums, counts, config):
    """
    Bin dense source samples into the selected EPSG:6933 grid and compute a
    simple arithmetic mean of finite source samples whose centres fall in each
    target cell.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & np.isfinite(values)
        & (lon >= -180.0)
        & (lon <= 180.0)
        & (lat >= -90.0)
        & (lat <= 90.0)
    )

    if not np.any(valid):
        return

    x, y = TO_EA.transform(lon[valid], lat[valid])
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    values = values[valid]

    valid_projected = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid_projected):
        return

    x = x[valid_projected]
    y = y[valid_projected]
    values = values[valid_projected]

    ix = np.floor((x - config.x_min) / config.cell_m).astype(np.int64)
    iy = np.floor((y - config.y_min) / config.cell_m).astype(np.int64)

    inside = (
        (ix >= 0)
        & (ix < config.nx)
        & (iy >= 0)
        & (iy < config.ny)
    )

    if not np.any(inside):
        return

    np.add.at(sums, (iy[inside], ix[inside]), values[inside])
    np.add.at(counts, (iy[inside], ix[inside]), 1)


# ============================================================
# TARGET GRID CENTRES
# ============================================================

def build_grid_centers(config):
    x_centres = (
        config.x_min
        + (np.arange(config.nx, dtype=np.float64) + 0.5) * config.cell_m
    )

    y_centres = (
        config.y_min
        + (np.arange(config.ny, dtype=np.float64) + 0.5) * config.cell_m
    )

    xx, yy = np.meshgrid(x_centres, y_centres)

    lon, lat = FROM_EA.transform(xx, yy)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)

    if not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise RuntimeError("Target grid contains invalid geographic coordinates.")

    return xx, yy, lon, lat


# ============================================================
# GEBCO
# ============================================================

def gebco_intermediate_file(config):
    return (
        config.intermediate_dir
        / f"GEBCO_2026_EPSG6933_{config.cell_km}km_{config.nx}x{config.ny}.tif"
    )


def prepare_gebco_grid(config):
    if not GEBCO_FILE.exists():
        raise FileNotFoundError(f"GEBCO file not found: {GEBCO_FILE.resolve()}")

    output = gebco_intermediate_file(config)

    if output.exists():
        print(
            f"Using existing GEBCO {config.cell_km} km raster: {output}"
        )
        return output

    print(f"\nReprojecting GEBCO to {config.cell_km} km EPSG:6933...")

    command = [
        "gdalwarp",
        "-overwrite",
        "-t_srs", "EPSG:6933",
        "-te",
        str(config.x_min),
        str(config.y_min),
        str(config.x_max),
        str(config.y_max),
        "-ts",
        str(config.nx),
        str(config.ny),
        "-r", "average",
        "-ot", "Float32",
        "-dstnodata", "-99999",
        "-co", "COMPRESS=DEFLATE",
        str(GEBCO_FILE),
        str(output),
    ]

    print(" ".join(command))

    import subprocess
    subprocess.run(command, check=True)

    print(f"Created: {output}")
    return output


def load_gebco_grid(config):
    output = prepare_gebco_grid(config)

    import rasterio

    with rasterio.open(output) as src:
        if src.crs is None or src.crs.to_epsg() != 6933:
            raise ValueError(
                f"GEBCO intermediate CRS is {src.crs}; expected EPSG:6933."
            )

        grid = src.read(1).astype(np.float32)

        # Raster row 0 is north; internal iy=0 is south.
        if src.transform.e < 0:
            grid = np.flipud(grid)

        nodata = src.nodata
        if nodata is not None:
            grid[np.isclose(grid, nodata)] = np.nan

    expected_shape = (config.ny, config.nx)
    if grid.shape != expected_shape:
        raise ValueError(
            f"GEBCO output shape {grid.shape} does not match {expected_shape}."
        )

    return grid


# ============================================================
# DENSE GRD INPUTS
# ============================================================

def add_from_grd(path, feature_name, config):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")

    with xr.open_dataset(path) as ds:
        x_name = "x" if "x" in ds.variables else "lon"
        y_name = "y" if "y" in ds.variables else "lat"

        if "z" not in ds.variables:
            raise ValueError(f"{path}: expected a 'z' variable.")

        lons = np.asarray(ds[x_name].values, dtype=np.float64).squeeze()
        lats = np.asarray(ds[y_name].values, dtype=np.float64).squeeze()
        z = ds["z"]

        lons = np.where(lons > 180.0, lons - 360.0, lons)
        sums, counts = empty_accumulators(config)

        for j, latitude in enumerate(lats):
            row = z.isel({y_name: j}).values.astype(np.float64).squeeze()

            if row.shape[0] != lons.shape[0]:
                raise ValueError(f"{path}: source row length mismatch.")

            valid = np.isfinite(row)
            if not np.any(valid):
                continue

            add_points(
                lons[valid],
                np.full(valid.sum(), float(latitude), dtype=np.float64),
                row[valid],
                sums,
                counts,
                config,
            )

    return feature_name, finalize_feature(sums, counts)


# ============================================================
# REGULAR COARSE-GRID INTERPOLATION
# ============================================================

def interpolate_regular_dataframe(
    df,
    value_col,
    target_lon,
    target_lat,
    scale,
    feature_name,
):
    d = df[["lon", "lat", value_col]].copy()

    d["lon"] = pd.to_numeric(d["lon"], errors="coerce")
    d["lat"] = pd.to_numeric(d["lat"], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["lon", "lat", value_col])

    d["lon"] = (d["lon"] + 180.0) % 360.0 - 180.0
    d[value_col] = d[value_col].astype(np.float64) * float(scale)

    pivot = (
        d.pivot_table(
            index="lat",
            columns="lon",
            values=value_col,
            aggfunc="mean",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    src_lat = pivot.index.to_numpy(dtype=np.float64)
    src_lon = pivot.columns.to_numpy(dtype=np.float64)
    src_grid = pivot.to_numpy(dtype=np.float64)

    if src_lat.size < 2 or src_lon.size < 2:
        raise ValueError(f"{feature_name}: source grid too small.")

    if not np.all(np.diff(src_lat) > 0):
        raise ValueError(f"{feature_name}: latitude axis is not strictly increasing.")

    if not np.all(np.diff(src_lon) > 0):
        raise ValueError(f"{feature_name}: longitude axis is not strictly increasing.")

    source_missing = int(np.count_nonzero(~np.isfinite(src_grid)))
    print(
        f"{feature_name}: source grid={src_grid.shape}, "
        f"source missing={source_missing:,}"
    )

    # Cyclic longitude seam.
    lon_extended = np.concatenate(
        ([src_lon[-1] - 360.0], src_lon, [src_lon[0] + 360.0])
    )
    grid_extended = np.concatenate(
        (src_grid[:, -1:], src_grid, src_grid[:, :1]),
        axis=1,
    )

    interpolator = RegularGridInterpolator(
        (src_lat, lon_extended),
        grid_extended,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    target_lon_norm = (
        (np.asarray(target_lon, dtype=np.float64) + 180.0) % 360.0 - 180.0
    )

    points = np.column_stack(
        (
            np.asarray(target_lat, dtype=np.float64).ravel(),
            target_lon_norm.ravel(),
        )
    )

    result = interpolator(points).reshape(target_lon.shape)
    return result.astype(np.float32)


def load_xyz_interpolated(
    path,
    feature_name,
    target_lon,
    target_lat,
    col_lon,
    col_lat,
    col_val,
    skiprows,
    scale,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+|,",
        engine="python",
        header=None,
        skiprows=skiprows,
        usecols=[col_lon, col_lat, col_val],
    )

    df.columns = ["lon", "lat", "value"]

    return interpolate_regular_dataframe(
        df=df,
        value_col="value",
        target_lon=target_lon,
        target_lat=target_lat,
        scale=scale,
        feature_name=feature_name,
    )


# ============================================================
# LITHOREF18
# ============================================================

def add_lithoref_features(features, target_lon, target_lat):
    if not LITHOREF.exists():
        raise FileNotFoundError(f"Missing LithoRef file: {LITHOREF}")

    columns = [
        "lon",
        "lat",
        "elev",
        "moho",
        "lab",
        "rho_c",
        "rho_l",
        "rho_sl",
        "bottom",
        "geoid",
        "fa",
        "gzz",
        "gxx",
        "gyy",
    ]

    df = pd.read_csv(
        LITHOREF,
        sep=r"\s+",
        engine="python",
        skiprows=9,
        names=columns,
    )

    specs = [
        ("moho", "lithoref_moho_depth_km", -0.001),
        ("lab", "lithoref_lab_depth_km", -0.001),
        ("rho_c", "lithoref_rho_c", 1.0),
        ("rho_l", "lithoref_rho_l", 1.0),
        ("rho_sl", "lithoref_rho_sl", 1.0),
        ("geoid", "lithoref_geoid_m", 1.0),
        ("fa", "lithoref_fa_mgal", 1.0),
    ]

    for source_col, output_col, scale in specs:
        print(f"Interpolating {output_col}...")
        features[output_col] = interpolate_regular_dataframe(
            df=df,
            value_col=source_col,
            target_lon=target_lon,
            target_lat=target_lat,
            scale=scale,
            feature_name=output_col,
        )


# ============================================================
# GEODESIC FAULT HELPERS
# ============================================================

def iter_lines(geometry) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return

    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from iter_lines(part)


def geodesic_geometry_length_m(geometry) -> float:
    total = 0.0
    for line in iter_lines(geometry):
        total += abs(float(GEOD.geometry_length(line)))
    return total


def densify_linestring_geodesic(line: LineString, spacing_m: float) -> np.ndarray:
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")

    coords = np.asarray(
        [(coord[0], coord[1]) for coord in line.coords],
        dtype=np.float64,
    )

    if len(coords) == 0:
        return np.empty((0, 2), dtype=np.float64)

    if len(coords) == 1:
        return coords.copy()

    out: list[tuple[float, float]] = [
        (float(coords[0, 0]), float(coords[0, 1]))
    ]

    for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:]):
        _, _, distance_m = GEOD.inv(lon1, lat1, lon2, lat2)
        distance_m = float(abs(distance_m))

        if not np.isfinite(distance_m):
            continue

        if distance_m == 0.0:
            endpoint = (float(lon2), float(lat2))
            if out[-1] != endpoint:
                out.append(endpoint)
            continue

        n_segments = max(1, int(math.ceil(distance_m / spacing_m)))

        if n_segments > 1:
            intermediate = GEOD.npts(
                lon1,
                lat1,
                lon2,
                lat2,
                n_segments - 1,
            )
            out.extend((float(lon), float(lat)) for lon, lat in intermediate)

        out.append((float(lon2), float(lat2)))

    return np.asarray(out, dtype=np.float64)


def densify_fault_geometries(gdf_wgs84, spacing_m):
    chunks = []

    for geom in gdf_wgs84.geometry:
        for line in iter_lines(geom):
            points = densify_linestring_geodesic(line, spacing_m)
            if len(points):
                chunks.append(points)

    if not chunks:
        raise RuntimeError("No usable fault-line samples were generated.")

    samples = np.vstack(chunks)

    valid = (
        np.isfinite(samples[:, 0])
        & np.isfinite(samples[:, 1])
        & (samples[:, 0] >= -180.0)
        & (samples[:, 0] <= 180.0)
        & (samples[:, 1] >= -90.0)
        & (samples[:, 1] <= 90.0)
    )

    return samples[valid]


def lonlat_to_unit_xyz(lon_deg, lat_deg):
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    cos_lat = np.cos(lat)

    return np.column_stack(
        (
            cos_lat * np.cos(lon),
            cos_lat * np.sin(lon),
            np.sin(lat),
        )
    )


def nearest_geodesic_distance_to_samples(
    query_lon,
    query_lat,
    sample_lon,
    sample_lat,
    k_candidates=DEFAULT_NEAREST_CANDIDATES,
    chunk_size=100_000,
):
    """
    Approximate point-to-fault distance using densely sampled fault traces.

    The cKDTree is used only to obtain nearby candidates on the unit sphere;
    reported distances are recomputed with WGS84 Geod.inv.
    """
    query_lon = np.asarray(query_lon, dtype=np.float64)
    query_lat = np.asarray(query_lat, dtype=np.float64)
    sample_lon = np.asarray(sample_lon, dtype=np.float64)
    sample_lat = np.asarray(sample_lat, dtype=np.float64)

    if query_lon.shape != query_lat.shape:
        raise ValueError("Query longitude/latitude shapes differ.")

    if sample_lon.shape != sample_lat.shape or sample_lon.size == 0:
        raise ValueError("Invalid fault samples.")

    k = max(1, min(int(k_candidates), int(sample_lon.size)))

    tree = cKDTree(
        lonlat_to_unit_xyz(sample_lon, sample_lat)
    )

    result = np.empty(query_lon.size, dtype=np.float64)

    for start in range(0, query_lon.size, chunk_size):
        stop = min(query_lon.size, start + chunk_size)

        qlon = query_lon[start:stop]
        qlat = query_lat[start:stop]

        _, indices = tree.query(
            lonlat_to_unit_xyz(qlon, qlat),
            k=k,
            workers=-1,
        )

        indices = np.asarray(indices, dtype=np.int64)
        if k == 1:
            indices = indices[:, None]

        candidate_lon = sample_lon[indices]
        candidate_lat = sample_lat[indices]

        qlon_2d = np.broadcast_to(qlon[:, None], candidate_lon.shape)
        qlat_2d = np.broadcast_to(qlat[:, None], candidate_lat.shape)

        candidate_shape = candidate_lon.shape
        _, _, distance_m_flat = GEOD.inv(
            qlon_2d.ravel(),
            qlat_2d.ravel(),
            candidate_lon.ravel(),
            candidate_lat.ravel(),
        )
        distance_m = np.asarray(distance_m_flat, dtype=np.float64).reshape(
            candidate_shape
        )

        result[start:stop] = np.min(
            np.abs(distance_m),
            axis=1,
        )

    return result


def compute_geodesic_fault_presence_and_length(gdf_wgs84, config):
    """
    Use EPSG:6933 only to determine which equal-area cell supports each fault
    fragment. Measure the clipped fragment itself geodesically after transforming
    it back to WGS84.
    """
    gdf_6933 = gdf_wgs84.to_crs(epsg=6933)

    presence = np.zeros(
        (config.ny, config.nx),
        dtype=np.uint8,
    )

    fault_length_m = np.zeros(
        (config.ny, config.nx),
        dtype=np.float64,
    )

    processed_components = 0

    for geom in gdf_6933.geometry:
        for line in iter_lines(geom):
            if line.is_empty:
                continue

            processed_components += 1

            minx, miny, maxx, maxy = line.bounds

            ix_min = max(
                0,
                int(math.floor((minx - config.x_min) / config.cell_m)),
            )
            ix_max = min(
                config.nx - 1,
                int(math.floor((maxx - config.x_min) / config.cell_m)),
            )
            iy_min = max(
                0,
                int(math.floor((miny - config.y_min) / config.cell_m)),
            )
            iy_max = min(
                config.ny - 1,
                int(math.floor((maxy - config.y_min) / config.cell_m)),
            )

            if ix_min > ix_max or iy_min > iy_max:
                continue

            for iy in range(iy_min, iy_max + 1):
                y0 = config.y_min + iy * config.cell_m
                y1 = y0 + config.cell_m

                for ix in range(ix_min, ix_max + 1):
                    x0 = config.x_min + ix * config.cell_m
                    x1 = x0 + config.cell_m

                    clipped = line.intersection(
                        box(x0, y0, x1, y1)
                    )

                    if clipped.is_empty:
                        continue

                    clipped_wgs84 = shapely_transform(
                        FROM_EA.transform,
                        clipped,
                    )

                    segment_m = geodesic_geometry_length_m(
                        clipped_wgs84
                    )

                    if segment_m <= 0.0:
                        continue

                    presence[iy, ix] = 1
                    fault_length_m[iy, ix] += segment_m

    print(
        f"Processed projected fault line components: "
        f"{processed_components:,}"
    )

    return presence, fault_length_m / 1000.0


def add_fault_features(
    config,
    target_lon,
    target_lat,
    fault_sample_spacing_km,
    candidate_k,
):
    """
    Final fault predictors:
      1. mapped-fault presence in each equal-area grid cell;
      2. WGS84 geodesic distance from the cell centre to the nearest densely
         sampled location along a mapped fault trace;
      3. geodesically measured mapped-fault length per 1000 km² of equal-area
         cell support.
    """
    if not FAULT_FILE.exists():
        raise FileNotFoundError(
            f"Required harmonized fault file missing: {FAULT_FILE}"
        )

    if fault_sample_spacing_km <= 0:
        raise ValueError("fault_sample_spacing_km must be positive.")

    gdf = gpd.read_file(FAULT_FILE)

    if gdf.crs is None:
        raise RuntimeError(
            "GEM fault dataset has no CRS metadata; refusing to guess its CRS."
        )

    gdf_wgs84 = gdf.to_crs(epsg=4326)
    gdf_wgs84 = gdf_wgs84[
        gdf_wgs84.geometry.notna()
        & ~gdf_wgs84.geometry.is_empty
    ].copy()

    print(f"Usable fault features: {len(gdf_wgs84):,}")

    spacing_m = float(fault_sample_spacing_km) * 1000.0

    print(
        f"Densifying faults at <= {fault_sample_spacing_km:g} km "
        "geodesic spacing..."
    )

    samples = densify_fault_geometries(
        gdf_wgs84,
        spacing_m,
    )

    print(f"Dense fault samples: {len(samples):,}")

    print(
        "Computing WGS84 nearest-fault distances "
        "for all grid-cell centres..."
    )

    dist_km = (
        nearest_geodesic_distance_to_samples(
            target_lon.ravel(),
            target_lat.ravel(),
            samples[:, 0],
            samples[:, 1],
            k_candidates=candidate_k,
        )
        / 1000.0
    ).reshape(
        (config.ny, config.nx)
    )

    print(
        "Clipping faults to equal-area cells and computing "
        "geodesic fault lengths..."
    )

    presence, fault_length_km = (
        compute_geodesic_fault_presence_and_length(
            gdf_wgs84,
            config,
        )
    )

    # EPSG:6933 is equal-area, so the projected cell area is the intended
    # equal-area support. The side lengths are not interpreted as true ground
    # distances here.
    cell_area_km2 = (
        config.cell_m * config.cell_m / 1e6
    )

    length_density = (
        fault_length_km
        / cell_area_km2
        * 1000.0
    )

    print(
        f"Fault-containing cells: {int(presence.sum()):,}"
    )
    print(
        f"Total represented geodesic fault length: "
        f"{fault_length_km.sum():,.1f} km"
    )
    print(
        f"Maximum geodesic fault length in one cell: "
        f"{fault_length_km.max():.2f} km"
    )

    return {
        "gem_fault_presence": presence.astype(np.float32),
        "gem_fault_dist_km": dist_km.astype(np.float32),
        "gem_fault_length_density_km_per_1000km2": (
            length_density.astype(np.float32)
        ),
    }


# ============================================================
# GEODESIC GRID-SCALE TERRAIN SLOPE
# ============================================================

def compute_geodesic_grid_slope_deg(
    elevation_m,
    lon_grid,
    lat_grid,
):
    """
    Compute the slope of the harmonized grid-scale elevation surface.

    The elevation values themselves are unchanged. Only the horizontal
    denominators are corrected: central differences use WGS84 geodesic
    distances between the two neighbouring cell centres, with one-sided
    differences at the outer grid edges.
    """
    z = np.asarray(elevation_m, dtype=np.float64)
    lon = np.asarray(lon_grid, dtype=np.float64)
    lat = np.asarray(lat_grid, dtype=np.float64)

    if z.shape != lon.shape or z.shape != lat.shape or z.ndim != 2:
        raise ValueError(
            "elevation/lon/lat must be matching two-dimensional grids."
        )

    ny, nx = z.shape
    if ny < 2 or nx < 2:
        raise ValueError("Grid must contain at least two rows and columns.")

    dz_dx = np.full_like(z, np.nan)
    dz_dy = np.full_like(z, np.nan)

    # East-west derivative.
    if nx > 2:
        _, _, distance_x = GEOD.inv(
            lon[:, :-2],
            lat[:, :-2],
            lon[:, 2:],
            lat[:, 2:],
        )
        dz_dx[:, 1:-1] = (
            (z[:, 2:] - z[:, :-2])
            / np.abs(distance_x)
        )

    _, _, distance_x0 = GEOD.inv(
        lon[:, 0],
        lat[:, 0],
        lon[:, 1],
        lat[:, 1],
    )
    _, _, distance_x1 = GEOD.inv(
        lon[:, -2],
        lat[:, -2],
        lon[:, -1],
        lat[:, -1],
    )

    dz_dx[:, 0] = (
        (z[:, 1] - z[:, 0])
        / np.abs(distance_x0)
    )
    dz_dx[:, -1] = (
        (z[:, -1] - z[:, -2])
        / np.abs(distance_x1)
    )

    # North-south derivative.
    if ny > 2:
        _, _, distance_y = GEOD.inv(
            lon[:-2, :],
            lat[:-2, :],
            lon[2:, :],
            lat[2:, :],
        )
        dz_dy[1:-1, :] = (
            (z[2:, :] - z[:-2, :])
            / np.abs(distance_y)
        )

    _, _, distance_y0 = GEOD.inv(
        lon[0, :],
        lat[0, :],
        lon[1, :],
        lat[1, :],
    )
    _, _, distance_y1 = GEOD.inv(
        lon[-2, :],
        lat[-2, :],
        lon[-1, :],
        lat[-1, :],
    )

    dz_dy[0, :] = (
        (z[1, :] - z[0, :])
        / np.abs(distance_y0)
    )
    dz_dy[-1, :] = (
        (z[-1, :] - z[-2, :])
        / np.abs(distance_y1)
    )

    slope = np.rad2deg(
        np.arctan(
            np.hypot(dz_dx, dz_dy)
        )
    )

    slope[~np.isfinite(slope)] = np.nan
    return slope


# ============================================================
# QC
# ============================================================

def print_feature_summary(features):
    print("\n")
    print("=" * 88)
    print("GLOBAL GEODESIC FEATURE QC")
    print("=" * 88)

    for name, array in features.items():
        finite_mask = np.isfinite(array)
        finite = array[finite_mask]
        missing = int(np.count_nonzero(~finite_mask))
        missing_pct = missing / array.size * 100.0

        if finite.size:
            print(
                f"{name:<48} "
                f"finite={finite.size:>9,} "
                f"missing={missing:>9,} "
                f"({missing_pct:6.2f}%) "
                f"range={finite.min():.5g} to {finite.max():.5g}"
            )
        else:
            print(f"{name:<48} NO FINITE VALUES")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    config = get_grid_config(args.cell_km)
    ensure_grid_directories(config)

    if args.fault_sample_spacing_km <= 0:
        raise ValueError("--fault-sample-spacing-km must be positive.")

    if args.candidate_k < 1:
        raise ValueError("--candidate-k must be at least 1.")

    print("=" * 88)
    print(
        f"GLOBAL GEOPHYSICAL FEATURE PREPARATION V5 — "
        f"{config.cell_km} KM — GEODESIC DISTANCE CORRECTION"
    )
    print("=" * 88)

    print(
        f"Grid: {config.nx} x {config.ny} "
        f"= {config.nx * config.ny:,} cells"
    )
    print(
        f"Projected X extent: {config.x_min:.2f} to {config.x_max:.2f}"
    )
    print(
        f"Projected Y extent: {config.y_min:.2f} to {config.y_max:.2f}"
    )
    print(
        "EPSG:6933 is retained as the equal-area sampling grid; "
        "distance/length/slope denominators are evaluated geodesically."
    )

    xx, yy, target_lon, target_lat = build_grid_centers(config)

    features = {}

    # --------------------------------------------------------
    # 1. GEBCO elevation / bathymetry
    # --------------------------------------------------------
    print("\nLoading GEBCO elevation...")
    features["elevation_m"] = load_gebco_grid(config)

    # --------------------------------------------------------
    # 2. WGM2012 gravity fields
    # --------------------------------------------------------
    wgm_specs = [
        (WGM_FREEAIR, "freeair_mgal"),
        (WGM_BOUGUER, "bouguer_mgal"),
        (WGM_ISOSTATIC, "isostatic_mgal"),
    ]

    for path, name in wgm_specs:
        print(f"\nProcessing {name}...")
        key, value = add_from_grd(path, name, config)
        features[key] = value

    # --------------------------------------------------------
    # 3. Continuous coarser regular grids
    # --------------------------------------------------------
    xyz_specs = [
        (GEOID_UMG3, "geoid_umg3", 0, 1, 2, 1, 1.0),
        (SEDTHK, "sedthk_km", 0, 1, 2, 0, 1.0),
        (CRSTHK, "crsthk_km", 0, 1, 2, 0, 1.0),
    ]

    for (
        path,
        name,
        col_lon,
        col_lat,
        col_val,
        skiprows,
        scale,
    ) in xyz_specs:
        print(f"\nInterpolating {name}...")
        features[name] = load_xyz_interpolated(
            path,
            name,
            target_lon,
            target_lat,
            col_lon,
            col_lat,
            col_val,
            skiprows,
            scale,
        )

    # --------------------------------------------------------
    # 4. LithoRef18
    # --------------------------------------------------------
    print("\nProcessing LithoRef18...")
    add_lithoref_features(
        features,
        target_lon,
        target_lat,
    )

    # --------------------------------------------------------
    # 5. EMAG2
    # --------------------------------------------------------
    # Intentionally excluded from the streamlined primary 18-feature set
    # after feature-family ablation found negligible incremental contribution.

    # --------------------------------------------------------
    # 6. GEM active-fault predictors — geodesically corrected
    # --------------------------------------------------------
    print("\nProcessing GEM harmonized active faults...")
    features.update(
        add_fault_features(
            config=config,
            target_lon=target_lon,
            target_lat=target_lat,
            fault_sample_spacing_km=args.fault_sample_spacing_km,
            candidate_k=args.candidate_k,
        )
    )

    # --------------------------------------------------------
    # 7. Grid-scale slope — geodesically corrected
    # --------------------------------------------------------
    print(
        f"\nCalculating geodesic-neighbour "
        f"{config.cell_km} km grid-scale slope..."
    )

    features["terrain_slope_deg"] = (
        compute_geodesic_grid_slope_deg(
            features["elevation_m"].astype(np.float64),
            target_lon,
            target_lat,
        )
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # QC — exact final 18-feature schema
    # --------------------------------------------------------
    expected_features = expected_feature_names()

    actual_feature_names = set(features.keys())

    if actual_feature_names != expected_features:
        raise RuntimeError(
            "Unexpected final physical feature set. "
            f"Missing: {sorted(expected_features - actual_feature_names)}; "
            f"Extra: {sorted(actual_feature_names - expected_features)}"
        )

    print(
        f"Final physical feature count: "
        f"{len(actual_feature_names)}"
    )

    print_feature_summary(features)

    # --------------------------------------------------------
    # MASTER TABLE
    # --------------------------------------------------------
    output = pd.DataFrame(
        {
            "x_6933_m": xx.ravel().astype(np.float32),
            "y_6933_m": yy.ravel().astype(np.float32),
            "lon": target_lon.ravel().astype(np.float32),
            "lat": target_lat.ravel().astype(np.float32),
        }
    )

    expected_shape = (config.ny, config.nx)

    for name, array in features.items():
        if array.shape != expected_shape:
            raise RuntimeError(
                f"{name}: shape {array.shape} does not match {expected_shape}."
            )

        output[name] = array.ravel().astype(np.float32)

    # --------------------------------------------------------
    # CELL IDS
    # --------------------------------------------------------
    output["ix"] = np.tile(
        np.arange(config.nx, dtype=np.int32),
        config.ny,
    )

    output["iy"] = np.repeat(
        np.arange(config.ny, dtype=np.int32),
        config.nx,
    )

    output["cell_id"] = (
        output["iy"].astype(str)
        + "_"
        + output["ix"].astype(str)
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------
    expected_rows = config.nx * config.ny

    if len(output) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows:,} rows, got {len(output):,}."
        )

    if output["cell_id"].duplicated().any():
        raise RuntimeError("Duplicate cell_id values detected.")

    if output[["x_6933_m", "y_6933_m", "lon", "lat"]].isna().any().any():
        raise RuntimeError("Spatial metadata contains missing values.")

    final_feature_columns = [
        column
        for column in output.columns
        if column not in {
            "x_6933_m",
            "y_6933_m",
            "lon",
            "lat",
            "ix",
            "iy",
            "cell_id",
        }
    ]

    if len(final_feature_columns) != 18:
        raise RuntimeError(
            f"Expected 18 physical feature columns, "
            f"found {len(final_feature_columns)}."
        )

    # --------------------------------------------------------
    # SAVE — do not overwrite old projected-distance feature table
    # --------------------------------------------------------
    output_file = output_feature_file(config)

    output.to_csv(
        output_file,
        index=False,
    )

    print("\n")
    print("=" * 88)
    print("SAVED V5 GEODESIC-CORRECTED FEATURE TABLE")
    print("=" * 88)
    print(f"Output: {output_file}")
    print(f"Rows: {len(output):,}")
    print(f"Columns: {len(output.columns)}")
    print(f"Physical features: {len(final_feature_columns)}")
    print("Slope column: terrain_slope_deg")

    print("\nColumns:")
    for column in output.columns:
        print(f"  {column}")


if __name__ == "__main__":
    main()
