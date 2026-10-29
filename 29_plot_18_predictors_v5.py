# 29_plot_all_18_predictors_30km_v5.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pyproj import CRS, Transformer


# ============================================================================
# USER SETTINGS
# ============================================================================
PRIMARY_INPUT = (
    Path("data")
    / "grid_30km"
    / "model"
    / "model_dataset_cell_30km_v5.csv"
)

FALLBACK_INPUT = (
    Path("data")
    / "grid_30km"
    / "features"
    / "global_features_30km_v5.csv"
)

OUTPUT_DIR = Path("plots_v5") / "predictors"

DPI = 600

FIGSIZE = (12.8, 7.2)  # inches; preserve the same dimensions for all 18 maps
FONT_SIZE = 15
FONT_WEIGHT = "bold"
FONT_FAMILY = "DejaVu Sans"

LONGITUDE_TICKS = (-180, -120, -60, 0, 60, 120, 180)
LATITUDE_TICKS = (-60, -30, 0, 30, 60)

# Longitude labels go on the straight top/bottom edges. Latitude labels follow
# the curved left/right boundary at their true projected geographic positions.
COORDINATE_LABEL_PAD_PT = 8
MAP_AXES_RECT = (0.070, 0.200, 0.860, 0.735)

# 1.00 = the rectangular colorbar matches the flat-edge width.
COLORBAR_WIDTH_FACTOR = 1.00
COLORBAR_HEIGHT_IN = 0.20
COLORBAR_GAP_IN = 0.52  # map bottom to colorbar top, leaving room for longitude labels
COLORBAR_LABEL_PAD_PT = 9

# Robust limits affect COLORS ONLY. Every finite input cell is still plotted.
# Under/over-range cells take the end colors, but the colorbar stays rectangular
# (no extension arrows/tips).
# Set False (or pass --full-range) to cover the complete observed data range.
ROBUST_COLOR_LIMITS = True
LOW_PERCENTILE = 0.5
HIGH_PERCENTILE = 99.5
SIGNED_ABS_PERCENTILE = 99.0
POSITIVE_PERCENTILE = 99.5

# Optional explicit limits: (minimum, maximum, tick_increment).
# They remain linear. Uncomment a line only when a fixed scale is desired.
LIMIT_OVERRIDES = {
    # "elevation_m": (-8000.0, 6000.0, 2000.0),
    # "terrain_slope_deg": (0.0, 4.0, 1.0),
    # "bouguer_mgal": (-600.0, 600.0, 200.0),
    # "geoid_umg3": (-30.0, 30.0, 10.0),
}

COASTLINE_RESOLUTION = "50m"
COASTLINE_WIDTH = 0.55
COASTLINE_COLOR = "#4a4f54"
COUNTRY_BORDER_WIDTH = 0.42
COUNTRY_BORDER_COLOR = "#62676c"
COUNTRY_BORDER_ALPHA = 0.82
GRIDLINE_WIDTH = 0.65
GRIDLINE_ALPHA = 0.5
MISSING_COLOR = "#eeeeec"

# Display CRS only. The source/predictor CRS and sampling remain EPSG:6933.
GEOGRAPHIC_CRS = CRS.from_epsg(4326)
MODEL_CRS = CRS.from_epsg(6933)
ROBINSON_CRS = CRS.from_proj4(
    "+proj=robin +lon_0=0 +datum=WGS84 +units=m +no_defs"
)
TO_GEOGRAPHIC = Transformer.from_crs(MODEL_CRS, GEOGRAPHIC_CRS, always_xy=True)
TO_MODEL = Transformer.from_crs(GEOGRAPHIC_CRS, MODEL_CRS, always_xy=True)
TO_ROBINSON = Transformer.from_crs(GEOGRAPHIC_CRS, ROBINSON_CRS, always_xy=True)


@dataclass(frozen=True)
class Predictor:
    feature: str
    title: str
    label: str
    mode: str
    palette: str = "viridis"

    @property
    def filename(self) -> str:
        return f"{self.title}.png"

PREDICTORS = (
    Predictor(
        "elevation_m",
        "GEBCO 2026 Global Topography and Bathymetry",
        "Elevation / bathymetry (m)",
        "elevation",
    ),

    Predictor(
        "terrain_slope_deg",
        "GEBCO 2026 Derived Terrain Slope",
        "Slope (degrees)",
        "positive",
        "slope",
    ),

    Predictor(
        "freeair_mgal",
        "WGM2012 Free-Air Gravity Anomaly",
        "Free-air gravity anomaly (mGal)",
        "signed",
    ),

    Predictor(
        "bouguer_mgal",
        "WGM2012 Bouguer Gravity Anomaly",
        "Bouguer gravity anomaly (mGal)",
        "signed",
        "bouguer",
    ),

    Predictor(
        "isostatic_mgal",
        "WGM2012 Isostatic Gravity Anomaly",
        "Isostatic gravity anomaly (mGal)",
        "signed",
    ),

    Predictor(
        "geoid_umg3",
        "UMG3 Upper-Mantle Geoid",
        "UMG3 upper-mantle geoid (m)",
        "signed",
    ),

    Predictor(
        "sedthk_km",
        "CRUST1.0 Sediment Thickness",
        "Sediment thickness (km)",
        "positive",
        "sediment",
    ),

    Predictor(
        "crsthk_km",
        "CRUST1.0 Total Crustal Thickness",
        "Total crustal thickness (km)",
        "sequential",
        "viridis",
    ),

    Predictor(
        "lithoref_moho_depth_km",
        "LithoRef18 Moho Depth",
        "Moho depth (km)",
        "sequential",
        "viridis",
    ),

    Predictor(
        "lithoref_lab_depth_km",
        "LithoRef18 Lithosphere–Asthenosphere Boundary Depth",
        "Lithosphere–asthenosphere boundary depth (km)",
        "sequential",
        "viridis",
    ),

    Predictor(
        "lithoref_rho_c",
        "LithoRef18 Average Crustal Density",
        "Average crustal density (kg m⁻³)",
        "sequential",
        "viridis",
    ),

    Predictor(
        "lithoref_rho_l",
        "LithoRef18 Lithospheric Mantle Density",
        "Lithospheric mantle density (kg m⁻³)",
        "sequential",
        "viridis",
    ),

    Predictor(
        "lithoref_rho_sl",
        "LithoRef18 Sublithospheric Mantle Density",
        "Sublithospheric mantle density (kg m⁻³)",
        "sequential",
        "viridis",
    ),

    Predictor(
        "lithoref_geoid_m",
        "LithoRef18 Geoid Height",
        "LithoRef18 geoid height (m)",
        "signed",
    ),

    Predictor(
        "lithoref_fa_mgal",
        "LithoRef18 Free-Air Gravity Anomaly",
        "LithoRef18 free-air gravity anomaly (mGal)",
        "signed",
    ),

    Predictor(
        "gem_fault_presence",
        "GEM Active Fault Presence",
        "Mapped active-fault presence",
        "binary",
    ),

    Predictor(
        "gem_fault_dist_km",
        "GEM Distance to Nearest Mapped Fault Trace",
        "Geodesic distance to nearest mapped fault trace (km)",
        "positive",
        "fault_distance",
    ),

    Predictor(
        "gem_fault_length_density_km_per_1000km2",
        "GEM Active Fault Length Density",
        "Mapped fault length (km per 1,000 km²)",
        "positive",
        "fault_density",
    ),
)

assert len(PREDICTORS) == 18
assert len({p.feature for p in PREDICTORS}) == 18

def configure_fonts() -> None:
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "font.weight": FONT_WEIGHT,
        "axes.labelsize": FONT_SIZE,
        "axes.labelweight": FONT_WEIGHT,
        "axes.titlesize": FONT_SIZE,
        "axes.titleweight": FONT_WEIGHT,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "axes.unicode_minus": True,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


# ============================================================================
# LOAD THE ACTUAL MODEL GRID; DO NOT IMPUTE OR SMOOTH PREDICTORS FOR THE MAPS
# ============================================================================
def choose_input(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit is not None else [PRIMARY_INPUT, FALLBACK_INPUT]
    for path in candidates:
        if path is not None and path.is_file():
            return path
    raise FileNotFoundError(
        "No input CSV found. Run in the project directory or use --input. Tried:\n"
        + "\n".join(str(p.resolve()) for p in candidates if p is not None)
    )


def numeric_column(df: pd.DataFrame, name: str) -> np.ndarray:
    # Non-numeric strings are errors, not silently converted into missing data.
    result = pd.to_numeric(df[name], errors="raise").to_numpy(dtype=np.float64)
    if np.isinf(result).any():
        raise ValueError(f"{name} contains infinity; inspect the source CSV.")
    return result


def load_grid(path: Path, selected: tuple[Predictor, ...]):
    required = ["ix", "iy", "x_6933_m", "y_6933_m"] + [p.feature for p in selected]
    df = pd.read_csv(path, usecols=required)
    for name in ("ix", "iy"):
        index = numeric_column(df, name)
        if not np.isfinite(index).all() or np.any(index < 0) or not np.all(index == np.floor(index)):
            raise ValueError(f"{name} must contain finite, non-negative integer grid indices.")
        df[name] = index.astype(np.int64)
    if df[["ix", "iy"]].duplicated().any():
        raise ValueError("Duplicate ix/iy cells: cannot reconstruct an unambiguous grid.")
    df = df.sort_values(["iy", "ix"]).reset_index(drop=True)
    nx, ny = int(df.ix.max()) + 1, int(df.iy.max()) + 1
    if nx < 2 or ny < 2 or len(df) != nx * ny:
        raise ValueError("The CSV must contain a complete rectangular grid, starting at ix=iy=0.")
    if not np.array_equal(df.ix.to_numpy(), np.tile(np.arange(nx), ny)):
        raise ValueError("Missing or inconsistent column indices.")
    if not np.array_equal(df.iy.to_numpy(), np.repeat(np.arange(ny), nx)):
        raise ValueError("Missing or inconsistent row indices.")

    xc = numeric_column(df, "x_6933_m").reshape(ny, nx)
    yc = numeric_column(df, "y_6933_m").reshape(ny, nx)
    if not np.isfinite(xc).all() or not np.isfinite(yc).all():
        raise ValueError("Missing projected cell-center coordinates.")
    x, y = xc[0], yc[:, 0]
    if not np.allclose(xc, x[None, :], rtol=0, atol=2.0):
        raise ValueError("Easting does not depend only on ix.")
    if not np.allclose(yc, y[:, None], rtol=0, atol=2.0):
        raise ValueError("Northing does not depend only on iy.")
    if not np.allclose(np.diff(x), 30000.0, rtol=0, atol=2.0):
        raise ValueError("The input is not the expected 30,000 m easting grid.")
    if not np.allclose(np.diff(y), 30000.0, rtol=0, atol=2.0):
        raise ValueError("The input is not the expected 30,000 m northing grid.")

    # Median origin estimates absorb sub-metre Float32 coordinate rounding.
    x0 = float(np.median(x - (np.arange(nx) + 0.5) * 30000.0))
    y0 = float(np.median(y - (np.arange(ny) + 0.5) * 30000.0))
    xe = x0 + np.arange(nx + 1) * 30000.0
    ye = y0 + np.arange(ny + 1) * 30000.0
    metadata = {"nx": nx, "ny": ny, "cell_size_m": 30000.0,
                "x_min": float(xe[0]), "x_max": float(xe[-1]),
                "y_min": float(ye[0]), "y_max": float(ye[-1])}
    print(f"Grid: {nx} x {ny} = {len(df):,} cells; 30 km projected spacing")
    if (nx, ny) != (1158, 488):
        warnings.warn("Grid dimensions differ from the supplied 1158 x 488 primary grid.")
    return df, xe, ye, metadata


def projected_cell_corners(xe: np.ndarray, ye: np.ndarray):
    """Use actual cell corners, clipping only overhanging edges to the globe.

    The original eastern edge extends slightly past 180 degrees. Clip that small
    overhang for display, rather than wrapping a long polygon across the map.
    Do NOT extrapolate the northern/southern grid to fill unobserved polar caps.
    """
    xw, _ = TO_MODEL.transform(-180.0, 0.0)
    xeast, _ = TO_MODEL.transform(180.0, 0.0)
    _, ys = TO_MODEL.transform(0.0, -90.0)
    _, yn = TO_MODEL.transform(0.0, 90.0)
    x_edges = np.clip(xe, xw, xeast)
    y_edges = np.clip(ye, ys, yn)
    lon, _ = TO_GEOGRAPHIC.transform(x_edges, np.zeros_like(x_edges))
    _, lat = TO_GEOGRAPHIC.transform(np.zeros_like(y_edges), y_edges)
    lon = np.clip(lon, -180.0, 180.0)
    lat = np.clip(lat, -90.0, 90.0)
    xx, yy = TO_ROBINSON.transform(*np.meshgrid(lon, lat))
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("Non-finite projected cell corners; check grid geometry.")
    return xx, yy


# ============================================================================
# COASTLINES AND PROJECTION GEOMETRY
# ============================================================================
def geometry_lines(geometry) -> Iterator[np.ndarray]:
    if geometry is None or geometry.is_empty:
        return
    kind = geometry.geom_type
    if kind in ("LineString", "LinearRing"):
        yield np.asarray(geometry.coords)[:, :2]
    elif kind == "Polygon":
        yield np.asarray(geometry.exterior.coords)[:, :2]
        for ring in geometry.interiors:
            yield np.asarray(ring.coords)[:, :2]
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from geometry_lines(part)


def load_coastlines(local_path: Path | None = None) -> list[np.ndarray]:
    if local_path is not None:
        import geopandas as gpd
        gdf = gpd.read_file(local_path)
        if gdf.crs is None:
            raise ValueError("The supplied coastline file has no CRS.")
        geometries = list(gdf.to_crs(4326).geometry)
    else:
        try:
            from cartopy.io import shapereader
        except ImportError as exc:
            raise ImportError(
                "Install Cartopy: conda install -c conda-forge cartopy\n"
                "Or supply --coastline path/to/a/local/coastline.shp (requires geopandas)."
            ) from exc
        try:
            coastfile = shapereader.natural_earth(
                resolution=COASTLINE_RESOLUTION, category="physical", name="coastline"
            )
            reader = shapereader.Reader(coastfile)
            try:
                geometries = list(reader.geometries())
            finally:
                reader.close()
        except Exception as exc:
            raise RuntimeError(
                "Could not load the Natural Earth coastline. Allow its first download, "
                "reuse your Cartopy data cache, or pass --coastline with a local file."
            ) from exc

    lines = []
    for geom in geometries:
        for coords in geometry_lines(geom):
            # Prevent a line segment from joining opposite sides of the dateline.
            cuts = np.where(np.abs(np.diff(coords[:, 0])) > 180.0)[0] + 1
            for part in np.split(coords, cuts):
                if len(part) < 2:
                    continue
                px, py = TO_ROBINSON.transform(part[:, 0], part[:, 1])
                segment = np.column_stack((px, py))
                if np.isfinite(segment).all():
                    lines.append(segment)
    if not lines:
        raise ValueError("No usable coastline segments were found.")
    return lines


def load_country_borders() -> list[np.ndarray]:
    """Load Natural Earth international land boundaries at publication resolution.

    Coastlines alone do not contain internal country boundaries.  This layer is
    intentionally separate so coastlines can remain slightly stronger than
    international borders.
    """
    try:
        from cartopy.io import shapereader
        borderfile = shapereader.natural_earth(
            resolution=COASTLINE_RESOLUTION,
            category="cultural",
            name="admin_0_boundary_lines_land",
        )
        reader = shapereader.Reader(borderfile)
        try:
            geometries = list(reader.geometries())
        finally:
            reader.close()
    except Exception as exc:
        warnings.warn(f"Could not load Natural Earth country borders: {exc}")
        return []

    lines: list[np.ndarray] = []
    for geom in geometries:
        for coords in geometry_lines(geom):
            cuts = np.where(np.abs(np.diff(coords[:, 0])) > 180.0)[0] + 1
            for part in np.split(coords, cuts):
                if len(part) < 2:
                    continue
                px, py = TO_ROBINSON.transform(part[:, 0], part[:, 1])
                segment = np.column_stack((px, py))
                if np.isfinite(segment).all():
                    lines.append(segment)
    return lines


def robinson_boundary() -> np.ndarray:
    lat = np.linspace(-90.0, 90.0, 721)
    xr, yr = TO_ROBINSON.transform(np.full(lat.size, 180.0), lat)
    xl, yl = TO_ROBINSON.transform(np.full(lat.size, -180.0), lat[::-1])
    return np.vstack((np.column_stack((xr, yr)), np.column_stack((xl, yl))))


def lon_label(value: int) -> str:
    return "0°" if value == 0 else f"{abs(value)}°{'W' if value < 0 else 'E'}"


def lat_label(value: int) -> str:
    return "0°" if value == 0 else f"{abs(value)}°{'S' if value < 0 else 'N'}"


def reference_line_style(spec: Predictor | None = None) -> dict[str, float | str]:
    """Return map-reference line styling, stronger for sparse fault-density maps."""
    style = {
        "coastline_width": COASTLINE_WIDTH,
        "coastline_color": COASTLINE_COLOR,
        "coastline_alpha": 0.92,
        "country_border_width": COUNTRY_BORDER_WIDTH,
        "country_border_color": COUNTRY_BORDER_COLOR,
        "country_border_alpha": COUNTRY_BORDER_ALPHA,
    }
    if spec is not None and (spec.palette == "fault_density" or spec.mode == "binary"):
        # Sparse fault maps are dominated by zero-valued cells.  Stronger map
        # references keep the continents readable without changing fault data.
        style.update({
            "coastline_width": 0.88,
            "coastline_color": "#354149",
            "coastline_alpha": 0.99,
            "country_border_width": 0.60,
            "country_border_color": "#566169",
            "country_border_alpha": 0.96,
        })
    return style


def setup_map(ax, coastline_segments: list[np.ndarray],
              country_border_segments: list[np.ndarray] | None = None,
              line_style: dict[str, float | str] | None = None):
    line_style = reference_line_style() if line_style is None else line_style
    boundary = robinson_boundary()
    ax.set_xlim(float(boundary[:, 0].min()), float(boundary[:, 0].max()))
    ax.set_ylim(float(boundary[:, 1].min()), float(boundary[:, 1].max()))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_autoscale_on(False)
    clip = Polygon(boundary, closed=True, facecolor=MISSING_COLOR,
                   edgecolor="none", linewidth=0, zorder=0)
    ax.add_patch(clip)
    # Do not draw a Robinson perimeter stroke.  Its straight polar segment is
    # visually mistaken for a data/grid line above 60°N/S.  The raster/clip
    # silhouette already defines the map edge cleanly.

    grid_lines = []
    latitude = np.linspace(-90.0, 90.0, 721)
    longitude = np.linspace(-180.0, 180.0, 1441)
    for value in LONGITUDE_TICKS[1:-1]:
        x, y = TO_ROBINSON.transform(np.full(latitude.size, value), latitude)
        grid_lines.append(np.column_stack((x, y)))
    for value in LATITUDE_TICKS:
        x, y = TO_ROBINSON.transform(longitude, np.full(longitude.size, value))
        grid_lines.append(np.column_stack((x, y)))
    graticule = LineCollection(grid_lines, colors=COASTLINE_COLOR,
                              linewidths=GRIDLINE_WIDTH, alpha=GRIDLINE_ALPHA,
                              linestyles=(0, (2, 3)), zorder=3)
    ax.add_collection(graticule)
    graticule.set_clip_path(clip)

    coast = LineCollection(
        coastline_segments, colors=line_style["coastline_color"],
        linewidths=line_style["coastline_width"],
        alpha=line_style["coastline_alpha"], zorder=4.2,
    )
    ax.add_collection(coast)
    coast.set_clip_path(clip)

    if country_border_segments:
        borders = LineCollection(
            country_border_segments, colors=line_style["country_border_color"],
            linewidths=line_style["country_border_width"],
            alpha=line_style["country_border_alpha"], zorder=4.1,
        )
        ax.add_collection(borders)
        borders.set_clip_path(clip)

    # Manual projected annotations guarantee ALL labels are drawn on Robinson.
    # Automatic gridliner overlap suppression can omit top/bottom labels.
    style = dict(fontsize=FONT_SIZE, fontweight=FONT_WEIGHT, color="#202020",
                 annotation_clip=False, clip_on=False, zorder=8)
    for longitude in LONGITUDE_TICKS:
        for latitude, sign in ((90.0, 1), (-90.0, -1)):
            anchor = TO_ROBINSON.transform(longitude, latitude)
            ax.annotate(lon_label(longitude), xy=anchor, xycoords="data",
                        xytext=(0, sign * COORDINATE_LABEL_PAD_PT), textcoords="offset points",
                        ha="center", va="bottom" if sign > 0 else "top", **style)
    for latitude in LATITUDE_TICKS:
        for longitude, sign in ((-180.0, -1), (180.0, 1)):
            anchor = TO_ROBINSON.transform(longitude, latitude)
            ax.annotate(lat_label(latitude), xy=anchor, xycoords="data",
                        xytext=(sign * COORDINATE_LABEL_PAD_PT, 0), textcoords="offset points",
                        ha="right" if sign < 0 else "left", va="center", **style)
    return clip


# ============================================================================
# LINEAR SCALES WITH EVENLY SPACED NUMERIC TICKS
# ============================================================================
def build_scale(values: np.ndarray, spec: Predictor, robust: bool):
    if spec.mode == "binary":
        if not np.isin(values, [0, 1]).all():
            raise ValueError(f"{spec.feature}: fault presence must contain only 0/1 or missing.")
        return BoundaryNorm([-0.5, 0.5, 1.5], 2), np.array([0.0, 1.0]), "neither"

    if spec.feature in LIMIT_OVERRIDES:
        low, high, step = LIMIT_OVERRIDES[spec.feature]
        if not (np.isfinite([low, high, step]).all() and high > low and step > 0):
            raise ValueError(f"Invalid limits for {spec.feature}.")
        ticks = np.arange(np.ceil(low / step) * step, high + step * 1e-8, step)
    else:
        if spec.mode == "signed":
            limit = float(np.percentile(np.abs(values), SIGNED_ABS_PERCENTILE)
                          if robust else np.max(np.abs(values)))
            if limit <= 0:
                limit = 1.0
            half_ticks = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(0, limit)
            step = float(half_ticks[1] - half_ticks[0])
            high = float(half_ticks[-1])
            low = -high
            ticks = np.arange(low, high + step * 0.1, step)
        else:
            if spec.mode == "positive":
                if values.min() < -1e-8:
                    raise ValueError(f"Unexpected negative value in {spec.feature}; check preprocessing.")
                low = 0.0
                # Use positive cells for sparse fields, rather than letting zeros
                # collapse the fault-density upper percentile to zero.
                pool = values[values > 0]
                if pool.size == 0:
                    high = 1.0
                else:
                    high = float(np.percentile(pool, POSITIVE_PERCENTILE)
                                 if robust else np.max(pool))
            elif spec.mode == "elevation":
                low, high = (np.percentile(values, [0.1, 99.9]) if robust
                             else (values.min(), values.max()))
                low, high = min(float(low), -1.0), max(float(high), 1.0)
            else:
                low, high = (np.percentile(values, [LOW_PERCENTILE, HIGH_PERCENTILE])
                             if robust else (values.min(), values.max()))
            if high <= low or np.isclose(high, low):
                padding = max(abs(float(low)) * 0.01, 0.5)
                low, high = float(low) - padding, float(high) + padding
                if spec.mode == "positive":
                    low = 0.0
            ticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(low, high)
            low, high = float(ticks[0]), float(ticks[-1])

    if len(ticks) < 2:
        raise ValueError(f"Fewer than two colorbar ticks for {spec.feature}.")
    if not np.allclose(np.diff(ticks), np.diff(ticks)[0]):
        raise ValueError("Numeric colorbar ticks are not evenly spaced.")
    # Keep robust clipping for the map colors, but never draw colorbar arrows.
    # Values outside [low, high] simply use the endpoint colors.
    extend = "neither"
    return Normalize(vmin=low, vmax=high, clip=False), ticks, extend


def make_palette(spec: Predictor, norm):
    """Return publication-friendly palettes tuned to each predictor family.

    The numeric normalization and colorbar remain linear.  Skewed predictors
    get colormaps with more perceptual contrast where their values are dense,
    so the map becomes readable without changing, smoothing, or transforming
    the underlying predictor values.
    """
    if spec.mode == "binary":
        # Use a visible neutral background rather than near-white so land/coast
        # context remains legible in this very sparse binary predictor.
        cmap = ListedColormap(["#e8e7e1", "#8b1e3f"], name="mapped_fault_presence")

    elif spec.mode == "elevation":
        sea_level = float((0.0 - norm.vmin) / (norm.vmax - norm.vmin))
        if not 0 < sea_level < 1:
            raise ValueError("Elevation display limits must straddle zero.")
        ocean = [
            "#08294b", "#0d4f78", "#167ba0", "#47a9b8",
            "#9dd5cb", "#d9eee2",
        ]
        land = [
            "#6f9258", "#91a867", "#b6ad73", "#c9af78",
            "#d7c395", "#ead9b5", "#f4ead5", "#fbfaf4",
        ]
        ocean_stops = np.linspace(0.0, sea_level - 1e-6, len(ocean))
        land_stops = np.linspace(sea_level, 1.0, len(land))
        stops = [(float(t), c) for t, c in zip(ocean_stops, ocean)]
        stops += [(float(t), c) for t, c in zip(land_stops, land)]
        cmap = LinearSegmentedColormap.from_list(
            "publication_topography_bathymetry", stops, N=2048
        )

    elif spec.palette == "bouguer":
        # Keep ±~300 mGal visually quiet; reserve strong blue/red for the tails.
        # This preserves the strong Tibetan negative anomaly without letting
        # broad moderate oceanic anomalies dominate the whole figure.
        cmap = LinearSegmentedColormap.from_list(
            "publication_bouguer",
            [
                (0.00, "#244a8f"),
                (0.12, "#5f8fc4"),
                (0.25, "#cbddeb"),
                (0.38, "#edf3f5"),
                (0.50, "#fbfaf7"),
                (0.62, "#f7ede8"),
                (0.75, "#efc9bd"),
                (0.88, "#d87561"),
                (1.00, "#982f43"),
            ],
            N=1024,
        )

    elif spec.mode == "signed":
        cmap = LinearSegmentedColormap.from_list(
            "publication_diverging",
            [
                (0.00, "#2b4c9a"),
                (0.22, "#6997c9"),
                (0.42, "#d2e0ec"),
                (0.50, "#faf9f6"),
                (0.58, "#efd8ce"),
                (0.78, "#cf7866"),
                (1.00, "#8f2d3d"),
            ],
            N=1024,
        )

    elif spec.palette == "warm":
        # Backward-compatible palette for custom/external use.
        cmap = LinearSegmentedColormap.from_list(
            "publication_warm",
            [
                (0.00, "#fffdf5"),
                (0.10, "#fff0b8"),
                (0.32, "#fdd276"),
                (0.58, "#f79045"),
                (0.80, "#df4b2e"),
                (1.00, "#b30000"),
            ],
            N=1024,
        )

    elif spec.palette == "slope":
        # Low-relief regions are light instead of dark purple.  Steep terrain
        # progresses through teal/green into restrained ochre-orange.
        cmap = LinearSegmentedColormap.from_list(
            "publication_slope",
            [
                (0.00, "#f8fbf8"),
                (0.07, "#e5f2eb"),
                (0.16, "#c8e6d8"),
                (0.32, "#93cfbc"),
                (0.52, "#56b09f"),
                (0.70, "#2f8f89"),
                (0.86, "#d7c35a"),
                (1.00, "#d96b45"),
            ],
            N=1024,
        )

    elif spec.palette == "sediment":
        # Thin sediment is almost neutral; thick basins become teal and then blue.
        cmap = LinearSegmentedColormap.from_list(
            "publication_sediment",
            [
                (0.00, "#fbfcf8"),
                (0.07, "#e8f4e4"),
                (0.16, "#cae7cf"),
                (0.32, "#96d1bf"),
                (0.52, "#5ab5ad"),
                (0.70, "#3b969f"),
                (0.86, "#3f7597"),
                (1.00, "#3f5f82"),
            ],
            N=1024,
        )

    elif spec.palette == "fault_distance":
        # Keep the entire scale relatively light.  Near-fault cells are cream,
        # while far-field cells reach only a medium blue-teal rather than navy.
        cmap = LinearSegmentedColormap.from_list(
            "publication_fault_distance",
            [
                (0.00, "#fffdf7"),
                (0.10, "#edf4e8"),
                (0.25, "#d3e8dc"),
                (0.45, "#a8d4ca"),
                (0.65, "#77b9b7"),
                (0.82, "#5a9aa9"),
                (1.00, "#4d7f9a"),
            ],
            N=1024,
        )

    elif spec.palette == "fault_density":
        # Zero occupies a neutral background.  Even very small positive values
        # immediately become visible, making the global fault network readable.
        cmap = LinearSegmentedColormap.from_list(
            "publication_fault_density",
            [
                # Zero is intentionally light grey-beige, not near-white.
                # That makes the globe/map context visible while preserving a
                # strong visual jump as soon as fault length becomes positive.
                (0.000, "#e8e7e1"),
                (0.010, "#f9d77f"),
                (0.020, "#f7ba55"),
                (0.050, "#f28a3b"),
                (0.120, "#e95d2f"),
                (0.300, "#d43b25"),
                (0.600, "#a91f19"),
                (1.000, "#781016"),
            ],
            N=1024,
        )

    else:
        base = plt.get_cmap("viridis")
        cmap = LinearSegmentedColormap.from_list(
            "publication_viridis",
            [base(x) for x in np.linspace(0.03, 0.97, 256)],
            N=1024,
        )

    cmap.set_bad(MISSING_COLOR)
    cmap.set_under(cmap(0.0))
    cmap.set_over(cmap(1.0))
    return cmap


def tick_formatter(ticks: np.ndarray) -> FuncFormatter:
    step = float(abs(ticks[1] - ticks[0]))
    decimals = 0
    while decimals < 6 and not np.isclose(step * 10**decimals, round(step * 10**decimals)):
        decimals += 1
    def format_value(value, position):
        if abs(value) < step * 1e-8:
            value = 0.0
        return f"{value:.{decimals}f}".replace("-", "−")
    return FuncFormatter(format_value)


def add_short_colorbar(fig, ax, mesh, spec: Predictor, ticks, extend):
    ax.apply_aspect()
    left_corner = TO_ROBINSON.transform(-180.0, -90.0)
    right_corner = TO_ROBINSON.transform(180.0, -90.0)
    left = fig.transFigure.inverted().transform(ax.transData.transform(left_corner))[0]
    right = fig.transFigure.inverted().transform(ax.transData.transform(right_corner))[0]
    width = (right - left) * COLORBAR_WIDTH_FACTOR
    center = (left + right) / 2.0
    bottom = ax.get_position().y0 - (COLORBAR_GAP_IN + COLORBAR_HEIGHT_IN) / FIGSIZE[1]
    cax = fig.add_axes([center - width / 2, bottom, width, COLORBAR_HEIGHT_IN / FIGSIZE[1]])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=ticks,
                      extend="neither")
    if spec.mode == "binary":
        cb.set_ticklabels(["Absent (0)", "Present (1)"])
    else:
        cb.ax.xaxis.set_major_formatter(tick_formatter(ticks))
    cb.minorticks_off()
    cb.ax.tick_params(axis="x", labelsize=FONT_SIZE, width=0.7, length=4, pad=5)
    cb.outline.set_linewidth(0.65)
    cb.set_label(spec.label, fontsize=FONT_SIZE, fontweight=FONT_WEIGHT,
                 labelpad=COLORBAR_LABEL_PAD_PT)
    for text in cb.ax.get_xticklabels():
        text.set_fontsize(FONT_SIZE)
        text.set_fontweight(FONT_WEIGHT)
    cb.ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE)
    cb.ax.xaxis.get_offset_text().set_fontweight(FONT_WEIGHT)
    return cb


# ============================================================================
# DRAW AND EXPORT
# ============================================================================
def plot_predictor(array, corner_x, corner_y, coastlines, country_borders,
                   spec, output_dir, dpi, robust):
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"{spec.feature}: all cells are missing.")
    norm, ticks, extend = build_scale(finite, spec, robust)
    cmap = make_palette(spec, norm)
    fig = plt.figure(figsize=FIGSIZE)
    try:
        ax = fig.add_axes(MAP_AXES_RECT)
        clip = setup_map(ax, coastlines, country_borders, reference_line_style(spec))
        mesh = ax.pcolormesh(corner_x, corner_y, np.ma.masked_invalid(array),
                             cmap=cmap, norm=norm, shading="flat", edgecolors="none",
                             linewidth=0, antialiased=False, rasterized=True, zorder=1)
        mesh.set_clip_path(clip)

        # Titles deliberately remain OFF. Use the filename in the paper caption.
        # ax.set_title(spec.title, fontsize=FONT_SIZE, fontweight=FONT_WEIGHT, pad=10)
        # fig.suptitle(...)

        add_short_colorbar(fig, ax, mesh, spec, ticks, extend)
        path = output_dir / spec.filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.16, facecolor="white")
    finally:
        plt.close(fig)

    below = int(np.count_nonzero(finite < norm.vmin)) if spec.mode != "binary" else 0
    above = int(np.count_nonzero(finite > norm.vmax)) if spec.mode != "binary" else 0
    print(f"Saved: {path.name}")
    print(f"  data [{finite.min():.6g}, {finite.max():.6g}]; "
          f"colors [{norm.vmin:.6g}, {norm.vmax:.6g}]; "
          f"missing={array.size - finite.size:,}; saturated={below + above:,}")
    return {
        "feature": spec.feature, "title": spec.title, "filename": spec.filename, "colorbar_label": spec.label,
        "grid_cell_size_m": 30000, "finite_cells": int(finite.size),
        "missing_cells": int(array.size - finite.size),
        "data_min": float(finite.min()), "data_max": float(finite.max()),
        "display_min": float(norm.vmin), "display_max": float(norm.vmax),
        "cells_below_display_min": below, "cells_above_display_max": above,
        "saturated_percent_of_finite": float(100 * (below + above) / finite.size),
        "colorbar_ticks": "|".join(f"{v:g}" for v in ticks),
        "numeric_scale": "categorical" if spec.mode == "binary" else "linear",
        "colorbar_extensions": extend,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="Override the model/global-feature CSV path.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--full-range", action="store_true", help="Use complete data ranges instead of robust limits.")
    parser.add_argument("--only", nargs="+", choices=[p.feature for p in PREDICTORS],
                        help="Optional quick check of selected features; default plots all 18.")
    parser.add_argument("--coastline", type=Path, help="Optional local coastline vector file in a declared CRS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("DPI must be positive.")
    configure_fonts()
    selected = tuple(p for p in PREDICTORS if args.only is None or p.feature in args.only)
    path = choose_input(args.input)
    print(f"Input: {path.resolve()}")
    args.output.mkdir(parents=True, exist_ok=True)
    df, xe, ye, grid_info = load_grid(path, selected)
    corner_x, corner_y = projected_cell_corners(xe, ye)
    coastlines = load_coastlines(args.coastline)
    country_borders = load_country_borders()
    robust = ROBUST_COLOR_LIMITS and not args.full_range
    manifest = []
    manifest_name = "predictor_plot_manifest_publication.csv" if args.only is None else "predictor_plot_manifest_publication_subset.csv"
    for spec in selected:
        array = numeric_column(df, spec.feature).reshape(grid_info["ny"], grid_info["nx"])
        manifest.append(plot_predictor(array, corner_x, corner_y, coastlines, country_borders,
                                       spec, args.output, args.dpi, robust))
        # Save after every map so partial progress is retained if a later map fails.
        pd.DataFrame(manifest).to_csv(args.output / manifest_name, index=False)
    config = {
        "input": str(path.resolve()), "output": str(args.output.resolve()),
        "grid": grid_info, "font_size_pt": FONT_SIZE, "font_weight": FONT_WEIGHT,
        "longitude_labels": LONGITUDE_TICKS, "latitude_labels": LATITUDE_TICKS,
        "titles_enabled": False, "filenames_derived_from_titles": True, "robust_display_limits": robust,
        "source_crs": "EPSG:6933", "display_crs": ROBINSON_CRS.to_string(),
        "rendering": "flat-colored projected cell-corner mesh; no smoothing",
        "predictors_plotted": [p.feature for p in selected],
        "note": "Grey means missing or outside model coverage, not zero."
    }
    config_name = "predictor_plot_config_publication.json" if args.only is None else "predictor_plot_config_publication_subset.json"
    (args.output / config_name).write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nDone: {len(manifest)} PNG maps in {args.output.resolve()}")
    print("Same-name PNGs replace earlier versions. Keep only filenames listed in the new manifest.")
    print("Caption note: 30 km equal-area predictor cells displayed in Robinson projection; "
          "grey denotes missing/outside-domain values; robust-limit saturation uses endpoint colours without colorbar arrows.")


if __name__ == "__main__":
    main()
