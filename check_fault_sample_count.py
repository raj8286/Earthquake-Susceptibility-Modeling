# check_fault_sample_count.py
import importlib.util
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Geod

SCRIPT = Path("02_prepare_global_features_grid_v5.py")
FAULTS = Path("gem_active_faults_harmonized.geojson")
SPACING_KM = 2.0

# Load your pipeline module (filename starts with a digit, so use importlib)
sys.path.insert(0, str(SCRIPT.resolve().parent))
spec = importlib.util.spec_from_file_location("prep", SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

geod = Geod(ellps="WGS84")
spacing_m = SPACING_KM * 1000.0

# Load faults exactly as the pipeline does
gdf = gpd.read_file(FAULTS).to_crs(epsg=4326)
gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
print(f"Fault features (rows):           {len(gdf):,}")  # expect 13,696

# ---- A. Count produced by the pipeline -------------------------------
samples = prep.densify_fault_geometries(gdf, spacing_m)
n_pipeline = len(samples)

# ---- B. Independent count from geometry ------------------------------
def lines_of(geom):
    if geom.geom_type == "LineString":
        yield geom
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            yield from lines_of(g)

n_lines = 0          # LineString components
n_segments = 0       # vertex-to-vertex segments
total_len_m = 0.0    # total geodesic length
expected = 0         # predicted number of sample points

for geom in gdf.geometry:
    for line in lines_of(geom):
        coords = np.asarray(line.coords)[:, :2]
        if len(coords) == 0:
            continue
        n_lines += 1
        expected += 1                                  # first vertex
        total_len_m += abs(geod.geometry_length(line))  # independent length
        for (lo1, la1), (lo2, la2) in zip(coords[:-1], coords[1:]):
            _, _, d = geod.inv(lo1, la1, lo2, la2)
            d = abs(d)
            n_segments += 1
            if d > 0:
                expected += math.ceil(d / spacing_m)   # points added per segment

total_len_km = total_len_m / 1000.0

# ---- C. Bounds from total length alone --------------------------------
lower = total_len_m / spacing_m + n_lines              # perfect packing
upper = lower + n_segments                             # ceil() adds <1 point per segment

# ---- D. Duplicate points (e.g. shared endpoints between traces) -------
n_unique = len(np.unique(np.round(samples, 6), axis=0))

print(f"LineString components:           {n_lines:,}")
print(f"Vertex-to-vertex segments:       {n_segments:,}")
print(f"Total geodesic length:           {total_len_km:,.0f} km")
print()
print(f"Pipeline sample count:           {n_pipeline:,}")
print(f"Independent predicted count:     {expected:,}")
print(f"Bounds from length alone:        {lower:,.0f} to {upper:,.0f}")
print(f"Unique points (6-decimal round): {n_unique:,}")
print(f"Mean spacing:                    {total_len_km / (n_pipeline - n_lines):.3f} km")
print()
print("Pipeline == predicted:           ", n_pipeline == expected)
print("Within bounds:                   ", lower <= n_pipeline <= upper)
print("Matches manuscript (433,200):    ", n_pipeline == 433_200)