# bird_split_01_features.py
# It splits the harmonized fault file by catalog_name and recomputes the three fault features for each subset.

import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from grid_config_v5 import get_grid_config

SCRIPT_DIR = Path(__file__).resolve().parent
GRID_KM = 30
BIRD_NAME = "Bird 2003"
OUT_DIR = SCRIPT_DIR / "bird_split_30km"
OUT_DIR.mkdir(exist_ok=True)


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


feat = load_module("feat02", "02_prepare_global_features_grid_v5.py")
config = get_grid_config(GRID_KM)

faults = gpd.read_file(feat.FAULT_FILE)
is_bird = faults["catalog_name"].astype("string").str.strip() == BIRD_NAME
print(f"Total traces: {len(faults):,} | Bird: {int(is_bird.sum()):,} | "
      f"non-Bird: {int((~is_bird).sum()):,}")
assert int(is_bird.sum()) == 4083, "Bird trace count differs from expected 4,083"

_, _, target_lon, target_lat = feat.build_grid_centers(config)
ix = np.tile(np.arange(config.nx, dtype=np.int32), config.ny)
iy = np.repeat(np.arange(config.ny, dtype=np.int32), config.nx)
cell_id = iy.astype(str) + "_" + ix.astype(str)

subsets = {"bird_only": faults[is_bird], "non_bird_only": faults[~is_bird]}

for name, subset in subsets.items():
    print("\n" + "=" * 80)
    print(f"SUBSET: {name} ({len(subset):,} traces)")
    print("=" * 80)

    subset_file = OUT_DIR / f"faults_{name}.geojson"
    subset[["geometry"]].to_file(subset_file, driver="GeoJSON")

    feat.FAULT_FILE = subset_file  # add_fault_features reads this module-level path

    result = feat.add_fault_features(
        config=config,
        target_lon=target_lon,
        target_lat=target_lat,
        fault_sample_spacing_km=feat.DEFAULT_FAULT_SAMPLE_SPACING_KM,
        candidate_k=feat.DEFAULT_NEAREST_CANDIDATES,
    )

    out = pd.DataFrame({"cell_id": cell_id})
    for col, arr in result.items():
        out[col] = arr.ravel().astype(np.float32)
    out.to_csv(OUT_DIR / f"fault_features_{name}.csv", index=False)
    print(f"Saved: {OUT_DIR / f'fault_features_{name}.csv'}")