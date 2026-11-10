# fault_audit.py
import geopandas as gpd
import pandas as pd

H = gpd.read_file("gem_active_faults_harmonized.geojson")
U = gpd.read_file("gem_active_faults.geojson")   # unharmonized release

def dup_traces(g):
    wkb = g.geometry.apply(lambda x: x.wkb)
    sizes = wkb.map(wkb.value_counts())
    return int((sizes > 1).sum())

print(f"Traces: unharmonized {len(U):,} -> harmonized {len(H):,}   (paper 16,195 -> 13,696)")
print(f"Traces in exact duplicate-geometry groups: "
      f"{dup_traces(U)} -> {dup_traces(H)}   (paper 166 -> 44)")

H["is_bird"] = H["catalog_name"].astype("string").str.strip() == "Bird 2003"
H["type_group"] = H["slip_type"].map(lambda s:
    "ridge" if s == "Spreading_Ridge" else
    "subduction" if s == "Subduction_Thrust" else
    "transform" if isinstance(s, str) and s.endswith("Transform") else
    "fold" if s in ("Anticline", "Syncline") else "other")
tab = pd.crosstab(H["type_group"], H["is_bird"].map({True: "Bird", False: "non-Bird"}),
                  margins=True)
print(tab)
tab.to_csv("fault_type_by_catalog.csv")