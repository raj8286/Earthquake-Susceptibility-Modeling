# spearman_correlation_check_30km.py

import pandas as pd
from grid_config_v5 import get_grid_config

FEATURES = ["elevation_m","freeair_mgal","bouguer_mgal","isostatic_mgal","geoid_umg3",
 "sedthk_km","crsthk_km","lithoref_moho_depth_km","lithoref_lab_depth_km","lithoref_rho_c",
 "lithoref_rho_l","lithoref_rho_sl","lithoref_geoid_m","lithoref_fa_mgal",
 "gem_fault_presence","gem_fault_dist_km","gem_fault_length_density_km_per_1000km2","terrain_slope_deg"]

c = get_grid_config(30)
df = pd.read_csv(c.model_dir / "model_dataset_cell_30km_v5.csv", usecols=FEATURES)
corr = df.corr(method="spearman")
corr.round(3).to_csv("spearman_corr_30km.csv")

pairs = corr.stack()
pairs = pairs[pairs.index.get_level_values(0) < pairs.index.get_level_values(1)]
print(pairs[pairs.abs() >= 0.7].sort_values(key=abs, ascending=False).round(2).to_string())
print("\nValues quoted in paper:")
print("bouguer vs elevation      :", round(corr.loc["bouguer_mgal","elevation_m"], 2), "(paper -0.97)")
print("bouguer vs crustal thick. :", round(corr.loc["bouguer_mgal","crsthk_km"], 2), "(paper -0.84)")