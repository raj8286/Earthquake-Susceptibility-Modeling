# Earthquake Susceptibility Modeling from Multi-Source Geophysical Fields

Code for the paper **"Earthquake Susceptibility Modeling from Multi-Source Geophysical Fields"** (Dubey and Verma, Indian Institute of Technology Kharagpur).

The project maps *static spatial earthquake susceptibility* on a global equal-area grid. Each grid cell is classified as earthquake-positive (at least one M ≥ 4 event in the 2000–2025 USGS catalog) or negative, using only 18 harmonized geophysical predictors. Four tree-based classifiers are compared under spatially blocked cross-validation, and the selected model (Extra Trees) is examined through feature-family ablation, magnitude-threshold, grid-resolution and coordinate sensitivity analyses, and probability calibration.

> **Scope.** The framework estimates where earthquakes have been recorded, not when they will occur. Calibrated outputs are the probability that a cell belongs to the historically M ≥ 4-positive class. They are not the probability of a future earthquake.

## Method summary

| Item | Setting |
|---|---|
| Grid | Nominal 15, 30 (primary) and 60 km cells in EPSG:6933 (WGS 84 / NSIDC EASE-Grid 2.0 Global); 30 km = 1,158 × 488 = 565,104 cells |
| Target | `target_m4_0plus`: 1 if a cell has at least one M ≥ 4.0 event in 2000–2025 (7.47% of cells at 30 km). Targets for M ≥ 4.5, 5.0, 5.5, 6.0 are also built |
| Predictors | 18 physical predictors in five families (below). Coordinates are not used as predictors in the primary model |
| Models | Extra Trees, Random Forest, XGBoost, LightGBM, with fixed, untuned hyperparameters |
| Validation | 600 km spatial blocks, 5-fold `GroupKFold`; inside each fold a 75/25 `GroupShuffleSplit` of the remaining blocks gives training and validation sets (seed = 42 + fold index) |
| Threshold | Maximizes F1 on the validation subset, then applied unchanged to the test subset |
| Imbalance | Class weighting (balanced / balanced-subsample for the sklearn forests, negative-to-positive ratio for XGBoost and LightGBM) |
| Metrics | PR-AUC (primary), ROC-AUC, F1, MCC, precision, recall, balanced accuracy; mean ± sample SD across folds |
| Calibration | Raw score, sigmoid and isotonic calibration, fitted on each fold's validation subset |

### The 18 predictors

| Family | Predictors (column names) | Source |
|---|---|---|
| Surface morphology (2) | `elevation_m`, `terrain_slope_deg` | GEBCO_2026 |
| Gravity and geoid (6) | `freeair_mgal`, `bouguer_mgal`, `isostatic_mgal`, `geoid_umg3`, `lithoref_geoid_m`, `lithoref_fa_mgal` | WGM2012, UMG3, LithoRef18 |
| Crust and sediment (3) | `sedthk_km`, `crsthk_km`, `lithoref_moho_depth_km` | CRUST1.0, LithoRef18 |
| Lithosphere and density (4) | `lithoref_lab_depth_km`, `lithoref_rho_c`, `lithoref_rho_l`, `lithoref_rho_sl` | LithoRef18 |
| Active faults (3) | `gem_fault_presence`, `gem_fault_dist_km`, `gem_fault_length_density_km_per_1000km2` | GEM Global Active Faults Database (harmonized release) |

EMAG2v3 magnetic anomalies were assessed separately (`08_emag2_incremental_screening_v5.py`) and are not part of the primary model.

## Repository contents

### Data preparation

| Script | Purpose |
|---|---|
| `download_usgs_earthquake.py` | Downloads the USGS FDSN catalog (M ≥ 4.0, 2000-01-01 to 2025-12-31) as 312 monthly CSV files |
| `01_prepare_usgs_earthquakes_grid.py` | Cleans and deduplicates the catalog, projects events to EPSG:6933 and builds cell-level labels at each magnitude threshold (`--cell-km 15\|30\|60`) |
| `02_prepare_global_features_grid_v5.py` | Harmonizes all raw geophysical sources to the grid and computes the 18 predictors, including geodesic terrain slope and fault distance and length density (`--cell-km`) |
| `03_build_model_dataset_grid_v5.py` | Merges predictors and labels into the model-ready dataset `model_dataset_cell_<km>km_v5.csv` (`--cell-km`) |

### Modeling and experiments

| Script | Purpose | Paper section |
|---|---|---|
| `05_extratrees_spatial_cv_grid_v5.py` | Extra Trees spatial cross-validation (`--grid-km`; optional `--save-oof`). Its fold metrics are the canonical Extra Trees baseline reused by scripts 06, 07, 08, 13 and 16 | 3.1 |
| `07_model_comparison_spatial_cv_grid_v5.py` | Trains XGBoost, LightGBM and Random Forest on identical folds and combines them with the Extra Trees baseline (`--grid-km`) | 3.1, 3.4 |
| `06_extratrees_feature_ablation_v5.py` | Removes each of the five feature families in turn (30 km) | 3.2 |
| `08_emag2_incremental_screening_v5.py` | Tests whether EMAG2 adds predictive signal beyond the 18 predictors (30 km) | 2.3.6, 3.2 |
| `13_magnitude_threshold_sensitivity_v5.py` | Repeats the analysis at M ≥ 4.5, 5.0, 5.5 and 6.0 (30 km) | 3.3 |
| `16_coordinate_augmentation_experiment_v5.py` | Adds EPSG:6933 x/y coordinates to the predictors (30 km) | 3.5 |
| `11_extra_trees_probability_calibration_v5.py` | Raw / sigmoid / isotonic calibration and the pooled out-of-fold prediction table `calibration_oof_predictions.csv` (30 km) | 3.7, 3.8 |
| `26_location_susceptibility_lookup_v5.py` | Looks up the susceptibility of a latitude/longitude from the out-of-fold table (`--lat`, `--lon`; interactive if omitted; optional `--faults`) | none |

### Figures

All plotting scripts write to `plots_v5/`.

| Script | Figure content | Paper figure |
|---|---|---|
| `29_plot_18_predictors_v5.py` | The 18 predictor maps (options: `--input`, `--output`, `--dpi`, `--full-range`, `--only`, `--coastline`) | 2–6 |
| `22_plot_model_comparison_key_results_v5.py` | Four-model comparison | 9 |
| `23_plot_feature_ablation_v5.py` | Feature-family ablation | 10 |
| `26_plot_magnitude_threshold_sensitivity_v5.py` | Magnitude-threshold sensitivity | 11 |
| `25_plot_all_models_grid_resolution_v5.py` | Grid-resolution sensitivity (`--summary15`, `--summary30`, `--summary60`) | 12 |
| `28_plot_coordinate_augmentation_v5.py` | Coordinate augmentation | 13 |
| `20_plot_india_actual_vs_predicted_m4_v5.py` | Indian subcontinent: historical vs. predicted positives | 14 |
| `24_plot_calibration_analysis_v5.py` | Calibration analysis | 15 |
| `18_plot_final_raw_susceptibility_maps_v5.py` | Global and regional raw susceptibility maps | 16 |
| `19_plot_calibrated_probability_maps_v5.py` | Global and regional isotonic-calibrated probability maps | 17 |

Figures 1, 7 and 8 (workflow diagrams) are not generated by this code.

## Requirements

Python 3.9 or newer. Packages imported by the scripts:

`numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `geopandas`, `shapely`, `pyproj`, `xarray` (with a NetCDF backend such as `netCDF4` or `h5netcdf`, needed for the GEBCO file), `cartopy`, `requests`.

```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm matplotlib \
            geopandas shapely pyproj xarray netCDF4 cartopy requests
```

## Input data

Raw datasets are not included in this repository. Script 02 reads them from the project directory under these names:

| File | Dataset | Download |
|---|---|---|
| `GEBCO_2026.nc` | GEBCO_2026 elevation and bathymetry (ice-surface elevation version) | [GEBCO_2026.zip](https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/ice_surface_elevation/netcdf/GEBCO_2026.zip?download=1) |
| `WGM2012_Freeair_ponc_2min.grd`, `WGM2012_Bouguer_ponc_2min.grd`, `WGM2012_Isostatic_ponc_2min.grd` | World Gravity Map 2012 (Bureau Gravimétrique International) | [BGI catalogue](https://bgi.obs-mip.fr/catalogue/?uuid=df2dab2d-a826-4776-b49f-61e8b284c409) |
| `Geoid_UMG3.xyz` | UMG3 upper-mantle geoid (Salajegheh and Afonso, 2023) | [Figshare](https://figshare.com/ndownloader/files/41388840) |
| `sedthk.xyz` | CRUST1.0 sediment thickness | [sedthk.xyz.zip](https://igppweb.ucsd.edu/~gabi/crust1/sedthk.xyz.zip) |
| `crsthk.xyz` | CRUST1.0 total crustal thickness | [crsthk.xyz.zip](https://igppweb.ucsd.edu/~gabi/crust1/crsthk.xyz.zip) |
| `LithoRef_model/LithoRef18.xyz` | LithoRef18 global reference model (Afonso et al., 2019) | [LithoRef_model_last.zip](https://www.juanafonso.com/_files/archives/7f2185_3d6c22f36e7346eb9fca4610ffed4255.zip?dn=LithoRef_model_last.zip) |
| `gem_active_faults_harmonized.geojson` | GEM Global Active Faults Database, harmonized release (Styron and Pagani, 2020) | [gem_active_faults_harmonized.geojson](https://github.com/GEMScienceTools/gem-global-active-faults/blob/master/geojson/gem_active_faults_harmonized.geojson) |
| `EMAG2_V3_20170530.csv` | EMAG2v3 magnetic anomaly grid (used only by script 08) | [NOAA NCEI](https://www.ncei.noaa.gov/products/earth-magnetic-model-anomaly-grid-2/download-data) |
| `data/usgs_raw_m4_2000_2025/usgs_m4_YYYY_MM.csv` | Monthly USGS catalog files, produced by `download_usgs_earthquake.py` | [USGS earthquake search](https://earthquake.usgs.gov/earthquakes/search/) (the script uses the FDSN web service) |
| `IndiaBoundary.shp`, `India-States.shp` | India boundary shapefiles used by scripts 18, 19 and 20 (must carry CRS metadata) | not included |

Notes on the downloads:

- **GEBCO.** GEBCO_2026 is distributed in two NetCDF versions, ice-surface elevation and sub-ice topography/bathymetry. The pipeline expects a file named `GEBCO_2026.nc`, which matches the ice-surface version.
- **GEM faults.** The repository's [`geojson` folder](https://github.com/GEMScienceTools/gem-global-active-faults/tree/master/geojson) holds both the unharmonized `gem_active_faults.geojson` and the harmonized `gem_active_faults_harmonized.geojson`. The paper uses the harmonized release (13,696 traces), so download that one.
- **Archives.** Several files come as zip archives. Unzip them into the project directory using the file names in the first column.

Citations for each dataset are in the paper's reference list.

## Running the pipeline

Run the scripts from the project directory. Steps 1–3 and 5 take `--cell-km` or `--grid-km` and should be run for 15, 30 and 60 where the grid-resolution analysis is wanted. Everything else runs at 30 km.

```bash
# 1. Catalog and labels
python download_usgs_earthquake.py
python 01_prepare_usgs_earthquakes_grid.py --cell-km 30

# 2. Predictors and model-ready dataset
python 02_prepare_global_features_grid_v5.py --cell-km 30
python 03_build_model_dataset_grid_v5.py --cell-km 30

# 3. Extra Trees baseline (must come before 06, 07, 08, 13, 16)
python 05_extratrees_spatial_cv_grid_v5.py --grid-km 30

# 4. Four-model comparison and grid-resolution analysis
python 07_model_comparison_spatial_cv_grid_v5.py --grid-km 30

# 5. Experiments (30 km)
python 06_extratrees_feature_ablation_v5.py
python 08_emag2_incremental_screening_v5.py
python 13_magnitude_threshold_sensitivity_v5.py
python 16_coordinate_augmentation_experiment_v5.py
python 11_extra_trees_probability_calibration_v5.py

# 6. Figures
python 29_plot_18_predictors_v5.py
python 22_plot_model_comparison_key_results_v5.py
python 23_plot_feature_ablation_v5.py
python 26_plot_magnitude_threshold_sensitivity_v5.py
python 25_plot_all_models_grid_resolution_v5.py
python 28_plot_coordinate_augmentation_v5.py
python 24_plot_calibration_analysis_v5.py
python 18_plot_final_raw_susceptibility_maps_v5.py
python 19_plot_calibrated_probability_maps_v5.py
python 20_plot_india_actual_vs_predicted_m4_v5.py

# 7. Look up a location
python 26_location_susceptibility_lookup_v5.py --lat 22.57 --lon 88.36
```

### Dependencies between scripts

| Script | Needs output of |
|---|---|
| 02, 03 | 01 (03 also needs 02) |
| 05, 06, 07, 08, 11, 13, 16 | 03 |
| 06, 07, 08, 13, 16 | 05 at the same grid size (Extra Trees baseline is reused, not retrained) |
| 22 | 07 at 30 km |
| 25 | 07 at 15, 30 and 60 km |
| 23 | 06 |
| 26_plot (magnitude) | 13 |
| 28 | 16 |
| 24, 18, 19, 20, 26_lookup | 11 (`calibration_oof_predictions.csv`) |
| 29 | 02 and 03 at 30 km |

### Output folders

Each script writes a folder next to itself, for example `model_comparison_spatial_cv_v5_m4_18features_30km/`, `extra_trees_ablation_v5_m4_18features_30km/`, `probability_calibration_v5_30km_m4_18features/`, `magnitude_threshold_sensitivity_v5_30km_18features/`, `emag2_incremental_screening_v5_30km/`. These contain per-fold metrics, summaries (mean, SD, min, max), configuration JSON files and, where relevant, out-of-fold predictions.

## Reproducibility notes

- Random seed 42 is used for all models. The inner train/validation split uses seed 42 + fold index (1–5). `GroupKFold` is deterministic.
- Hyperparameters are fixed in advance and not tuned (Extra Trees and Random Forest: 500 trees; XGBoost and LightGBM: 600 iterations, learning rate 0.05).
- Imputation is fitted on each fold's training subset only. The 18-feature dataset has no missing values, so imputation is a safeguard.
- PR-AUC is the trapezoidal area under the precision-recall curve.
- Neighboring 600 km blocks can fall in different folds, and no buffer is imposed between them.
- Regional maps are extracted from the global out-of-fold field. Straight-edged discontinuities in them are block boundaries between fold-specific models, not physical features.
