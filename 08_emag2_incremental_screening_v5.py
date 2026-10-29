# 08_emag2_incremental_screening_v5.py
# V5 incremental-value screening of EMAG2 against the canonical 18-feature 30-km model.
#
# This is intentionally separate from the canonical v5 model dataset. The primary model
# remains the 18-feature physical specification; this script asks only whether adding EMAG2
# contributes measurable incremental predictive signal under identical spatial folds.

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json

import numpy as np
import pandas as pd
from pyproj import Transformer

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from grid_config_v5 import PROJECT_DIR, get_grid_config
from baseline_reuse_v5 import (
    canonical_extratrees_baseline_file,
    load_canonical_extratrees_baseline,
)


SCRIPT_DIR = Path(__file__).resolve().parent
GRID_KM = 30
TARGET_COL = "target_m4_0plus"
SPATIAL_BLOCK_M = 600_000.0
N_SPLITS = 5
RANDOM_STATE = 42
N_JOBS = 10

EMAG2_FILE = PROJECT_DIR / "EMAG2_V3_20170530.csv"

TO_EA = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:6933",
    always_xy=True,
)


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


EMAG2_VALUE_COL = "emag2_nT"
EMAG2_MISSING_COL = "emag2_missing"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether EMAG2 adds incremental predictive signal beyond "
            "the canonical v5 18-feature 30-km Extra Trees model."
        )
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--rebuild-emag2-cache",
        action="store_true",
        help="Rebuild the cached 30-km EMAG2 table even if it already exists.",
    )
    return parser.parse_args()


def resolve_input_file(input_arg):
    config = get_grid_config(GRID_KM)
    if input_arg:
        path = Path(input_arg)
        if not path.is_absolute():
            path = SCRIPT_DIR / path
        return path
    return config.model_dir / "model_dataset_cell_30km_v5.csv"


def resolve_output_dir(output_arg):
    if output_arg:
        path = Path(output_arg)
        if not path.is_absolute():
            path = SCRIPT_DIR / path
    else:
        path = SCRIPT_DIR / "emag2_incremental_screening_v5_30km"
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_points(lon, lat, values, sums, counts, config):
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

    projected = np.isfinite(x) & np.isfinite(y)
    if not np.any(projected):
        return

    x = x[projected]
    y = y[projected]
    values = values[projected]

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


def build_emag2_cache(cache_file):
    config = get_grid_config(GRID_KM)

    if not EMAG2_FILE.exists():
        raise FileNotFoundError(
            f"EMAG2 source file not found: {EMAG2_FILE.resolve()}"
        )

    sums = np.zeros((config.ny, config.nx), dtype=np.float64)
    counts = np.zeros((config.ny, config.nx), dtype=np.int32)

    accepted_samples = 0

    print(f"Reading EMAG2: {EMAG2_FILE.resolve()}")

    with EMAG2_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
        errors="replace",
    ) as stream:
        reader = csv.reader(stream)

        batch_lon = []
        batch_lat = []
        batch_values = []

        for row in reader:
            if len(row) < 7:
                continue

            try:
                lon = float(row[2])
                lat = float(row[3])
                value = float(row[5])
                source_code = int(float(row[6]))
            except ValueError:
                continue

            if (
                not np.isfinite(lon)
                or not np.isfinite(lat)
                or not np.isfinite(value)
                or value == 99999.0
                or source_code == 999
            ):
                continue

            if lon > 180.0:
                lon -= 360.0

            batch_lon.append(lon)
            batch_lat.append(lat)
            batch_values.append(value)
            accepted_samples += 1

            if len(batch_lon) >= 200_000:
                add_points(
                    np.asarray(batch_lon, dtype=np.float64),
                    np.asarray(batch_lat, dtype=np.float64),
                    np.asarray(batch_values, dtype=np.float64),
                    sums,
                    counts,
                    config,
                )
                batch_lon.clear()
                batch_lat.clear()
                batch_values.clear()

        if batch_lon:
            add_points(
                np.asarray(batch_lon, dtype=np.float64),
                np.asarray(batch_lat, dtype=np.float64),
                np.asarray(batch_values, dtype=np.float64),
                sums,
                counts,
                config,
            )

    emag2 = np.full((config.ny, config.nx), np.nan, dtype=np.float32)
    valid = counts > 0
    emag2[valid] = (sums[valid] / counts[valid]).astype(np.float32)

    ix = np.tile(np.arange(config.nx, dtype=np.int32), config.ny)
    iy = np.repeat(np.arange(config.ny, dtype=np.int32), config.nx)
    cell_id = np.char.add(
        np.char.add(iy.astype(str), "_"),
        ix.astype(str),
    )

    cache = pd.DataFrame(
        {
            "cell_id": cell_id,
            EMAG2_VALUE_COL: emag2.ravel(),
            "emag2_source_sample_count": counts.ravel().astype(np.int32),
        }
    )
    cache.to_csv(cache_file, index=False)

    print(f"Accepted raw EMAG2 samples: {accepted_samples:,}")
    print(
        f"30-km cells with EMAG2 coverage: "
        f"{int(np.count_nonzero(valid)):,} / {config.nx * config.ny:,}"
    )
    print(f"Saved EMAG2 cache: {cache_file.resolve()}")

    return cache


def load_or_build_emag2(cache_file, rebuild):
    if cache_file.exists() and not rebuild:
        print(f"Using cached EMAG2 table: {cache_file.resolve()}")
        cache = pd.read_csv(cache_file)
    else:
        cache = build_emag2_cache(cache_file)

    required = {
        "cell_id",
        EMAG2_VALUE_COL,
        "emag2_source_sample_count",
    }
    missing = required - set(cache.columns)
    if missing:
        raise RuntimeError(
            f"EMAG2 cache is missing columns: {sorted(missing)}"
        )

    if cache["cell_id"].duplicated().any():
        raise RuntimeError("EMAG2 cache contains duplicate cell_id values.")

    return cache


def make_groups(df):
    block_cells = int(round(SPATIAL_BLOCK_M / (GRID_KM * 1000.0)))
    if block_cells != 20:
        raise RuntimeError("Expected 20 cells per 600-km block edge.")

    ix = pd.to_numeric(df["ix"], errors="raise").to_numpy(dtype=np.int32)
    iy = pd.to_numeric(df["iy"], errors="raise").to_numpy(dtype=np.int32)

    return np.char.add(
        np.char.add((iy // block_cells).astype(str), "_"),
        (ix // block_cells).astype(str),
    )


def build_folds(n_rows, y, groups):
    dummy = np.zeros((n_rows, 1), dtype=np.float32)
    outer = GroupKFold(n_splits=N_SPLITS)
    folds = []

    for fold, (outer_train_idx, test_idx) in enumerate(
        outer.split(dummy, y, groups=groups),
        start=1,
    ):
        outer_train_idx = np.asarray(outer_train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)

        inner = GroupShuffleSplit(
            n_splits=1,
            train_size=0.75,
            random_state=RANDOM_STATE + fold,
        )
        train_rel, val_rel = next(
            inner.split(
                dummy[outer_train_idx],
                y.iloc[outer_train_idx],
                groups=groups[outer_train_idx],
            )
        )

        train_idx = outer_train_idx[train_rel]
        val_idx = outer_train_idx[val_rel]

        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        test_groups = set(groups[test_idx])

        if train_groups & val_groups:
            raise RuntimeError(f"Fold {fold}: train/validation share blocks.")
        if train_groups & test_groups:
            raise RuntimeError(f"Fold {fold}: train/test share blocks.")
        if val_groups & test_groups:
            raise RuntimeError(f"Fold {fold}: validation/test share blocks.")

        folds.append(
            {
                "fold": fold,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            }
        )

    return folds


def make_model():
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=False,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


def choose_f1_threshold(y_val, p):
    precision, recall, thresholds = precision_recall_curve(y_val, p)
    if len(thresholds) == 0:
        return 0.5

    f1 = (
        2.0 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    if not np.isfinite(f1).any():
        return 0.5

    return float(thresholds[int(np.nanargmax(f1))])


def evaluate(y_true, p, threshold):
    pred = (p >= threshold).astype(np.int8)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, p)

    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
        "brier_score": float(brier_score_loss(y_true, p)),
        "best_threshold": float(threshold),
        "precision_at_best": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "recall_at_best": float(
            recall_score(y_true, pred, zero_division=0)
        ),
        "f1_at_best": float(
            f1_score(y_true, pred, zero_division=0)
        ),
        "balanced_accuracy_at_best": float(
            balanced_accuracy_score(y_true, pred)
        ),
        "mcc_at_best": float(matthews_corrcoef(y_true, pred)),
    }


def run_experiment(name, X, y, folds):
    rows = []

    print("\n" + "#" * 96)
    print(f"EXPERIMENT: {name} | columns={X.shape[1]}")
    print("#" * 96)

    for info in folds:
        fold = info["fold"]
        train_idx = info["train_idx"]
        val_idx = info["val_idx"]
        test_idx = info["test_idx"]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        y_test = y.iloc[test_idx]

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X.iloc[train_idx]).astype(
            np.float32, copy=False
        )
        X_val = imputer.transform(X.iloc[val_idx]).astype(
            np.float32, copy=False
        )
        X_test = imputer.transform(X.iloc[test_idx]).astype(
            np.float32, copy=False
        )

        model = make_model()
        model.fit(X_train, y_train)

        val_p = model.predict_proba(X_val)[:, 1]
        threshold = choose_f1_threshold(y_val, val_p)

        test_p = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, test_p, threshold)

        rows.append(
            {
                "experiment": name,
                "fold": fold,
                "n_model_columns": int(X.shape[1]),
                "test_positive_rate": float(y_test.mean()),
                **metrics,
            }
        )

        print(
            f"Fold {fold}: "
            f"ROC={metrics['roc_auc']:.6f} | "
            f"PR={metrics['pr_auc']:.6f} | "
            f"F1={metrics['f1_at_best']:.6f} | "
            f"MCC={metrics['mcc_at_best']:.6f}"
        )

    return pd.DataFrame(rows)



def physical_baseline_to_emag2(baseline):
    return pd.DataFrame({
        "experiment": "physical_only",
        "fold": baseline["fold"].astype(int),
        "n_model_columns": 18,
        "test_positive_rate": baseline["test_positive_rate"].astype(float),
        "roc_auc": baseline["roc_auc"].astype(float),
        "pr_auc": baseline["pr_auc"].astype(float),
        "brier_score": baseline["brier_score"].astype(float),
        "best_threshold": baseline["best_threshold"].astype(float),
        "precision_at_best": baseline["precision_at_best"].astype(float),
        "recall_at_best": baseline["recall_at_best"].astype(float),
        "f1_at_best": baseline["f1_at_best"].astype(float),
        "balanced_accuracy_at_best": baseline[
            "balanced_accuracy_at_best"
        ].astype(float),
        "mcc_at_best": baseline["mcc_at_best"].astype(float),
    }).sort_values("fold").reset_index(drop=True)

def main():
    args = parse_args()
    input_file = resolve_input_file(args.input)
    output_dir = resolve_output_dir(args.output_dir)
    cache_file = output_dir / "emag2_30km_v5.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Canonical v5 model dataset not found: {input_file.resolve()}"
        )

    required = PHYSICAL_FEATURES + [TARGET_COL, "cell_id", "ix", "iy"]
    df = pd.read_csv(input_file, usecols=required)
    config = get_grid_config(GRID_KM)
    expected_rows = config.nx * config.ny

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows:,} model rows, found {len(df):,}."
        )

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Canonical dataset is missing columns: {missing}")

    if df["cell_id"].duplicated().any():
        raise RuntimeError("Canonical dataset contains duplicate cell_id values.")

    emag2 = load_or_build_emag2(
        cache_file=cache_file,
        rebuild=args.rebuild_emag2_cache,
    )

    df = df.merge(
        emag2,
        on="cell_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(df) != expected_rows:
        raise RuntimeError("Row count changed while attaching EMAG2.")

    df[EMAG2_MISSING_COL] = (
        df[EMAG2_VALUE_COL].isna().astype(np.int8)
    )

    y = pd.to_numeric(df[TARGET_COL], errors="raise").astype(np.int8)
    if sorted(y.unique().tolist()) != [0, 1]:
        raise RuntimeError("Target must contain both binary classes.")

    groups = make_groups(df)
    folds = build_folds(len(df), y, groups)

    plus_cols = PHYSICAL_FEATURES + [
        EMAG2_VALUE_COL,
        EMAG2_MISSING_COL,
    ]
    X_plus = df[plus_cols].copy()
    for col in plus_cols:
        X_plus[col] = pd.to_numeric(
            X_plus[col], errors="coerce"
        ).astype(np.float32)

    print("=" * 100)
    print("V5 EMAG2 INCREMENTAL SCREENING — 30 KM — M>=4")
    print("=" * 100)
    print(f"Dataset: {input_file.resolve()}")
    print(f"EMAG2 source: {EMAG2_FILE.resolve()}")
    print(f"Cells: {len(df):,}")
    print(
        f"EMAG2 missing cells: "
        f"{int(df[EMAG2_MISSING_COL].sum()):,} "
        f"({100.0 * df[EMAG2_MISSING_COL].mean():.3f}%)"
    )

    baseline = load_canonical_extratrees_baseline(
        SCRIPT_DIR,
        GRID_KM,
    )
    physical = physical_baseline_to_emag2(baseline)
    print(
        "Reusing physical-only Extra Trees baseline from: "
        f"{canonical_extratrees_baseline_file(SCRIPT_DIR, GRID_KM).resolve()}"
    )

    plus_emag2 = run_experiment(
        "physical_plus_emag2",
        X_plus,
        y,
        folds,
    )

    all_results = pd.concat(
        [physical, plus_emag2],
        ignore_index=True,
    )
    all_results.to_csv(
        output_dir / "emag2_incremental_fold_metrics.csv",
        index=False,
    )

    summary_rows = []
    for name, d in all_results.groupby("experiment"):
        summary_rows.append(
            {
                "experiment": name,
                "n_model_columns": int(d["n_model_columns"].iloc[0]),
                "roc_auc_mean": float(d["roc_auc"].mean()),
                "roc_auc_std": float(d["roc_auc"].std(ddof=1)),
                "pr_auc_mean": float(d["pr_auc"].mean()),
                "pr_auc_std": float(d["pr_auc"].std(ddof=1)),
                "brier_score_mean": float(d["brier_score"].mean()),
                "f1_at_best_mean": float(d["f1_at_best"].mean()),
                "f1_at_best_std": float(d["f1_at_best"].std(ddof=1)),
                "mcc_at_best_mean": float(d["mcc_at_best"].mean()),
                "mcc_at_best_std": float(d["mcc_at_best"].std(ddof=1)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / "emag2_incremental_summary.csv",
        index=False,
    )

    baseline = summary[
        summary["experiment"] == "physical_only"
    ].iloc[0]
    candidate = summary[
        summary["experiment"] == "physical_plus_emag2"
    ].iloc[0]

    deltas = pd.DataFrame(
        [
            {
                "metric": "roc_auc",
                "physical_only": baseline["roc_auc_mean"],
                "physical_plus_emag2": candidate["roc_auc_mean"],
                "delta": candidate["roc_auc_mean"] - baseline["roc_auc_mean"],
            },
            {
                "metric": "pr_auc",
                "physical_only": baseline["pr_auc_mean"],
                "physical_plus_emag2": candidate["pr_auc_mean"],
                "delta": candidate["pr_auc_mean"] - baseline["pr_auc_mean"],
            },
            {
                "metric": "f1_at_best",
                "physical_only": baseline["f1_at_best_mean"],
                "physical_plus_emag2": candidate["f1_at_best_mean"],
                "delta": (
                    candidate["f1_at_best_mean"]
                    - baseline["f1_at_best_mean"]
                ),
            },
            {
                "metric": "mcc_at_best",
                "physical_only": baseline["mcc_at_best_mean"],
                "physical_plus_emag2": candidate["mcc_at_best_mean"],
                "delta": (
                    candidate["mcc_at_best_mean"]
                    - baseline["mcc_at_best_mean"]
                ),
            },
        ]
    )
    deltas.to_csv(
        output_dir / "emag2_incremental_deltas.csv",
        index=False,
    )

    with (output_dir / "emag2_incremental_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "version": "v5",
                "dataset": str(input_file.resolve()),
                "emag2_source": str(EMAG2_FILE.resolve()),
                "emag2_cache": str(cache_file.resolve()),
                "target": TARGET_COL,
                "grid_km": GRID_KM,
                "physical_feature_count": 18,
                "physical_features": PHYSICAL_FEATURES,
                "candidate_extra_columns": [
                    EMAG2_VALUE_COL,
                    EMAG2_MISSING_COL,
                ],
                "physical_only_baseline_reused_from": str(
                    canonical_extratrees_baseline_file(
                        SCRIPT_DIR, GRID_KM
                    ).resolve()
                ),
                "spatial_block_m": SPATIAL_BLOCK_M,
                "n_spatial_folds": N_SPLITS,
                "model": "ExtraTreesClassifier",
                "random_state": RANDOM_STATE,
                "pr_auc_definition": (
                    "trapezoidal area under precision-recall curve"
                ),
            },
            stream,
            indent=2,
        )

    print("\n" + "=" * 112)
    print("EMAG2 INCREMENTAL SCREENING SUMMARY")
    print("=" * 112)
    print(summary.to_string(index=False))
    print("\nINCREMENTAL DELTAS")
    print(deltas.to_string(index=False))
    print("\nSaved under:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
