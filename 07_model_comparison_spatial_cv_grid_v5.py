# 07_model_comparison_spatial_cv_grid_v5.py
# Canonical v5 four-model comparison for 15/30/60-km grids using identical spatial folds.

from __future__ import annotations

from pathlib import Path
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


# Some sklearn/lightgbm combinations emit this harmless compatibility warning
# even though the imputed arrays have the correct fixed feature order.
warnings.filterwarnings(
    "ignore",
    message=(
        r"X does not have valid feature names, "
        r"but LGBMClassifier was fitted with feature names"
    ),
    category=UserWarning,
)

from grid_config_v5 import get_grid_config
from baseline_reuse_v5 import (
    canonical_extratrees_baseline_file,
    load_canonical_extratrees_baseline,
)


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_COL = "target_m4_0plus"
RANDOM_STATE = 42
N_JOBS = 10
SPATIAL_BLOCK_M = 600_000.0
N_SPLITS = 5

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

TRAIN_MODELS = [
    "xgboost",
    "lightgbm",
    "random_forest",
]

ALL_MODELS = TRAIN_MODELS + ["extra_trees"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare XGBoost, LightGBM, Random Forest, and Extra Trees "
            "using the v5 18-feature M>=4 spatial-CV framework."
        )
    )
    parser.add_argument(
        "--grid-km",
        type=int,
        required=True,
        choices=[15, 30, 60],
        help="Nominal grid resolution in kilometres.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Optional model-dataset CSV. By default uses "
            "data/grid_<km>/model/model_dataset_cell_<km>km_v5.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args()


def resolve_input_file(grid_km, input_arg):
    config = get_grid_config(grid_km)
    if input_arg:
        path = Path(input_arg)
        if not path.is_absolute():
            path = SCRIPT_DIR / path
        return path
    return config.model_dir / f"model_dataset_cell_{grid_km}km_v5.csv"


def resolve_output_dir(grid_km, output_arg):
    if output_arg:
        path = Path(output_arg)
        if not path.is_absolute():
            path = SCRIPT_DIR / path
    else:
        path = (
            SCRIPT_DIR
            / f"model_comparison_spatial_cv_v5_m4_18features_{grid_km}km"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(input_file, grid_km):
    if not input_file.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_file.resolve()}")

    print(f"Loading: {input_file.resolve()}")
    required = PHYSICAL_FEATURES + [
        TARGET_COL,
        "cell_id",
        "ix",
        "iy",
    ]
    df = pd.read_csv(input_file, usecols=required)

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df["cell_id"].duplicated().any():
        raise ValueError("Duplicate cell_id values detected.")

    config = get_grid_config(grid_km)
    expected_rows = config.nx * config.ny
    if len(df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows:,} rows for the {grid_km} km grid, "
            f"found {len(df):,}."
        )

    ix = pd.to_numeric(df["ix"], errors="raise")
    iy = pd.to_numeric(df["iy"], errors="raise")

    if (
        int(ix.min()) != 0
        or int(ix.max()) != config.nx - 1
        or int(ix.nunique()) != config.nx
    ):
        raise RuntimeError("Dataset ix coverage does not match the configured grid.")

    if (
        int(iy.min()) != 0
        or int(iy.max()) != config.ny - 1
        or int(iy.nunique()) != config.ny
    ):
        raise RuntimeError("Dataset iy coverage does not match the configured grid.")

    X = df[PHYSICAL_FEATURES].copy()
    for column in PHYSICAL_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        ).astype(np.float32)

    y = pd.to_numeric(df[TARGET_COL], errors="raise").astype(np.int8)

    if sorted(y.unique().tolist()) != [0, 1]:
        raise RuntimeError("Target must contain both binary classes 0 and 1.")

    print(f"Rows: {len(df):,}")
    print(
        f"M>=4 positive cells: {int(y.sum()):,} "
        f"({100.0 * y.mean():.3f}%)"
    )
    print(f"Physical predictors: {len(PHYSICAL_FEATURES)}")
    return df, X, y


def make_spatial_groups(df, grid_km):
    cell_m = float(grid_km) * 1000.0
    cells_per_block_edge = int(round(SPATIAL_BLOCK_M / cell_m))

    if not np.isclose(
        cells_per_block_edge * cell_m,
        SPATIAL_BLOCK_M,
    ):
        raise RuntimeError(
            "600-km spatial block size is not an integer multiple of cell size."
        )

    ix = pd.to_numeric(df["ix"], errors="raise").to_numpy(dtype=np.int32)
    iy = pd.to_numeric(df["iy"], errors="raise").to_numpy(dtype=np.int32)

    block_x = ix // cells_per_block_edge
    block_y = iy // cells_per_block_edge

    groups = np.char.add(
        np.char.add(block_y.astype(str), "_"),
        block_x.astype(str),
    )
    return groups, cells_per_block_edge


def build_fixed_spatial_folds(n_rows, y, groups):
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


def choose_f1_threshold(y_val, val_proba):
    precision, recall, thresholds = precision_recall_curve(
        y_val,
        val_proba,
    )
    if len(thresholds) == 0:
        return 0.5

    f1 = (
        2.0 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    if not np.isfinite(f1).any():
        return 0.5

    return float(thresholds[int(np.nanargmax(f1))])


def evaluate_predictions(y_true, proba, threshold):
    pred = (proba >= threshold).astype(np.int8)

    precision, recall, _ = precision_recall_curve(
        y_true,
        proba,
    )
    pr_auc = auc(recall, precision)

    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier_score_loss(y_true, proba)),
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
        "mcc_at_best": float(
            matthews_corrcoef(y_true, pred)
        ),
    }


def build_model(model_name, scale_pos_weight):
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            colsample_bytree=0.80,
            objective="binary:logistic",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            tree_method="hist",
        )

    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError(
                "lightgbm is not installed in the active Python environment."
            )
        return LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            colsample_bytree=0.80,
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=-1,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )

    raise ValueError(f"Unknown model: {model_name}")


def run_model(model_name, X, y, folds):
    rows = []

    print("\n" + "#" * 88)
    print(f"MODEL: {model_name}")
    print("#" * 88)

    for info in folds:
        fold = info["fold"]
        train_idx = info["train_idx"]
        val_idx = info["val_idx"]
        test_idx = info["test_idx"]

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        y_test = y.iloc[test_idx]

        imputer = SimpleImputer(strategy="median")
        X_train_i = imputer.fit_transform(X_train).astype(np.float32, copy=False)
        X_val_i = imputer.transform(X_val).astype(np.float32, copy=False)
        X_test_i = imputer.transform(X_test).astype(np.float32, copy=False)

        positives = float(np.sum(y_train == 1))
        negatives = float(np.sum(y_train == 0))
        if positives <= 0:
            raise RuntimeError(f"Fold {fold} training split has zero positives.")

        scale_pos_weight = negatives / positives

        model = build_model(model_name, scale_pos_weight)

        start = time.perf_counter()
        model.fit(X_train_i, y_train)
        fit_seconds = time.perf_counter() - start

        val_proba = model.predict_proba(X_val_i)[:, 1]
        threshold = choose_f1_threshold(y_val, val_proba)

        test_proba = model.predict_proba(X_test_i)[:, 1]
        metrics = evaluate_predictions(
            y_true=y_test,
            proba=test_proba,
            threshold=threshold,
        )

        metrics.update(
            {
                "model": model_name,
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "test_rows": int(len(test_idx)),
                "test_positive_rate": float(y_test.mean()),
                "scale_pos_weight": float(scale_pos_weight),
                "fit_seconds": float(fit_seconds),
            }
        )
        rows.append(metrics)

        print(
            f"{model_name} | fold {fold}/{N_SPLITS} | "
            f"ROC={metrics['roc_auc']:.6f} | "
            f"PR={metrics['pr_auc']:.6f} | "
            f"F1={metrics['f1_at_best']:.6f} | "
            f"MCC={metrics['mcc_at_best']:.6f} | "
            f"time={fit_seconds:.1f}s"
        )

        del X_train_i, X_val_i, X_test_i, model

    return pd.DataFrame(rows)



def extra_trees_baseline_to_model_comparison(baseline):
    """Convert canonical script-05 fold metrics to script-07 schema."""
    out = pd.DataFrame({
        "model": "extra_trees",
        "fold": baseline["fold"].astype(int),
        "train_rows": baseline.get("train_rows", np.nan),
        "validation_rows": baseline.get("validation_rows", np.nan),
        "test_rows": baseline.get("test_rows", np.nan),
        "test_positive_rate": baseline["test_positive_rate"].astype(float),
        "scale_pos_weight": np.nan,
        "fit_seconds": np.nan,
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
    })
    return out.sort_values("fold").reset_index(drop=True)


def load_partial_results(output_dir):
    """Reuse only fully completed five-fold trained-model checkpoints."""
    partial_file = output_dir / "model_comparison_fold_metrics_partial.csv"
    if not partial_file.exists():
        return [], set()

    partial = pd.read_csv(partial_file)
    if not {"model", "fold"}.issubset(partial.columns):
        return [], set()

    frames = []
    completed_models = set()
    for model_name in TRAIN_MODELS:
        d = partial[partial["model"] == model_name].copy()
        folds = sorted(
            pd.to_numeric(d.get("fold"), errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ) if not d.empty else []

        if len(d) == N_SPLITS and folds == list(range(1, N_SPLITS + 1)):
            frames.append(d)
            completed_models.add(model_name)

    if completed_models:
        print(
            "Resuming completed trained models from partial checkpoint: "
            + ", ".join(
                m for m in TRAIN_MODELS if m in completed_models
            )
        )

    return frames, completed_models

def summarize_results(all_results):
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "brier_score",
        "f1_at_best",
        "precision_at_best",
        "recall_at_best",
        "balanced_accuracy_at_best",
        "mcc_at_best",
        "best_threshold",
        "fit_seconds",
    ]

    rows = []
    for model_name, group in all_results.groupby("model"):
        row = {"model": model_name}

        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("pr_auc_mean", ascending=False)
        .reset_index(drop=True)
    )


def main():
    args = parse_args()
    grid_km = int(args.grid_km)
    input_file = resolve_input_file(grid_km, args.input)
    output_dir = resolve_output_dir(grid_km, args.output_dir)

    df, X, y = load_dataset(input_file, grid_km)
    groups, cells_per_block_edge = make_spatial_groups(df, grid_km)
    folds = build_fixed_spatial_folds(len(df), y, groups)

    print("=" * 100)
    print(
        f"V5 {grid_km} KM — FOUR-MODEL M>=4 "
        "SPATIAL CV COMPARISON"
    )
    print("=" * 100)
    print(f"Input: {input_file.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Target: {TARGET_COL}")
    print(f"Cell size: {grid_km} km nominal")
    print(f"Spatial block size: {SPATIAL_BLOCK_M/1000:.0f} km nominal")
    print(f"Cells per block edge: {cells_per_block_edge}")
    print(f"Total spatial blocks: {len(np.unique(groups)):,}")
    print(f"Fixed spatial folds prepared: {len(folds)}")

    baseline = load_canonical_extratrees_baseline(
        SCRIPT_DIR,
        grid_km,
    )
    baseline_results = extra_trees_baseline_to_model_comparison(
        baseline
    )
    print(
        "Reusing completed canonical Extra Trees baseline from: "
        f"{canonical_extratrees_baseline_file(SCRIPT_DIR, grid_km).resolve()}"
    )

    trained_results, completed_models = load_partial_results(
        output_dir
    )

    for model_name in TRAIN_MODELS:
        if model_name in completed_models:
            print(f"SKIP completed trained model: {model_name}")
            continue

        result = run_model(
            model_name,
            X,
            y,
            folds,
        )
        trained_results.append(result)

        pd.concat(
            trained_results,
            ignore_index=True,
        ).to_csv(
            output_dir / "model_comparison_fold_metrics_partial.csv",
            index=False,
        )

    all_results_df = pd.concat(
        [baseline_results] + trained_results,
        ignore_index=True,
    )

    all_results_df.to_csv(
        output_dir / "model_comparison_fold_metrics.csv",
        index=False,
    )

    summary = summarize_results(all_results_df)
    summary.to_csv(
        output_dir / "model_comparison_summary.csv",
        index=False,
    )

    with (
        output_dir / "model_comparison_config.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "version": "v5",
                "input_file": str(input_file.resolve()),
                "target": TARGET_COL,
                "cell_size_m": grid_km * 1000.0,
                "grid_km": grid_km,
                "spatial_block_m": SPATIAL_BLOCK_M,
                "cells_per_block_edge": cells_per_block_edge,
                "n_folds": N_SPLITS,
                "physical_feature_count": len(PHYSICAL_FEATURES),
                "physical_features": PHYSICAL_FEATURES,
                "coordinates_used_as_predictors": False,
                "models": ALL_MODELS,
                "models_trained_in_this_run": TRAIN_MODELS,
                "extra_trees_reused_from": str(
                    canonical_extratrees_baseline_file(
                        SCRIPT_DIR, grid_km
                    ).resolve()
                ),
                "random_state": RANDOM_STATE,
                "pr_auc_definition": "trapezoidal area under precision-recall curve",
            },
            stream,
            indent=2,
        )

    print("\n" + "=" * 118)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 118)

    display_cols = [
        "model",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "f1_at_best_mean",
        "f1_at_best_std",
        "precision_at_best_mean",
        "recall_at_best_mean",
        "mcc_at_best_mean",
        "mcc_at_best_std",
        "fit_seconds_mean",
    ]

    print(summary[display_cols].to_string(index=False))
    print("\nSaved results to:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
