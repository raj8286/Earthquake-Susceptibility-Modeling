# 05_extratrees_spatial_cv_grid_v5.py
# Canonical v5 Extra Trees spatial cross-validation for 15/30/60-km grids.

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from grid_config_v5 import get_grid_config


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_COL = "target_m4_0plus"
RANDOM_STATE = 42
SPATIAL_BLOCK_M = 600_000.0
N_SPLITS = 5

PHYSICAL_FEATURES = [
    "elevation_m",
    "freeair_mgal",
    "bouguer_mgal",
    "isostatic_mgal",
    # "geoid_umg3",
    "sedthk_km",
    "crsthk_km",
    "lithoref_moho_depth_km",
    "lithoref_lab_depth_km",
    "lithoref_rho_c",
    "lithoref_rho_l",
    "lithoref_rho_sl",
    # "lithoref_geoid_m",
    "lithoref_fa_mgal",
    "gem_fault_presence",
    "gem_fault_dist_km",
    "gem_fault_length_density_km_per_1000km2",
    "terrain_slope_deg",
]


# assert len(PHYSICAL_FEATURES) == 18


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical v5 Extra Trees M>=4 spatial cross-validation "
            "experiment for a 15, 30, or 60 km equal-area grid."
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
    parser.add_argument(
        "--save-oof",
        action="store_true",
        help=(
            "Also save the full per-cell OOF prediction table. "
            "Disabled by default because calibration script 11 creates the "
            "30-km OOF table needed downstream and 15/60-km OOF files are "
            "not required for the current figures."
        ),
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
            / f"extra_trees_spatial_cv_v5_m4_18features_{grid_km}km no geoid"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(input_file, grid_km):
    if not input_file.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_file.resolve()}")

    print(f"Loading: {input_file.resolve()}")
    df = pd.read_csv(input_file)

    required = PHYSICAL_FEATURES + [
        TARGET_COL,
        "x_6933_m",
        "y_6933_m",
        "ix",
        "iy",
        "cell_id",
        "lon",
        "lat",
    ]
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
        f"Positive cells ({TARGET_COL}): {int(y.sum()):,} "
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


def make_model():
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=False,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=8,
    )


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
    pred_05 = (proba >= 0.5).astype(np.int8)
    pred_best = (proba >= threshold).astype(np.int8)

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_true,
        proba,
    )
    pr_auc = auc(recall_curve, precision_curve)

    cm_05 = confusion_matrix(y_true, pred_05, labels=[0, 1])
    cm_best = confusion_matrix(y_true, pred_best, labels=[0, 1])

    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "best_threshold": float(threshold),
        "precision_at_0.5": float(
            precision_score(y_true, pred_05, zero_division=0)
        ),
        "recall_at_0.5": float(
            recall_score(y_true, pred_05, zero_division=0)
        ),
        "f1_at_0.5": float(
            f1_score(y_true, pred_05, zero_division=0)
        ),
        "balanced_accuracy_at_0.5": float(
            balanced_accuracy_score(y_true, pred_05)
        ),
        "mcc_at_0.5": float(matthews_corrcoef(y_true, pred_05)),
        "precision_at_best": float(
            precision_score(y_true, pred_best, zero_division=0)
        ),
        "recall_at_best": float(
            recall_score(y_true, pred_best, zero_division=0)
        ),
        "f1_at_best": float(
            f1_score(y_true, pred_best, zero_division=0)
        ),
        "balanced_accuracy_at_best": float(
            balanced_accuracy_score(y_true, pred_best)
        ),
        "mcc_at_best": float(matthews_corrcoef(y_true, pred_best)),
        "tn_at_0.5": int(cm_05[0, 0]),
        "fp_at_0.5": int(cm_05[0, 1]),
        "fn_at_0.5": int(cm_05[1, 0]),
        "tp_at_0.5": int(cm_05[1, 1]),
        "tn_at_best": int(cm_best[0, 0]),
        "fp_at_best": int(cm_best[0, 1]),
        "fn_at_best": int(cm_best[1, 0]),
        "tp_at_best": int(cm_best[1, 1]),
    }


def summarize_metrics(metrics_df):
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "brier_score",
        "best_threshold",
        "precision_at_0.5",
        "recall_at_0.5",
        "f1_at_0.5",
        "balanced_accuracy_at_0.5",
        "mcc_at_0.5",
        "precision_at_best",
        "recall_at_best",
        "f1_at_best",
        "balanced_accuracy_at_best",
        "mcc_at_best",
    ]

    rows = []
    for metric in metric_cols:
        values = metrics_df[metric].astype(float).to_numpy()
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    grid_km = int(args.grid_km)
    input_file = resolve_input_file(grid_km, args.input)
    output_dir = resolve_output_dir(grid_km, args.output_dir)

    df, X, y = load_dataset(input_file, grid_km)
    groups, cells_per_block_edge = make_spatial_groups(df, grid_km)
    folds = build_fixed_spatial_folds(len(df), y, groups)

    print("=" * 88)
    print(
        f"V5 EXTRA TREES M>=4 SPATIAL CV — {grid_km} KM — \n no geoid"
        # f"18 CANONICAL FEATURES"
    )
    print("=" * 88)
    print(f"Input: {input_file.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Spatial block size: {SPATIAL_BLOCK_M/1000:.0f} km nominal")
    print(f"Cells per block edge: {cells_per_block_edge}")
    print(f"Total spatial blocks: {len(np.unique(groups)):,}")

    metric_rows = []
    oof_rows = [] if args.save_oof else None

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

        if (
            y_train.nunique() < 2
            or y_val.nunique() < 2
            or y_test.nunique() < 2
        ):
            raise RuntimeError(f"Fold {fold} contains a single-class split.")

        imputer = SimpleImputer(strategy="median")
        X_train_i = imputer.fit_transform(X_train)
        X_val_i = imputer.transform(X_val)
        X_test_i = imputer.transform(X_test)

        model = make_model()
        model.fit(X_train_i, y_train)

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
                "fold": fold,
                "target": TARGET_COL,
                "grid_km": grid_km,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "test_rows": int(len(test_idx)),
                "train_positive": int(y_train.sum()),
                "validation_positive": int(y_val.sum()),
                "test_positive": int(y_test.sum()),
                "train_positive_rate": float(y_train.mean()),
                "validation_positive_rate": float(y_val.mean()),
                "test_positive_rate": float(y_test.mean()),
                "train_blocks": int(len(set(groups[train_idx]))),
                "validation_blocks": int(len(set(groups[val_idx]))),
                "test_blocks": int(len(set(groups[test_idx]))),
            }
        )
        metric_rows.append(metrics)

        if args.save_oof:
            fold_oof = df.iloc[test_idx][
                [
                    "cell_id",
                    "ix",
                    "iy",
                    "x_6933_m",
                    "y_6933_m",
                    "lon",
                    "lat",
                    TARGET_COL,
                ]
            ].copy()
            fold_oof.insert(0, "row_index", test_idx)
            fold_oof["oof_fold"] = fold
            fold_oof["raw_susceptibility_score"] = test_proba
            fold_oof["fold_operating_threshold"] = threshold
            fold_oof["predicted_m4_positive"] = (
                test_proba >= threshold
            ).astype(np.int8)
            oof_rows.append(fold_oof)

        print(
            f"Fold {fold}: "
            f"ROC={metrics['roc_auc']:.6f} | "
            f"PR={metrics['pr_auc']:.6f} | "
            f"F1={metrics['f1_at_best']:.6f} | "
            f"MCC={metrics['mcc_at_best']:.6f} | "
            f"threshold={threshold:.6f}"
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("fold")
    metrics_df.to_csv(
        output_dir / "spatial_cv_fold_metrics.csv",
        index=False,
    )

    summary_df = summarize_metrics(metrics_df)
    summary_df.to_csv(
        output_dir / "spatial_cv_metric_summary.csv",
        index=False,
    )

    if args.save_oof:
        oof = (
            pd.concat(oof_rows, ignore_index=True)
            .sort_values("row_index")
            .reset_index(drop=True)
        )

        if (
            len(oof) != len(df)
            or oof["row_index"].duplicated().any()
            or oof["cell_id"].duplicated().any()
        ):
            raise RuntimeError("Invalid pooled OOF prediction table.")

        if not np.array_equal(
            oof["row_index"].to_numpy(dtype=np.int64),
            np.arange(len(df), dtype=np.int64),
        ):
            raise RuntimeError(
                "OOF predictions do not cover every row exactly once."
            )

        oof.to_csv(
            output_dir / "oof_predictions.csv",
            index=False,
        )

    with (
        output_dir / "spatial_cv_config.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "version": "v5",
                "input_file": str(input_file.resolve()),
                "target": TARGET_COL,
                "grid_km": grid_km,
                "cell_size_m": grid_km * 1000.0,
                "spatial_block_m": SPATIAL_BLOCK_M,
                "cells_per_block_edge": cells_per_block_edge,
                "n_splits": N_SPLITS,
                "physical_feature_count": len(PHYSICAL_FEATURES),
                "physical_features": PHYSICAL_FEATURES,
                "coordinates_used_as_predictors": False,
                "model": "ExtraTreesClassifier",
                "random_state": RANDOM_STATE,
                "pr_auc_definition": "trapezoidal area under precision-recall curve",
                "oof_predictions_saved": bool(args.save_oof),
            },
            stream,
            indent=2,
        )

    print("\n" + "=" * 88)
    print("MEAN +/- STANDARD DEVIATION")
    print("=" * 88)

    key_metrics = [
        "roc_auc",
        "pr_auc",
        "brier_score",
        "f1_at_best",
        "precision_at_best",
        "recall_at_best",
        "balanced_accuracy_at_best",
        "mcc_at_best",
    ]
    key_summary = (
        summary_df[summary_df["metric"].isin(key_metrics)]
        .set_index("metric")
    )

    for metric in key_metrics:
        row = key_summary.loc[metric]
        print(
            f"{metric:<28} "
            f"{row['mean']:.6f} +/- {row['std']:.6f} "
            f"(min={row['min']:.6f}, max={row['max']:.6f})"
        )

    print("\nSaved results to:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
