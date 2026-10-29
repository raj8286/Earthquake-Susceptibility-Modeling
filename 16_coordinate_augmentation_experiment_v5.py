# 16_coordinate_augmentation_experiment_v5.py
# V5 coordinate-augmentation sensitivity analysis for the canonical 30-km Extra Trees model.

from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

from grid_config_v5 import get_grid_config
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


COORD_FEATURES = [
    "x_6933_m",
    "y_6933_m",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the canonical 18 physical predictors against the same "
            "Extra Trees model augmented with EPSG:6933 x/y coordinates."
        )
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
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
        path = (
            SCRIPT_DIR
            / "coordinate_augmentation_v5_30km_m4_18features"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def best_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    if len(thresholds) == 0:
        return 0.5

    f1 = (
        2.0 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    if not np.isfinite(f1).any():
        return 0.5

    return float(thresholds[int(np.nanargmax(f1))])


def evaluate(y, p, threshold):
    pred = (p >= threshold).astype(np.int8)
    precision_curve, recall_curve, _ = precision_recall_curve(y, p)

    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
        "brier_score": float(brier_score_loss(y, p)),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "mcc": float(matthews_corrcoef(y, pred)),
        "threshold": float(threshold),
    }


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


def prepare_X(df, include_xy):
    cols = PHYSICAL_FEATURES.copy()
    if include_xy:
        cols += COORD_FEATURES

    X = df[cols].copy()
    for col in cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(np.float32)

    return X


def run_experiment(name, X, y, folds):
    rows = []

    print("\n" + "#" * 92)
    print(f"EXPERIMENT: {name}")
    print("#" * 92)

    for info in folds:
        fold = info["fold"]
        train_idx = info["train_idx"]
        val_idx = info["val_idx"]
        test_idx = info["test_idx"]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        y_test = y.iloc[test_idx]

        if (
            y_train.nunique() < 2
            or y_val.nunique() < 2
            or y_test.nunique() < 2
        ):
            raise RuntimeError(f"{name}, fold {fold} has a single-class split.")

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

        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        val_p = model.predict_proba(X_val)[:, 1]
        threshold = best_threshold(y_val, val_p)

        test_p = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(
            y_test,
            test_p,
            threshold,
        )

        rows.append(
            {
                "experiment": name,
                "fold": fold,
                "n_model_columns": X.shape[1],
                "fit_seconds": float(fit_seconds),
                **metrics,
            }
        )

        print(
            f"Fold {fold}: "
            f"ROC={metrics['roc_auc']:.6f} | "
            f"PR={metrics['pr_auc']:.6f} | "
            f"F1={metrics['f1']:.6f} | "
            f"MCC={metrics['mcc']:.6f}"
        )

    return pd.DataFrame(rows)



def physical_baseline_to_coordinate(baseline):
    return pd.DataFrame({
        "experiment": "physical_only",
        "fold": baseline["fold"].astype(int),
        "n_model_columns": 18,
        "fit_seconds": np.nan,
        "roc_auc": baseline["roc_auc"].astype(float),
        "pr_auc": baseline["pr_auc"].astype(float),
        "brier_score": baseline["brier_score"].astype(float),
        "precision": baseline["precision_at_best"].astype(float),
        "recall": baseline["recall_at_best"].astype(float),
        "f1": baseline["f1_at_best"].astype(float),
        "balanced_accuracy": baseline[
            "balanced_accuracy_at_best"
        ].astype(float),
        "mcc": baseline["mcc_at_best"].astype(float),
        "threshold": baseline["best_threshold"].astype(float),
    }).sort_values("fold").reset_index(drop=True)

def main():
    args = parse_args()
    input_file = resolve_input_file(args.input)
    output_dir = resolve_output_dir(args.output_dir)

    if not input_file.exists():
        raise FileNotFoundError(f"Dataset not found: {input_file.resolve()}")

    required = (
        PHYSICAL_FEATURES
        + COORD_FEATURES
        + [TARGET_COL, "cell_id", "ix", "iy"]
    )
    df = pd.read_csv(input_file, usecols=required)
    config = get_grid_config(GRID_KM)
    expected_rows = config.nx * config.ny
    if len(df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows:,} rows, found {len(df):,}."
        )

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["cell_id"].duplicated().any():
        raise ValueError("Duplicate cell_id values detected.")

    y = pd.to_numeric(df[TARGET_COL], errors="raise").astype(np.int8)
    if sorted(y.unique().tolist()) != [0, 1]:
        raise RuntimeError("Target must contain both classes 0 and 1.")

    groups = make_groups(df)
    folds = build_folds(len(df), y, groups)

    print("=" * 96)
    print("V5 COORDINATE-AUGMENTATION EXPERIMENT — 30 KM — M>=4")
    print("=" * 96)
    print(f"Input: {input_file.resolve()}")
    print(f"Cells: {len(df):,}")
    print(f"M>=4 positives: {int(y.sum()):,} ({100*y.mean():.3f}%)")
    print(f"Physical predictors: {len(PHYSICAL_FEATURES)}")
    print(f"Coordinate predictors: {len(COORD_FEATURES)}")
    print(f"Spatial blocks: {len(np.unique(groups)):,}")

    baseline = load_canonical_extratrees_baseline(
        SCRIPT_DIR,
        GRID_KM,
    )
    physical = physical_baseline_to_coordinate(baseline)
    print(
        "Reusing physical-only baseline from: "
        f"{canonical_extratrees_baseline_file(SCRIPT_DIR, GRID_KM).resolve()}"
    )

    plus_xy = run_experiment(
        "physical_plus_xy",
        prepare_X(df, include_xy=True),
        y,
        folds,
    )

    all_results = pd.concat(
        [physical, plus_xy],
        ignore_index=True,
    )
    all_results.to_csv(
        output_dir / "coordinate_ablation_fold_metrics.csv",
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
                "brier_score_std": float(d["brier_score"].std(ddof=1)),
                "f1_mean": float(d["f1"].mean()),
                "f1_std": float(d["f1"].std(ddof=1)),
                "precision_mean": float(d["precision"].mean()),
                "recall_mean": float(d["recall"].mean()),
                "balanced_accuracy_mean": float(
                    d["balanced_accuracy"].mean()
                ),
                "mcc_mean": float(d["mcc"].mean()),
                "mcc_std": float(d["mcc"].std(ddof=1)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / "coordinate_ablation_summary.csv",
        index=False,
    )

    base = summary[
        summary["experiment"] == "physical_only"
    ].iloc[0]
    coord = summary[
        summary["experiment"] == "physical_plus_xy"
    ].iloc[0]

    comparison = pd.DataFrame(
        [
            {
                "metric": "roc_auc",
                "physical_only": base["roc_auc_mean"],
                "physical_plus_xy": coord["roc_auc_mean"],
                "delta": coord["roc_auc_mean"] - base["roc_auc_mean"],
            },
            {
                "metric": "pr_auc",
                "physical_only": base["pr_auc_mean"],
                "physical_plus_xy": coord["pr_auc_mean"],
                "delta": coord["pr_auc_mean"] - base["pr_auc_mean"],
            },
            {
                "metric": "f1",
                "physical_only": base["f1_mean"],
                "physical_plus_xy": coord["f1_mean"],
                "delta": coord["f1_mean"] - base["f1_mean"],
            },
            {
                "metric": "mcc",
                "physical_only": base["mcc_mean"],
                "physical_plus_xy": coord["mcc_mean"],
                "delta": coord["mcc_mean"] - base["mcc_mean"],
            },
        ]
    )
    comparison.to_csv(
        output_dir / "coordinate_ablation_comparison.csv",
        index=False,
    )

    labels = ["ROC-AUC", "PR-AUC", "F1", "MCC"]
    base_vals = [
        base["roc_auc_mean"],
        base["pr_auc_mean"],
        base["f1_mean"],
        base["mcc_mean"],
    ]
    coord_vals = [
        coord["roc_auc_mean"],
        coord["pr_auc_mean"],
        coord["f1_mean"],
        coord["mcc_mean"],
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        base_vals,
        width,
        label="Physical only",
    )
    ax.bar(
        x + width / 2,
        coord_vals,
        width,
        label="Physical + x/y",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean 5-fold spatial-CV score")
    ax.set_title("V5 Effect of Explicit Geographic Coordinates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "coordinate_ablation_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    with (output_dir / "coordinate_ablation_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "version": "v5",
                "dataset": str(input_file.resolve()),
                "target": TARGET_COL,
                "grid_km": GRID_KM,
                "physical_feature_count": 18,
                "physical_features": PHYSICAL_FEATURES,
                "coordinate_features": COORD_FEATURES,
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
    print("COORDINATE-AUGMENTATION SUMMARY")
    print("=" * 112)
    print(summary.to_string(index=False))

    print("\nCOORDINATE EFFECT")
    print(comparison.to_string(index=False))

    print("\nSaved under:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
