# 13_magnitude_threshold_sensitivity_v5.py
# Optimized v5 magnitude-threshold sensitivity for the canonical 30-km Extra Trees model.
# Reuses the already-completed M>=4 canonical baseline and trains only higher thresholds.

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

TARGETS = {
    "M>=4.0": "target_m4_0plus",
    "M>=4.5": "target_m4_5plus",
    "M>=5.0": "target_m5_0plus",
    "M>=5.5": "target_m5_5plus",
    "M>=6.0": "target_m6_0plus",
}

TRAIN_TARGETS = {
    "M>=4.5": "target_m4_5plus",
    "M>=5.0": "target_m5_0plus",
    "M>=5.5": "target_m5_5plus",
    "M>=6.0": "target_m6_0plus",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run optimized v5 30-km Extra Trees magnitude-threshold "
            "sensitivity, reusing the completed M>=4 baseline."
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
        path = SCRIPT_DIR / "magnitude_threshold_sensitivity_v5_30km_18features"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_spatial_groups(df):
    block_cells = int(round(SPATIAL_BLOCK_M / (GRID_KM * 1000.0)))
    if block_cells != 20:
        raise RuntimeError("Expected 20 cells per 600-km block edge.")

    ix = pd.to_numeric(df["ix"], errors="raise").to_numpy(dtype=np.int32)
    iy = pd.to_numeric(df["iy"], errors="raise").to_numpy(dtype=np.int32)
    return np.char.add(
        np.char.add((iy // block_cells).astype(str), "_"),
        (ix // block_cells).astype(str),
    )


def build_outer_folds(n_rows, groups):
    dummy = np.zeros((n_rows, 1), dtype=np.float32)
    dummy_y = np.zeros(n_rows, dtype=np.int8)
    outer = GroupKFold(n_splits=N_SPLITS)
    folds = []
    for fold, (outer_train_idx, test_idx) in enumerate(
        outer.split(dummy, dummy_y, groups=groups),
        start=1,
    ):
        folds.append(
            {
                "fold": fold,
                "outer_train_idx": np.asarray(outer_train_idx, dtype=np.int64),
                "test_idx": np.asarray(test_idx, dtype=np.int64),
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


def choose_f1_threshold(y_val, val_proba):
    precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    if not np.isfinite(f1).any():
        return 0.5
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(np.int8)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, proba)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "best_threshold": float(threshold),
        "precision_at_best": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "recall_at_best": float(recall_score(y_true, pred, zero_division=0)),
        "f1_at_best": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy_at_best": float(
            balanced_accuracy_score(y_true, pred)
        ),
        "mcc_at_best": float(matthews_corrcoef(y_true, pred)),
    }


def m4_baseline_to_magnitude(baseline, y_m4):
    positive_cells = int(y_m4.sum())
    prevalence = float(y_m4.mean())
    return pd.DataFrame(
        {
            "magnitude_threshold": "M>=4.0",
            "target_column": "target_m4_0plus",
            "fold": baseline["fold"].astype(int),
            "global_positive_cells": positive_cells,
            "global_prevalence": prevalence,
            "random_pr_baseline": prevalence,
            "train_rows": baseline.get("train_rows", np.nan),
            "validation_rows": baseline.get("validation_rows", np.nan),
            "test_rows": baseline.get("test_rows", np.nan),
            "train_positive_rate": baseline.get("train_positive_rate", np.nan),
            "validation_positive_rate": baseline.get(
                "validation_positive_rate", np.nan
            ),
            "test_positive_rate": baseline["test_positive_rate"].astype(float),
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
        }
    ).sort_values("fold").reset_index(drop=True)


def load_partial_targets(output_dir):
    partial_file = output_dir / "magnitude_threshold_fold_metrics_partial.csv"
    if not partial_file.exists():
        return [], set()

    partial = pd.read_csv(partial_file)
    if not {"magnitude_threshold", "fold"}.issubset(partial.columns):
        return [], set()

    frames = []
    completed = set()
    for label in TRAIN_TARGETS:
        d = partial[partial["magnitude_threshold"] == label].copy()
        folds = sorted(
            pd.to_numeric(d.get("fold"), errors="coerce")
            .dropna().astype(int).unique().tolist()
        ) if not d.empty else []
        if len(d) == N_SPLITS and folds == list(range(1, N_SPLITS + 1)):
            frames.append(d)
            completed.add(label)

    if completed:
        print(
            "Resuming completed magnitude targets: "
            + ", ".join(label for label in TRAIN_TARGETS if label in completed)
        )
    return frames, completed


def run_target(threshold_label, target_col, X, df, groups, outer_folds):
    y = pd.to_numeric(df[target_col], errors="raise").astype(np.int8)
    positive_cells = int(y.sum())
    prevalence = float(y.mean())

    print("\n" + "#" * 100)
    print(
        f"TARGET: {threshold_label} | positive={positive_cells:,} "
        f"({100*prevalence:.3f}%)"
    )
    print("#" * 100)

    rows = []
    for fold_info in outer_folds:
        fold = fold_info["fold"]
        outer_train_idx = fold_info["outer_train_idx"]
        test_idx = fold_info["test_idx"]

        inner = GroupShuffleSplit(
            n_splits=1,
            train_size=0.75,
            random_state=RANDOM_STATE + fold,
        )
        dummy_inner = np.zeros((len(outer_train_idx), 1), dtype=np.float32)
        train_rel, val_rel = next(
            inner.split(
                dummy_inner,
                y.iloc[outer_train_idx],
                groups=groups[outer_train_idx],
            )
        )
        train_idx = outer_train_idx[train_rel]
        val_idx = outer_train_idx[val_rel]

        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        test_groups = set(groups[test_idx])
        if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
            raise RuntimeError(
                f"{threshold_label}, fold {fold}: spatial block leakage."
            )

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        y_test = y.iloc[test_idx]
        if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
            raise RuntimeError(
                f"{threshold_label}, fold {fold} has a single-class split."
            )

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X.iloc[train_idx]).astype(
            np.float32, copy=False
        )
        X_val = imputer.transform(X.iloc[val_idx]).astype(np.float32, copy=False)
        X_test = imputer.transform(X.iloc[test_idx]).astype(np.float32, copy=False)

        model = make_model()
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = choose_f1_threshold(y_val, val_proba)
        test_proba = model.predict_proba(X_test)[:, 1]
        result = evaluate(y_test, test_proba, threshold)

        rows.append(
            {
                "magnitude_threshold": threshold_label,
                "target_column": target_col,
                "fold": fold,
                "global_positive_cells": positive_cells,
                "global_prevalence": prevalence,
                "random_pr_baseline": prevalence,
                "train_rows": len(train_idx),
                "validation_rows": len(val_idx),
                "test_rows": len(test_idx),
                "train_positive_rate": float(y_train.mean()),
                "validation_positive_rate": float(y_val.mean()),
                "test_positive_rate": float(y_test.mean()),
                "fit_seconds": float(fit_seconds),
                **result,
            }
        )

        print(
            f"Fold {fold}: ROC={result['roc_auc']:.6f} | "
            f"PR={result['pr_auc']:.6f} | F1={result['f1_at_best']:.6f} | "
            f"MCC={result['mcc_at_best']:.6f} | threshold={threshold:.6f}"
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    input_file = resolve_input_file(args.input)
    output_dir = resolve_output_dir(args.output_dir)

    if not input_file.exists():
        raise FileNotFoundError(f"Dataset not found: {input_file.resolve()}")

    required_cols = PHYSICAL_FEATURES + list(TARGETS.values()) + [
        "cell_id", "ix", "iy"
    ]
    df = pd.read_csv(input_file, usecols=required_cols)

    config = get_grid_config(GRID_KM)
    expected_rows = config.nx * config.ny
    if len(df) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows:,} rows, found {len(df):,}.")
    if df["cell_id"].duplicated().any():
        raise ValueError("Duplicate cell_id values detected.")

    X = df[PHYSICAL_FEATURES].copy()
    for col in PHYSICAL_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(np.float32)

    for target_col in TARGETS.values():
        values = pd.to_numeric(df[target_col], errors="raise").astype(np.int8)
        if sorted(values.unique().tolist()) != [0, 1]:
            raise RuntimeError(f"{target_col} must contain both binary classes.")

    groups = make_spatial_groups(df)
    outer_folds = build_outer_folds(len(df), groups)

    baseline = load_canonical_extratrees_baseline(SCRIPT_DIR, GRID_KM)
    y_m4 = pd.to_numeric(df["target_m4_0plus"], errors="raise").astype(np.int8)
    m4_baseline = m4_baseline_to_magnitude(baseline, y_m4)

    print("=" * 104)
    print("V5 30 KM EXTRA TREES — MAGNITUDE-THRESHOLD SENSITIVITY")
    print("=" * 104)
    print(f"Input: {input_file.resolve()}")
    print(f"Cells: {len(df):,}")
    print(f"Physical predictors: {len(PHYSICAL_FEATURES)}")
    print(f"Spatial blocks: {len(np.unique(groups)):,}")
    print(
        "Reusing M>=4 baseline from: "
        f"{canonical_extratrees_baseline_file(SCRIPT_DIR, GRID_KM).resolve()}"
    )

    trained_frames, completed = load_partial_targets(output_dir)
    for threshold_label, target_col in TRAIN_TARGETS.items():
        if threshold_label in completed:
            print(f"SKIP completed magnitude target: {threshold_label}")
            continue

        result = run_target(
            threshold_label,
            target_col,
            X,
            df,
            groups,
            outer_folds,
        )
        trained_frames.append(result)
        pd.concat(trained_frames, ignore_index=True).to_csv(
            output_dir / "magnitude_threshold_fold_metrics_partial.csv",
            index=False,
        )

    fold_results = pd.concat(
        [m4_baseline] + trained_frames,
        ignore_index=True,
    )
    fold_results.to_csv(
        output_dir / "magnitude_threshold_fold_metrics.csv",
        index=False,
    )

    metric_cols = [
        "roc_auc", "pr_auc", "brier_score", "f1_at_best",
        "precision_at_best", "recall_at_best", "balanced_accuracy_at_best",
        "mcc_at_best", "best_threshold", "fit_seconds",
    ]
    summary_rows = []
    for threshold_label in TARGETS:
        d = fold_results[fold_results["magnitude_threshold"] == threshold_label]
        row = {
            "magnitude_threshold": threshold_label,
            "positive_cells": int(d["global_positive_cells"].iloc[0]),
            "prevalence_percent": 100.0 * float(d["global_prevalence"].iloc[0]),
            "random_pr_baseline": float(d["random_pr_baseline"].iloc[0]),
        }
        for metric in metric_cols:
            vals = pd.to_numeric(d[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "magnitude_threshold_summary.csv", index=False)

    for metric, ylabel, filename in [
        ("pr_auc", "Mean PR-AUC", "magnitude_threshold_pr_auc.png"),
        ("roc_auc", "Mean ROC-AUC", "magnitude_threshold_roc_auc.png"),
        ("f1_at_best", "Mean F1", "magnitude_threshold_f1.png"),
        ("mcc_at_best", "Mean MCC", "magnitude_threshold_mcc.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            summary["magnitude_threshold"],
            summary[f"{metric}_mean"],
            yerr=summary[f"{metric}_std"],
            marker="o",
            capsize=4,
        )
        ax.set_xlabel("Magnitude threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(f"V5 Magnitude-Threshold Sensitivity — {ylabel}")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    with (output_dir / "magnitude_threshold_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "version": "v5",
                "dataset": str(input_file.resolve()),
                "grid_km": GRID_KM,
                "targets": TARGETS,
                "targets_trained_in_this_run": TRAIN_TARGETS,
                "m4_baseline_reused_from": str(
                    canonical_extratrees_baseline_file(
                        SCRIPT_DIR, GRID_KM
                    ).resolve()
                ),
                "physical_feature_count": 18,
                "physical_features": PHYSICAL_FEATURES,
                "spatial_block_m": SPATIAL_BLOCK_M,
                "n_spatial_folds": N_SPLITS,
                "model": "ExtraTreesClassifier",
                "n_jobs": N_JOBS,
                "random_state": RANDOM_STATE,
                "pr_auc_definition": (
                    "trapezoidal area under precision-recall curve"
                ),
                "pr_auc_random_baseline": (
                    "positive-class prevalence for each magnitude target"
                ),
            },
            stream,
            indent=2,
        )

    print("\n" + "=" * 120)
    print("MAGNITUDE-THRESHOLD SENSITIVITY SUMMARY")
    print("=" * 120)
    display = [
        "magnitude_threshold", "positive_cells", "prevalence_percent",
        "random_pr_baseline", "roc_auc_mean", "roc_auc_std", "pr_auc_mean",
        "pr_auc_std", "f1_at_best_mean", "f1_at_best_std",
        "precision_at_best_mean", "recall_at_best_mean", "mcc_at_best_mean",
        "mcc_at_best_std",
    ]
    print(summary[display].to_string(index=False))
    print("\nSaved under:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
