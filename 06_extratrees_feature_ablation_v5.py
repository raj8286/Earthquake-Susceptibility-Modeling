# 06_extratrees_feature_ablation_v5.py
# V5 feature-family ablation for the final 18-feature, 30-km Extra Trees model.
#
# Important:
#   This script ablates ONLY the canonical final 18-feature v5 model.
#   EMAG2 is not part of the v5 model dataset and is therefore not included here.
#   Magnetic screening should be treated as a separate pre-selection experiment.

from __future__ import annotations

from pathlib import Path
import argparse
import json

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


FEATURE_FAMILIES = {
    "surface_morphology": [
        "elevation_m",
        "terrain_slope_deg",
    ],
    "gravity_geoid": [
        "freeair_mgal",
        "bouguer_mgal",
        "isostatic_mgal",
        "geoid_umg3",
        "lithoref_geoid_m",
        "lithoref_fa_mgal",
    ],
    "crust_sediment": [
        "sedthk_km",
        "crsthk_km",
        "lithoref_moho_depth_km",
    ],
    "lithosphere_density": [
        "lithoref_lab_depth_km",
        "lithoref_rho_c",
        "lithoref_rho_l",
        "lithoref_rho_sl",
    ],
    "faults": [
        "gem_fault_presence",
        "gem_fault_dist_km",
        "gem_fault_length_density_km_per_1000km2",
    ],
}


def validate_feature_families():
    flattened = [
        feature
        for family in FEATURE_FAMILIES.values()
        for feature in family
    ]
    missing = sorted(set(PHYSICAL_FEATURES) - set(flattened))
    extra = sorted(set(flattened) - set(PHYSICAL_FEATURES))
    duplicates = sorted(
        {
            feature
            for feature in flattened
            if flattened.count(feature) > 1
        }
    )
    if missing or extra or duplicates:
        raise RuntimeError(
            "FEATURE_FAMILIES must be an exact non-overlapping partition of "
            f"the 18 canonical predictors. Missing={missing}, "
            f"extra={extra}, duplicated={duplicates}"
        )


validate_feature_families()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run v5 Extra Trees feature-family ablation on the canonical "
            "30-km M>=4 model."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Optional model dataset. Default: "
            "data/grid_30km/model/model_dataset_cell_30km_v5.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output-directory override.",
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
        path = SCRIPT_DIR / "extra_trees_ablation_v5_m4_18features_30km"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(input_file):
    if not input_file.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_file.resolve()}")

    required = PHYSICAL_FEATURES + [
        TARGET_COL,
        "cell_id",
        "ix",
        "iy",
    ]
    df = pd.read_csv(input_file, usecols=required)
    config = get_grid_config(GRID_KM)
    expected_rows = config.nx * config.ny

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows:,} rows, found {len(df):,}."
        )

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df["cell_id"].duplicated().any():
        raise ValueError("Duplicate cell_id values detected.")

    y = pd.to_numeric(df[TARGET_COL], errors="raise").astype(np.int8)
    if sorted(y.unique().tolist()) != [0, 1]:
        raise RuntimeError("Target must contain both classes 0 and 1.")

    return df, y


def make_spatial_groups(df):
    cells_per_block_edge = int(round(SPATIAL_BLOCK_M / (GRID_KM * 1000.0)))
    if cells_per_block_edge != 20:
        raise RuntimeError("Expected 20 cells per 600-km block edge at 30 km.")

    ix = pd.to_numeric(df["ix"], errors="raise").to_numpy(dtype=np.int32)
    iy = pd.to_numeric(df["iy"], errors="raise").to_numpy(dtype=np.int32)

    bx = ix // cells_per_block_edge
    by = iy // cells_per_block_edge

    groups = np.char.add(
        np.char.add(by.astype(str), "_"),
        bx.astype(str),
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


def build_ablation_experiments():
    experiments = {}
    for family_name, removed in FEATURE_FAMILIES.items():
        experiments[f"no_{family_name}"] = [
            feature
            for feature in PHYSICAL_FEATURES
            if feature not in removed
        ]
    return experiments


ABLATION_EXPERIMENTS = build_ablation_experiments()
ALL_EXPERIMENTS = {
    "full": PHYSICAL_FEATURES.copy(),
    **ABLATION_EXPERIMENTS,
}


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
    f1 = (
        2.0 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    if not np.isfinite(f1).any():
        return 0.5
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate_predictions(y_true, proba, threshold):
    pred = (proba >= threshold).astype(np.int8)
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_true,
        proba,
    )
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
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
        "mcc_at_best": float(matthews_corrcoef(y_true, pred)),
    }


def run_experiment(name, features, df, y, folds):
    X = df[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(np.float32)

    removed = sorted(set(PHYSICAL_FEATURES) - set(features))
    rows = []

    print("\n" + "#" * 96)
    print(f"ABLATION: {name}")
    print(f"Retained: {len(features)} | Removed: {removed or 'none'}")
    print("#" * 96)

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
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = choose_f1_threshold(y_val, val_proba)
        test_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_predictions(
            y_test,
            test_proba,
            threshold,
        )
        metrics.update(
            {
                "experiment": name,
                "fold": fold,
                "physical_feature_count": len(features),
                "removed_features": "|".join(removed),
                "test_positive_rate": float(y_test.mean()),
            }
        )
        rows.append(metrics)

        print(
            f"Fold {fold}: ROC={metrics['roc_auc']:.6f} | "
            f"PR={metrics['pr_auc']:.6f} | "
            f"F1={metrics['f1_at_best']:.6f} | "
            f"MCC={metrics['mcc_at_best']:.6f}"
        )

    return pd.DataFrame(rows)



def full_baseline_to_ablation(baseline):
    return pd.DataFrame({
        "experiment": "full",
        "fold": baseline["fold"].astype(int),
        "physical_feature_count": 18,
        "removed_features": "",
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


def load_ablation_partial(output_dir):
    partial_file = output_dir / "ablation_fold_metrics_partial.csv"
    if not partial_file.exists():
        return [], set()

    partial = pd.read_csv(partial_file)
    if not {"experiment", "fold"}.issubset(partial.columns):
        return [], set()

    frames = []
    completed = set()
    for name in ABLATION_EXPERIMENTS:
        d = partial[partial["experiment"] == name].copy()
        folds = sorted(
            pd.to_numeric(d.get("fold"), errors="coerce")
            .dropna().astype(int).unique().tolist()
        ) if not d.empty else []
        if len(d) == N_SPLITS and folds == list(range(1, N_SPLITS + 1)):
            frames.append(d)
            completed.add(name)

    if completed:
        print(
            "Resuming completed ablation families: "
            + ", ".join(name for name in ABLATION_EXPERIMENTS if name in completed)
        )

    return frames, completed

def summarize(all_results):
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
    ]
    rows = []

    for name, group in all_results.groupby("experiment"):
        row = {
            "experiment": name,
            "physical_feature_count": int(
                group["physical_feature_count"].iloc[0]
            ),
            "removed_features": group["removed_features"].iloc[0],
        }
        for metric in metric_cols:
            values = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["_order"] = np.where(summary["experiment"] == "full", 0, 1)
    summary = (
        summary.sort_values(["_order", "experiment"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    full = summary[summary["experiment"] == "full"].iloc[0]
    for metric in ["roc_auc", "pr_auc", "f1_at_best", "mcc_at_best"]:
        summary[f"delta_{metric}_vs_full"] = (
            summary[f"{metric}_mean"] - full[f"{metric}_mean"]
        )

    return summary


def main():
    args = parse_args()
    input_file = resolve_input_file(args.input)
    output_dir = resolve_output_dir(args.output_dir)

    df, y = load_dataset(input_file)
    groups, cells_per_block_edge = make_spatial_groups(df)
    folds = build_fixed_spatial_folds(len(df), y, groups)

    print("=" * 100)
    print("V5 EXTRA TREES FEATURE-FAMILY ABLATION — 30 KM — M>=4")
    print("=" * 100)
    print(f"Input: {input_file.resolve()}")
    print(f"Rows: {len(df):,}")
    print(f"Positive cells: {int(y.sum()):,} ({100*y.mean():.3f}%)")
    print(f"Spatial blocks: {len(np.unique(groups)):,}")
    print(f"Cells per block edge: {cells_per_block_edge}")
    print("Magnetic family: not part of this final-18 v5 ablation")

    baseline = load_canonical_extratrees_baseline(
        SCRIPT_DIR,
        GRID_KM,
    )
    full_baseline = full_baseline_to_ablation(baseline)
    print(
        "Reusing full-model baseline from: "
        f"{canonical_extratrees_baseline_file(SCRIPT_DIR, GRID_KM).resolve()}"
    )

    results, completed = load_ablation_partial(output_dir)

    for name, features in ABLATION_EXPERIMENTS.items():
        if name in completed:
            print(f"SKIP completed ablation family: {name}")
            continue

        result = run_experiment(
            name,
            features,
            df,
            y,
            folds,
        )
        results.append(result)

        pd.concat(results, ignore_index=True).to_csv(
            output_dir / "ablation_fold_metrics_partial.csv",
            index=False,
        )

    all_results = pd.concat(
        [full_baseline] + results,
        ignore_index=True,
    )
    all_results.to_csv(
        output_dir / "ablation_fold_metrics.csv",
        index=False,
    )

    summary = summarize(all_results)
    summary.to_csv(
        output_dir / "ablation_summary.csv",
        index=False,
    )

    plot_df = summary.sort_values("pr_auc_mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        plot_df["experiment"],
        plot_df["pr_auc_mean"],
        xerr=plot_df["pr_auc_std"],
    )
    ax.set_xlabel("Mean spatial CV PR-AUC")
    ax.set_title("V5 Extra Trees Feature-Family Ablation (30 km, M>=4)")
    fig.tight_layout()
    fig.savefig(
        output_dir / "ablation_pr_auc_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    with (output_dir / "ablation_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "version": "v5",
                "input_file": str(input_file.resolve()),
                "target": TARGET_COL,
                "grid_km": GRID_KM,
                "spatial_block_m": SPATIAL_BLOCK_M,
                "cells_per_block_edge": cells_per_block_edge,
                "n_spatial_folds": N_SPLITS,
                "physical_feature_count": 18,
                "physical_features": PHYSICAL_FEATURES,
                "feature_families": FEATURE_FAMILIES,
                "experiments": ALL_EXPERIMENTS,
                "full_baseline_reused_from": str(
                    canonical_extratrees_baseline_file(
                        SCRIPT_DIR, GRID_KM
                    ).resolve()
                ),
                "coordinates_used_as_predictors": False,
                "emag2_used": False,
                "magnetic_screening_in_this_script": False,
                "model": "ExtraTreesClassifier",
                "random_state": RANDOM_STATE,
                "pr_auc_definition": (
                    "trapezoidal area under precision-recall curve"
                ),
            },
            stream,
            indent=2,
        )

    print("\n" + "=" * 118)
    print("FEATURE-FAMILY ABLATION SUMMARY")
    print("=" * 118)
    display = [
        "experiment",
        "physical_feature_count",
        "roc_auc_mean",
        "delta_roc_auc_vs_full",
        "pr_auc_mean",
        "delta_pr_auc_vs_full",
        "f1_at_best_mean",
        "delta_f1_at_best_vs_full",
        "mcc_at_best_mean",
        "delta_mcc_at_best_vs_full",
    ]
    print(summary[display].to_string(index=False))
    print("\nSaved under:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
