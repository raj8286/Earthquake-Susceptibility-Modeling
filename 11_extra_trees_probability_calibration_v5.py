# 11_extra_trees_probability_calibration_v5.py
# V5 probability-calibration analysis for the canonical 30-km Extra Trees M>=4 model.

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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from grid_config_v5 import get_grid_config


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



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run v5 raw/sigmoid/isotonic probability calibration for the "
            "canonical 30-km Extra Trees M>=4 model."
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
        path = SCRIPT_DIR / "probability_calibration_v5_30km_m4_18features"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ece(y_true, p, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(p, edges[1:-1], right=False)
    out = 0.0
    for b in range(n_bins):
        mask = ids == b
        if mask.any():
            out += mask.mean() * abs(
                p[mask].mean() - y_true[mask].mean()
            )
    return float(out)


def reliability_table(y_true, p, method, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(p, edges[1:-1], right=False)

    rows = []
    for b in range(n_bins):
        mask = ids == b
        rows.append(
            {
                "method": method,
                "bin": b + 1,
                "bin_left": float(edges[b]),
                "bin_right": float(edges[b + 1]),
                "count": int(mask.sum()),
                "mean_predicted_probability": (
                    float(p[mask].mean()) if mask.any() else np.nan
                ),
                "observed_positive_fraction": (
                    float(y_true[mask].mean()) if mask.any() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def metrics(y_true, p):
    p = np.asarray(p, dtype=float)
    safe = np.clip(p, 1e-15, 1.0 - 1e-15)

    precision, recall, _ = precision_recall_curve(y_true, p)
    pr_auc = auc(recall, precision)

    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, safe, labels=[0, 1])),
        "ece_10bin": ece(y_true, p, 10),
        "mean_predicted_probability": float(p.mean()),
        "observed_positive_fraction": float(np.mean(y_true)),
    }


def choose_f1_threshold(y_val, val_score):
    precision, recall, thresholds = precision_recall_curve(
        y_val,
        val_score,
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


def load_dataset(input_file):
    if not input_file.exists():
        raise FileNotFoundError(f"Dataset not found: {input_file.resolve()}")

    required = PHYSICAL_FEATURES + [
        TARGET_COL,
        "cell_id",
        "ix",
        "iy",
        "x_6933_m",
        "y_6933_m",
        "lon",
        "lat",
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
        raise ValueError(f"Missing required columns: {missing}")

    if df["cell_id"].duplicated().any():
        raise RuntimeError("Duplicate cell_id values found.")

    X = df[PHYSICAL_FEATURES].copy()
    for col in PHYSICAL_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(np.float32)

    y = pd.to_numeric(df[TARGET_COL], errors="raise").astype(np.int8)
    if sorted(y.unique().tolist()) != [0, 1]:
        raise RuntimeError("Target must contain both classes 0 and 1.")

    return df, X, y


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


def main():
    args = parse_args()
    input_file = resolve_input_file(args.input)
    output_dir = resolve_output_dir(args.output_dir)

    df, X, y = load_dataset(input_file)
    groups = make_groups(df)

    print("=" * 96)
    print("V5 30 KM EXTRA TREES PROBABILITY CALIBRATION")
    print("=" * 96)
    print(f"Input: {input_file.resolve()}")
    print(f"Cells: {len(df):,}")
    print(f"M>=4 positives: {int(y.sum()):,} ({100*y.mean():.3f}%)")
    print(f"Spatial blocks: {len(np.unique(groups)):,}")

    outer = GroupKFold(n_splits=N_SPLITS)
    dummy = np.zeros((len(df), 1), dtype=np.float32)

    fold_rows = []
    oof_rows = []

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

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        y_test = y.iloc[test_idx]

        if (
            y_train.nunique() < 2
            or y_val.nunique() < 2
            or y_test.nunique() < 2
        ):
            raise RuntimeError(f"Fold {fold} has a single-class split.")

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

        val_raw = model.predict_proba(X_val)[:, 1]
        test_raw = model.predict_proba(X_test)[:, 1]

        fold_threshold = choose_f1_threshold(y_val, val_raw)
        test_positive = (test_raw >= fold_threshold).astype(np.int8)

        sigmoid = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        sigmoid.fit(val_raw.reshape(-1, 1), y_val)
        test_sigmoid = sigmoid.predict_proba(
            test_raw.reshape(-1, 1)
        )[:, 1]

        isotonic = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        isotonic.fit(val_raw, y_val)
        test_isotonic = isotonic.predict(test_raw)

        predictions = {
            "raw": test_raw,
            "sigmoid": test_sigmoid,
            "isotonic": test_isotonic,
        }

        print(
            f"\nFold {fold}: train={len(train_idx):,}, "
            f"val={len(val_idx):,}, test={len(test_idx):,}"
        )

        for method, pvals in predictions.items():
            m = metrics(y_test, pvals)
            fold_rows.append({"fold": fold, "method": method, **m})
            print(
                f"{method:<8} "
                f"Brier={m['brier_score']:.6f} "
                f"LogLoss={m['log_loss']:.6f} "
                f"ECE={m['ece_10bin']:.6f} "
                f"ROC={m['roc_auc']:.6f} "
                f"PR={m['pr_auc']:.6f}"
            )

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
        fold_oof["raw_susceptibility_score"] = test_raw
        fold_oof["sigmoid_calibrated_probability"] = test_sigmoid
        fold_oof["isotonic_calibrated_probability"] = test_isotonic
        fold_oof["fold_operating_threshold"] = fold_threshold
        fold_oof["predicted_m4_positive"] = test_positive
        oof_rows.append(fold_oof)

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(
        output_dir / "calibration_fold_metrics.csv",
        index=False,
    )

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
        raise RuntimeError("OOF predictions do not cover every row exactly once.")

    oof.to_csv(
        output_dir / "calibration_oof_predictions.csv",
        index=False,
    )

    methods = {
        "raw": "raw_susceptibility_score",
        "sigmoid": "sigmoid_calibrated_probability",
        "isotonic": "isotonic_calibrated_probability",
    }

    summary_rows = []
    for method in methods:
        d = fold_metrics[fold_metrics["method"] == method]
        row = {"method": method}
        for col in [
            "brier_score",
            "log_loss",
            "ece_10bin",
            "roc_auc",
            "pr_auc",
            "mean_predicted_probability",
            "observed_positive_fraction",
        ]:
            row[f"{col}_mean"] = float(d[col].mean())
            row[f"{col}_std"] = float(d[col].std(ddof=1))
        summary_rows.append(row)

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["brier_score_mean", "log_loss_mean"])
        .reset_index(drop=True)
    )
    summary.to_csv(
        output_dir / "calibration_summary.csv",
        index=False,
    )

    pooled_rows = []
    reliability_frames = []

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    for method, col in methods.items():
        m = metrics(oof[TARGET_COL], oof[col])
        pooled_rows.append({"method": method, **m})

        table = reliability_table(
            oof[TARGET_COL],
            oof[col],
            method,
            n_bins=10,
        )
        reliability_frames.append(table)

        good = table["count"] > 0
        ax.plot(
            table.loc[good, "mean_predicted_probability"],
            table.loc[good, "observed_positive_fraction"],
            marker="o",
            label=method,
        )

    pooled = (
        pd.DataFrame(pooled_rows)
        .sort_values(["brier_score", "log_loss"])
        .reset_index(drop=True)
    )
    pooled.to_csv(
        output_dir / "calibration_pooled_metrics.csv",
        index=False,
    )

    reliability = pd.concat(reliability_frames, ignore_index=True)
    reliability.to_csv(
        output_dir / "calibration_reliability_bins.csv",
        index=False,
    )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("V5 30 km Extra Trees M>=4 — Spatial Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "calibration_reliability_diagram.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    with (output_dir / "calibration_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "version": "v5",
                "dataset": str(input_file.resolve()),
                "target": TARGET_COL,
                "grid_km": GRID_KM,
                "spatial_block_km": 600,
                "outer_folds": N_SPLITS,
                "inner_train_fraction": 0.75,
                "physical_feature_count": 18,
                "physical_features": PHYSICAL_FEATURES,
                "coordinates_used_as_predictors": False,
                "base_model": "ExtraTreesClassifier",
                "base_model_class_weight": "balanced",
                "calibration_methods": ["raw", "sigmoid", "isotonic"],
                "calibrator_fit_data": (
                    "inner spatial validation predictions only"
                ),
                "evaluation_data": (
                    "untouched outer spatial test folds only"
                ),
                "pr_auc_definition": (
                    "trapezoidal area under precision-recall curve"
                ),
                "random_state": RANDOM_STATE,
            },
            stream,
            indent=2,
        )

    print("\n" + "=" * 112)
    print("CALIBRATION SUMMARY — MEAN +/- SD ACROSS 5 SPATIAL FOLDS")
    print("=" * 112)
    print(
        summary[
            [
                "method",
                "brier_score_mean",
                "brier_score_std",
                "log_loss_mean",
                "log_loss_std",
                "ece_10bin_mean",
                "ece_10bin_std",
                "roc_auc_mean",
                "pr_auc_mean",
                "mean_predicted_probability_mean",
                "observed_positive_fraction_mean",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 112)
    print("POOLED OUT-OF-FOLD CALIBRATION METRICS")
    print("=" * 112)
    print(pooled.to_string(index=False))

    print("\nSaved under:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
