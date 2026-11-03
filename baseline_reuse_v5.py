# baseline_reuse_v5.py
# Shared validation/loading for the already-completed canonical Extra Trees v5 baseline.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


N_SPLITS = 5
TARGET_COL = "target_m4_0plus"

REQUIRED_BASELINE_COLUMNS = {
    "fold",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "best_threshold",
    "precision_at_best",
    "recall_at_best",
    "f1_at_best",
    "balanced_accuracy_at_best",
    "mcc_at_best",
    "test_positive_rate",
}


def canonical_extratrees_baseline_file(script_dir: Path, grid_km: int) -> Path:
    return (
        Path(script_dir)
        / f"extra_trees_spatial_cv_v5_m4_18features_{int(grid_km)}km"
        / "spatial_cv_fold_metrics.csv"
    )


def load_canonical_extratrees_baseline(
    script_dir: Path,
    grid_km: int,
) -> pd.DataFrame:
    path = canonical_extratrees_baseline_file(script_dir, grid_km)

    if not path.exists():
        raise FileNotFoundError(
            "Canonical Extra Trees baseline is required before optimized "
            f"reuse. Missing: {path.resolve()}"
        )

    df = pd.read_csv(path)

    missing = sorted(REQUIRED_BASELINE_COLUMNS - set(df.columns))
    if missing:
        raise RuntimeError(
            f"Baseline file is missing required columns: {missing}"
        )

    folds = sorted(
        pd.to_numeric(df["fold"], errors="raise")
        .astype(int)
        .tolist()
    )
    expected_folds = list(range(1, N_SPLITS + 1))

    if len(df) != N_SPLITS or folds != expected_folds:
        raise RuntimeError(
            "Canonical Extra Trees baseline must contain exactly one row for "
            f"folds {expected_folds}; got {folds}."
        )

    if "grid_km" in df.columns:
        grid_values = sorted(
            pd.to_numeric(df["grid_km"], errors="raise")
            .astype(int)
            .unique()
            .tolist()
        )
        if grid_values != [int(grid_km)]:
            raise RuntimeError(
                f"Baseline grid mismatch: expected {grid_km}, got {grid_values}."
            )

    if "target" in df.columns:
        targets = sorted(df["target"].astype(str).unique().tolist())
        if targets != [TARGET_COL]:
            raise RuntimeError(
                f"Baseline target mismatch: expected {TARGET_COL}, got {targets}."
            )

    numeric_metric_cols = [
        "roc_auc",
        "pr_auc",
        "brier_score",
        "best_threshold",
        "precision_at_best",
        "recall_at_best",
        "f1_at_best",
        "balanced_accuracy_at_best",
        "mcc_at_best",
        "test_positive_rate",
    ]
    for col in numeric_metric_cols:
        values = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Baseline column {col} contains non-finite values.")

    return df.sort_values("fold").reset_index(drop=True)
