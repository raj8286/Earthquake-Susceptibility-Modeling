# calibration_logloss_clipping_check.py


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

OOF = Path("probability_calibration_v5_30km_m4_18features") / "calibration_oof_predictions.csv"
T = "target_m4_0plus"
COLS = {
    "raw": "raw_susceptibility_score",
    "sigmoid": "sigmoid_calibrated_probability",
    "isotonic": "isotonic_calibrated_probability",
}
EPS = (1e-15, 1e-4)

oof = pd.read_csv(OOF)
print(f"cells: {len(oof):,} | positives: {int(oof[T].sum()):,} "
      f"({oof[T].mean()*100:.3f}%) | folds: {sorted(oof['oof_fold'].unique())}")

rows = []
for name, col in COLS.items():
    p = oof[col]
    row = {"method": name}
    for eps in EPS:
        per_fold = [
            log_loss(g[T], np.clip(g[col], eps, 1 - eps), labels=[0, 1])
            for _, g in oof.groupby("oof_fold")
        ]
        row[f"logloss_mean_eps{eps:g}"] = np.mean(per_fold)
        row[f"logloss_sd_eps{eps:g}"] = np.std(per_fold, ddof=1)
    row["positives_scored_0"] = int(((oof[T] == 1) & (p <= 0)).sum())
    row["negatives_scored_1"] = int(((oof[T] == 0) & (p >= 1)).sum())
    row["mean_probability"] = p.mean()
    rows.append(row)

out = pd.DataFrame(rows).set_index("method")
pd.set_option("display.width", 200)
print(out.round(5).to_string())
out.to_csv("logloss_clipping_check.csv")

print("\nPaper values:")
print("  eps=1e-15 mean log loss: raw 0.1431, sigmoid 0.1426, isotonic 0.1325")
print("  eps=1e-4  mean log loss: raw 0.1345, sigmoid 0.1426, isotonic 0.1323")
print("  raw positives scored 0: 194 | mean isotonic probability: 0.0744 | prevalence: 0.0747")