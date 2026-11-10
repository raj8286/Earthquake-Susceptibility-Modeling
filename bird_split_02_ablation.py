# bird_split_02_ablation.py  (improved)

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

SCRIPT_DIR = Path(__file__).resolve().parent
GRID_KM = 30
FEAT_DIR = SCRIPT_DIR / "bird_split_30km"
FAULT_COLS = ["gem_fault_presence", "gem_fault_dist_km",
              "gem_fault_length_density_km_per_1000km2"]

def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cv = load_module("cv05", "05_extratrees_spatial_cv_grid_v5.py")
df, X_full, y = cv.load_dataset(cv.resolve_input_file(GRID_KM, None), GRID_KM)
folds = cv.build_fixed_spatial_folds(len(df), y, cv.make_spatial_groups(df, GRID_KM)[0])
nonfault = [c for c in cv.PHYSICAL_FEATURES if c not in FAULT_COLS]

def subset_faults(name):
    sub = pd.read_csv(FEAT_DIR / f"fault_features_{name}.csv").set_index("cell_id")
    return sub.loc[df["cell_id"].to_numpy(), FAULT_COLS].astype(np.float32)

def with_faults(*names):
    X = df[nonfault].astype(np.float32).copy()
    for n in names:
        sub = subset_faults(n)
        for c in FAULT_COLS:
            X[f"{n}__{c}"] = sub[c].to_numpy()
    return X

experiments = {
    "no_faults": df[nonfault].astype(np.float32),
    "non_bird_only": with_faults("non_bird_only"),
    "bird_only": with_faults("bird_only"),
    "bird_plus_non_bird_separate": with_faults("bird_only", "non_bird_only"),
    "all_faults_full_model": X_full[cv.PHYSICAL_FEATURES].astype(np.float32),
}

rows = []
for name, X in experiments.items():
    print(f"\nEXPERIMENT: {name} | columns={X.shape[1]}")
    for info in folds:
        tr, va, te = info["train_idx"], info["val_idx"], info["test_idx"]
        imp = SimpleImputer(strategy="median")
        Xtr, Xva, Xte = (imp.fit_transform(X.iloc[tr]),
                         imp.transform(X.iloc[va]), imp.transform(X.iloc[te]))
        model = cv.make_model().fit(Xtr, y.iloc[tr])
        thr = cv.choose_f1_threshold(y.iloc[va], model.predict_proba(Xva)[:, 1])
        m = cv.evaluate_predictions(y.iloc[te], model.predict_proba(Xte)[:, 1], thr)
        m.update(experiment=name, fold=info["fold"])
        rows.append(m)
        print(f"  fold {info['fold']}: PR={m['pr_auc']:.4f} F1={m['f1_at_best']:.4f} MCC={m['mcc_at_best']:.4f}")

res = pd.DataFrame(rows)
res.to_csv(FEAT_DIR / "bird_split_fold_metrics.csv", index=False)

metrics = ["roc_auc", "pr_auc", "f1_at_best", "mcc_at_best"]
order = list(experiments)
summ = res.groupby("experiment")[metrics].agg(["mean", "std"]).loc[order]
summ.to_csv(FEAT_DIR / "bird_split_summary.csv")
print("\n" + summ.round(4).to_string())

pr = res.pivot(index="fold", columns="experiment", values="pr_auc")
gain = pd.DataFrame({
    "mean_gain_vs_no_faults": [(pr[n] - pr["no_faults"]).mean() for n in order],
}, index=order)
gain.to_csv(FEAT_DIR / "bird_split_pr_auc_gain.csv")
print("\nPR-AUC gain over no-fault model (unrounded means)\n", gain.round(4).to_string())

print("\nPer-fold gain of adding non-Bird traces to Bird traces:")
print((pr["bird_plus_non_bird_separate"] - pr["bird_only"]).round(4).to_string())
print("\nPer-fold gain (all faults vs Bird only):")
print((pr["all_faults_full_model"] - pr["bird_only"]).round(4).to_string())









# # bird_split_02_ablation.py
# # It runs four Extra Trees experiments on identical spatial folds.

# import importlib.util
# from pathlib import Path

# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer

# SCRIPT_DIR = Path(__file__).resolve().parent
# GRID_KM = 30
# FEAT_DIR = SCRIPT_DIR / "bird_split_30km"

# FAULT_COLS = [
#     "gem_fault_presence",
#     "gem_fault_dist_km",
#     "gem_fault_length_density_km_per_1000km2",
# ]


# def load_module(name, filename):
#     spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
#     mod = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(mod)
#     return mod


# cv = load_module("cv05", "05_extratrees_spatial_cv_grid_v5.py")

# input_file = cv.resolve_input_file(GRID_KM, None)
# df, X_full, y = cv.load_dataset(input_file, GRID_KM)
# groups, _ = cv.make_spatial_groups(df, GRID_KM)
# folds = cv.build_fixed_spatial_folds(len(df), y, groups)

# nonfault = [c for c in cv.PHYSICAL_FEATURES if c not in FAULT_COLS]


# def with_subset_faults(name):
#     sub = pd.read_csv(FEAT_DIR / f"fault_features_{name}.csv").set_index("cell_id")
#     sub = sub.loc[df["cell_id"].to_numpy(), FAULT_COLS]
#     X = df[nonfault].copy().astype(np.float32)
#     for c in FAULT_COLS:
#         X[c] = sub[c].to_numpy(dtype=np.float32)
#     return X


# experiments = {
#     "no_faults": df[nonfault].astype(np.float32),
#     "non_bird_faults_only": with_subset_faults("non_bird_only"),
#     "bird_faults_only": with_subset_faults("bird_only"),
#     "all_faults_full_model": X_full[cv.PHYSICAL_FEATURES].astype(np.float32),
# }

# rows = []
# for name, X in experiments.items():
#     print("\n" + "#" * 80)
#     print(f"EXPERIMENT: {name} | columns={X.shape[1]}")
#     print("#" * 80)

#     for info in folds:
#         tr, va, te = info["train_idx"], info["val_idx"], info["test_idx"]

#         imputer = SimpleImputer(strategy="median")
#         X_tr = imputer.fit_transform(X.iloc[tr])
#         X_va = imputer.transform(X.iloc[va])
#         X_te = imputer.transform(X.iloc[te])

#         model = cv.make_model()
#         model.fit(X_tr, y.iloc[tr])

#         thr = cv.choose_f1_threshold(y.iloc[va], model.predict_proba(X_va)[:, 1])
#         m = cv.evaluate_predictions(y.iloc[te], model.predict_proba(X_te)[:, 1], thr)
#         m.update({"experiment": name, "fold": info["fold"]})
#         rows.append(m)

#         print(f"Fold {info['fold']}: PR={m['pr_auc']:.6f} | "
#               f"F1={m['f1_at_best']:.6f} | MCC={m['mcc_at_best']:.6f}")

# results = pd.DataFrame(rows)
# results.to_csv(FEAT_DIR / "bird_split_fold_metrics.csv", index=False)

# order = list(experiments.keys())
# summary = (
#     results.groupby("experiment")[["roc_auc", "pr_auc", "f1_at_best", "mcc_at_best"]]
#     .agg(["mean", "std"])
#     .loc[order]
# )
# summary.to_csv(FEAT_DIR / "bird_split_summary.csv")

# base = results[results.experiment == "no_faults"].set_index("fold")["pr_auc"]
# print("\n" + "=" * 90)
# print("SUMMARY (mean over 5 spatial folds; std uses n-1)")
# print("=" * 90)
# print(summary.round(4).to_string())

# print("\nPR-AUC gain over no-fault model")
# for name in order:
#     d = results[results.experiment == name].set_index("fold")["pr_auc"]
#     print(f"{name:<24} {float((d - base).mean()):+.4f}")