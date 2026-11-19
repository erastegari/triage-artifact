#!/usr/bin/env python3
"""Why LR / shallow tree and not something fancier? Expanded bake-off on the eligible
population: LR, shallow tree (d3), full tree, random forest, gradient boosting.
Same 8 features, same 5-fold CV, same cohorts/checkpoints. Accuracy vs interpretability cost.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from repro import build_model_df, NETCOLS

BURDEN = 0.20
COHORTS = [("BBB", "2013J"), ("BBB", "2014J"), ("DDD", "2013J"), ("DDD", "2014J")]
CPS = [28, 56, 84]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def eligible(mod, pres, cp):
    m = build_model_df(mod, pres, cp)
    m = m[~(m["unreg"] < cp)].copy()
    cols = [c for c in NETCOLS if m[c].nunique() > 1]
    nd = m[cols].copy()
    for c in ["total_clicks", "avg_clicks_per_active_day", "mean_lateness"]:
        if c in nd: nd[c] = nd[c].clip(nd[c].quantile(.01), nd[c].quantile(.99))
    return nd.values, m["at_risk"].values

def hgb():
    try:    return HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
    except TypeError: return HistGradientBoostingClassifier(random_state=42)

MODELS = {
    "LogReg":        lambda: LogisticRegression(class_weight="balanced", max_iter=2000),
    "Tree(d3)":      lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42),
    "Tree(full)":    lambda: DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "RandomForest":  lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    "GradBoost":     hgb,
}
# human-readable size (how much a person must read to audit the model)
PARAMS = {"LogReg": "8 coefficients", "Tree(d3)": "<=8 leaves (explicit rules)",
          "Tree(full)": "100s of leaves", "RandomForest": "300 trees", "GradBoost": "100s of trees"}

def cap_at_burden(y, score, burden=BURDEN):
    n = len(y); k = max(1, round(burden * n))
    idx = np.argsort(-score)[:k]
    tp = int(y[idx].sum())
    return tp / max(1, y.sum())

rows = []
for mod, pres in COHORTS:
    for cp in CPS:
        X, y = eligible(mod, pres, cp)
        if y.sum() < 20 or (y == 0).sum() < 20:
            continue
        Xs = StandardScaler().fit_transform(X)
        rec = {"cohort": f"{mod}-{pres}", "cp": cp}
        for name, mk in MODELS.items():
            Xin = Xs if name == "LogReg" else X
            p = cross_val_predict(mk(), Xin, y, cv=cv, method="predict_proba")[:, 1]
            rec[f"AUC_{name}"] = roc_auc_score(y, p)
            rec[f"cap_{name}"] = cap_at_burden(y, p)
        rows.append(rec)

df = pd.DataFrame(rows)
auc_cols = [f"AUC_{n}" for n in MODELS]
cap_cols = [f"cap_{n}" for n in MODELS]

print("=== Mean across 12 cohort x checkpoint settings (eligible population) ===")
summ = pd.DataFrame({
    "model": list(MODELS),
    "mean_AUC": [df[f"AUC_{n}"].mean() for n in MODELS],
    "mean_cap@20": [df[f"cap_{n}"].mean() for n in MODELS],
    "AUC_vs_LR": [df[f"AUC_{n}"].mean() - df["AUC_LogReg"].mean() for n in MODELS],
    "times_best_AUC": [int((df[auc_cols].idxmax(axis=1) == f"AUC_{n}").sum()) for n in MODELS],
    "to_audit": [PARAMS[n] for n in MODELS],
}).round(3)
print(summ.to_string(index=False))

print("\n=== AUC by cohort (mean over checkpoints) ===")
by_cohort = df.groupby("cohort")[auc_cols].mean().round(3)
by_cohort.columns = [c.replace("AUC_", "") for c in by_cohort.columns]
print(by_cohort.to_string())

# head-to-head gap: best ensemble minus LR, per setting
df["best_ensemble"] = df[["AUC_RandomForest", "AUC_GradBoost"]].max(axis=1)
df["ensemble_minus_LR"] = df["best_ensemble"] - df["AUC_LogReg"]
print(f"\nBest ensemble (RF or GBM) minus LR, per setting:")
print(f"   mean={df['ensemble_minus_LR'].mean():+.3f}  median={df['ensemble_minus_LR'].median():+.3f}"
      f"  min={df['ensemble_minus_LR'].min():+.3f}  max={df['ensemble_minus_LR'].max():+.3f}")
print(f"   settings where an ensemble beats LR by >0.02 AUC: {(df['ensemble_minus_LR']>0.02).sum()} / {len(df)}")
df.round(3).to_csv("model_bakeoff.csv", index=False)
print("\nsaved model_bakeoff.csv")
