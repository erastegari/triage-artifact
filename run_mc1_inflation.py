#!/usr/bin/env python3
"""MC1: quantify eligibility inflation. Same model and cohort, evaluated two ways:
CONVENTIONAL = all students registered at the checkpoint (already-withdrawn included);
ELIGIBLE     = only students still enrolled at the checkpoint.
Report N, prevalence, AUC, and precision/recall/F1 at a 20% follow-up budget for both.
Within-cohort 5-fold CV isolates the evaluation-population effect."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from analysis_core import eligible, Preproc, FEATURES, CLIP_COLS, MODULES, DEV, CHECKPOINTS
cv = StratifiedKFold(5, shuffle=True, random_state=42)

def prep(df):
    nd = df[FEATURES].copy()
    for c in CLIP_COLS:
        nd[c] = nd[c].clip(nd[c].quantile(.01), nd[c].quantile(.99))
    return StandardScaler().fit_transform(nd)

def metrics(df):
    y = df["at_risk"].values
    X = prep(df)
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    k = max(1, round(0.20 * len(y)))
    idx = np.argsort(-p)[:k]
    tp = int(y[idx].sum())
    prec = tp / k
    rec = tp / max(1, y.sum())
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return len(y), y.mean(), auc, prec, rec, f1

rows = []
for mod in MODULES:
    for cp in CHECKPOINTS:
        full = eligible(mod, DEV, cp)              # all registered (conventional)
        elig = full[full.eligible].copy()          # still-enrolled (eligible)
        for label, d in [("Conventional", full), ("Eligible", elig)]:
            n, prev, auc, prec, rec, f1 = metrics(d)
            rows.append({"module": mod, "cp": cp, "evaluation": label, "n": n,
                         "prevalence": round(prev, 3), "AUC": round(auc, 3),
                         "precision@20": round(prec, 3), "recall@20": round(rec, 3),
                         "F1@20": round(f1, 3)})

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("=== MC1: CONVENTIONAL vs ELIGIBLE evaluation (development cohorts, 5-fold CV) ===")
print(res.to_string(index=False))

# deltas (conventional minus eligible)
print("\n=== Inflation (Conventional - Eligible) ===")
piv = res.pivot_table(index=["module", "cp"], columns="evaluation",
                      values=["AUC", "precision@20", "recall@20", "F1@20", "prevalence", "n"])
delt = []
for (mod, cp), g in res.groupby(["module", "cp"]):
    c = g[g.evaluation == "Conventional"].iloc[0]; e = g[g.evaluation == "Eligible"].iloc[0]
    delt.append({"module": mod, "cp": cp,
                 "d_AUC": round(c.AUC - e.AUC, 3),
                 "d_prec@20": round(c["precision@20"] - e["precision@20"], 3),
                 "d_recall@20": round(c["recall@20"] - e["recall@20"], 3),
                 "d_prevalence": round(c.prevalence - e.prevalence, 3),
                 "pct_already_withdrawn": round(100 * (1 - e.n / c.n), 1)})
dd = pd.DataFrame(delt)
print(dd.to_string(index=False))
res.to_csv("results/table_mc1_inflation.csv", index=False)
dd.to_csv("results/table_mc1_inflation_deltas.csv", index=False)
print("\nsaved results/table_mc1_inflation.csv")
