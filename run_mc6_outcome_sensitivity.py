#!/usr/bin/env python3
"""MC6: outcome-definition sensitivity. The label combines Fail and Withdrawn. Here we
(1) report the Withdrawn:Fail composition of the eligible at-risk group, and
(2) train the same features to separate each failure mode from success:
   Withdraw-vs-Pass (drop Fail) and Fail-vs-Pass (drop Withdrawn), eligible population, CV AUC."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from analysis_core import eligible_only, FEATURES, CLIP_COLS, MODULES, DEV, CHECKPOINTS
cv = StratifiedKFold(5, shuffle=True, random_state=42)

def auc(m, pos_mask, neg_mask):
    d = m[pos_mask | neg_mask].copy()
    y = pos_mask[pos_mask | neg_mask].astype(int).values
    nd = d[FEATURES].copy()
    for c in CLIP_COLS: nd[c] = nd[c].clip(nd[c].quantile(.01), nd[c].quantile(.99))
    X = StandardScaler().fit_transform(nd)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 10: return np.nan, int(y.sum()), int((y==0).sum())
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p), int(y.sum()), int((y==0).sum())

rows = []
for mod in MODULES:
    for cp in CHECKPOINTS:
        m = eligible_only(mod, DEV, cp)
        fr = m["final_result"]
        is_w = fr.eq("Withdrawn"); is_f = fr.eq("Fail"); is_pass = fr.isin(["Pass", "Distinction"])
        n_atrisk = int((is_w | is_f).sum())
        w_auc, nw, _ = auc(m, is_w, is_pass)     # withdraw vs pass/distinction
        f_auc, nf, _ = auc(m, is_f, is_pass)     # fail vs pass/distinction
        rows.append({"module": mod, "cp": cp, "n_eligible": len(m),
                     "n_withdrawn": int(is_w.sum()), "n_fail": int(is_f.sum()),
                     "withdrawn_pct_of_atrisk": round(100*is_w.sum()/max(1, n_atrisk), 1),
                     "AUC_withdraw_vs_pass": round(w_auc, 3) if w_auc==w_auc else np.nan,
                     "AUC_fail_vs_pass": round(f_auc, 3) if f_auc==f_auc else np.nan})
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("=== MC6: outcome-definition sensitivity (eligible, development cohorts, CV) ===")
print(res.to_string(index=False))
res.to_csv("results/table_mc6_outcome_sensitivity.csv", index=False)
print("saved results/table_mc6_outcome_sensitivity.csv")
