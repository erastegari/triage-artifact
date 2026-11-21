#!/usr/bin/env python3
"""MC3: evaluate ALL models under cross-presentation transfer (train 2013J, test 2014J),
not only cross-validation, on the eligible population. Reports transfer AUC and
capture@20 / precision@20 for LR, shallow tree, random forest, gradient boosting."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from analysis_core import eligible_only, Preproc, MODULES, DEV, VAL, CHECKPOINTS

def cap_prec(y, p, b=0.20):
    k = max(1, round(b*len(y))); idx = np.argsort(-p)[:k]
    tp = int(y[idx].sum()); return tp/max(1, y.sum()), tp/k

def hgb():
    try: return HistGradientBoostingClassifier(random_state=42)
    except TypeError: return HistGradientBoostingClassifier(random_state=42)

rows = []
for mod in MODULES:
    for cp in CHECKPOINTS:
        dev = eligible_only(mod, DEV, cp); val = eligible_only(mod, VAL, cp)
        pp = Preproc().fit(dev)
        Xtr_s, Xte_s = pp.transform_scaled(dev), pp.transform_scaled(val)
        Xtr_r, Xte_r = pp.transform_raw(dev), pp.transform_raw(val)
        ytr, yte = dev.at_risk.values, val.at_risk.values
        models = {
            "LogReg": (LogisticRegression(max_iter=2000), Xtr_s, Xte_s),
            "Tree(d3)": (DecisionTreeClassifier(max_depth=3, random_state=42), Xtr_r, Xte_r),
            "RandomForest": (RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1), Xtr_r, Xte_r),
            "GradBoost": (hgb(), Xtr_r, Xte_r),
        }
        rec = {"module": mod, "cp": cp}
        for name, (m, Xtr, Xte) in models.items():
            m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
            cap, prec = cap_prec(yte, p)
            rec[f"{name}_AUC"] = round(roc_auc_score(yte, p), 3)
            rec[f"{name}_cap20"] = round(cap, 3)
        rows.append(rec)

res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("=== MC3: TRANSFER (train 2013J -> test 2014J) AUC by model ===")
auc_cols = ["module", "cp"] + [c for c in res.columns if c.endswith("_AUC")]
print(res[auc_cols].to_string(index=False))
print("\nMean transfer AUC:")
for name in ["LogReg", "Tree(d3)", "RandomForest", "GradBoost"]:
    print(f"  {name:13s} {res[name+'_AUC'].mean():.3f}   capture@20 {res[name+'_cap20'].mean():.3f}")
# is LR >= best ensemble at each setting?
res["best_ens_AUC"] = res[["RandomForest_AUC", "GradBoost_AUC"]].max(axis=1)
res["LR_minus_bestEns"] = res["LogReg_AUC"] - res["best_ens_AUC"]
n_ge = (res["LR_minus_bestEns"] >= -0.005).sum()
print(f"\nLR within 0.005 of or above best ensemble in {n_ge}/{len(res)} transfer settings")
print(f"LR - best ensemble: mean {res['LR_minus_bestEns'].mean():+.3f}, min {res['LR_minus_bestEns'].min():+.3f}, max {res['LR_minus_bestEns'].max():+.3f}")
res.to_csv("results/table_mc3_transfer_bakeoff.csv", index=False)
print("saved results/table_mc3_transfer_bakeoff.csv")
