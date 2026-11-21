#!/usr/bin/env python3
"""Block 10: fairness/equity audit on the eligible population, validation transfer, Day 56.
Global 20% flagging cutoff (as deployed), then per-subgroup flagged-rate, capture, precision,
base rate, AUC and calibration. Blocks 11-12 (generalization, null) synthesized in the memo."""
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from analysis_core import eligible_only, Preproc, MODULES, DEV, VAL
os.makedirs("results", exist_ok=True)
CP = 56; BURDEN = 0.20

def imd_group(s):
    if s == "Unknown" or pd.isna(s): return "Unknown"
    lo = int(str(s).split("-")[0].replace("%", ""))
    return "Low IMD (0-30%)" if lo < 30 else ("Mid IMD (30-70%)" if lo < 70 else "High IMD (70-100%)")

def scored_val(mod):
    dev = eligible_only(mod, DEV, CP); val = eligible_only(mod, VAL, CP)
    pp = Preproc().fit(dev)
    lr = LogisticRegression(max_iter=2000).fit(pp.transform_scaled(dev), dev.at_risk.values)
    val = val.copy(); val["score"] = lr.predict_proba(pp.transform_scaled(val))[:, 1]
    cut = np.quantile(val["score"], 1 - BURDEN)      # global cutoff
    val["flagged"] = val["score"] >= cut
    val["imd_grp"] = val["imd_band"].map(imd_group)
    val["age_grp"] = val["age_band"]
    return val

ATTRS = {"gender": "gender", "age_grp": "age band", "imd_grp": "IMD band", "disability": "disability"}
rows = []
for mod in MODULES:
    v = scored_val(mod)
    overall_cap = (v.flagged & (v.at_risk == 1)).sum() / max(1, (v.at_risk == 1).sum())
    for col, label in ATTRS.items():
        for lvl, g in v.groupby(col):
            if len(g) < 30: continue
            y = g.at_risk.values; fl = g.flagged.values
            tp = int((fl & (y == 1)).sum())
            rows.append({"module": mod, "attribute": label, "group": str(lvl), "n": len(g),
                         "base_rate": round(y.mean(), 3),
                         "flagged_rate": round(fl.mean(), 3),
                         "capture": round(tp / max(1, y.sum()), 3),
                         "precision": round(tp / max(1, fl.sum()), 3),
                         "AUC": round(roc_auc_score(y, g.score.values), 3) if len(np.unique(y)) > 1 else np.nan,
                         "mean_pred": round(g.score.mean(), 3)})
fair = pd.DataFrame(rows)
fair.to_csv("results/table6_fairness.csv", index=False)
pd.set_option("display.width", 200)
print(f"=== TABLE 6: fairness audit (validation transfer, Day {CP}, global {int(BURDEN*100)}% flag cutoff) ===")
for mod in MODULES:
    print(f"\n--- {mod}-{VAL} ---")
    print(fair[fair.module == mod].drop(columns="module").to_string(index=False))

# disparity summary: max-min gap per attribute/module for capture & flagged_rate
print("\n=== Disparity gaps (max - min across groups within attribute) ===")
gap = (fair.groupby(["module", "attribute"])
          .agg(capture_gap=("capture", lambda s: round(s.max()-s.min(), 3)),
               flagged_gap=("flagged_rate", lambda s: round(s.max()-s.min(), 3)),
               precision_gap=("precision", lambda s: round(s.max()-s.min(), 3))).reset_index())
print(gap.to_string(index=False))
