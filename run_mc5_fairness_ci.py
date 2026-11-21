#!/usr/bin/env python3
"""MC5: fairness audit with bootstrap confidence intervals. Validation transfer, Day 56,
global 20% flagging threshold. Reports per-subgroup base rate, flagged rate, capture and
precision with 95% bootstrap CIs, plus AUC. IMD direction stated explicitly."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from analysis_core import eligible_only, Preproc, MODULES, DEV, VAL
CP = 56; BURDEN = 0.20; B = 2000
rng = np.random.default_rng(42)

def imd_group(s):
    # OULAD imd_band: lower band = MORE deprived (0-10% = most deprived decile).
    if s == "Unknown" or pd.isna(s): return "Unknown"
    lo = int(str(s).split("-")[0].replace("%", ""))
    return "Most deprived (0-30%)" if lo < 30 else ("Mid (30-70%)" if lo < 70 else "Least deprived (70-100%)")

def scored(mod):
    dev = eligible_only(mod, DEV, CP); val = eligible_only(mod, VAL, CP).copy()
    pp = Preproc().fit(dev)
    lr = LogisticRegression(max_iter=2000).fit(pp.transform_scaled(dev), dev.at_risk.values)
    val["score"] = lr.predict_proba(pp.transform_scaled(val))[:, 1]
    cut = np.quantile(val["score"], 1 - BURDEN)
    val["flagged"] = val["score"] >= cut
    val["imd_grp"] = val["imd_band"].map(imd_group)
    return val

def boot_ci(y, fl):
    y = np.asarray(y); fl = np.asarray(fl); n = len(y)
    caps, precs = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n); yy, ff = y[idx], fl[idx]
        tp = (ff & (yy == 1)).sum()
        caps.append(tp / max(1, (yy == 1).sum()))
        precs.append(tp / max(1, ff.sum()))
    ci = lambda a: (round(np.percentile(a, 2.5), 2), round(np.percentile(a, 97.5), 2))
    return ci(caps), ci(precs)

ATTRS = {"gender": "Gender", "age_band": "Age band", "imd_grp": "IMD (deprivation)", "disability": "Disability"}
rows = []
for mod in MODULES:
    v = scored(mod)
    for col, label in ATTRS.items():
        for lvl, g in v.groupby(col):
            if len(g) < 30: continue
            y = g.at_risk.values; fl = g.flagged.values
            tp = int((fl & (y == 1)).sum())
            cap = tp / max(1, y.sum()); prec = tp / max(1, fl.sum())
            (cl, ch), (pl, ph) = boot_ci(y, fl)
            rows.append({"module": mod, "attribute": label, "group": str(lvl), "n": len(g),
                         "base_rate": round(y.mean(), 3), "flagged_rate": round(fl.mean(), 3),
                         "capture": round(cap, 2), "capture_CI": f"[{cl},{ch}]",
                         "precision": round(prec, 2), "precision_CI": f"[{pl},{ph}]",
                         "AUC": round(roc_auc_score(y, g.score.values), 2) if len(np.unique(y)) > 1 else np.nan})
fair = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(f"=== MC5: fairness audit with 95% bootstrap CIs (val transfer, Day {CP}, global 20% flag) ===")
print("IMD direction: lower band = MORE deprived (0-10% = most-deprived decile).")
for mod in MODULES:
    print(f"\n--- {mod}-{VAL} ---")
    print(fair[fair.module == mod].drop(columns="module").to_string(index=False))
fair.to_csv("results/table_mc5_fairness_ci.csv", index=False)
print("\nsaved results/table_mc5_fairness_ci.csv")
