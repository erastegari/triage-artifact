#!/usr/bin/env python3
"""MC4: calibration, done correctly. For each module/checkpoint (transfer to 2014J):
- reference Brier = predicting the base rate for everyone  [prev*(1-prev)]
- Brier, calibration slope & intercept BEFORE recalibration
- a TWO-parameter logistic recalibration (intercept AND slope), fit out-of-fold on the
  target presentation (available once outcomes accrue), and the same metrics AFTER.
Also a before/after reliability figure for BBB (the module that drifts)."""
import numpy as np, pandas as pd, warnings, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from analysis_core import eligible_only, Preproc, MODULES, DEV, VAL, CHECKPOINTS
cvr = StratifiedKFold(5, shuffle=True, random_state=0)

def logit(p):
    eps = 1e-6; p = np.clip(p, eps, 1-eps); return np.log(p/(1-p))

def slope_intercept(y, p):
    m = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
    return float(m.coef_[0][0]), float(m.intercept_[0])

def recalibrate(y, p):
    # two-parameter (Platt) recalibration, out-of-fold on the target presentation
    return cross_val_predict(LogisticRegression(C=1e6), logit(p).reshape(-1, 1), y,
                             cv=cvr, method="predict_proba")[:, 1]

rows = []; curves = {}
for mod in MODULES:
    for cp in CHECKPOINTS:
        dev = eligible_only(mod, DEV, cp); val = eligible_only(mod, VAL, cp)
        pp = Preproc().fit(dev)
        lr = LogisticRegression(max_iter=2000).fit(pp.transform_scaled(dev), dev.at_risk.values)
        y = val.at_risk.values; p = lr.predict_proba(pp.transform_scaled(val))[:, 1]
        prev = y.mean(); ref = prev * (1 - prev)
        s0, i0 = slope_intercept(y, p); b0 = brier_score_loss(y, p)
        pr = recalibrate(y, p)
        s1, i1 = slope_intercept(y, pr); b1 = brier_score_loss(y, pr)
        rows.append({"module": mod, "cp": cp, "prev": round(prev, 3), "ref_Brier": round(ref, 3),
                     "Brier_before": round(b0, 3), "slope_before": round(s0, 2), "intcpt_before": round(i0, 2),
                     "Brier_after": round(b1, 3), "slope_after": round(s1, 2), "intcpt_after": round(i1, 2)})
        if cp in (56, 84):
            curves[(mod, cp)] = (y, p, pr)

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("=== MC4: calibration before/after two-parameter recalibration (transfer to 2014J) ===")
print(res.to_string(index=False))
print("\nref_Brier = Brier of predicting the base rate for everyone (no-skill reference).")
print("A model Brier above ref_Brier means worse than predicting the base rate.")
res.to_csv("results/table_mc4_calibration.csv", index=False)

# before/after reliability figure for BBB Day 56
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, which in zip(axes, ["before", "after"]):
    for mod, col in [("BBB", "#d1495b"), ("DDD", "#2e86ab")]:
        y, p, pr = curves[(mod, 56)]
        pp_ = p if which == "before" else pr
        frac, mean_pred = calibration_curve(y, pp_, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", color=col, lw=2, label=mod)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_title(f"Calibration {which} recalibration (Day 56)", fontweight="bold")
    ax.set_xlabel("predicted risk"); ax.set_ylabel("observed at-risk rate")
    ax.legend(frameon=False); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig("results/fig_calibration_before_after.png", dpi=130, bbox_inches="tight")
print("saved results/fig_calibration_before_after.png")
