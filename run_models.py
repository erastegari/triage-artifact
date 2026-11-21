#!/usr/bin/env python3
"""Blocks 4-5: LR primary (CV on dev + train-2013J->test-2014J transfer + calibration) and
shallow tree companion. Leakage-safe: preprocessing fit on development only."""
import numpy as np, pandas as pd, os, warnings, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from analysis_core import eligible_only, Preproc, FEATURES, MODULES, DEV, VAL, CHECKPOINTS
os.makedirs("results", exist_ok=True)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

def burden_metrics(y, p, b=0.20):
    k = max(1, round(b*len(y))); idx = np.argsort(-p)[:k]
    tp = int(y[idx].sum()); return tp/max(1, y.sum()), tp/k

def cal_slope_intercept(y, p):
    eps = 1e-6; lp = np.log(np.clip(p, eps, 1-eps)/(1-np.clip(p, eps, 1-eps)))
    m = LogisticRegression(C=1e6, solver="lbfgs").fit(lp.reshape(-1, 1), y)
    return float(m.coef_[0][0]), float(m.intercept_[0])

rows, coef_rows = [], []
rel = {}   # reliability data for figure
for mod in MODULES:
    for cp in CHECKPOINTS:
        dev = eligible_only(mod, DEV, cp); val = eligible_only(mod, VAL, cp)
        pp = Preproc().fit(dev)
        Xtr_s, Xte_s = pp.transform_scaled(dev), pp.transform_scaled(val)
        Xtr_r, Xte_r = pp.transform_raw(dev), pp.transform_raw(val)
        ytr, yte = dev.at_risk.values, val.at_risk.values
        # LR: plain (naturally calibrated; outcomes near-balanced so no class weighting needed)
        lr = LogisticRegression(max_iter=2000)
        auc_cv = roc_auc_score(ytr, cross_val_predict(lr, Xtr_s, ytr, cv=cv, method="predict_proba")[:, 1])
        lr.fit(Xtr_s, ytr); p_val = lr.predict_proba(Xte_s)[:, 1]
        auc_val = roc_auc_score(yte, p_val)
        cap, prec = burden_metrics(yte, p_val)
        brier = brier_score_loss(yte, p_val); slope, intc = cal_slope_intercept(yte, p_val)
        # sanity: balanced variant AUC (to document weighting was unnecessary)
        lrb = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Xtr_s, ytr)
        auc_val_bal = roc_auc_score(yte, lrb.predict_proba(Xte_s)[:, 1])
        # tree: CV on dev, transfer to val
        dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
        auc_cv_t = roc_auc_score(ytr, cross_val_predict(dt, Xtr_r, ytr, cv=cv, method="predict_proba")[:, 1])
        dt.fit(Xtr_r, ytr); auc_val_t = roc_auc_score(yte, dt.predict_proba(Xte_r)[:, 1])
        rows.append({"module": mod, "cp": cp,
                     "LR_AUC_devCV": round(auc_cv, 3), "LR_AUC_valTransfer": round(auc_val, 3),
                     "LR_cap@20_val": round(cap, 3), "LR_prec@20_val": round(prec, 3),
                     "LR_Brier_val": round(brier, 3), "LR_calSlope": round(slope, 2), "LR_calIntcpt": round(intc, 2),
                     "LRbal_AUC_val": round(auc_val_bal, 3),
                     "Tree_AUC_devCV": round(auc_cv_t, 3), "Tree_AUC_valTransfer": round(auc_val_t, 3)})
        if cp == 56:
            for f, c in zip(FEATURES, lr.coef_[0]):
                coef_rows.append({"module": mod, "feature": f, "coef_std": round(float(c), 3)})
            rel[(mod, cp)] = (yte, p_val)

res = pd.DataFrame(rows); res.to_csv("results/table3_models.csv", index=False)
print("=== TABLE 3: LR primary + tree (dev CV and 2013J->2014J transfer, calibration on val) ===")
print(res.to_string(index=False))
print("\ncalSlope ideal=1.0 (>1 under-confident, <1 over-confident); calIntcpt ideal=0; Brier lower=better")

coef = pd.DataFrame(coef_rows); coef.to_csv("results/table_lr_coefficients.csv", index=False)
print("\n=== LR standardized coefficients (dev fit, Day 56) ===")
print(coef.pivot(index="feature", columns="module", values="coef_std").to_string())

# reliability figure (val transfer, Day 56)
fig, axes = plt.subplots(1, len(MODULES), figsize=(11, 4.6))
for ax, mod in zip(axes, MODULES):
    yte, p = rel[(mod, 56)]
    frac, mean_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
    ax.plot(mean_pred, frac, "o-", color="#2e86ab", lw=2, label="LR")
    ax.set_title(f"{mod}: 2013J->2014J, Day 56", fontweight="bold", fontsize=11)
    ax.set_xlabel("predicted risk"); ax.set_ylabel("observed at-risk rate")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=.25)
fig.suptitle("Calibration on held-out validation presentation", fontweight="bold")
fig.tight_layout(); fig.savefig("results/fig_calibration.png", dpi=130, bbox_inches="tight")
print("saved results/fig_calibration.png")

# tree rules (dev, Day 56) for both modules
with open("results/tree_rules.txt", "w") as fh:
    for mod in MODULES:
        dev = eligible_only(mod, DEV, 56); pp = Preproc().fit(dev)
        dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42).fit(
            pp.transform_raw(dev), dev.at_risk.values)
        fh.write(f"\n### {mod}-2013J Day 56 shallow tree (depth 3, {dt.get_n_leaves()} leaves) ###\n")
        fh.write(export_text(dt, feature_names=FEATURES))
print("saved results/tree_rules.txt")
