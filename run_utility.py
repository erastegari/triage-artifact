#!/usr/bin/env python3
"""Block 7: decision utility (capture/precision at realistic follow-up burdens) and earliness.
Primary estimates on held-out validation presentation (train 2013J -> test 2014J)."""
import numpy as np, pandas as pd, os, warnings, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from analysis_core import eligible_only, Preproc, MODULES, DEV, VAL, CHECKPOINTS
os.makedirs("results", exist_ok=True)
BURDENS = [0.10, 0.20, 0.30]

def fit_transfer(mod, cp):
    dev = eligible_only(mod, DEV, cp); val = eligible_only(mod, VAL, cp)
    pp = Preproc().fit(dev)
    lr = LogisticRegression(max_iter=2000).fit(pp.transform_scaled(dev), dev.at_risk.values)
    p = lr.predict_proba(pp.transform_scaled(val))[:, 1]
    return val.at_risk.values, p

def cap_prec(y, p, b):
    k = max(1, round(b*len(y))); idx = np.argsort(-p)[:k]
    tp = int(y[idx].sum()); return tp/max(1, y.sum()), tp/k, k

# ---- Table 5: utility at burdens (validation transfer) ----
rows = []
for mod in MODULES:
    for cp in CHECKPOINTS:
        y, p = fit_transfer(mod, cp)
        rec = {"module": mod, "cp": cp, "n_val": len(y), "base_rate": round(y.mean(), 3),
               "AUC": round(roc_auc_score(y, p), 3)}
        for b in BURDENS:
            cap, prec, k = cap_prec(y, p, b)
            rec[f"cap@{int(b*100)}"] = round(cap, 3); rec[f"prec@{int(b*100)}"] = round(prec, 3)
        rows.append(rec)
t5 = pd.DataFrame(rows); t5.to_csv("results/table5_utility.csv", index=False)
print("=== TABLE 5: decision utility on validation (capture/precision at follow-up burden) ===")
print(t5.to_string(index=False))

# ---- Figure A: capture-vs-burden curve (val, Day 56) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
grid = np.linspace(0.02, 0.6, 30)
for ax, mod in zip(axes, MODULES):
    y, p = fit_transfer(mod, 56)
    caps = [cap_prec(y, p, b)[0] for b in grid]
    ax.plot(grid*100, np.array(caps)*100, "-", color="#2e86ab", lw=2.2, label="LR triage")
    ax.plot(grid*100, np.minimum(grid/ y.mean(), 1)*0 + grid*0, alpha=0)  # spacer
    ax.plot([0, 100], [0, 100], "k:", lw=1, label="flag-everyone ceiling / random base")
    # random baseline: capture == burden (proportional)
    ax.plot(grid*100, grid*100, color="gray", ls="--", lw=1.2, label="random flagging")
    ax.set_title(f"{mod}: 2013J->2014J, Day 56", fontweight="bold")
    ax.set_xlabel("students followed up (% of eligible)"); ax.set_ylabel("at-risk students caught (%)")
    ax.set_xlim(0, 60); ax.set_ylim(0, 100); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8)
fig.suptitle("Capture vs follow-up burden — how many at-risk students you reach per unit of staff effort",
             fontweight="bold")
fig.tight_layout(); fig.savefig("results/fig_capture_burden.png", dpi=130, bbox_inches="tight")
print("saved results/fig_capture_burden.png")

# ---- Figure B: earliness (AUC & capture@20 vs checkpoint, val transfer) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for mod in MODULES:
    aucs, caps = [], []
    for cp in CHECKPOINTS:
        y, p = fit_transfer(mod, cp); aucs.append(roc_auc_score(y, p)); caps.append(cap_prec(y, p, 0.20)[0])
    axes[0].plot(CHECKPOINTS, aucs, "o-", lw=2, label=mod)
    axes[1].plot(CHECKPOINTS, np.array(caps)*100, "o-", lw=2, label=mod)
axes[0].set_title("Discrimination by checkpoint (validation)", fontweight="bold")
axes[0].set_ylabel("ROC-AUC"); axes[0].axhline(0.5, color="gray", ls=":", lw=1); axes[0].set_ylim(0.5, 0.9)
axes[1].set_title("Capture at 20% follow-up by checkpoint", fontweight="bold")
axes[1].set_ylabel("at-risk caught (%)")
for ax in axes:
    ax.set_xlabel("checkpoint (day)"); ax.set_xticks(CHECKPOINTS); ax.grid(alpha=.25); ax.legend(frameon=False)
fig.suptitle("Earliness: the sooner-vs-later tradeoff on held-out validation", fontweight="bold")
fig.tight_layout(); fig.savefig("results/fig_earliness.png", dpi=130, bbox_inches="tight")
print("saved results/fig_earliness.png")
