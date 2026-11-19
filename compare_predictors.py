#!/usr/bin/env python3
"""Head-to-head: Logistic Regression vs shallow Decision Tree as the flagging model.
Accuracy (cross-validated AUC + capture/precision at matched follow-up burden) and
interpretability (tree rules vs LR coefficients). Eligible (still-enrolled) population,
per checkpoint, BBB + DDD, dev + validation presentations.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from repro import build_model_df, NETCOLS

BURDEN = 0.20   # flag top 20% by score -> matched follow-up capacity
COHORTS = [("BBB", "2013J"), ("BBB", "2014J"), ("DDD", "2013J"), ("DDD", "2014J")]
CPS = [28, 56, 84]

def eligible(mod, pres, cp):
    m = build_model_df(mod, pres, cp)
    m = m[~(m["unreg"] < cp)].copy()            # still enrolled at checkpoint
    cols = [c for c in NETCOLS if m[c].nunique() > 1]
    nd = m[cols].copy()
    for c in ["total_clicks", "avg_clicks_per_active_day", "mean_lateness"]:
        if c in nd: nd[c] = nd[c].clip(nd[c].quantile(.01), nd[c].quantile(.99))
    return m, nd, cols, m["at_risk"].values

def metrics_at_burden(y, score, burden):
    n = len(y); k = max(1, round(burden * n))
    order = np.argsort(-score)
    flagged = np.zeros(n, bool); flagged[order[:k]] = True
    tp = int((flagged & (y == 1)).sum())
    cap = tp / max(1, (y == 1).sum())
    prec = tp / k
    return cap, prec, k

rows = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for mod, pres in COHORTS:
    for cp in CPS:
        m, nd, cols, y = eligible(mod, pres, cp)
        if y.sum() < 20 or (y == 0).sum() < 20:
            continue
        Xs = StandardScaler().fit_transform(nd)          # for LR
        Xt = nd.values                                    # tree: raw features
        lr = LogisticRegression(class_weight="balanced", max_iter=2000)
        dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
        p_lr = cross_val_predict(lr, Xs, y, cv=cv, method="predict_proba")[:, 1]
        p_dt = cross_val_predict(dt, Xt, y, cv=cv, method="predict_proba")[:, 1]
        auc_lr, auc_dt = roc_auc_score(y, p_lr), roc_auc_score(y, p_dt)
        cap_lr, prec_lr, k = metrics_at_burden(y, p_lr, BURDEN)
        cap_dt, prec_dt, _ = metrics_at_burden(y, p_dt, BURDEN)
        rows.append({"cohort": f"{mod}-{pres}", "cp": cp, "n": len(m), "base_rate": round(y.mean(), 3),
                     "AUC_LR": round(auc_lr, 3), "AUC_tree": round(auc_dt, 3),
                     "cap@20_LR": round(cap_lr, 3), "cap@20_tree": round(cap_dt, 3),
                     "prec@20_LR": round(prec_lr, 3), "prec@20_tree": round(prec_dt, 3)})

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("=== PREDICTOR METRICS (5-fold CV, eligible population) ===")
print(res.to_string(index=False))
res.to_csv("predictor_metrics.csv", index=False)
print(f"\nMean AUC  LR={res.AUC_LR.mean():.3f}  tree={res.AUC_tree.mean():.3f}"
      f"   (AUC_LR - AUC_tree) mean={np.mean(res.AUC_LR-res.AUC_tree):+.3f}")
print(f"Mean capture@20%  LR={res['cap@20_LR'].mean():.3f}  tree={res['cap@20_tree'].mean():.3f}")

# depth sensitivity (BBB-2013J D56): is shallow enough?
m, nd, cols, y = eligible("BBB", "2013J", 56)
print("\n=== Tree depth sensitivity  (BBB-2013J Day 56, CV AUC) ===")
for d in [2, 3, 4, 6, None]:
    dt = DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=42)
    p = cross_val_predict(dt, nd.values, y, cv=cv, method="predict_proba")[:, 1]
    lab = "full" if d is None else d
    print(f"   depth={lab:>4}  AUC={roc_auc_score(y,p):.3f}")

# interpretability artifacts (BBB-2013J D56)
print("\n=== Shallow tree rules (depth 3, BBB-2013J Day 56) ===")
dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42).fit(nd.values, y)
print(export_text(dt, feature_names=list(cols), show_weights=False))
print(f"   leaves={dt.get_n_leaves()}")
lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(StandardScaler().fit_transform(nd), y)
coef = pd.Series(lr.coef_[0], index=cols).sort_values(key=abs, ascending=False)
print("=== LR standardized coefficients (BBB-2013J Day 56) ===")
print(coef.round(3).to_string())

# figure: AUC by checkpoint, LR vs tree, per cohort
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
for ax, (mod, pres) in zip(axes, COHORTS):
    sub = res[res.cohort == f"{mod}-{pres}"]
    if len(sub) == 0:
        ax.set_visible(False); continue
    ax.plot(sub.cp, sub.AUC_LR, "o-", color="#2e86ab", label="Logistic Reg.", lw=2)
    ax.plot(sub.cp, sub.AUC_tree, "s--", color="#d1495b", label="Shallow tree (d=3)", lw=2)
    ax.set_title(f"{mod}-{pres}", fontweight="bold")
    ax.set_xticks(CPS); ax.set_xlabel("Checkpoint (day)")
    ax.axhline(0.5, color="gray", ls=":", lw=1); ax.set_ylim(0.45, 0.95); ax.grid(alpha=.25)
axes[0].set_ylabel("Cross-validated ROC-AUC"); axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Flagging accuracy by checkpoint — Logistic Regression vs shallow Decision Tree (eligible students)",
             fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("fig_predictors_auc.png", dpi=130, bbox_inches="tight")
print("\nsaved fig_predictors_auc.png")
