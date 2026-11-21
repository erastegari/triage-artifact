#!/usr/bin/env python3
"""Block 8: longitudinal tracking. Out-of-fold LR risk score per student at each checkpoint
(development cohort), then ranking stability, tier transitions, and the value of tracking
over a single snapshot. Block 9 (final viz): PCA map + direct feature plot."""
import numpy as np, pandas as pd, os, warnings, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.stats import spearmanr
from analysis_core import eligible_only, Preproc, archetypes, ARCHETYPE_ORDER, MODULES, DEV, CHECKPOINTS
os.makedirs("results", exist_ok=True)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

def oof_scores(mod, pres, cp):
    d = eligible_only(mod, pres, cp); pp = Preproc().fit(d)
    X = pp.transform_scaled(d)
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, d.at_risk.values, cv=cv, method="predict_proba")[:, 1]
    return pd.DataFrame({"id_student": d.id_student.values, "score": p, "at_risk": d.at_risk.values})

for MOD in MODULES:
    sc = {cp: oof_scores(MOD, DEV, cp).set_index("id_student") for cp in CHECKPOINTS}
    # ranking stability (students eligible at both checkpoints)
    print(f"\n### {MOD}-{DEV} longitudinal ###")
    stab = []
    for a, b in [(28, 56), (56, 84), (28, 84)]:
        common = sc[a].index.intersection(sc[b].index)
        rho = spearmanr(sc[a].loc[common, "score"], sc[b].loc[common, "score"]).statistic
        stab.append({"from": a, "to": b, "n_common": len(common), "spearman": round(rho, 3)})
    print("  ranking stability (Spearman of risk score):")
    print(pd.DataFrame(stab).to_string(index=False))

    # tertile tiers from Day-28 distribution, transitions 28 -> 56
    q = sc[28]["score"].quantile([1/3, 2/3]).values
    def tier(s): return np.where(s <= q[0], "Low", np.where(s <= q[1], "Med", "High"))
    common = sc[28].index.intersection(sc[56].index)
    t28 = pd.Series(tier(sc[28].loc[common, "score"]), index=common)
    t56 = pd.Series(tier(sc[56].loc[common, "score"]), index=common)
    ar = sc[56].loc[common, "at_risk"]
    trans = pd.crosstab(t28, t56).reindex(index=["Low", "Med", "High"], columns=["Low", "Med", "High"]).fillna(0).astype(int)
    print("  tier transition counts (Day28 row -> Day56 col):")
    print(trans.to_string())
    # value of tracking: students LOW at Day 28 who rise to HIGH by Day 56
    rose = common[(t28 == "Low") & (t56 == "High")]
    stayed_low = common[(t28 == "Low") & (t56 == "Low")]
    print(f"  Low@28 -> High@56: n={len(rose)}  at-risk rate={ar.loc[rose].mean()*100:.0f}%"
          f"   (vs Low@28->Low@56 at-risk {ar.loc[stayed_low].mean()*100:.0f}%)")
    if MOD == "BBB":
        BBB_sc, BBB_q, BBB_tier = sc, q, tier

# ---- Figure: mean risk trajectory by final outcome (BBB) ----
sc = BBB_sc
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
# left: mean trajectory by final outcome among students eligible at all 3
allc = sc[28].index
for cp in CHECKPOINTS: allc = allc.intersection(sc[cp].index)
out = sc[84].loc[allc, "at_risk"]
for lab, val, col in [("ended at-risk", 1, "#d1495b"), ("ended not at-risk", 0, "#2e86ab")]:
    ids = allc[out == val]
    means = [sc[cp].loc[ids, "score"].mean() for cp in CHECKPOINTS]
    axes[0].plot(CHECKPOINTS, means, "o-", color=col, lw=2.4, label=lab)
axes[0].set_title("BBB: mean predicted risk trajectory", fontweight="bold")
axes[0].set_xlabel("checkpoint (day)"); axes[0].set_ylabel("mean predicted risk")
axes[0].set_xticks(CHECKPOINTS); axes[0].legend(frameon=False); axes[0].grid(alpha=.25)
# right: sample individual trajectories
rng = np.random.default_rng(0); samp = rng.choice(allc, size=60, replace=False)
for sid in samp:
    ys = [sc[cp].loc[sid, "score"] for cp in CHECKPOINTS]
    axes[1].plot(CHECKPOINTS, ys, "-", color="#d1495b" if out.loc[sid] else "#2e86ab", alpha=.35, lw=1)
axes[1].set_title("BBB: individual risk trajectories (60 sampled)", fontweight="bold")
axes[1].set_xlabel("checkpoint (day)"); axes[1].set_ylabel("predicted risk")
axes[1].set_xticks(CHECKPOINTS); axes[1].grid(alpha=.25)
fig.suptitle("Longitudinal risk tracking (development cohort, out-of-fold scores)", fontweight="bold")
fig.tight_layout(); fig.savefig("results/fig_trajectories.png", dpi=130, bbox_inches="tight")
print("\nsaved results/fig_trajectories.png")

# ---- Block 9: final PCA map + direct feature plot (BBB-2013J Day 56) ----
d = eligible_only("BBB", DEV, 56); pp = Preproc().fit(d); X = pp.transform_scaled(d)
arch = archetypes(d); pca = PCA(2, random_state=0); emb = pca.fit_transform(X)
evr = pca.explained_variance_ratio_
palette = {"Disengaged (<=1 active day)": "#d1495b", "Engaged, no submission": "#edae49",
           "Submitting, low score (<40)": "#8e44ad", "On track (score >=40)": "#2e86ab"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for a in ARCHETYPE_ORDER:
    sel = arch == a
    axes[0].scatter(emb[sel, 0], emb[sel, 1], s=9, c=palette[a], alpha=.6, label=a, linewidths=0)
axes[0].set_title(f"PCA of 8 features (PC1 {evr[0]*100:.0f}%, PC2 {evr[1]*100:.0f}%)", fontweight="bold")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend(fontsize=7, frameon=False, loc="best")
# direct feature plot: the practitioner view
axes[1].scatter(d["mean_score"], np.log1p(d["total_clicks"]), s=9,
                c=np.where(d.at_risk == 1, "#d1495b", "#2e86ab"), alpha=.5, linewidths=0)
axes[1].set_title("Direct view: assessment score x engagement", fontweight="bold")
axes[1].set_xlabel("mean assessment score"); axes[1].set_ylabel("log(1 + total clicks)")
fig.suptitle("Visualization layer — PCA (reproducible map) and the direct feature plot practitioners read",
             fontweight="bold")
fig.tight_layout(); fig.savefig("results/fig_pca_feature_view.png", dpi=130, bbox_inches="tight")
print("saved results/fig_pca_feature_view.png")
