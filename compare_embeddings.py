#!/usr/bin/env python3
"""Head-to-head: PCA vs t-SNE vs UMAP as a VISUALIZATION layer over the 8 features.
Two questions:
  (1) Reproducibility  -- does the map look the same if you run it again? (Procrustes disparity across seeds)
  (2) Archetype clarity -- does the map keep same-kind students together and preserve
                           high-D neighborhoods? (silhouette vs archetypes; trustworthiness)
Population: still-enrolled (eligible) students at the checkpoint.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from scipy.spatial import procrustes
import umap
from repro import build_model_df, NETCOLS

MOD, PRES, CP = "BBB", "2013J", 56   # representative map; embeddings are visualization, one cohort/cp is enough
SEEDS = [0, 1, 2, 3, 4]

def eligible_X(mod, pres, cp):
    m = build_model_df(mod, pres, cp)
    m = m[~(m["unreg"] < cp)].copy()          # still enrolled at checkpoint
    cols = [c for c in NETCOLS if m[c].nunique() > 1]
    nd = m[cols].copy()
    for c in ["total_clicks", "avg_clicks_per_active_day", "mean_lateness"]:
        if c in nd: nd[c] = nd[c].clip(nd[c].quantile(.01), nd[c].quantile(.99))
    X = StandardScaler().fit_transform(nd)
    # interpretable archetypes defined on the features (what we'd actually present)
    arch = np.where(m["active_days"] <= 1, "Disengaged (<=1 active day)",
           np.where(m["has_early_assessment"] == 0, "Active, no submission",
           np.where(m["mean_score"] < 40, "Submitted, low score (<40)",
                                          "Submitted, OK (>=40)")))
    return m, X, arch

m, X, arch = eligible_X(MOD, PRES, CP)
at_risk = m["at_risk"].values
print(f"{MOD}-{PRES} Day {CP}: eligible n={len(m)}  at-risk={at_risk.mean()*100:.1f}%")
for a in pd.unique(arch):
    sel = arch == a
    print(f"   {a:32s} n={sel.sum():4d}  at-risk={at_risk[sel].mean()*100:5.1f}%")

def embed(method, seed):
    if method == "PCA":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "t-SNE":
        return TSNE(n_components=2, perplexity=30, init="random",
                    random_state=seed, max_iter=1000).fit_transform(X)
    if method == "UMAP":
        return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                         random_state=seed).fit_transform(X)

# ---- metrics across seeds ----
rows = []
runs = {}
for method in ["PCA", "t-SNE", "UMAP"]:
    embs = [embed(method, s) for s in SEEDS]
    runs[method] = embs
    # reproducibility: mean Procrustes disparity over all seed pairs (0=identical shape,1=unrelated)
    disp = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            _, _, d = procrustes(embs[i], embs[j])
            disp.append(d)
    disp = np.array(disp)
    # quality (averaged over seeds)
    tw = np.mean([trustworthiness(X, e, n_neighbors=10) for e in embs])
    sil_arch = np.mean([silhouette_score(e, arch) for e in embs])
    sil_risk = np.mean([silhouette_score(e, at_risk) for e in embs])
    rows.append({"method": method,
                 "procrustes_disparity_mean": round(float(disp.mean()), 4),
                 "procrustes_disparity_max": round(float(disp.max()), 4),
                 "trustworthiness": round(float(tw), 4),
                 "silhouette_archetype": round(float(sil_arch), 4),
                 "silhouette_atrisk": round(float(sil_risk), 4)})

tbl = pd.DataFrame(rows)
print("\n=== EMBEDDING METRICS (mean over 5 seeds) ===")
print(tbl.to_string(index=False))
print("\nReproducibility: 0 = identical shape every run; higher = layout changes run-to-run.")
print("Trustworthiness: 1.0 = high-D neighbors preserved in 2D (distances trustworthy).")
print("Silhouette: higher = same-group points sit together (>0 separated, <0 mixed).")
tbl.to_csv("embedding_metrics.csv", index=False)

# ---- figure 1: archetype map, seed 0 for each method ----
palette = {"Disengaged (<=1 active day)": "#d1495b", "Active, no submission": "#edae49",
           "Submitted, low score (<40)": "#66a182", "Submitted, OK (>=40)": "#2e86ab"}
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, method in zip(axes, ["PCA", "t-SNE", "UMAP"]):
    e = runs[method][0]
    for a, col in palette.items():
        sel = arch == a
        ax.scatter(e[sel, 0], e[sel, 1], s=7, c=col, alpha=.6, label=a, linewidths=0)
    ax.set_title(f"{method}", fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(loc="upper center", bbox_to_anchor=(1.65, -0.03), ncol=4, fontsize=9, frameon=False)
fig.suptitle(f"Feature-space maps colored by archetype — {MOD}-{PRES}, Day {CP} (eligible n={len(m)})",
             fontsize=13, y=1.00)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig("fig_embeddings_archetypes.png", dpi=130, bbox_inches="tight")
print("\nsaved fig_embeddings_archetypes.png")

# ---- figure 2: reproducibility — seed 0 vs seed 1 for each method ----
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for col, method in enumerate(["PCA", "t-SNE", "UMAP"]):
    for row in range(2):
        e = runs[method][row]
        ax = axes[row, col]
        sc = ax.scatter(e[:, 0], e[:, 1], s=6, c=at_risk, cmap="coolwarm", alpha=.55, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0: ax.set_title(method, fontsize=13, fontweight="bold")
        ax.set_ylabel(f"seed {SEEDS[row]}", fontsize=11) if col == 0 else None
fig.suptitle("Same data, two random seeds (points colored by at-risk). PCA is identical; t-SNE/UMAP reshuffle.",
             fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("fig_embeddings_reproducibility.png", dpi=130, bbox_inches="tight")
print("saved fig_embeddings_reproducibility.png")
