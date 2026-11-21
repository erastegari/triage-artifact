#!/usr/bin/env python3
"""Blocks 1-3: sample description (Table 1) + archetypes (Table 2) + feature distributions."""
import numpy as np, pandas as pd, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analysis_core import (eligible, eligible_only, archetypes, ARCHETYPE_ORDER,
                           FEATURES, MODULES, DEV, VAL, CHECKPOINTS)
os.makedirs("results", exist_ok=True)

# ---------- Table 1: sample description ----------
rows = []
for mod in MODULES:
    for pres in [DEV, VAL]:
        for cp in CHECKPOINTS:
            m = eligible(mod, pres, cp)
            elig = m[m.eligible]
            rows.append({"module": mod, "presentation": pres, "role": "dev" if pres == DEV else "val",
                         "checkpoint": cp, "n_cohort": len(m),
                         "n_withdrawn_before": int(m.withdrew_before.sum()),
                         "pct_withdrawn_before": round(100*m.withdrew_before.mean(), 1),
                         "n_eligible": len(elig),
                         "base_rate_eligible": round(elig.at_risk.mean(), 3)})
t1 = pd.DataFrame(rows)
t1.to_csv("results/table1_sample.csv", index=False)
print("=== TABLE 1: sample description (eligible = still enrolled at checkpoint) ===")
print(t1.to_string(index=False))

# ---------- Table 2: archetypes (development cohorts, earliest checkpoint = Day 28) ----------
arch_rows = []
for mod in MODULES:
    m = eligible_only(mod, DEV, 28)
    m["arch"] = archetypes(m)
    for a in ARCHETYPE_ORDER:
        sel = m.arch == a
        if sel.sum() == 0:
            arch_rows.append({"module": mod, "archetype": a, "n": 0, "pct_of_cohort": 0.0, "at_risk_rate": np.nan}); continue
        arch_rows.append({"module": mod, "archetype": a, "n": int(sel.sum()),
                          "pct_of_cohort": round(100*sel.mean(), 1),
                          "at_risk_rate": round(m.loc[sel, "at_risk"].mean(), 3)})
t2 = pd.DataFrame(arch_rows)
t2.to_csv("results/table2_archetypes.csv", index=False)
print("\n=== TABLE 2: archetypes (development cohorts, Day 28 eligible) ===")
print(t2.to_string(index=False))

# ---------- Figure: feature distributions by outcome (BBB-2013J, Day 28) ----------
m = eligible_only("BBB", DEV, 28); y = m.at_risk.values
key = ["mean_score", "active_days", "total_clicks", "has_early_assessment",
       "avg_clicks_per_active_day", "mean_lateness"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, f in zip(axes.ravel(), key):
    d0 = m.loc[y == 0, f]; d1 = m.loc[y == 1, f]
    ax.hist([d0, d1], bins=20, color=["#2e86ab", "#d1495b"], label=["not at-risk", "at-risk"], density=True)
    ax.set_title(f, fontsize=10, fontweight="bold"); ax.set_yticks([])
axes[0, 0].legend(fontsize=8, frameon=False)
fig.suptitle("Feature distributions by outcome — BBB-2013J, Day 28 eligible", fontweight="bold")
fig.tight_layout(); fig.savefig("results/fig_feature_distributions.png", dpi=130, bbox_inches="tight")
print("\nsaved results/fig_feature_distributions.png")
