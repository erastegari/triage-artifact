#!/usr/bin/env python3
"""FROZEN analysis spec for the interpretable-triage paper.
One source of truth for: eligible population, label, features, leakage-safe preprocessing,
archetypes. Imported by every results script so numbers can't drift between sections.
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from repro import build_model_df, NETCOLS

# ---- frozen spec ----
FEATURES = list(NETCOLS)   # 8 features, fixed order
CLIP_COLS = ["total_clicks", "avg_clicks_per_active_day", "mean_lateness"]
MODULES = ["BBB", "DDD"]
DEV, VAL = "2013J", "2014J"
CHECKPOINTS = [28, 56, 84]

def eligible(mod, pres, cp):
    """Whole module cohort frame + eligibility flags. Eligible = still enrolled at checkpoint."""
    m = build_model_df(mod, pres, cp)
    m["withdrew_before"] = m["unreg"] < cp          # NaN unreg -> False -> still enrolled
    m["eligible"] = ~m["withdrew_before"]
    return m

def eligible_only(mod, pres, cp):
    return eligible(mod, pres, cp).loc[lambda d: d["eligible"]].copy()

class Preproc:
    """Clip + standardize. FIT ON DEVELOPMENT ONLY, applied to dev and validation (no leakage)."""
    def __init__(self):
        self.bounds = {}; self.scaler = StandardScaler()
    def fit(self, df):
        nd = df[FEATURES].copy()
        for c in CLIP_COLS:
            lo, hi = nd[c].quantile(.01), nd[c].quantile(.99)
            self.bounds[c] = (lo, hi); nd[c] = nd[c].clip(lo, hi)
        self.scaler.fit(nd.values)
        self._raw_clipped = nd            # for tree (raw, clipped, unscaled)
        return self
    def _clip(self, df):
        nd = df[FEATURES].copy()
        for c in CLIP_COLS:
            lo, hi = self.bounds[c]; nd[c] = nd[c].clip(lo, hi)
        return nd
    def transform_scaled(self, df):       # for LR
        return self.scaler.transform(self._clip(df).values)
    def transform_raw(self, df):          # for tree
        return self._clip(df).values

def archetypes(m):
    """Interpretable groups defined on the features (early-signal)."""
    return np.where(m["active_days"] <= 1, "Disengaged (<=1 active day)",
           np.where(m["has_early_assessment"] == 0, "Engaged, no submission",
           np.where(m["mean_score"] < 40, "Submitting, low score (<40)",
                                          "On track (score >=40)")))

ARCHETYPE_ORDER = ["Disengaged (<=1 active day)", "Engaged, no submission",
                   "Submitting, low score (<40)", "On track (score >=40)"]
