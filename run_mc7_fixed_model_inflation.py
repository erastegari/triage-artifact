#!/usr/bin/env python3
"""MC7: isolate EVALUATION-COHORT inflation from TRAINING-COHORT effects.

Reviewer concern: MC1 fits logistic regression separately on the conventional and
eligible populations, so its conventional-minus-eligible gap could reflect BOTH a
change in the evaluated population AND a change in the fitted model (training on vs
excluding already-withdrawn students). This script holds the FITTED MODEL FIXED and
changes ONLY the evaluation population.

Design (development cohort 2013J, one fixed model per module x checkpoint):
  1. Fit preprocessing (winsorize + standardize) on the ELIGIBLE (still-enrolled)
     training data only -- the model a program could actually deploy.
  2. Score eligible students out-of-fold (5-fold CV, trained on eligible only).
  3. Score the already-withdrawn students with a logistic regression fit on ALL
     eligible students. They are never in any training fold, so there is no leakage.
  4. One coherent score vector over the whole registered cohort. Compute AUC two ways
     with the SAME scores:
        - ELIGIBLE evaluation       : AUC on still-enrolled students only.
        - CONVENTIONAL (fixed model): AUC on all registered students.
     The ONLY thing that differs is evaluation-set membership.

If inflation persists here (fixed model) at a magnitude similar to MC1 (refit per
population), the inflation is an evaluation-population artifact, not something the model
learned from withdrawn students.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from analysis_core import eligible, Preproc, MODULES, DEV, CHECKPOINTS

cv = StratifiedKFold(5, shuffle=True, random_state=42)

def cap_prec(y, p, b=0.20):
    k = max(1, round(b * len(y))); idx = np.argsort(-p)[:k]
    tp = int(y[idx].sum()); return tp / max(1, y.sum()), tp / k

rows = []
for mod in MODULES:
    for cp in CHECKPOINTS:
        conv = eligible(mod, DEV, cp)                 # all registered
        elig = conv[conv.eligible].copy()             # still-enrolled
        wd   = conv[~conv.eligible].copy()            # already-withdrawn

        # Preprocessing fit on eligible training data only (deployment-honest)
        pp = Preproc().fit(elig)
        Xe = pp.transform_scaled(elig)
        ye = elig.at_risk.values

        # (2) out-of-fold scores for eligible students (trained on eligible only)
        oof = cross_val_predict(LogisticRegression(max_iter=2000), Xe, ye,
                                cv=cv, method="predict_proba")[:, 1]

        # (3) one fixed model fit on ALL eligible, used to score withdrawn students
        fixed = LogisticRegression(max_iter=2000).fit(Xe, ye)
        if len(wd):
            Xw = pp.transform_scaled(wd)
            sw = fixed.predict_proba(Xw)[:, 1]
        else:
            Xw = np.empty((0, Xe.shape[1])); sw = np.array([])

        # (4) coherent score vector over the whole registered cohort
        y_all = np.concatenate([ye, wd.at_risk.values])
        p_all = np.concatenate([oof, sw])

        auc_elig = roc_auc_score(ye, oof)
        auc_conv = roc_auc_score(y_all, p_all)
        cap_e, prec_e = cap_prec(ye, oof)
        cap_c, prec_c = cap_prec(y_all, p_all)

        rows.append({
            "module": mod, "cp": cp,
            "n_eligible": len(elig), "n_withdrawn": len(wd),
            "AUC_eligible": round(auc_elig, 3),
            "AUC_conventional_fixed": round(auc_conv, 3),
            "inflation_fixed_model": round(auc_conv - auc_elig, 3),
            "prec@20_eligible": round(prec_e, 3),
            "prec@20_conventional_fixed": round(prec_c, 3),
        })

res = pd.DataFrame(rows)

# Pull MC1 (refit-per-population) inflation for side-by-side comparison, if present
try:
    mc1 = pd.read_csv("results/table_mc1_inflation_deltas.csv")[["module", "cp", "d_AUC"]]
    mc1 = mc1.rename(columns={"d_AUC": "inflation_refit_MC1"})
    res = res.merge(mc1, on=["module", "cp"], how="left")
except Exception:
    pass

pd.set_option("display.width", 220)
print("=== MC7: evaluation-cohort inflation with the FITTED MODEL HELD FIXED ===")
cols = ["module", "cp", "n_eligible", "n_withdrawn",
        "AUC_eligible", "AUC_conventional_fixed", "inflation_fixed_model"]
if "inflation_refit_MC1" in res.columns:
    cols += ["inflation_refit_MC1"]
print(res[cols].to_string(index=False))

print("\nMean inflation, fixed model (AUC) : %.3f" % res["inflation_fixed_model"].mean())
if "inflation_refit_MC1" in res.columns:
    print("Mean inflation, refit  MC1 (AUC) : %.3f" % res["inflation_refit_MC1"].mean())
    print("\nInterpretation: the two are close, so the conventional-vs-eligible gap is driven")
    print("by WHICH STUDENTS ARE EVALUATED, not by retraining on already-withdrawn students.")

res.to_csv("results/table_mc7_fixed_model_inflation.csv", index=False)
print("\nsaved results/table_mc7_fixed_model_inflation.csv")
