#!/usr/bin/env python3
"""MC7: isolate EVALUATION-COHORT inflation with a single, fixed fitted model.

Reviewer concern (resolved here): an earlier version of this check mixed out-of-fold
predictions for eligible students with a full-fit model for withdrawn students, so the
two populations were not scored by one identical model. This version removes that
ambiguity by using one clearly defined prediction procedure for both populations:

  For each module x checkpoint:
    1. Fit preprocessing (winsorize + standardize) on the ELIGIBLE (still-enrolled)
       students of the DEVELOPMENT presentation (2013J).
    2. Train ONE logistic regression on that eligible development sample.
    3. Apply that identical fitted model, unchanged, to the held-out VALIDATION
       presentation (2014J) -- an independent sample the model never saw.
    4. Score every registered 2014J student once, then read AUC two ways from the
       same score vector:
          - ELIGIBLE evaluation       : still-enrolled 2014J students only.
          - CONVENTIONAL evaluation   : all registered 2014J students.
       The ONLY thing that differs between the two numbers is which students are in
       the evaluation set. One model, one scoring pass, no leakage (2014J is never in
       training).

Inflation = AUC(conventional) - AUC(eligible). The eligible-evaluation number equals
the logistic-regression transfer AUC already reported in Table 5, by construction.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from analysis_core import eligible, Preproc, MODULES, DEV, VAL, CHECKPOINTS

def cap_prec(y, p, b=0.20):
    k = max(1, round(b * len(y))); idx = np.argsort(-p)[:k]
    tp = int(y[idx].sum()); return tp / max(1, y.sum()), tp / k

rows = []
for mod in MODULES:
    for cp in CHECKPOINTS:
        dev = eligible(mod, DEV, cp)
        dev_elig = dev[dev.eligible].copy()           # train only on still-enrolled dev students

        pp = Preproc().fit(dev_elig)
        Xtr = pp.transform_scaled(dev_elig); ytr = dev_elig.at_risk.values
        model = LogisticRegression(max_iter=2000).fit(Xtr, ytr)   # ONE fixed model

        val = eligible(mod, VAL, cp)                  # held-out later presentation
        m = val.eligible.values                       # still-enrolled mask in VAL
        Xval = pp.transform_scaled(val)
        p = model.predict_proba(Xval)[:, 1]           # one scoring pass over all registered VAL
        y = val.at_risk.values

        auc_conv = roc_auc_score(y, p)
        auc_elig = roc_auc_score(y[m], p[m])
        cap_c, prec_c = cap_prec(y, p)
        cap_e, prec_e = cap_prec(y[m], p[m])

        rows.append({
            "module": mod, "cp": cp,
            "n_eligible": int(m.sum()), "n_withdrawn": int((~m).sum()),
            "AUC_eligible": round(auc_elig, 3),
            "AUC_conventional_fixed": round(auc_conv, 3),
            "inflation_fixed_model": round(auc_conv - auc_elig, 3),
            "prec@20_eligible": round(prec_e, 3),
            "prec@20_conventional_fixed": round(prec_c, 3),
        })

res = pd.DataFrame(rows)

# Side-by-side with the refit-per-population inflation (MC1), if available
try:
    mc1 = pd.read_csv("results/table_mc1_inflation_deltas.csv")[["module", "cp", "d_AUC"]]
    res = res.merge(mc1.rename(columns={"d_AUC": "inflation_refit_MC1"}), on=["module", "cp"], how="left")
except Exception:
    pass

pd.set_option("display.width", 220)
print("=== MC7: evaluation-cohort inflation, ONE fixed model (train dev-eligible -> score held-out VAL) ===")
cols = ["module", "cp", "n_eligible", "n_withdrawn",
        "AUC_eligible", "AUC_conventional_fixed", "inflation_fixed_model"]
if "inflation_refit_MC1" in res.columns:
    cols += ["inflation_refit_MC1"]
print(res[cols].to_string(index=False))
print("\nMean inflation, fixed model (AUC) : %.3f" % res["inflation_fixed_model"].mean())
if "inflation_refit_MC1" in res.columns:
    print("Mean inflation, refit  MC1 (AUC) : %.3f" % res["inflation_refit_MC1"].mean())

res.to_csv("results/table_mc7_fixed_model_inflation.csv", index=False)
print("\nsaved results/table_mc7_fixed_model_inflation.csv")
