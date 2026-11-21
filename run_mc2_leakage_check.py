#!/usr/bin/env python3
"""MC2: temporal-leakage robustness check. The main pipeline builds assessment features from
assessments DUE by the checkpoint. A strict alternative uses only assessments actually
SUBMITTED by the checkpoint (date_submitted <= checkpoint), i.e., strictly available at
prediction time. We recompute eligible-population AUC under both and compare."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import repro
from repro import studentInfo, studentReg, studentVle, vle, sa_all, NETCOLS
from analysis_core import FEATURES, CLIP_COLS, MODULES, DEV, CHECKPOINTS
cv = StratifiedKFold(5, shuffle=True, random_state=42)

def build_strict(mod, pres, early_end, strict):
    """strict=False reproduces the main pipeline (assessment due date <= cp);
       strict=True uses only submissions with date_submitted <= cp."""
    si = studentInfo[(studentInfo.code_module==mod)&(studentInfo.code_presentation==pres)].copy()
    sr = studentReg[(studentReg.code_module==mod)&(studentReg.code_presentation==pres)].copy()
    sv = studentVle[(studentVle.code_module==mod)&(studentVle.code_presentation==pres)].copy()
    sa = sa_all[(sa_all.code_module==mod)&(sa_all.code_presentation==pres)].copy()
    base = si.merge(sr[["id_student","date_registration","date_unregistration"]], on="id_student", how="left")
    svm = sv.merge(vle[["id_site","activity_type"]], on="id_site", how="left")
    sve = svm[svm.date<=early_end]
    daily = sve.groupby(["id_student","date"])["sum_click"].sum().reset_index()
    avgd = daily.groupby("id_student")["sum_click"].mean().reset_index(name="avg_clicks_per_active_day")
    vf = sve.groupby("id_student").agg(total_clicks=("sum_click","sum"), active_days=("date","nunique"),
        unique_sites=("id_site","nunique"), activity_type_diversity=("activity_type","nunique")).reset_index().merge(avgd, on="id_student", how="left")
    # assessment availability filter
    if strict:
        avail = sa[(sa.date_submitted.notna())&(sa.date_submitted<=early_end)&(sa.assessment_type!="Exam")]
        perf_src = avail[avail.weight>0].copy()
    else:
        avail = sa[(sa.date<=early_end)&(sa.assessment_type!="Exam")]
        perf_src = sa[(sa.date<=early_end)&(sa.assessment_type!="Exam")&(sa.weight>0)].copy()
    pf = avail.groupby("id_student").agg(assessments_submitted=("date_submitted", lambda x: x.notna().sum())).reset_index()
    if len(perf_src):
        perf_src["lateness"] = perf_src["date_submitted"] - perf_src["date"]
        perf = perf_src.groupby("id_student").agg(mean_score=("score","mean"), mean_lateness=("lateness","mean")).reset_index()
    else:
        perf = pd.DataFrame(columns=["id_student","mean_score","mean_lateness"])
    af = pf.merge(perf, on="id_student", how="left")
    m = base.merge(vf, on="id_student", how="left").merge(af, on="id_student", how="left")
    for c in ["total_clicks","active_days","unique_sites","activity_type_diversity","avg_clicks_per_active_day","assessments_submitted"]:
        if c in m: m[c] = m[c].fillna(0)
    m["has_early_assessment"] = (m["assessments_submitted"]>0).astype(int)
    for c in ["mean_score","mean_lateness"]:
        if c in m: m[c] = m[c].fillna(0)
    m["at_risk"] = m["final_result"].isin(["Fail","Withdrawn"]).astype(int)
    m["unreg"] = pd.to_numeric(m["date_unregistration"], errors="coerce")
    m = m[~(m["unreg"]<early_end)].copy()   # eligible only
    return m

def auc_of(m):
    nd = m[FEATURES].copy()
    for c in CLIP_COLS: nd[c] = nd[c].clip(nd[c].quantile(.01), nd[c].quantile(.99))
    X = StandardScaler().fit_transform(nd); y = m["at_risk"].values
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

rows = []
for mod in MODULES:
    for cp in CHECKPOINTS:
        a = auc_of(build_strict(mod, DEV, cp, strict=False))
        b = auc_of(build_strict(mod, DEV, cp, strict=True))
        rows.append({"module": mod, "cp": cp, "AUC_due_date": round(a, 3),
                     "AUC_strict_submitted": round(b, 3), "diff": round(b - a, 3)})
res = pd.DataFrame(rows)
print("=== MC2: leakage robustness (eligible AUC, assessment availability rule) ===")
print(res.to_string(index=False))
print(f"\nMean |difference| = {res['diff'].abs().mean():.3f}  (max {res['diff'].abs().max():.3f})")
res.to_csv("results/table_mc2_leakage_check.csv", index=False)
print("saved results/table_mc2_leakage_check.csv")
