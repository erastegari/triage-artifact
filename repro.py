#!/usr/bin/env python3
"""Independent reimplementation of the BBB triage pipeline from the reported methods,
with an eligibility (still-enrolled) layer. Verifies disconnected-component and
Immediate-band at-risk rates vs the manuscript, and tests the already-withdrawn effect."""
import pandas as pd, numpy as np, networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import os
from config import DATA_DIR
D = DATA_DIR
VLE_FILE = os.path.join(DATA_DIR, "vle_bbb_ddd.csv")

studentInfo=pd.read_csv(D+"/studentInfo.csv")
studentReg=pd.read_csv(D+"/studentRegistration.csv")
studentAssessment=pd.read_csv(D+"/studentAssessment.csv")
assessments=pd.read_csv(D+"/assessments.csv")
vle=pd.read_csv(D+"/vle.csv")
studentVle=pd.read_csv(VLE_FILE)

studentInfo["imd_band"]=studentInfo["imd_band"].fillna("Unknown")
sa_all=studentAssessment.merge(assessments[["id_assessment","code_module","code_presentation","assessment_type","date","weight"]],on="id_assessment",how="left")

NETCOLS=["total_clicks","active_days","unique_sites","activity_type_diversity","avg_clicks_per_active_day","has_early_assessment","mean_score","mean_lateness"]

def build_model_df(mod,pres,early_end):
    si=studentInfo[(studentInfo.code_module==mod)&(studentInfo.code_presentation==pres)].copy()
    sr=studentReg[(studentReg.code_module==mod)&(studentReg.code_presentation==pres)].copy()
    sv=studentVle[(studentVle.code_module==mod)&(studentVle.code_presentation==pres)].copy()
    sa=sa_all[(sa_all.code_module==mod)&(sa_all.code_presentation==pres)].copy()
    base=si.merge(sr[["id_student","date_registration","date_unregistration"]],on="id_student",how="left")
    # vle features (date<=early_end)
    svm=sv.merge(vle[["id_site","activity_type"]],on="id_site",how="left")
    sve=svm[svm.date<=early_end]
    daily=sve.groupby(["id_student","date"])["sum_click"].sum().reset_index()
    avgd=daily.groupby("id_student")["sum_click"].mean().reset_index(name="avg_clicks_per_active_day")
    vf=sve.groupby("id_student").agg(total_clicks=("sum_click","sum"),active_days=("date","nunique"),
        unique_sites=("id_site","nunique"),activity_type_diversity=("activity_type","nunique")).reset_index().merge(avgd,on="id_student",how="left")
    # assessment features (date<=early_end, non-Exam)
    sap=sa[(sa.date<=early_end)&(sa.assessment_type!="Exam")]
    pf=sap.groupby("id_student").agg(assessments_seen=("id_assessment","nunique"),
        assessments_submitted=("date_submitted",lambda x:x.notna().sum())).reset_index()
    sperf=sa[(sa.date<=early_end)&(sa.assessment_type!="Exam")&(sa.weight>0)].copy()
    if len(sperf):
        sperf["lateness"]=sperf["date_submitted"]-sperf["date"]
        perf=sperf.groupby("id_student").agg(mean_score=("score","mean"),mean_weight=("weight","mean"),
            missed_scores=("score",lambda x:x.isna().sum()),mean_lateness=("lateness","mean"),max_lateness=("lateness","max")).reset_index()
    else:
        perf=pd.DataFrame(columns=["id_student","mean_score","mean_weight","missed_scores","mean_lateness","max_lateness"])
    af=pf.merge(perf,on="id_student",how="left")
    m=base.merge(vf,on="id_student",how="left").merge(af,on="id_student",how="left")
    for c in ["total_clicks","active_days","unique_sites","activity_type_diversity","avg_clicks_per_active_day","assessments_seen","assessments_submitted"]:
        if c in m: m[c]=m[c].fillna(0)
    m["has_early_assessment"]=(m["assessments_submitted"]>0).astype(int)
    for c in ["mean_score","mean_weight","mean_lateness","max_lateness","missed_scores"]:
        if c in m: m[c]=m[c].fillna(0)
    m["at_risk"]=m["final_result"].isin(["Fail","Withdrawn"]).astype(int)
    m["unreg"]=pd.to_numeric(m["date_unregistration"],errors="coerce")
    return m

def build_graph(m,k=10):
    cols=[c for c in NETCOLS if m[c].nunique()>1]
    nd=m[cols].copy()
    for c in ["total_clicks","avg_clicks_per_active_day","mean_lateness"]:
        if c in nd: nd[c]=nd[c].clip(nd[c].quantile(.01),nd[c].quantile(.99))
    X=StandardScaler().fit_transform(nd)
    nn=NearestNeighbors(n_neighbors=k+1,metric="euclidean").fit(X)
    dist,idx=nn.kneighbors(X); dist,idx=dist[:,1:],idx[:,1:]
    ids=m["id_student"].tolist()
    G=nx.Graph()
    for i,r in m.iterrows(): G.add_node(r["id_student"],at_risk=int(r["at_risk"]))
    seen={}
    for i in range(len(ids)):
        for j in range(k):
            a,b=ids[i],ids[idx[i,j]]; w=1/(1+dist[i,j]); key=(min(a,b),max(a,b))
            if key not in seen or dist[i,j]<seen[key][0]: seen[key]=(dist[i,j],w)
    for (a,b),(dd,w) in seen.items(): G.add_edge(a,b,weight=w)
    return G

def analyze(mod,pres,early_end,label):
    m=build_model_df(mod,pres,early_end)
    G=build_graph(m)
    comps=sorted(nx.connected_components(G),key=len,reverse=True)
    main=set(max(comps,key=len))
    m["in_main"]=m["id_student"].isin(main)
    # eligibility
    m["withdrew_before"]=m["unreg"]<early_end
    m["still"]=~m["withdrew_before"]
    disc=m[~m.in_main]           # disconnected => auto Immediate (band 1)
    out={"cohort":label,"cp":early_end,"n":len(m),"n_components":len(comps),
         "n_disconnected":len(disc),"disc_atrisk_rate":disc.at_risk.mean() if len(disc) else np.nan,
         "disc_withdrawn_before_cp_pct":100*disc.withdrew_before.mean() if len(disc) else np.nan,
         "cohort_withdrew_before_cp_pct":100*m.withdrew_before.mean()}
    # disc still-enrolled at-risk rate
    ds=disc[disc.still]
    out["disc_n_still"]=len(ds)
    out["disc_still_atrisk_rate"]=ds.at_risk.mean() if len(ds) else np.nan
    return m,G,disc,out

rows=[]
for (mod,pres,lab) in [("BBB","2013J","BBB-2013J(dev)"),("BBB","2014J","BBB-2014J(val)")]:
    for cp in [28,56,84]:
        m,G,disc,out=analyze(mod,pres,cp,lab)
        rows.append(out)
        print(f"\n### {lab}  Day {cp} ###")
        print(f"  students={out['n']}  components={out['n_components']}  disconnected={out['n_disconnected']}")
        print(f"  DISCONNECTED group: at-risk rate = {out['disc_atrisk_rate']*100:.1f}%  (the paper's 'near-perfect' groups)")
        print(f"    -> of these disconnected, ALREADY WITHDREW before Day {cp}: {out['disc_withdrawn_before_cp_pct']:.1f}%")
        print(f"    -> still-enrolled disconnected: n={out['disc_n_still']}  at-risk rate among them = {out['disc_still_atrisk_rate']*100 if out['disc_n_still'] else float('nan'):.1f}%")
        print(f"  Whole cohort already withdrew before Day {cp}: {out['cohort_withdrew_before_cp_pct']:.1f}%")

print("\n\n=== SUMMARY TABLE ===")
print(pd.DataFrame(rows).to_string(index=False))
