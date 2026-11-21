#!/usr/bin/env python3
"""Eligibility-corrected evaluation + banded-LR ablation.
Builds the artifact's risk ordering (disconnected -> top, then main-community at-risk rate)
and a logistic-regression baseline on the SAME 8 features, then compares capture/precision
at matched burden on (a) ALL students [naive, reproduces paper] and (b) STILL-ENROLLED only
[honest, actionable population]."""
import pandas as pd, numpy as np, networkx as nx, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import repro  # reuse build_model_df, build_graph, NETCOLS

def cap_prec_at_burden(y, score, burden):
    """Flag the top `burden` fraction by score; return (capture=recall, precision, k)."""
    n=len(y); k=max(1,int(round(burden*n)))
    order=np.argsort(-score, kind="mergesort")
    flagged=np.zeros(n,bool); flagged[order[:k]]=True
    tp=int((flagged & (y==1)).sum()); P=int((y==1).sum())
    cap=tp/P if P else np.nan; prec=tp/k
    return cap, prec, k

def artifact_scores(m):
    """Risk ordering from the method: disconnected => 1.0, else community at-risk rate."""
    G=repro.build_graph(m)
    comps=sorted(nx.connected_components(G),key=len,reverse=True)
    main=set(max(comps,key=len))
    m=m.copy(); m["in_main"]=m["id_student"].isin(main)
    # Louvain on main component
    Gm=G.subgraph(main).copy()
    comm=nx.community.louvain_communities(Gm,weight="weight",seed=42)
    cmap={}
    for i,c in enumerate(comm):
        for node in c: cmap[node]=i
    m["comm"]=m["id_student"].map(cmap)
    rate=m[m.in_main].groupby("comm")["at_risk"].mean()
    m["score"]=np.where(m.in_main, m["comm"].map(rate).fillna(0.0), 1.0)  # disconnected=1.0
    return m["score"].values

def lr_scores(m):
    cols=[c for c in repro.NETCOLS if m[c].nunique()>1]
    X=StandardScaler().fit_transform(m[cols])
    y=m["at_risk"].values
    lr=LogisticRegression(class_weight="balanced",max_iter=2000)
    proba=cross_val_predict(lr,X,y,cv=5,method="predict_proba")[:,1]
    return proba

def run(mod,pres,cp,label):
    m=repro.build_model_df(mod,pres,cp)
    m["still"]=~(m["unreg"]<cp)
    art=artifact_scores(m); lr=lr_scores(m)
    m["art"]=art; m["lr"]=lr
    rows=[]
    for popname, sub in [("ALL (naive)", m), ("STILL-ENROLLED", m[m.still])]:
        y=sub["at_risk"].values
        if y.sum()==0 or y.sum()==len(y):
            continue
        aauc=roc_auc_score(y, sub["art"]); lauc=roc_auc_score(y, sub["lr"])
        # disconnected-group burden (natural Immediate-ish op point) and 20% / 30%
        disc_burden=float((sub["art"]>=1.0).mean())
        for bname,burden in [("disc-grp",disc_burden),("20%",0.20),("30%",0.30)]:
            if burden<=0: continue
            ca,pa,k=cap_prec_at_burden(y, sub["art"].values, burden)
            cl,pl,_=cap_prec_at_burden(y, sub["lr"].values, burden)
            rows.append(dict(cohort=label,cp=cp,pop=popname,n=len(sub),atrisk=int(y.sum()),
                base_rate=round(y.mean(),3), op=bname, burden=round(burden,3),
                art_cap=round(ca,3),art_prec=round(pa,3), lr_cap=round(cl,3),lr_prec=round(pl,3),
                art_auc=round(aauc,3),lr_auc=round(lauc,3)))
    return rows

allrows=[]
for mod,pres,lab in [("BBB","2014J","BBB-2014J"),("BBB","2013J","BBB-2013J")]:
    for cp in [28,56,84]:
        allrows+=run(mod,pres,cp,lab)
df=pd.DataFrame(allrows)
pd.set_option("display.width",200,"display.max_columns",30)
print(df.to_string(index=False))
df.to_csv("/home/claude/eligibility_corrected_results.csv",index=False)
print("\nsaved eligibility_corrected_results.csv")
