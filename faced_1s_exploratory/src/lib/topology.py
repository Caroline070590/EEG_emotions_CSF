"""topology.py — higher-order topology via persistent homology (parallelised).
Works on any matrix size (Spearman 30x30, MTF/QG qxq)."""
import numpy as np, pandas as pd, gudhi
from joblib import Parallel, delayed
from lib.representations import vec_to_mat

TCOLS = ["H0_total","H0_entropy","H1_n","H1_total","H1_max","H1_entropy"]

def _tda_one(v, n_nodes, max_edge=2.0):
    M = vec_to_mat(v, n_nodes)
    D = np.sqrt(np.maximum(0, 2*(1-M))); np.fill_diagonal(D, 0)
    st = gudhi.RipsComplex(distance_matrix=D, max_edge_length=max_edge).create_simplex_tree(max_dimension=2)
    st.compute_persistence(); out={}
    for dim in (0,1):
        life=np.array([d-b for b,d in st.persistence_intervals_in_dimension(dim) if np.isfinite(d)])
        out[f"H{dim}_n"]=float(len(life)); out[f"H{dim}_total"]=float(life.sum()) if len(life) else 0.0
        out[f"H{dim}_max"]=float(life.max()) if len(life) else 0.0
        if len(life): p=life/life.sum(); p=p[p>0]; out[f"H{dim}_entropy"]=float(-(p*np.log(p)).sum())
        else: out[f"H{dim}_entropy"]=0.0
    return out

def tda_features(agg_feat, n_nodes, max_edge=2.0, n_jobs=-1):
    res=Parallel(n_jobs=n_jobs)(delayed(_tda_one)(v,n_nodes,max_edge) for v in agg_feat)
    return pd.DataFrame(res)

def tda_cutoff_sweep(agg_feat, n_nodes, cutoffs=(1.6,1.8,2.0), n_jobs=-1):
    return {c: tda_features(agg_feat, n_nodes, max_edge=c, n_jobs=n_jobs)["H1_total"].values for c in cutoffs}
