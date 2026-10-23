"""compute_features.py — Part 3 heavy compute (parallel, cached) per representation.
Spearman uses 30 channels; MTF/QG use q=Q_EMBED state matrices."""
import sys, json; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, EMOS, EI, REPR_N
from lib.representations import build_representation, aggregate
from lib.topology import tda_features, TCOLS
from lib.embedding import emotion_rdm
from lib.stats import lmm_table, subject_cluster_bootstrap, pseudo_eta2, valence_contrast
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

def run(tag="spearman", n_jobs=-1, n_boot=1000):
    n_nodes = REPR_N[tag]
    feat, meta = build_representation(tag=tag, n_jobs=n_jobs)
    agg, am = aggregate(feat, meta)
    emo_idx = am["emotion"].map(EI).values
    subj_codes = pd.factorize(am["subject"].values)[0]
    tda = pd.concat([am.reset_index(drop=True), tda_features(agg, n_nodes, n_jobs=n_jobs)], axis=1)
    lmm = lmm_table(tda, TCOLS)
    Y = tda[TCOLS].values; eta={}
    for k,c in enumerate(TCOLS):
        r=subject_cluster_bootstrap(am, lambda rows,kk=k: pseudo_eta2(rows,Y,emo_idx,subj_codes,kk),
                                    n_boot=n_boot, n_jobs=n_jobs)
        eta[c]=dict(obs=float(r["obs"]),lo=float(r["lo"]),hi=float(r["hi"]))
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
    yp=cross_val_predict(clf,agg,am["emotion"].values,groups=am["subject"].values,cv=LeaveOneGroupOut())
    y=am["emotion"].values
    rb=subject_cluster_bootstrap(am, lambda rows: balanced_accuracy_score(y[rows],yp[rows]), n_boot=n_boot, n_jobs=n_jobs)
    decode=dict(obs=float(rb["obs"]),lo=float(rb["lo"]),hi=float(rb["hi"]),chance=1/len(EMOS))
    rv=subject_cluster_bootstrap(am, lambda rows: valence_contrast(rows,agg,emo_idx), n_boot=n_boot, n_jobs=n_jobs)
    valence=dict(obs=float(rv["obs"]),lo=float(rv["lo"]),hi=float(rv["hi"]),p=float(np.mean(rv["boots"]<=0)))
    np.save(CACHE/f"agg_{tag}.npy",agg); am.to_csv(CACHE/f"agg_meta_{tag}.csv",index=False)
    tda.to_csv(CACHE/f"tda_{tag}.csv",index=False)
    emotion_rdm(agg,emo_idx).to_csv(CACHE/f"rdm_{tag}.csv")
    lmm.to_csv(CACHE/f"lmm_{tag}.csv",index=False)
    json.dump({"eta":eta,"decode":decode,"valence":valence}, open(CACHE/f"boot_{tag}.json","w"), indent=2)
    print(f"[{tag}] done. decode={decode['obs']:.3f} CI[{decode['lo']:.3f},{decode['hi']:.3f}] valence p={valence['p']:.3f}")

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--tag",default="spearman")
    ap.add_argument("--n_jobs",type=int,default=-1); ap.add_argument("--n_boot",type=int,default=1000)
    a=ap.parse_args(); run(tag=a.tag,n_jobs=a.n_jobs,n_boot=a.n_boot)
