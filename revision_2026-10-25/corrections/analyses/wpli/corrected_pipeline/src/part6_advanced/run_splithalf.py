"""
run_splithalf.py — split-half reliability of the emotion RSA geometry (Spearman).
Random subject splits; RDM (emotion x emotion correlation-distance of centroids) in
each half; Spearman correlation of the two RDMs across splits. Reports the stability
of the emotion taxonomy.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from lib.config import CACHE, FIG_DIR, EMOS, set_nature_style, save, W2, panel_letter, nice_heatmap, slim_colorbar

def rdm(agg, am, subs):
    cents=[]
    for e in EMOS:
        m=(am["emotion"]==e).values & am["subject"].isin(subs).values
        cents.append(agg[m].mean(0))
    C=np.vstack(cents); return squareform(pdist(C, metric="correlation"))

def main():
    set_nature_style()
    if not (CACHE/"agg_spearman.npy").exists(): print("[skip] agg missing"); return
    agg=np.load(CACHE/"agg_spearman.npy"); am=pd.read_csv(CACHE/"agg_meta_spearman.csv")
    subs=np.array(sorted(am["subject"].unique())); rng=np.random.default_rng(0)
    iu=np.triu_indices(len(EMOS),1); rs=[]
    for _ in range(300):
        rng.shuffle(subs); h=len(subs)//2
        A=rdm(agg,am,subs[:h]); B=rdm(agg,am,subs[h:])
        rs.append(spearmanr(A[iu],B[iu]).correlation)
    rs=np.array(rs)
    fig=plt.figure(figsize=(W2*0.8, W2*0.34)); gs=GridSpec(1,2,figure=fig,wspace=0.45,top=0.86,bottom=0.2,width_ratios=[1,1.1])
    axA=fig.add_subplot(gs[0,0])
    axA.hist(rs,bins=24,color="#4c72b0",edgecolor="white",linewidth=0.3)
    axA.axvline(np.median(rs),color="#c0392b",lw=1.2,ls="--")
    axA.set_xlabel("split-half RDM correlation (ρ)"); axA.set_ylabel("count")
    axA.set_title(f"RSA reliability (median ρ = {np.median(rs):.2f})", pad=6)
    for sp in ["top","right"]: axA.spines[sp].set_visible(False); 
    panel_letter(axA,"a",dx=-0.18)
    axB=fig.add_subplot(gs[0,1]); full=rdm(agg,am,subs)
    im=nice_heatmap(axB,full,EMOS,EMOS,cmap="magma",aspect="equal",fontsize=5.0)
    axB.set_title("Full-sample RDM", pad=6); panel_letter(axB,"b",dx=-0.14); slim_colorbar(fig,im,axB,"1−r")
    save(fig, FIG_DIR/"part6"/"Figure_Part6_rsa_reliability"); plt.close(fig)
    print(f"  split-half median rho={np.median(rs):.3f}  [{np.percentile(rs,5):.2f},{np.percentile(rs,95):.2f}]")

if __name__=="__main__": main()
