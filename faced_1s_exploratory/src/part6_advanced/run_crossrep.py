"""
run_crossrep.py — cross-representation decoding (shared emotion code test).
Train an emotion decoder on one representation and test on another. Off-diagonal
cells bridge representations through a CCA shared space learned on training subjects;
diagonal cells are standard within-representation decoding. Subject-grouped 5-fold CV.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import CCA
from sklearn.metrics import balanced_accuracy_score
from lib.config import CACHE, FIG_DIR, EMOS, set_nature_style, save, W2, panel_letter, nice_heatmap, slim_colorbar

METHODS = {"spearman":"Spearman","mtf":"MTF","qg":"QG"}

def load_aligned():
    data={}; key=None
    for t in METHODS:
        if not (CACHE/f"agg_{t}.npy").exists(): continue
        X=np.load(CACHE/f"agg_{t}.npy"); m=pd.read_csv(CACHE/f"agg_meta_{t}.csv")
        m=m.reset_index(drop=True); m["k"]=m["subject"].astype(str)+"|"+m["emotion"]
        data[t]=(X,m)
    tags=list(data); base=data[tags[0]][1]["k"].tolist()
    aligned={}
    for t in tags:
        X,m=data[t]; idx=[m.index[m["k"]==k][0] for k in base]
        aligned[t]=X[idx]
    meta=data[tags[0]][1]
    return tags, aligned, meta["emotion"].values, meta["subject"].values

def main():
    set_nature_style()
    tags, X, y, groups = load_aligned()
    k_can=8; gkf=GroupKFold(5)
    ACC=np.full((len(tags),len(tags)),np.nan)
    for ai,A in enumerate(tags):
        for bi,B in enumerate(tags):
            yt_all=[]; yp_all=[]
            for tr,te in gkf.split(X[A],y,groups):
                if A==B:
                    sc=StandardScaler().fit(X[A][tr])
                    clf=LogisticRegression(max_iter=2000).fit(sc.transform(X[A][tr]),y[tr])
                    yp=clf.predict(sc.transform(X[A][te]))
                else:
                    sA=StandardScaler().fit(X[A][tr]); sB=StandardScaler().fit(X[B][tr])
                    At,Bt=sA.transform(X[A][tr]),sB.transform(X[B][tr])
                    cca=CCA(n_components=k_can,max_iter=500).fit(At,Bt)
                    Ac=cca.transform(At)                       # A train in shared space
                    Bc_te=sB.transform(X[B][te]) @ cca.y_rotations_   # B test into shared space
                    clf=LogisticRegression(max_iter=2000).fit(Ac,y[tr])
                    yp=clf.predict(Bc_te)
                yt_all.append(y[te]); yp_all.append(yp)
            ACC[ai,bi]=balanced_accuracy_score(np.concatenate(yt_all),np.concatenate(yp_all))
    fig,ax=plt.subplots(figsize=(W2*0.46,W2*0.42)); 
    im=nice_heatmap(ax,ACC,[METHODS[t] for t in tags],[METHODS[t] for t in tags],
                    cmap="viridis",vmin=0.11,vmax=max(0.2,np.nanmax(ACC)),annot=True,fmt="{:.3f}",
                    aspect="equal",fontsize=6,annot_size=6)
    ax.set_ylabel("train"); ax.set_xlabel("test")
    ax.set_title("Cross-representation decoding\n(balanced acc; chance 0.111)", pad=8)
    slim_colorbar(fig,im,ax,"balanced accuracy")
    fig.subplots_adjust(left=0.2,bottom=0.2,top=0.82,right=0.85)
    save(fig, FIG_DIR/"part6"/"Figure_Part6_cross_representation"); plt.close(fig)
    print("  ACC matrix:\n", np.round(ACC,3))

if __name__=="__main__": main()
