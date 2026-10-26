"""
run_gudhi_simplices.py — higher-order structure via GUDHI simplex counts (Spearman).
For each emotion: clique (flag) complex of the 10%-density |Spearman| graph; number of
k-simplices (f-vector) computed with GUDHI. These are higher-dimensional structures
induced by a pairwise-connectivity flag complex; they are not independent measures
of physiological interactions among three or more neural sources.
a  emotion × simplex dimension counts (k = 2..5).
b  total higher-order simplices (k ≥ 2) per emotion + Euler characteristic.
c  density-robust view: number of simplices vs Rips filtration (1−|corr|) per emotion.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, gudhi
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lib.config import CACHE, FIG_DIR, EMOS, EMO_COLORS, CH, set_nature_style, save, W2, panel_letter, nice_heatmap, slim_colorbar
from lib.representations import vec_to_mat

N=len(CH); MAXDIM=5

def thr_edges(M, dens=0.10):
    A=np.abs(M.copy()); np.fill_diagonal(A,0); iu=np.triu_indices(N,1); w=A[iu]
    k=max(1,int(round(dens*len(w)))); thr=np.sort(w)[-k]
    return [(int(iu[0][t]),int(iu[1][t])) for t in np.where(w>=thr)[0]]

def fvector(M):
    st=gudhi.SimplexTree()
    for i in range(N): st.insert([i])
    for (i,j) in thr_edges(M): st.insert([i,j])
    st.expansion(MAXDIM)
    c=Counter()
    for s,_ in st.get_simplices(): c[len(s)-1]+=1
    return np.array([c[k] for k in range(MAXDIM+1)])

def rips_curve(M, ts):
    D=1.0-np.abs(M); np.fill_diagonal(D,0.0)
    st=gudhi.RipsComplex(distance_matrix=D, max_edge_length=float(max(ts))).create_simplex_tree(max_dimension=4)
    filt=np.sort([f for _,f in st.get_simplices()])
    return [int(np.searchsorted(filt,t,side="right")) for t in ts]

def main():
    set_nature_style()
    if not (CACHE/"agg_spearman.npy").exists(): print("[skip] agg missing"); return
    agg=np.load(CACHE/"agg_spearman.npy"); am=pd.read_csv(CACHE/"agg_meta_spearman.csv")
    mats={e: vec_to_mat(agg[(am.emotion==e).values].mean(0),N) for e in EMOS}
    F={e: fvector(mats[e]) for e in EMOS}
    euler={e: int(sum(((-1)**k)*F[e][k] for k in range(MAXDIM+1))) for e in EMOS}
    higher={e: int(F[e][2:].sum()) for e in EMOS}
    ts=np.linspace(0.3,0.95,16); curves={e: rips_curve(mats[e],ts) for e in EMOS}

    fig=plt.figure(figsize=(W2, W2*0.34)); gs=GridSpec(1,3,figure=fig,wspace=0.5,top=0.84,bottom=0.26,width_ratios=[1.3,1,1.1])
    # a: grouped barplot of simplex counts by dimension, per emotion
    axA=fig.add_subplot(gs[0,0])
    dims=[2,3,4]; shades=["#3b4cc0","#7b9ff9","#c9d3f0"]; wbar=0.26; x=np.arange(len(EMOS))
    eo=sorted(EMOS,key=lambda e:F[e][2],reverse=True)
    for di,d in enumerate(dims):
        axA.bar(x+(di-1)*wbar,[F[e][d] for e in eo],width=wbar,color=shades[di],
                edgecolor="k",linewidth=0.3,label=f"{d}-simplex")
    axA.set_xticks(x); axA.set_xticklabels(eo,rotation=90,fontsize=5.0)
    axA.set_ylabel("count"); axA.set_title("Simplex counts by dimension",pad=6)
    axA.legend(fontsize=4.8,frameon=False,loc="upper right")
    for sp in ["top","right"]: axA.spines[sp].set_visible(False)
    panel_letter(axA,"a",dx=-0.14)
    # b: higher-order total + euler
    axB=fig.add_subplot(gs[0,1])
    order=sorted(EMOS,key=lambda e:higher[e],reverse=True)
    axB.barh(range(len(order)),[higher[e] for e in order],color=[EMO_COLORS[e] for e in order],edgecolor="k",linewidth=0.3)
    for k,e in enumerate(order): axB.text(higher[e], k, f"  χ={euler[e]}", va="center", fontsize=4.4, color="0.3")
    axB.set_yticks(range(len(order))); axB.set_yticklabels(order,fontsize=5.2); axB.invert_yaxis()
    axB.set_xlabel("higher-order simplices (k≥2)"); axB.set_title("Higher-order load",pad=6)
    for sp in ["top","right"]: axB.spines[sp].set_visible(False)
    panel_letter(axB,"b",dx=-0.14)
    # c: rips filtration curves
    axC=fig.add_subplot(gs[0,2])
    for e in EMOS: axC.plot(ts,curves[e],color=EMO_COLORS[e],lw=1.2,alpha=0.9,label=e)
    axC.set_xlabel("filtration (1−|corr|)"); axC.set_ylabel("# simplices")
    axC.set_title("Rips filtration",pad=6); axC.set_yscale("log")
    axC.legend(fontsize=4.2,ncol=2,loc="upper left",frameon=False)
    for sp in ["top","right"]: axC.spines[sp].set_visible(False)
    panel_letter(axC,"c",dx=-0.16)
    save(fig, FIG_DIR/"part6"/"Figure_Part6_gudhi_simplices"); plt.close(fig)
    print("  higher-order simplices:", {e:higher[e] for e in order})
    print("  euler char:", {e:euler[e] for e in EMOS})

if __name__=="__main__": main()
