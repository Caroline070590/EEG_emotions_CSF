"""
run_richclub.py — mesoscale description per emotion (Spearman, 10% density).
a  rich-club coefficient curve phi(k) per emotion (do high-degree hubs interconnect?)
b  degree assortativity per emotion.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lib.config import CACHE, FIG_DIR, EMOS, EMO_COLORS, CH, set_nature_style, save, W2, panel_letter
from lib.representations import vec_to_mat

def thr_graph(M, dens=0.10):
    A=np.abs(M.copy()); np.fill_diagonal(A,0); n=A.shape[0]; iu=np.triu_indices(n,1)
    w=A[iu]; k=max(1,int(round(dens*len(w)))); thr=np.sort(w)[-k]
    return nx.from_numpy_array((np.maximum((A>=thr),0)).astype(float))

def rich_club_raw(G):
    deg=dict(G.degree()); kmax=max(deg.values()) if deg else 0
    phi={}
    for k in range(0,kmax):
        nodes=[u for u,d in deg.items() if d>k]
        if len(nodes)<2: break
        sub=G.subgraph(nodes); E=sub.number_of_edges(); n=len(nodes)
        phi[k]=2*E/(n*(n-1)) if n>1 else np.nan
    return phi

def main():
    set_nature_style()
    if not (CACHE/"agg_spearman.npy").exists(): print("[skip] agg missing"); return
    agg=np.load(CACHE/"agg_spearman.npy"); am=pd.read_csv(CACHE/"agg_meta_spearman.csv")
    fig=plt.figure(figsize=(W2*0.82, W2*0.36)); gs=GridSpec(1,2,figure=fig,wspace=0.4,top=0.86,bottom=0.2,width_ratios=[1.3,1])
    axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1])
    asso={}
    for e in EMOS:
        M=vec_to_mat(agg[(am["emotion"]==e).values].mean(0),len(CH)); G=thr_graph(M)
        phi=rich_club_raw(G)
        if phi: axA.plot(list(phi.keys()),list(phi.values()),color=EMO_COLORS[e],lw=1.3,label=e,alpha=0.9)
        try: asso[e]=nx.degree_assortativity_coefficient(G)
        except Exception: asso[e]=np.nan
    axA.set_xlabel("degree threshold k"); axA.set_ylabel("rich-club φ(k)")
    axA.set_title("Rich-club organization", pad=6)
    for sp in ["top","right"]: axA.spines[sp].set_visible(False)
    axA.legend(fontsize=4.6, ncol=2, loc="lower right", frameon=False); panel_letter(axA,"a",dx=-0.16)
    eo=sorted(EMOS,key=lambda e:asso[e],reverse=True)
    axB.barh(range(len(eo)),[asso[e] for e in eo],color=[EMO_COLORS[e] for e in eo],edgecolor="k",linewidth=0.3)
    axB.axvline(0,color="0.6",lw=0.6)
    axB.set_yticks(range(len(eo))); axB.set_yticklabels(eo,fontsize=5.4); axB.invert_yaxis()
    axB.set_xlabel("degree assortativity"); axB.set_title("Assortativity", pad=6)
    for sp in ["top","right"]: axB.spines[sp].set_visible(False); 
    panel_letter(axB,"b")
    save(fig, FIG_DIR/"part6"/"Figure_Part6_richclub_assortativity"); plt.close(fig)
    print("  assortativity range:", round(min(asso.values()),3),"to",round(max(asso.values()),3))

if __name__=="__main__": main()
