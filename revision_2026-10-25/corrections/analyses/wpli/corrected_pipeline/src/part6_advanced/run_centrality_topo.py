"""
run_centrality_topo.py — emotion-specific hub topographies (Spearman channel graph).
Eigenvector and betweenness centrality DIFFERENCE from neutral, per emotion, as
scalp maps (diverging). Neutral panel = zero reference. Eigenvector centrality is
computed on the full weighted |Spearman| graph (robust to thresholding/disconnection);
betweenness on the 10%-density graph.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lib.config import CACHE, FIG_DIR, EMOS, CH, set_nature_style, save, W2
from lib.layout import topomap
from lib.representations import vec_to_mat

def eig_full(M):
    A=np.abs(M.copy()); np.fill_diagonal(A,0)
    G=nx.from_numpy_array(A); ev=nx.eigenvector_centrality_numpy(G,weight="weight")
    return np.array([ev[i] for i in range(len(CH))])

def betw_thr(M, dens=0.10):
    A=np.abs(M.copy()); np.fill_diagonal(A,0); n=A.shape[0]; iu=np.triu_indices(n,1)
    w=A[iu]; k=max(1,int(round(dens*len(w)))); thr=np.sort(w)[-k]; B=(A>=thr)*A
    G=nx.from_numpy_array(np.maximum(B,B.T))
    bt=nx.betweenness_centrality(G,weight=lambda u,v,d:1.0/(d["weight"]+1e-9),normalized=True)
    return np.array([bt[i] for i in range(n)])

def grid_diff(D, name, neutral_key="neutral"):
    set_nature_style()
    vmax=float(np.nanpercentile(np.abs(np.vstack([D[e] for e in EMOS])),98)) or 1e-6
    fig=plt.figure(figsize=(W2*0.8, W2*0.8))
    gs=GridSpec(3,3,figure=fig,wspace=0.15,hspace=0.3,top=0.93,bottom=0.05,right=0.87)
    im=None
    for k,e in enumerate(EMOS):
        ax=fig.add_subplot(gs[k//3,k%3])
        im=topomap(ax,CH,D[e],cmap="RdBu_r",vmin=-vmax,vmax=vmax)
        ax.set_title(("neutral (ref)" if e==neutral_key else e),pad=2,fontsize=7)
    cax=fig.add_axes([0.89,0.35,0.015,0.3])
    fig.colorbar(im,cax=cax).set_label(f"Δ {name.replace('_',' ')} vs neutral",fontsize=6)
    save(fig, FIG_DIR/"part6"/f"Figure_Part6_{name}_topo"); plt.close(fig)

def main():
    if not (CACHE/"agg_spearman.npy").exists(): print("[skip] agg missing"); return
    agg=np.load(CACHE/"agg_spearman.npy"); am=pd.read_csv(CACHE/"agg_meta_spearman.csv")
    EV={}; BT={}
    for e in EMOS:
        M=vec_to_mat(agg[(am["emotion"]==e).values].mean(0),len(CH))
        EV[e]=eig_full(M); BT[e]=betw_thr(M)
    EVd={e: EV[e]-EV["neutral"] for e in EMOS}
    BTd={e: BT[e]-BT["neutral"] for e in EMOS}
    grid_diff(EVd,"eigenvector_centrality")
    grid_diff(BTd,"betweenness_centrality")
    print("  saved eigenvector + betweenness difference topographies")

if __name__=="__main__": main()
