"""
run_figure4_spatial.py — KEYSTONE Figure 4: where emotion reorganizes the network.
a,b  NBS subnetwork vs neutral for sadness (increases) and anger (decreases).
c    signed NBS edge counts per emotion (increases vs decreases; * = permutation p<.05).
d,e  eigenvector-centrality difference-from-neutral topographies (sadness, anger).
f    regional emotion-modulation strength (σ range across emotions; MTF & QG).
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, networkx as nx, warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")
from lib.config import CACHE, FIG_DIR, EMOS, EMO_COLORS, CH, set_nature_style, save, W2, panel_letter
from lib.layout import scalp_edges, topomap, head
from lib.regions import REGIONS
from lib.representations import vec_to_mat

# Match the manuscript NBS specification. Do not reduce this for publication runs.
N=len(CH); IU=np.triu_indices(N,1); T_THR=3.0; NPERM=5000

def edge_diff(agg, am, emo):
    rows=[]
    for s in am["subject"].unique():
        me=(am.subject==s)&(am.emotion==emo); mn=(am.subject==s)&(am.emotion=="neutral")
        if me.values.any() and mn.values.any(): rows.append(agg[me.values][0]-agg[mn.values][0])
    return np.vstack(rows)
def tvec(D): return D.mean(0)/(D.std(0,ddof=1)/np.sqrt(len(D))+1e-12)
def big_component(tv):
    supra=np.abs(tv)>T_THR
    if supra.sum()==0: return [],0
    G=nx.Graph()
    for idx in np.where(supra)[0]: G.add_edge(int(IU[0][idx]),int(IU[1][idx]),idx=idx)
    comp=max(nx.connected_components(G),key=lambda c:G.subgraph(c).number_of_edges())
    SG=G.subgraph(comp); return [(i,j,SG[i][j]["idx"]) for i,j in SG.edges()], SG.number_of_edges()
def nbs(D, rng):
    tv=tvec(D); comp,obs=big_component(tv)
    null=np.array([big_component(tvec(D*rng.choice([-1,1],size=len(D))[:,None]))[1] for _ in range(NPERM)])
    p=(np.sum(null>=obs)+1)/(NPERM+1)
    inc=sum(1 for _,_,idx in comp if tv[idx]>0); dec=len(comp)-inc
    return tv, comp, p, inc, dec

def eig_diff(agg, am, emo, neutral_ev):
    M=vec_to_mat(agg[(am.emotion==emo).values].mean(0),N); A=np.abs(M); np.fill_diagonal(A,0)
    ev=nx.eigenvector_centrality_numpy(nx.from_numpy_array(A),weight="weight")
    return np.array([ev[i] for i in range(N)])-neutral_ev

def main():
    set_nature_style()
    agg=np.load(CACHE/"agg_spearman.npy"); am=pd.read_csv(CACHE/"agg_meta_spearman.csv")
    rng=np.random.default_rng(0)
    res={}
    for e in EMOS:
        if e=="neutral": continue
        res[e]=nbs(edge_diff(agg,am,e),rng)
    Mn=vec_to_mat(agg[(am.emotion=="neutral").values].mean(0),N); An=np.abs(Mn); np.fill_diagonal(An,0)
    evn=np.array([nx.eigenvector_centrality_numpy(nx.from_numpy_array(An),weight="weight")[i] for i in range(N)])

    fig=plt.figure(figsize=(W2, W2*0.66))
    gs=GridSpec(2,3,figure=fig,wspace=0.25,hspace=0.3,top=0.92,bottom=0.1)
    # a,b NBS subnetworks
    for col,e in zip([0,1],["sadness","anger"]):
        ax=fig.add_subplot(gs[0,col]); tv,comp,p,inc,dec=res[e]
        scalp_edges(ax,CH,[(i,j) for i,j,_ in comp],[np.sign(tv[idx]) for _,_,idx in comp])
        ax.set_title(f"{e}: NBS vs neutral\n({len(comp)} edges, p={p:.3f})",fontsize=6.5,pad=3)
        panel_letter(ax,"ab"[col])
    # c signed NBS summary
    axC=fig.add_subplot(gs[0,2])
    order=sorted(res,key=lambda e:res[e][3]+res[e][4],reverse=True)
    for k,e in enumerate(order):
        _,comp,p,inc,dec=res[e]
        axC.barh(k,inc,color="#c0392b",edgecolor="k",linewidth=0.3)
        axC.barh(k,-dec,color="#2c5fa8",edgecolor="k",linewidth=0.3)
        if p<0.05: axC.text(inc+2,k,"*",fontsize=7,va="center")
    axC.axvline(0,color="0.5",lw=0.6)
    axC.set_yticks(range(len(order))); axC.set_yticklabels(order,fontsize=5.4); axC.invert_yaxis()
    axC.set_xlabel("← decreased   edges   increased →",fontsize=5.4)
    axC.set_title("NBS edge direction",pad=4)
    for sp in ["top","right"]: axC.spines[sp].set_visible(False)
    panel_letter(axC,"c")
    # d,e centrality topographies
    vmax=None
    diffs={e:eig_diff(agg,am,e,evn) for e in ["sadness","anger"]}
    vmax=float(np.percentile(np.abs(np.concatenate(list(diffs.values()))),98))
    for col,e in zip([0,1],["sadness","anger"]):
        ax=fig.add_subplot(gs[1,col]); im=topomap(ax,CH,diffs[e],cmap="RdBu_r",vmin=-vmax,vmax=vmax)
        ax.set_title(f"{e}: hub Δ (eigenvector)",fontsize=6.5,pad=3); panel_letter(ax,"de"[col])
    cax=fig.add_axes([0.355,0.12,0.012,0.18]); fig.colorbar(im,cax=cax).set_label("Δ centrality",fontsize=5)
    # f regional modulation
    axF=fig.add_subplot(gs[1,2])
    d=pd.read_csv(CACHE/"region_trial_metrics.csv")
    sm=d.groupby(["emotion","region","method"])["sigma_small_world"].mean().reset_index()
    w=0.38
    for mi,method in enumerate(["MTF","QG"]):
        rng_vals=[]
        for r in REGIONS:
            v=sm[(sm.method==method)&(sm.region==r)]["sigma_small_world"]
            rng_vals.append(v.max()-v.min() if len(v) else 0)
        axF.barh(np.arange(len(REGIONS))+(mi-0.5)*w,rng_vals,height=w,
                 color="#34495e" if method=="MTF" else "#16a085",label=method,edgecolor="k",linewidth=0.3)
    axF.set_yticks(range(len(REGIONS))); axF.set_yticklabels([r.capitalize() for r in REGIONS],fontsize=5.4)
    axF.invert_yaxis(); axF.set_xlabel("σ range across emotions",fontsize=5.6)
    axF.set_title("Regional modulation",pad=4); axF.legend(fontsize=5,loc="lower right")
    for sp in ["top","right"]: axF.spines[sp].set_visible(False)
    panel_letter(axF,"f")
    save(fig, FIG_DIR/"part4"/"Figure_Part4_spatial_specificity"); plt.close(fig)
    print("  NBS signed:",{e:(res[e][3],res[e][4],round(res[e][2],3)) for e in order})

if __name__=="__main__": main()
