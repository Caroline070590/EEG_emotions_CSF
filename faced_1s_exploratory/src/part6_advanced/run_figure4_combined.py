"""
run_figure4_combined.py — COMBINED keystone Figure 4: network reorganization of emotion.
Row 1 (a-c): NBS subnetworks vs neutral (sadness, anger) + signed edge-direction summary.
Row 2 (d-f): eigenvector-centrality hub topographies (sadness, anger) + regional modulation.
Row 3 (g-i): GUDHI higher-order topology — simplex-count barplot, higher-order load + Euler
             characteristic, and density-robust Rips filtration.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, networkx as nx, gudhi, warnings
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")
from lib.config import CACHE, FIG_DIR, EMOS, EMO_COLORS, CH, set_nature_style, save, W2, panel_letter
from lib.layout import scalp_edges, topomap
from lib.regions import REGIONS
from lib.representations import vec_to_mat

# Match the manuscript NBS specification. Do not reduce this for publication runs.
N=len(CH); IU=np.triu_indices(N,1); T_THR=3.0; NPERM=5000; MAXDIM=5

# ---------- NBS ----------
def edge_diff(agg,am,emo):
    rows=[]
    for s in am.subject.unique():
        me=(am.subject==s)&(am.emotion==emo); mn=(am.subject==s)&(am.emotion=="neutral")
        if me.values.any() and mn.values.any(): rows.append(agg[me.values][0]-agg[mn.values][0])
    return np.vstack(rows)
def tvec(D): return D.mean(0)/(D.std(0,ddof=1)/np.sqrt(len(D))+1e-12)
def big_comp(tv):
    supra=np.abs(tv)>T_THR
    if supra.sum()==0: return [],0
    G=nx.Graph()
    for idx in np.where(supra)[0]: G.add_edge(int(IU[0][idx]),int(IU[1][idx]),idx=idx)
    comp=max(nx.connected_components(G),key=lambda c:G.subgraph(c).number_of_edges()); SG=G.subgraph(comp)
    return [(i,j,SG[i][j]["idx"]) for i,j in SG.edges()], SG.number_of_edges()
def nbs(D,rng):
    tv=tvec(D); comp,obs=big_comp(tv)
    null=np.array([big_comp(tvec(D*rng.choice([-1,1],size=len(D))[:,None]))[1] for _ in range(NPERM)])
    p=(np.sum(null>=obs)+1)/(NPERM+1); inc=sum(1 for _,_,idx in comp if tv[idx]>0)
    return tv,comp,p,inc,len(comp)-inc

# ---------- centrality ----------
def eig_full(M):
    A=np.abs(M.copy()); np.fill_diagonal(A,0)
    ev=nx.eigenvector_centrality_numpy(nx.from_numpy_array(A),weight="weight")
    return np.array([ev[i] for i in range(N)])

# ---------- gudhi ----------
def thr_edges(M,dens=0.10):
    A=np.abs(M.copy()); np.fill_diagonal(A,0); w=A[IU]; k=max(1,int(round(dens*len(w)))); thr=np.sort(w)[-k]
    return [(int(IU[0][t]),int(IU[1][t])) for t in np.where(w>=thr)[0]]
def fvector(M):
    st=gudhi.SimplexTree()
    for i in range(N): st.insert([i])
    for (i,j) in thr_edges(M): st.insert([i,j])
    st.expansion(MAXDIM); c=Counter()
    for s,_ in st.get_simplices(): c[len(s)-1]+=1
    return np.array([c[k] for k in range(MAXDIM+1)])
def rips_curve(M,ts):
    D=1.0-np.abs(M); np.fill_diagonal(D,0.0)
    st=gudhi.RipsComplex(distance_matrix=D,max_edge_length=float(max(ts))).create_simplex_tree(max_dimension=4)
    filt=np.sort([f for _,f in st.get_simplices()])
    return [int(np.searchsorted(filt,t,side="right")) for t in ts]

def main():
    set_nature_style()
    agg=np.load(CACHE/"agg_spearman.npy"); am=pd.read_csv(CACHE/"agg_meta_spearman.csv")
    rng=np.random.default_rng(0)
    mats={e:vec_to_mat(agg[(am.emotion==e).values].mean(0),N) for e in EMOS}
    res={e:nbs(edge_diff(agg,am,e),rng) for e in EMOS if e!="neutral"}
    evn=eig_full(mats["neutral"]); hub={e:eig_full(mats[e])-evn for e in ["sadness","anger"]}
    F={e:fvector(mats[e]) for e in EMOS}
    euler={e:int(sum(((-1)**k)*F[e][k] for k in range(MAXDIM+1))) for e in EMOS}
    higher={e:int(F[e][2:].sum()) for e in EMOS}
    ts=np.linspace(0.3,0.95,16); curves={e:rips_curve(mats[e],ts) for e in EMOS}

    fig=plt.figure(figsize=(W2, W2*1.02))
    gs=GridSpec(3,3,figure=fig,wspace=0.35,hspace=0.45,top=0.95,bottom=0.11)
    # Row1 NBS
    for col,e in zip([0,1],["sadness","anger"]):
        ax=fig.add_subplot(gs[0,col]); tv,comp,p,inc,dec=res[e]
        scalp_edges(ax,CH,[(i,j) for i,j,_ in comp],[np.sign(tv[idx]) for _,_,idx in comp])
        ax.set_title(f"{e} · {len(comp)} edges (p={p:.3f})",fontsize=6.2,pad=8); panel_letter(ax,"ab"[col])
    axC=fig.add_subplot(gs[0,2]); order=sorted(res,key=lambda e:res[e][3]+res[e][4],reverse=True)
    for k,e in enumerate(order):
        _,_,p,inc,dec=res[e]; axC.barh(k,inc,color="#c0392b",edgecolor="k",linewidth=0.3); axC.barh(k,-dec,color="#2c5fa8",edgecolor="k",linewidth=0.3)
        if p<0.05: axC.text(inc+2,k,"*",fontsize=7,va="center")
    axC.axvline(0,color="0.5",lw=0.6); axC.set_yticks(range(len(order))); axC.set_yticklabels(order,fontsize=5.2); axC.invert_yaxis()
    axC.set_xlabel("← decreased  edges  increased →",fontsize=5.2); axC.set_title("NBS edge direction",pad=4)
    for sp in ["top","right"]: axC.spines[sp].set_visible(False)
    panel_letter(axC,"c")
    # Row2 hubs + regional
    vmax=float(np.percentile(np.abs(np.concatenate(list(hub.values()))),98))
    for col,e in zip([0,1],["sadness","anger"]):
        ax=fig.add_subplot(gs[1,col]); im=topomap(ax,CH,hub[e],cmap="RdBu_r",vmin=-vmax,vmax=vmax)
        ax.set_title(f"{e} · hub Δ (eigenvector)",fontsize=6.2,pad=8); panel_letter(ax,"de"[col])
    cax=fig.add_axes([0.355,0.40,0.011,0.13]); fig.colorbar(im,cax=cax).set_label("Δ centrality",fontsize=5)
    axF=fig.add_subplot(gs[1,2]); d=pd.read_csv(CACHE/"region_trial_metrics.csv")
    sm=d.groupby(["emotion","region","method"])["sigma_small_world"].mean().reset_index(); w=0.38
    for mi,method in enumerate(["MTF","QG"]):
        rv=[ (sm[(sm.method==method)&(sm.region==r)]["sigma_small_world"].max()-sm[(sm.method==method)&(sm.region==r)]["sigma_small_world"].min()) if len(sm[(sm.method==method)&(sm.region==r)]) else 0 for r in REGIONS]
        axF.barh(np.arange(len(REGIONS))+(mi-0.5)*w,rv,height=w,color="#34495e" if method=="MTF" else "#16a085",label=method,edgecolor="k",linewidth=0.3)
    axF.set_yticks(range(len(REGIONS))); axF.set_yticklabels([r.capitalize() for r in REGIONS],fontsize=5.2); axF.invert_yaxis()
    axF.set_xlabel("σ range across emotions",fontsize=5.4); axF.set_title("Regional modulation",pad=4); axF.legend(fontsize=5,loc="lower right")
    for sp in ["top","right"]: axF.spines[sp].set_visible(False)
    panel_letter(axF,"f")
    # Row3 GUDHI
    axG=fig.add_subplot(gs[2,0]); dims=[2,3,4]; shades=["#3b4cc0","#7b9ff9","#c9d3f0"]; wbar=0.26; x=np.arange(len(EMOS))
    eo=sorted(EMOS,key=lambda e:F[e][2],reverse=True)
    for di,dd in enumerate(dims):
        axG.bar(x+(di-1)*wbar,[F[e][dd] for e in eo],width=wbar,color=shades[di],edgecolor="k",linewidth=0.3,label=f"{dd}-simplex")
    axG.set_xticks(x); axG.set_xticklabels(eo,rotation=90,fontsize=4.8); axG.set_ylabel("count")
    axG.set_title("Simplex counts (GUDHI)",pad=4); axG.legend(fontsize=4.6,frameon=False,loc="upper right")
    for sp in ["top","right"]: axG.spines[sp].set_visible(False)
    panel_letter(axG,"g")
    axH=fig.add_subplot(gs[2,1]); ho=sorted(EMOS,key=lambda e:higher[e],reverse=True)
    axH.barh(range(len(ho)),[higher[e] for e in ho],color=[EMO_COLORS[e] for e in ho],edgecolor="k",linewidth=0.3)
    for k,e in enumerate(ho): axH.text(higher[e],k,f"  χ={euler[e]}",va="center",fontsize=4.4,color="0.3")
    axH.set_yticks(range(len(ho))); axH.set_yticklabels(ho,fontsize=5.2); axH.invert_yaxis()
    axH.set_xlabel("higher-order simplices (k≥2)",fontsize=5.4); axH.set_title("Higher-order load",pad=4)
    for sp in ["top","right"]: axH.spines[sp].set_visible(False)
    panel_letter(axH,"h")
    axI=fig.add_subplot(gs[2,2])
    for e in EMOS: axI.plot(ts,curves[e],color=EMO_COLORS[e],lw=1.1,alpha=0.9)
    axI.set_xlabel("filtration (1−|corr|)",fontsize=5.4); axI.set_ylabel("# simplices"); axI.set_yscale("log")
    axI.set_title("Rips filtration",pad=4)
    for sp in ["top","right"]: axI.spines[sp].set_visible(False)
    panel_letter(axI,"i")
    handles=[mpatch.Patch(facecolor=EMO_COLORS[e],edgecolor="k",lw=0.3,label=e) for e in EMOS]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(0.5,0.005),ncol=9,
               fontsize=5.4,frameon=False,handlelength=1.0,columnspacing=0.9,
               title="emotion (panels c, h, i)",title_fontsize=5.6)
    save(fig, FIG_DIR/"part4"/"Figure_Part4_network_reorganization"); plt.close(fig)
    print("  combined Figure 4 saved")

if __name__=="__main__": main()
