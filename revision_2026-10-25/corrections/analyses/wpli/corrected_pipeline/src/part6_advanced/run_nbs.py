"""
run_nbs.py — network-based statistic (NBS) for emotion vs neutral (Spearman edges).
Per subject, edge-wise difference (emotion - neutral) over the 435 channel pairs;
one-sample t per edge; supra-threshold edges form a graph; connected-component size
is tested against a sign-flip permutation null (cluster-level family-wise correction).
Significant subnetworks are drawn on the scalp per emotion.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, networkx as nx
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lib.config import CACHE, FIG_DIR, EMOS, CH, set_nature_style, save, W2
from lib.layout import scalp_edges

N = len(CH); IU = np.triu_indices(N, 1); T_THR = 3.0; NPERM = 5000

def edge_diff(agg, am, emo):
    piv = {}
    for s in am["subject"].unique():
        me=(am.subject==s)&(am.emotion==emo); mn=(am.subject==s)&(am.emotion=="neutral")
        if me.values.any() and mn.values.any():
            piv[s]=agg[me.values][0]-agg[mn.values][0]
    return np.vstack(list(piv.values()))   # subjects x 435

def t_edges(D):
    m=D.mean(0); s=D.std(0,ddof=1)+1e-12; return m/(s/np.sqrt(len(D)))

def max_component(tvec):
    supra=np.abs(tvec)>T_THR
    if supra.sum()==0: return 0, supra
    G=nx.Graph()
    for idx in np.where(supra)[0]:
        i,j=IU[0][idx],IU[1][idx]; G.add_edge(int(i),int(j))
    if G.number_of_edges()==0: return 0, supra
    comp=max(nx.connected_components(G), key=lambda c: G.subgraph(c).number_of_edges())
    return G.subgraph(comp).number_of_edges(), supra

def main():
    set_nature_style()
    if not (CACHE/"agg_spearman.npy").exists(): print("[skip] agg missing"); return
    agg=np.load(CACHE/"agg_spearman.npy"); am=pd.read_csv(CACHE/"agg_meta_spearman.csv")
    rng=np.random.default_rng(0)
    # First compute all contrasts, then correct the eight emotion-vs-neutral
    # cluster-FWE p-values across emotions with Benjamini-Hochberg FDR.
    res = {}
    for e in EMOS:
        if e == "neutral":
            continue
        D = edge_diff(agg, am, e)
        tv = t_edges(D)
        obs, supra = max_component(tv)
        null = np.empty(NPERM)
        for p in range(NPERM):
            sign = rng.choice([-1, 1], size=len(D))[:, None]
            null[p], _ = max_component(t_edges(D * sign))
        pval = (np.sum(null >= obs) + 1) / (NPERM + 1)
        res[e] = dict(obs_edges=int(obs), p_cluster_fwe=float(pval), tv=tv, supra=supra)

    emotions = [e for e in EMOS if e != "neutral"]
    qs = multipletests([res[e]["p_cluster_fwe"] for e in emotions], method="fdr_bh")[1]
    for e, q in zip(emotions, qs):
        res[e]["q_bh_across_emotions"] = float(q)

    rows = [{"emotion": e, "t_threshold": T_THR, "n_permutations": NPERM,
             "component_edges": res[e]["obs_edges"],
             "p_cluster_fwe": res[e]["p_cluster_fwe"],
             "q_bh_across_emotions": res[e]["q_bh_across_emotions"]}
            for e in emotions]
    pd.DataFrame(rows).to_csv(CACHE / "nbs_spearman_vs_neutral.csv", index=False)

    fig = plt.figure(figsize=(W2*0.8, W2*0.8))
    gs = GridSpec(3, 3, figure=fig, wspace=0.1, hspace=0.28, top=0.93, bottom=0.04)
    for k, e in enumerate(EMOS):
        ax = fig.add_subplot(gs[k//3, k%3])
        if e == "neutral":
            from lib.layout import head
            head(ax)
            ax.set_title("neutral (ref)", fontsize=7, pad=2)
            continue
        tv = res[e]["tv"]; supra = res[e]["supra"]; obs = res[e]["obs_edges"]
        pval = res[e]["p_cluster_fwe"]; qval = res[e]["q_bh_across_emotions"]
        edges = []; signs = []
        if obs > 0:
            G = nx.Graph()
            for idx in np.where(supra)[0]:
                i, j = int(IU[0][idx]), int(IU[1][idx])
                G.add_edge(i, j, idx=idx)
            comp = max(nx.connected_components(G), key=lambda c: G.subgraph(c).number_of_edges())
            for i, j in G.subgraph(comp).edges():
                idx = G[i][j]["idx"]
                edges.append((i, j))
                signs.append(np.sign(tv[idx]))
        scalp_edges(ax, CH, edges, signs)
        star = "*" if qval < 0.05 else ""
        ax.set_title(f"{e}  ({obs} edges, p={pval:.3f}, q={qval:.3f}){star}", fontsize=5.4, pad=2)
    save(fig, FIG_DIR/"part6"/"Figure_Part6_nbs_subnetworks")
    plt.close(fig)
    print("  NBS:", {e:(res[e]['obs_edges'], round(res[e]['p_cluster_fwe'],4), round(res[e]['q_bh_across_emotions'],4)) for e in emotions})

if __name__=="__main__": main()
