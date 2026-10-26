"""
run_network_enrichment.py — deeper complex-network analysis (Spearman channel graph).
a  emotion x channel nodal strength (z across emotions); channels grouped by region.
b  modularity Q per emotion (Louvain on 10%-density graph).
c  minimum-spanning-tree leaf fraction per emotion (density-independent).
Reads cache/agg_spearman.npy (+ meta).
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity
from matplotlib.gridspec import GridSpec
from lib.config import (CACHE, FIG_DIR, EMOS, EMO_COLORS, CH, set_nature_style, save, W2,
                        panel_letter, nice_heatmap, slim_colorbar)
from lib.regions import REGION_MAP
from lib.representations import vec_to_mat

def _order_channels():
    order, ticks = [], []
    for reg, chs in REGION_MAP.items():
        for c in chs:
            if c in CH: order.append(CH.index(c)); ticks.append(c)
    # append any leftover channels not in region map
    for i, c in enumerate(CH):
        if i not in order: order.append(i); ticks.append(c)
    return order, ticks

def _emotion_mats(agg, am):
    out = {}
    for e in EMOS:
        v = agg[(am["emotion"] == e).values].mean(0)
        out[e] = vec_to_mat(v, len(CH))
    return out

def _density_graph(M, dens=0.10):
    n = M.shape[0]; A = np.abs(M.copy()); np.fill_diagonal(A, 0)
    iu = np.triu_indices(n, 1); w = A[iu]; k = max(1, int(round(dens*len(w))))
    thr = np.sort(w)[-k]; B = (A >= thr).astype(float); np.fill_diagonal(B, 0)
    G = nx.from_numpy_array(np.maximum(B, B.T)); return G

def main():
    set_nature_style()
    if not (CACHE/"agg_spearman.npy").exists(): print("  [skip] agg_spearman missing"); return
    agg = np.load(CACHE/"agg_spearman.npy"); am = pd.read_csv(CACHE/"agg_meta_spearman.csv")
    mats = _emotion_mats(agg, am)
    order, ticks = _order_channels()

    # (a) nodal strength emotion x channel
    strength = np.vstack([np.abs(mats[e]).sum(1)[order] for e in EMOS])  # 9 x 30
    z = (strength - strength.mean(0)) / (strength.std(0) + 1e-9)

    # (b) modularity Q + (c) MST leaf fraction per emotion
    Q = {}; leaf = {}
    for e in EMOS:
        G = _density_graph(mats[e])
        comms = louvain_communities(G, seed=42)
        Q[e] = modularity(G, comms)
        D = nx.from_numpy_array(1 - np.abs(mats[e]))   # distance = 1-|corr|
        T = nx.minimum_spanning_tree(D)
        deg = np.array([d for _, d in T.degree()])
        leaf[e] = np.mean(deg == 1)

    fig = plt.figure(figsize=(W2, W2*0.42))
    gs = GridSpec(1, 3, figure=fig, wspace=0.5, width_ratios=[1.7, 1, 1], top=0.84, bottom=0.28)
    axA = fig.add_subplot(gs[0,0])
    im = nice_heatmap(axA, z, EMOS, ticks, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto", fontsize=4.4)
    axA.set_title("Nodal strength (emotion × channel, z)", pad=6); panel_letter(axA,"a",dx=-0.1)
    slim_colorbar(fig, im, axA, "z")
    axB = fig.add_subplot(gs[0,1])
    eo = sorted(EMOS, key=lambda e: Q[e], reverse=True)
    axB.barh(range(len(eo)), [Q[e] for e in eo], color=[EMO_COLORS[e] for e in eo], edgecolor="k", linewidth=0.3)
    axB.set_yticks(range(len(eo))); axB.set_yticklabels(eo, fontsize=5.4); axB.invert_yaxis()
    axB.set_xlabel("modularity Q"); axB.set_title("Modular organization", pad=6)
    for sp in ["top","right"]: axB.spines[sp].set_visible(False)
    panel_letter(axB,"b")
    axC = fig.add_subplot(gs[0,2])
    eo2 = sorted(EMOS, key=lambda e: leaf[e], reverse=True)
    axC.barh(range(len(eo2)), [leaf[e] for e in eo2], color=[EMO_COLORS[e] for e in eo2], edgecolor="k", linewidth=0.3)
    axC.set_yticks(range(len(eo2))); axC.set_yticklabels(eo2, fontsize=5.4); axC.invert_yaxis()
    axC.set_xlabel("MST leaf fraction"); axC.set_title("Tree integration", pad=6)
    for sp in ["top","right"]: axC.spines[sp].set_visible(False)
    panel_letter(axC,"c")
    save(fig, FIG_DIR/"part5"/"Figure_Part5_network_enrichment"); plt.close(fig)
    print("  Q range:", round(min(Q.values()),3), "-", round(max(Q.values()),3),
          "| leaf range:", round(min(leaf.values()),3), "-", round(max(leaf.values()),3))

if __name__ == "__main__":
    main()
