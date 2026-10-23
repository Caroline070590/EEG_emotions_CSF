"""
run_richclub_mtf_q30.py — Fig. S13 extended to MTF (q=30).

The original Fig. S13 (part6_advanced/run_richclub.py) is Spearman-only
(confirmed: it loads agg_spearman.npy exclusively). There was previously
no MTF version -- attempting one at the original q=13 would have been of
limited value, since a 13-node graph at 10% density has only ~8 edges and
therefore very few nodes with high enough degree to form a meaningful
"rich club" curve (max degree is mechanically capped very low).

Now that MTF uses q=30 (node-count-matched to Spearman's 30 channels),
this analysis is directly comparable in graph size to the Spearman
panel, so it is a natural and much more meaningful addition.

Method is IDENTICAL to the original Spearman script: reconstruct the
per-emotion averaged connectivity matrix, threshold to 10% density,
compute the rich-club coefficient phi(k) curve and degree assortativity,
per emotion.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, FIG_DIR, EMOS, EMO_COLORS, set_nature_style, save, W2, panel_letter
from lib.representations import vec_to_mat

TAGS = {"spearman": ("Spearman", 30), "mtf_q30": ("MTF (q=30)", 30)}


def thr_graph(M, dens=0.10):
    A = np.abs(M.copy()); np.fill_diagonal(A, 0); n = A.shape[0]; iu = np.triu_indices(n, 1)
    w = A[iu]; k = max(1, int(round(dens * len(w)))); thr = np.sort(w)[-k]
    return nx.from_numpy_array((np.maximum((A >= thr), 0)).astype(float))


def rich_club_raw(G):
    deg = dict(G.degree()); kmax = max(deg.values()) if deg else 0
    phi = {}
    for k in range(0, kmax):
        nodes = [u for u, d in deg.items() if d > k]
        if len(nodes) < 2:
            break
        sub = G.subgraph(nodes); E = sub.number_of_edges(); n = len(nodes)
        phi[k] = 2 * E / (n * (n - 1)) if n > 1 else np.nan
    return phi


def run_one(tag, n_nodes, axA, axB, title):
    f_agg = CACHE / f"agg_{tag}.npy"; f_meta = CACHE / f"agg_meta_{tag}.csv"
    if not (f_agg.exists() and f_meta.exists()):
        print(f"  [skip] {tag}: missing agg files"); return None
    agg = np.load(f_agg); am = pd.read_csv(f_meta)
    asso = {}
    for e in EMOS:
        M = vec_to_mat(agg[(am["emotion"] == e).values].mean(0), n_nodes)
        G = thr_graph(M)
        phi = rich_club_raw(G)
        if phi:
            axA.plot(list(phi.keys()), list(phi.values()), color=EMO_COLORS[e], lw=1.3, label=e, alpha=0.9)
        try:
            asso[e] = nx.degree_assortativity_coefficient(G)
        except Exception:
            asso[e] = np.nan
    axA.set_xlabel("degree threshold k"); axA.set_ylabel("rich-club φ(k)")
    axA.set_title(f"Rich-club organization\n({title})", pad=6, fontsize=7)
    for sp in ["top", "right"]:
        axA.spines[sp].set_visible(False)
    axA.legend(fontsize=4.2, ncol=2, loc="lower right", frameon=False)

    eo = sorted(EMOS, key=lambda e: asso[e], reverse=True)
    axB.barh(range(len(eo)), [asso[e] for e in eo], color=[EMO_COLORS[e] for e in eo], edgecolor="k", linewidth=0.3)
    axB.axvline(0, color="0.6", lw=0.6)
    axB.set_yticks(range(len(eo))); axB.set_yticklabels(eo, fontsize=5.2); axB.invert_yaxis()
    axB.set_xlabel("degree assortativity"); axB.set_title(f"Assortativity\n({title})", pad=6, fontsize=7)
    for sp in ["top", "right"]:
        axB.spines[sp].set_visible(False)
    return asso


def main():
    set_nature_style()
    fig = plt.figure(figsize=(W2, W2 * 0.72))
    gs = GridSpec(2, 2, figure=fig, wspace=0.4, hspace=0.55, top=0.90, bottom=0.10, width_ratios=[1.3, 1])

    axA1 = fig.add_subplot(gs[0, 0]); axB1 = fig.add_subplot(gs[0, 1])
    asso_sp = run_one("spearman", 30, axA1, axB1, "Spearman")
    panel_letter(axA1, "a", dx=-0.16); panel_letter(axB1, "b")

    axA2 = fig.add_subplot(gs[1, 0]); axB2 = fig.add_subplot(gs[1, 1])
    asso_mtf = run_one("mtf_q30", 30, axA2, axB2, "MTF, q=30 [new]")
    panel_letter(axA2, "c", dx=-0.16); panel_letter(axB2, "d")

    fig.suptitle("Fig. S13 extended: rich-club organization and assortativity, Spearman + MTF (q=30)",
                 fontsize=7.8, y=0.975)
    save(fig, FIG_DIR / "part6" / "Figure_Part6_richclub_assortativity_mtf_q30")
    plt.close(fig)

    if asso_sp and asso_mtf:
        print("  Spearman assortativity range:", round(min(asso_sp.values()), 3), "to", round(max(asso_sp.values()), 3))
        print("  MTF(q=30) assortativity range:", round(min(asso_mtf.values()), 3), "to", round(max(asso_mtf.values()), 3))
        tab_dir = Path(CACHE).parent / "supplementary_tables"; tab_dir.mkdir(exist_ok=True, parents=True)
        rows = [dict(emotion=e, assortativity_spearman=round(asso_sp[e], 3), assortativity_mtf_q30=round(asso_mtf[e], 3))
                for e in EMOS]
        pd.DataFrame(rows).to_csv(tab_dir / "TableS_richclub_assortativity_mtf_q30.csv", index=False)
    print("[Fig S13 extended] saved Figure_Part6_richclub_assortativity_mtf_q30.pdf/.png")


if __name__ == "__main__":
    main()
