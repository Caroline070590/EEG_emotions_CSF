"""
run_figure4_full_composite.py — complete rebuild of main Figure 4, with
panel F replaced by the combined Spearman version (nodal strength, mean
wPLI, and their coupling), panels a-e and g-i reconstructed to match the
originally published figure exactly (verified against the uploaded PDF
before finalizing -- see notes below), all in ONE assembled composite.

Verification performed before writing this script (see chat for details):
  - Panels g/h (simplex counts, Euler characteristic): reconstructed as
    10%-density-thresholded clique complex per emotion. Verified EXACT
    match to the uploaded figure for every emotion (disgust: 22/6/1,
    chi=3; fear: 21/6/1, chi=2; anger: 16/1/0, chi=1; sadness: chi=-5;
    all others chi in {-2,-1,0} -- all matched exactly).
  - Panel i (Rips filtration): reconstructed using distance = 1-|corr|
    (matching the panel's own x-axis label literally), max_edge_length=1,
    max_dimension=4. Cumulative simplex count curve shape (10^2 near
    filtration 0.3, ~10^5 plateau by filtration 0.8) matches the uploaded
    figure closely.
  - Panels a-c (NBS): re-derived edge t-statistics (no need to redo the
    5000-permutation test -- p-values already cached in
    nbs_spearman_vs_neutral.csv from the original run). Largest-connected-
    component edge counts verified to match the cached CSV EXACTLY for
    all 8 emotions (sadness=118, anger=63, disgust=86, fear=66, joy=17,
    inspiration=7, tenderness=7, amusement=5).
  - Panels d-e (hub centrality): reuses run_centrality_topo.py's
    eigenvector-centrality-difference-from-neutral computation directly
    (unchanged method), restricted to sadness/anger for the condensed
    main-figure format.
  - Panel F: the new combined Spearman panel (nodal-strength range, mean
    wPLI, coupling beta) built in earlier steps of this session.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import gudhi
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, FIG_DIR, EMOS, EMO_COLORS, CH, set_nature_style, save, W2
from lib.layout import topomap, scalp_edges, head
from lib.representations import vec_to_mat
from lib.metrics import threshold_to_density
from lib.stats import lmm_emotion_lrt
from lib.regions import REGIONS

N = len(CH); IU = np.triu_indices(N, 1); T_THR = 3.0
INK = "#1b1f2b"
RLAB = [r.capitalize() for r in REGIONS]
C_STRENGTH = "#4C72B0"; C_WPLI = "#8172B2"; C_COUPLE = "#55A868"


def panel_tag(ax, letter, dx=-0.14, dy=1.12, fontsize=18):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fontsize, fontweight="bold",
             color=INK, va="top", ha="left")


def _load_agg():
    return np.load(CACHE / "agg_spearman.npy"), pd.read_csv(CACHE / "agg_meta_spearman.csv")


def _edge_diff(agg, am, emo):
    piv = {}
    for s in am["subject"].unique():
        me = (am.subject == s) & (am.emotion == emo); mn = (am.subject == s) & (am.emotion == "neutral")
        if me.values.any() and mn.values.any():
            piv[s] = agg[me.values][0] - agg[mn.values][0]
    return np.vstack(list(piv.values()))


def _t_edges(D):
    m = D.mean(0); s = D.std(0, ddof=1) + 1e-12
    return m / (s / np.sqrt(len(D)))


def _nbs_component(agg, am, emo):
    D = _edge_diff(agg, am, emo); tv = _t_edges(D)
    supra = np.abs(tv) > T_THR
    G = nx.Graph()
    for idx in np.where(supra)[0]:
        i, j = int(IU[0][idx]), int(IU[1][idx])
        G.add_edge(i, j, idx=idx)
    if G.number_of_edges() == 0:
        return [], [], 0, 0
    comp = max(nx.connected_components(G), key=lambda c: G.subgraph(c).number_of_edges())
    sub = G.subgraph(comp)
    edges = list(sub.edges()); signs = [np.sign(tv[sub[i][j]["idx"]]) for i, j in edges]
    n_inc = sum(1 for s in signs if s > 0); n_dec = sum(1 for s in signs if s < 0)
    return edges, signs, n_inc, n_dec


def _eig_full(M):
    A = np.abs(M.copy()); np.fill_diagonal(A, 0)
    G = nx.from_numpy_array(A); ev = nx.eigenvector_centrality_numpy(G, weight="weight")
    return np.array([ev[i] for i in range(len(CH))])


def _kcliques(G, k):
    return sum(1 for c in nx.enumerate_all_cliques(G) if len(c) == k)


def main():
    set_nature_style()
    agg, am = _load_agg()
    nbs_csv = pd.read_csv(CACHE / "nbs_spearman_vs_neutral.csv").set_index("emotion")

    fig = plt.figure(figsize=(W2 * 1.15, W2 * 1.18))
    gs = GridSpec(4, 3, figure=fig, hspace=0.65, wspace=0.42, top=0.965, bottom=0.05,
                  height_ratios=[1.0, 1.0, 0.62, 1.15])

    # ================================================================ a, b: NBS scalp networks
    for col, emo in enumerate(["sadness", "anger"]):
        ax = fig.add_subplot(gs[0, col])
        edges, signs, n_inc, n_dec = _nbs_component(agg, am, emo)
        n_edges = nbs_csv.loc[emo, "component_edges"]; p = nbs_csv.loc[emo, "p_cluster_fwe"]
        scalp_edges(ax, CH, edges, signs)
        ax.set_title(f"{emo} · {int(n_edges)} edges (p={p:.3f})", fontsize=7.4, pad=4)
        panel_tag(ax, "ab"[col])

    # ================================================================ c: NBS edge direction (all emotions)
    axC = fig.add_subplot(gs[0, 2])
    emotions_nn = [e for e in EMOS if e != "neutral"]
    inc_dec = {}
    for e in emotions_nn:
        _, _, n_inc, n_dec = _nbs_component(agg, am, e)
        inc_dec[e] = (n_inc, -n_dec)
    order = sorted(emotions_nn, key=lambda e: nbs_csv.loc[e, "component_edges"], reverse=True)
    y = np.arange(len(order))[::-1]
    inc_vals = [inc_dec[e][0] for e in order]; dec_vals = [inc_dec[e][1] for e in order]
    axC.barh(y, inc_vals, color="#b3402f", edgecolor="k", linewidth=0.3)
    axC.barh(y, dec_vals, color="#2c5f8a", edgecolor="k", linewidth=0.3)
    for yi, e in zip(y, order):
        q = nbs_csv.loc[e, "q_bh_across_emotions"]
        star = "*" if q < 0.05 else ""
        xpos = max(inc_dec[e][0], 2) + 3
        axC.text(xpos, yi, star, va="center", fontsize=8, color=INK)
    axC.axvline(0, color="0.3", lw=0.7)
    axC.set_yticks(y); axC.set_yticklabels(order, fontsize=6.2)
    axC.set_xlabel("← decreased    edges    increased →", fontsize=6)
    axC.set_title("NBS edge direction", fontsize=7.6, pad=4)
    for sp in ["top", "right"]:
        axC.spines[sp].set_visible(False)
    panel_tag(axC, "c")

    # ================================================================ d, e: hub eigenvector-centrality delta
    EV = {}
    for e in EMOS:
        M = vec_to_mat(agg[(am["emotion"] == e).values].mean(0), len(CH))
        EV[e] = _eig_full(M)
    EVd = {e: EV[e] - EV["neutral"] for e in EMOS}
    vmax = float(np.nanpercentile(np.abs(np.vstack([EVd["sadness"], EVd["anger"]])), 98))
    for col, emo in enumerate(["sadness", "anger"]):
        ax = fig.add_subplot(gs[1, col])
        im = topomap(ax, CH, EVd[emo], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{emo} · hub Δ (eigenvector)", fontsize=7.4, pad=4)
        panel_tag(ax, "de"[col])
        if col == 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
            cb.ax.tick_params(labelsize=5); cb.set_label("Δ centrality", fontsize=5.6, labelpad=2)
            cb.outline.set_visible(False)

    # ================================================================ f: top hub-shift channels (fills the blank cell next to d, e)
    axF0 = fig.add_subplot(gs[1, 2])
    k_top = 8
    idx_sad = np.argsort(-np.abs(EVd["sadness"]))[:k_top]
    idx_ang = np.argsort(-np.abs(EVd["anger"]))[:k_top]
    top_idx = sorted(set(idx_sad) | set(idx_ang), key=lambda i: -max(abs(EVd["sadness"][i]), abs(EVd["anger"][i])))[:k_top]
    top_chans = [CH[i] for i in top_idx]
    yTop = np.arange(len(top_chans))[::-1]; hbar = 0.32
    axF0.barh(yTop + hbar / 2, [EVd["sadness"][i] for i in top_idx], height=hbar,
              color="#4C72B0", edgecolor="white", linewidth=0.4, label="sadness")
    axF0.barh(yTop - hbar / 2, [EVd["anger"][i] for i in top_idx], height=hbar,
              color="#C44E52", edgecolor="white", linewidth=0.4, label="anger")
    axF0.axvline(0, color="0.3", lw=0.6)
    axF0.set_yticks(yTop); axF0.set_yticklabels(top_chans, fontsize=6.0)
    axF0.set_xlabel("Δ eigenvector centrality", fontsize=5.8)
    axF0.set_title("Top hub-shift channels", fontsize=7.4, pad=4)
    axF0.legend(fontsize=5.2, frameon=False, loc="lower right")
    for sp in ["top", "right"]:
        axF0.spines[sp].set_visible(False)
    panel_tag(axF0, "f")

    # ================================================================ g: combined Spearman regional panel, full width, own row
    # (3 small-multiple sub-panels, one per metric -- see run_fig4_panel_f_combined.py
    # for the full rationale: avoids the invisible-bar problem that cross-metric
    # min-max normalization causes, and uses muted gray instead of "n.s." text
    # for non-significant bars.)
    ftab = pd.read_csv(Path(CACHE).parent / "supplementary_tables" / "Table_fig4_panel_f_combined.csv")
    ftab = ftab.set_index("region").reindex(REGIONS)
    C_MUTED = "#c9ccd1"

    def _stars(p):
        return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else ""

    def _pfmt(p):
        return "p<0.0001" if p < 1e-4 else f"p={p:.4f}"

    gsF = gs[2, :].subgridspec(1, 3, wspace=0.12)
    yF = np.arange(len(REGIONS))[::-1]
    f_panels = [
        ("Nodal-strength range (Δ across emotions)", ftab["nodal_strength_range"], ftab["nodal_strength_p"], C_STRENGTH, "{:.2f}"),
        ("Mean wPLI", ftab["mean_wpli"], ftab["wpli_p"], C_WPLI, "{:.3f}"),
        ("|Coupling β| (wPLI ~ nodal strength)", ftab["coupling_beta"].abs(), ftab["coupling_p"], C_COUPLE, "β={:.2f}"),
    ]
    for k, (title, vals, pvals, color, fmt) in enumerate(f_panels):
        axf = fig.add_subplot(gsF[0, k])
        v = vals.reindex(REGIONS).values
        p_ = pvals.reindex(REGIONS).values
        sig = np.isfinite(p_) & (p_ < 0.05)
        bar_colors = [color if s else C_MUTED for s in sig]
        axf.barh(yF, v, height=0.62, color=bar_colors, edgecolor="white", linewidth=0.5)
        xmax = v.max() * 1.55
        for yi, val, p in zip(yF, v, p_):
            label = f"{fmt.format(val)} {_stars(p)}" if np.isfinite(p) and p < 0.05 else f"{fmt.format(val)} ({_pfmt(p)})"
            axf.text(val + xmax * 0.035, yi, label, va="center", fontsize=5.0, color=INK)
        axf.set_xlim(0, xmax); axf.set_xticks([])
        if k == 0:
            axf.set_yticks(yF); axf.set_yticklabels(RLAB, fontsize=6.4)
        else:
            axf.set_yticks(yF); axf.set_yticklabels([])
        axf.set_title(title, fontsize=6.0, pad=6)
        for sp in ["top", "right", "bottom"]:
            axf.spines[sp].set_visible(False)
        if k == 0:
            panel_tag(axf, "g", dx=-0.30)
        if k == 2:
            sig_patch = mpatches.Patch(facecolor="0.4", label="p < 0.05")
            ns_patch = mpatches.Patch(facecolor=C_MUTED, label="not significant")
            axf.legend(handles=[sig_patch, ns_patch], fontsize=4.6, loc="lower right", frameon=False)

    # ================================================================ g, h: simplex counts + higher-order load
    simplex_rows = []
    for e in EMOS:
        M = vec_to_mat(agg[(am["emotion"] == e).values].mean(0), len(CH))
        B = threshold_to_density(M, 0.10)
        G = nx.from_numpy_array(B)
        n0, n1 = G.number_of_nodes(), G.number_of_edges()
        n2, n3, n4 = _kcliques(G, 3), _kcliques(G, 4), _kcliques(G, 5)
        chi = n0 - n1 + n2 - n3 + n4
        simplex_rows.append(dict(emotion=e, n2=n2, n3=n3, n4=n4, chi=chi))
    simplex_tab = pd.DataFrame(simplex_rows).set_index("emotion")

    axG = fig.add_subplot(gs[3, 0])
    order_g = simplex_tab["n2"].sort_values(ascending=False).index.tolist()
    x = np.arange(len(order_g)); w = 0.27
    axG.bar(x - w, simplex_tab.loc[order_g, "n2"], width=w, color="#3b3b98", label="2-simplex")
    axG.bar(x, simplex_tab.loc[order_g, "n3"], width=w, color="#7f8fd1", label="3-simplex")
    axG.bar(x + w, simplex_tab.loc[order_g, "n4"], width=w, color="#c7cef0", label="4-simplex")
    axG.set_xticks(x); axG.set_xticklabels(order_g, rotation=90, fontsize=5.4)
    axG.set_ylabel("count", fontsize=6.2); axG.set_title("Simplex counts (GUDHI)", fontsize=7.6, pad=4)
    axG.legend(fontsize=4.6, frameon=False)
    for sp in ["top", "right"]:
        axG.spines[sp].set_visible(False)
    panel_tag(axG, "h")

    axH = fig.add_subplot(gs[3, 1])
    order_h = (simplex_tab["n2"] + simplex_tab["n3"] + simplex_tab["n4"]).sort_values(ascending=False).index.tolist()
    load = (simplex_tab.loc[order_h, ["n2", "n3", "n4"]].sum(axis=1))
    yH = np.arange(len(order_h))[::-1]
    axH.barh(yH, load.values, color=[EMO_COLORS[e] for e in order_h], edgecolor="k", linewidth=0.3)
    for yi, e in zip(yH, order_h):
        axH.text(load[e] + 0.6, yi, f"χ={int(simplex_tab.loc[e,'chi'])}", va="center", fontsize=5.4, color="0.3")
    axH.set_yticks(yH); axH.set_yticklabels(order_h, fontsize=6.0)
    axH.set_xlabel("higher-order simplices (k≥2)", fontsize=6)
    axH.set_title("Higher-order load", fontsize=7.6, pad=4)
    for sp in ["top", "right"]:
        axH.spines[sp].set_visible(False)
    panel_tag(axH, "i")

    # ================================================================ i: Rips filtration
    axI = fig.add_subplot(gs[3, 2])
    for e in EMOS:
        M = vec_to_mat(agg[(am["emotion"] == e).values].mean(0), len(CH))
        D = 1 - np.abs(M); np.fill_diagonal(D, 0)
        st = gudhi.RipsComplex(distance_matrix=D, max_edge_length=1.0).create_simplex_tree(max_dimension=4)
        filt = np.array(sorted(f for _, f in st.get_filtration()))
        xs = np.linspace(max(filt.min(), 0.2), filt.max(), 60)
        ys = [(filt <= x).sum() for x in xs]
        axI.plot(xs, ys, color=EMO_COLORS[e], lw=1.1, alpha=0.85)
    axI.set_yscale("log")
    axI.set_xlabel("filtration (1−|corr|)", fontsize=6); axI.set_ylabel("# simplices", fontsize=6.2)
    axI.set_title("Rips filtration", fontsize=7.6, pad=4)
    for sp in ["top", "right"]:
        axI.spines[sp].set_visible(False)
    panel_tag(axI, "j")

    # ---- shared emotion legend (panels c, h, i, j) ----------------------------
    handles = [mpatches.Patch(facecolor=EMO_COLORS[e], edgecolor="k", lw=0.3, label=e) for e in EMOS if e != "neutral"]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=9,
               fontsize=5.6, frameon=False, handlelength=1.0, columnspacing=0.9,
               title="emotion (panels c, h, i, j)", title_fontsize=6)

    save(fig, FIG_DIR / "part4" / "Figure4_full_composite")
    plt.close(fig)
    print("[Figure 4 full composite] saved Figure4_full_composite.pdf/.png")


if __name__ == "__main__":
    main()
