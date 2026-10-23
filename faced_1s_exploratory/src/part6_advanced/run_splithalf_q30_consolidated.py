"""
run_splithalf_q30_consolidated.py — single consolidated figure combining:

  Row 1 (a-c): RSA split-half reliability histograms, Spearman / MTF(q30) / QG(q30)
  Row 2 (d-f): Full-sample RDM heatmaps, Spearman / MTF(q30) / QG(q30)
  Row 3 (g-i): g = RDM alignment across methods (+ participation ratio dim);
               h = emotion taxonomy dendrogram (Spearman);
               i = emotion taxonomy dendrogram (MTF, q=30) -- newly plotted,
               did not previously exist anywhere in the pipeline.

Everything reuses cached data already built in this session (agg_*.npy,
rdm_*.csv); the only genuinely new computation is the MTF (q=30) dendrogram
(panel i), built the same way as the existing Spearman dendrogram.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import (CACHE, FIG_DIR, EMOS, EMO_COLORS, set_nature_style, save, W2,
                        panel_letter, nice_heatmap, slim_colorbar)

TAGS = {"spearman": "Spearman", "mtf_q30": "MTF (q=30)", "qg_q30": "QG (q=30)"}
N_SPLITS = 300
SEED = 0


def rdm_from_agg(agg, am, subs):
    cents = []
    for e in EMOS:
        m = (am["emotion"] == e).values & am["subject"].isin(subs).values
        cents.append(agg[m].mean(0))
    C = np.vstack(cents)
    return squareform(pdist(C, metric="correlation"))


def splithalf(agg, am, n_splits=N_SPLITS, seed=SEED):
    subs = np.array(sorted(am["subject"].unique()))
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(len(EMOS), 1)
    rs = []
    for _ in range(n_splits):
        rng.shuffle(subs); h = len(subs) // 2
        A = rdm_from_agg(agg, am, subs[:h]); B = rdm_from_agg(agg, am, subs[h:])
        r = spearmanr(A[iu], B[iu]).correlation
        if np.isfinite(r):
            rs.append(r)
    return np.array(rs)


def _part_ratio(X):
    C = np.cov(X.T); ev = np.linalg.eigvalsh(C); ev = ev[ev > 0]
    return (ev.sum() ** 2) / (np.sum(ev ** 2)) if len(ev) else np.nan


def main():
    set_nature_style()
    tab_dir = Path(CACHE).parent / "supplementary_tables"; tab_dir.mkdir(exist_ok=True, parents=True)

    # ---- load everything -------------------------------------------------
    data = {}
    for tag in TAGS:
        f_agg = CACHE / f"agg_{tag}.npy"; f_meta = CACHE / f"agg_meta_{tag}.csv"
        if f_agg.exists() and f_meta.exists():
            data[tag] = (np.load(f_agg), pd.read_csv(f_meta))
    if len(data) < 3:
        print(f"  [warn] only found: {list(data)}"); 

    results = {}
    for tag, (agg, am) in data.items():
        rs = splithalf(agg, am)
        full = rdm_from_agg(agg, am, np.array(sorted(am["subject"].unique())))
        results[tag] = dict(rs=rs, full=full, median=float(np.median(rs)))
        print(f"  {TAGS[tag]}: split-half median rho={results[tag]['median']:.3f}")

    # cached rdm_*.csv (from Fig S6/S1 work) -- used for panels g, h, i
    rdms = {}
    for tag in TAGS:
        f = CACHE / f"rdm_{tag}.csv"
        if f.exists():
            rdms[tag] = pd.read_csv(f, index_col=0).reindex(index=EMOS, columns=EMOS).values

    # ---- figure: 3 rows x 3 cols ------------------------------------------
    fig = plt.figure(figsize=(W2, W2 * 0.95))
    gs = GridSpec(3, 3, figure=fig, wspace=0.5, hspace=0.75, top=0.94, bottom=0.08)
    letters = iter("abcdefghi")
    tags = list(results.keys())

    # row 1: split-half histograms
    for k, tag in enumerate(tags):
        v = results[tag]
        ax = fig.add_subplot(gs[0, k])
        ax.hist(v["rs"], bins=24, color="#4c72b0", edgecolor="white", linewidth=0.3)
        ax.axvline(v["median"], color="#c0392b", lw=1.2, ls="--")
        ax.set_xlabel("split-half RDM correlation (ρ)", fontsize=6); ax.set_ylabel("count", fontsize=6)
        ax.set_title(f"{TAGS[tag]}\nRSA reliability (median ρ = {v['median']:.2f})", fontsize=6.8, pad=6)
        ax.tick_params(labelsize=5.4)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        panel_letter(ax, next(letters), dx=-0.22)

    # row 2: full-sample RDMs
    im_last = None
    for k, tag in enumerate(tags):
        ax = fig.add_subplot(gs[1, k])
        im_last = nice_heatmap(ax, results[tag]["full"], EMOS, EMOS, cmap="magma", aspect="equal", fontsize=4.6)
        ax.set_title(f"Full-sample RDM ({TAGS[tag]})", fontsize=6.2, pad=6)
        panel_letter(ax, next(letters), dx=-0.20)
        if k == len(tags) - 1:
            slim_colorbar(fig, im_last, ax, "1−r")

    # row 3, panel g: RDM alignment across methods + PR dim
    axG = fig.add_subplot(gs[2, 0])
    avail = [t for t in TAGS if t in rdms]
    iu = np.triu_indices(len(EMOS), 1)
    vecs = {t: rdms[t][iu] for t in avail}
    A = np.eye(len(avail))
    for i, a in enumerate(avail):
        for j, b in enumerate(avail):
            if i < j:
                A[i, j] = A[j, i] = spearmanr(vecs[a], vecs[b]).correlation
    masked = np.where(np.tril(np.ones_like(A)) > 0, A, np.nan)
    imG = nice_heatmap(axG, masked, [TAGS[t] for t in avail], [TAGS[t] for t in avail],
                       cmap="PuOr_r", vmin=-1, vmax=1, annot=True, fmt="{:.2f}", aspect="equal",
                       fontsize=5.6, annot_size=5.4)
    axG.set_title("RDM alignment (ρ), q=30", pad=6); panel_letter(axG, "g", dx=-0.20)
    slim_colorbar(fig, imG, axG, "ρ")
    pr = {t: _part_ratio(data[t][0]) for t in avail if t in data}
    axG.set_xlabel("PR dim: " + ", ".join(f"{TAGS[t]} {pr[t]:.1f}" for t in avail if t in pr), fontsize=4.6)

    # row 3, panel h: emotion taxonomy dendrogram, Spearman
    axH = fig.add_subplot(gs[2, 1])
    if "spearman" in rdms:
        rdm_sp = rdms["spearman"]; rdm_sp = (rdm_sp + rdm_sp.T) / 2; np.fill_diagonal(rdm_sp, 0)
        Z = linkage(squareform(rdm_sp, checks=False), method="average")
        dendrogram(Z, labels=EMOS, ax=axH, color_threshold=0, above_threshold_color="0.4", leaf_font_size=5.4)
        for lbl in axH.get_xmajorticklabels():
            lbl.set_color(EMO_COLORS.get(lbl.get_text(), "k")); lbl.set_rotation(90)
    axH.set_title("Emotion taxonomy (RSA)\nSpearman", fontsize=6.8, pad=6)
    axH.set_ylabel("dissimilarity", fontsize=6)
    for sp in ["top", "right"]:
        axH.spines[sp].set_visible(False)
    panel_letter(axH, "h", dx=-0.16)

    # row 3, panel i: emotion taxonomy dendrogram, MTF (q=30) -- NEW
    axI = fig.add_subplot(gs[2, 2])
    if "mtf_q30" in rdms:
        rdm_mtf = rdms["mtf_q30"]; rdm_mtf = (rdm_mtf + rdm_mtf.T) / 2; np.fill_diagonal(rdm_mtf, 0)
        Z2 = linkage(squareform(rdm_mtf, checks=False), method="average")
        dendrogram(Z2, labels=EMOS, ax=axI, color_threshold=0, above_threshold_color="0.4", leaf_font_size=5.4)
        for lbl in axI.get_xmajorticklabels():
            lbl.set_color(EMO_COLORS.get(lbl.get_text(), "k")); lbl.set_rotation(90)
    axI.set_title("Emotion taxonomy (RSA)\nMTF (q=30) [new]", fontsize=6.8, pad=6)
    axI.set_ylabel("dissimilarity", fontsize=6)
    for sp in ["top", "right"]:
        axI.spines[sp].set_visible(False)
    panel_letter(axI, "i", dx=-0.16)

    fig.suptitle("Fig. S7 consolidated (q=30): RSA reliability, RDMs, cross-representation alignment, and emotion taxonomy",
                 fontsize=7.8, y=0.975)
    save(fig, FIG_DIR / "part6" / "Figure_Part6_rsa_reliability_q30_consolidated"); plt.close(fig)
    print("[consolidated Fig S7] saved Figure_Part6_rsa_reliability_q30_consolidated.pdf/.png")

    # save the alignment numbers too
    align_rows = [dict(pair=f"{TAGS[a]}-{TAGS[b]}", rho=round(A[i, j], 3))
                  for i, a in enumerate(avail) for j, b in enumerate(avail) if i < j]
    pd.DataFrame(align_rows).to_csv(tab_dir / "TableS_RDM_alignment_q30_consolidated.csv", index=False)
    print(pd.DataFrame(align_rows).to_string(index=False))


if __name__ == "__main__":
    main()
