"""
run_embedding_enrichment_q30.py — Fig. S6 panel c (RDM alignment +
participation ratio) regenerated with q=30 (node-count-matched) MTF/QG.

IMPORTANT: panels a (LOSO confusion) and b (RSA dendrogram) are Spearman-
ONLY in the original figure (confirmed by reading run_embedding_enrichment.py:
they load agg_spearman.npy / rdm_spearman.csv exclusively) -- they do NOT
depend on MTF/QG at all, so they are reproduced UNCHANGED here for context.
Only panel c actually combines all three representations and needed
rebuilding at q=30.

Also note: Fig. S7 (RSA reliability split-half + full-sample RDM) is
ALSO Spearman-only in its entirety (part6_advanced/run_splithalf.py uses
only agg_spearman.npy) -- there is nothing to regenerate there for q=30;
it is already correct as-is and unaffected by this fix.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import (CACHE, FIG_DIR, EMOS, EI, EMO_COLORS, set_nature_style, save, W2,
                        panel_letter, nice_heatmap, slim_colorbar)
from lib.embedding import emotion_rdm

METHODS_Q30 = {"spearman": "Spearman", "mtf_q30": "MTF (q=30)", "qg_q30": "QG (q=30)"}


def _part_ratio(X):
    C = np.cov(X.T); ev = np.linalg.eigvalsh(C); ev = ev[ev > 0]
    return (ev.sum() ** 2) / (np.sum(ev ** 2)) if len(ev) else np.nan


def _ensure_rdm(tag):
    """Build rdm_{tag}.csv from the cached agg_{tag}.npy if not already
    present (mirrors part3_embedding_topology/compute_features.py's RDM
    step, applied here to the q=30 aggregates built earlier for Fig. S5)."""
    f = CACHE / f"rdm_{tag}.csv"
    if f.exists():
        return
    agg = np.load(CACHE / f"agg_{tag}.npy"); am = pd.read_csv(CACHE / f"agg_meta_{tag}.csv")
    emo_idx = am["emotion"].map(EI).values
    rdm = emotion_rdm(agg, emo_idx)
    rdm.to_csv(f)
    print(f"  [built] rdm_{tag}.csv from cached agg_{tag}.npy")


def main():
    set_nature_style()
    for tag in ["mtf_q30", "qg_q30"]:
        _ensure_rdm(tag)

    if not (CACHE / "agg_spearman.npy").exists():
        print("  [skip] agg_spearman.npy missing"); return
    agg = np.load(CACHE / "agg_spearman.npy"); am = pd.read_csv(CACHE / "agg_meta_spearman.csv")
    fig = plt.figure(figsize=(W2, W2 * 0.42))
    gs = GridSpec(1, 3, figure=fig, wspace=0.55, top=0.84, bottom=0.26)

    # (a) confusion matrix -- Spearman-only, UNCHANGED from original Fig S6
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    yp = cross_val_predict(clf, agg, am["emotion"].values, groups=am["subject"].values, cv=LeaveOneGroupOut())
    Cm = confusion_matrix(am["emotion"].values, yp, labels=EMOS).astype(float)
    Cm /= Cm.sum(1, keepdims=True)
    axA = fig.add_subplot(gs[0, 0])
    im = nice_heatmap(axA, Cm, EMOS, EMOS, cmap="magma", vmin=0, vmax=0.5, aspect="equal", fontsize=4.8)
    axA.set_title("LOSO confusion (Spearman)\n[unchanged]", pad=6, fontsize=6.8)
    axA.set_ylabel("true"); axA.set_xlabel("predicted", fontsize=6)
    panel_letter(axA, "a", dx=-0.16); slim_colorbar(fig, im, axA, "P(pred|true)")

    # (b) dendrogram -- Spearman-only, UNCHANGED from original Fig S6
    rdm_sp = pd.read_csv(CACHE / "rdm_spearman.csv", index_col=0).reindex(index=EMOS, columns=EMOS).values
    rdm_sp = (rdm_sp + rdm_sp.T) / 2; np.fill_diagonal(rdm_sp, 0)
    Z = linkage(squareform(rdm_sp, checks=False), method="average")
    axB = fig.add_subplot(gs[0, 1])
    dn = dendrogram(Z, labels=EMOS, ax=axB, color_threshold=0, above_threshold_color="0.4", leaf_font_size=5.4)
    for lbl in axB.get_xmajorticklabels():
        lbl.set_color(EMO_COLORS.get(lbl.get_text(), "k")); lbl.set_rotation(90)
    axB.set_title("Emotion taxonomy (RSA)\n[unchanged]", pad=6, fontsize=6.8)
    axB.set_ylabel("dissimilarity", fontsize=6)
    for sp in ["top", "right"]:
        axB.spines[sp].set_visible(False)
    panel_letter(axB, "b", dx=-0.12)

    # (c) RDM alignment across methods + dimensionality -- q=30 MTF/QG
    tags = [t for t in METHODS_Q30 if (CACHE / f"rdm_{t}.csv").exists()]
    iu = np.triu_indices(len(EMOS), 1)
    vecs = {}
    for t in tags:
        r = pd.read_csv(CACHE / f"rdm_{t}.csv", index_col=0).reindex(index=EMOS, columns=EMOS).values
        vecs[t] = r[iu]
    A = np.eye(len(tags))
    for i, a in enumerate(tags):
        for j, b in enumerate(tags):
            if i < j:
                A[i, j] = A[j, i] = spearmanr(vecs[a], vecs[b]).correlation
    axC = fig.add_subplot(gs[0, 2])
    masked = np.where(np.tril(np.ones_like(A)) > 0, A, np.nan)
    im2 = nice_heatmap(axC, masked, [METHODS_Q30[t] for t in tags], [METHODS_Q30[t] for t in tags],
                       cmap="PuOr_r", vmin=-1, vmax=1, annot=True, fmt="{:.2f}", aspect="equal",
                       fontsize=5.6, annot_size=5.4)
    axC.set_title("RDM alignment (ρ), q=30", pad=6); panel_letter(axC, "c", dx=-0.14)
    slim_colorbar(fig, im2, axC, "ρ")
    pr = {t: _part_ratio(np.load(CACHE / f"agg_{t}.npy")) for t in tags}
    axC.set_xlabel("PR dim: " + ", ".join(f"{METHODS_Q30[t]} {pr[t]:.1f}" for t in tags), fontsize=4.6)

    fig.suptitle("Fig. S6 (q=30 robustness version): panels a-b unchanged (Spearman-only); panel c uses MTF/QG at q=30",
                 fontsize=6.8, y=0.98)
    save(fig, FIG_DIR / "part5" / "Figure_Part5_embedding_enrichment_q30"); plt.close(fig)
    print("RDM alignment (q=30):", {f"{a}-{b}": round(A[i, j], 2) for i, a in enumerate(tags) for j, b in enumerate(tags) if i < j})
    print("PR dim (q=30):", {METHODS_Q30[t]: round(pr[t], 1) for t in tags})


if __name__ == "__main__":
    main()
