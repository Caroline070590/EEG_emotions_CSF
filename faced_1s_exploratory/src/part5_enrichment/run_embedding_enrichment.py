"""
run_embedding_enrichment.py — deeper embedding / representational analysis.
a  LOSO decoding confusion matrix (Spearman): which emotions are confused.
b  emotion dendrogram (hierarchical clustering of the RSA).
c  representational alignment across methods (Spearman correlation of the three
   RDMs) + representation dimensionality (participation ratio) per method.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
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
from lib.config import (CACHE, FIG_DIR, EMOS, EMO_COLORS, set_nature_style, save, W2,
                        panel_letter, nice_heatmap, slim_colorbar)

METHODS = {"spearman":"Spearman","mtf":"MTF","qg":"QG"}

def _part_ratio(X):
    C = np.cov(X.T); ev = np.linalg.eigvalsh(C); ev = ev[ev > 0]
    return (ev.sum()**2) / (np.sum(ev**2)) if len(ev) else np.nan

def main():
    set_nature_style()
    if not (CACHE/"agg_spearman.npy").exists(): print("  [skip] agg missing"); return
    agg = np.load(CACHE/"agg_spearman.npy"); am = pd.read_csv(CACHE/"agg_meta_spearman.csv")
    fig = plt.figure(figsize=(W2, W2*0.42))
    gs = GridSpec(1, 3, figure=fig, wspace=0.55, top=0.84, bottom=0.26)

    # (a) confusion matrix
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    yp = cross_val_predict(clf, agg, am["emotion"].values, groups=am["subject"].values, cv=LeaveOneGroupOut())
    Cm = confusion_matrix(am["emotion"].values, yp, labels=EMOS).astype(float)
    Cm /= Cm.sum(1, keepdims=True)
    axA = fig.add_subplot(gs[0,0])
    im = nice_heatmap(axA, Cm, EMOS, EMOS, cmap="magma", vmin=0, vmax=0.5, aspect="equal", fontsize=4.8)
    axA.set_title("LOSO confusion (Spearman)", pad=6); axA.set_ylabel("true"); axA.set_xlabel("predicted", fontsize=6)
    panel_letter(axA,"a",dx=-0.16); slim_colorbar(fig, im, axA, "P(pred|true)")

    # (b) dendrogram from RSA
    rdm = pd.read_csv(CACHE/"rdm_spearman.csv", index_col=0).reindex(index=EMOS, columns=EMOS).values
    rdm = (rdm + rdm.T)/2; np.fill_diagonal(rdm, 0)
    Z = linkage(squareform(rdm, checks=False), method="average")
    axB = fig.add_subplot(gs[0,1])
    dn = dendrogram(Z, labels=EMOS, ax=axB, color_threshold=0, above_threshold_color="0.4",
                    leaf_font_size=5.4)
    for lbl in axB.get_xmajorticklabels():
        lbl.set_color(EMO_COLORS.get(lbl.get_text(), "k")); lbl.set_rotation(90)
    axB.set_title("Emotion taxonomy (RSA)", pad=6); axB.set_ylabel("dissimilarity", fontsize=6)
    for sp in ["top","right"]: axB.spines[sp].set_visible(False)
    panel_letter(axB,"b",dx=-0.12)

    # (c) RDM alignment across methods + dimensionality
    tags = [t for t in METHODS if (CACHE/f"rdm_{t}.csv").exists()]
    iu = np.triu_indices(len(EMOS),1)
    vecs = {}
    for t in tags:
        r = pd.read_csv(CACHE/f"rdm_{t}.csv", index_col=0).reindex(index=EMOS, columns=EMOS).values
        vecs[t] = r[iu]
    A = np.eye(len(tags))
    for i,a in enumerate(tags):
        for j,b in enumerate(tags):
            if i<j: A[i,j]=A[j,i]=spearmanr(vecs[a],vecs[b]).correlation
    axC = fig.add_subplot(gs[0,2])
    masked = np.where(np.tril(np.ones_like(A))>0, A, np.nan)
    im2 = nice_heatmap(axC, masked, [METHODS[t] for t in tags], [METHODS[t] for t in tags],
                       cmap="PuOr_r", vmin=-1, vmax=1, annot=True, fmt="{:.2f}", aspect="equal", fontsize=5.6, annot_size=5.4)
    axC.set_title("RDM alignment (ρ)", pad=6); panel_letter(axC,"c",dx=-0.14)
    slim_colorbar(fig, im2, axC, "ρ")
    pr = {t: _part_ratio(np.load(CACHE/f"agg_{t}.npy")) for t in tags}
    axC.set_xlabel("PR dim: " + ", ".join(f"{METHODS[t]} {pr[t]:.1f}" for t in tags), fontsize=4.8)

    save(fig, FIG_DIR/"part5"/"Figure_Part5_embedding_enrichment"); plt.close(fig)
    print("  RDM alignment:", {f"{a}-{b}": round(A[i,j],2) for i,a in enumerate(tags) for j,b in enumerate(tags) if i<j})

if __name__ == "__main__":
    main()
