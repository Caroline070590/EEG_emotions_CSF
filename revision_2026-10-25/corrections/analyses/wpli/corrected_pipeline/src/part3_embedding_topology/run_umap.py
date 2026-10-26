"""
run_umap.py — nonlinear embedding companions to the PCA emotion-geometry map.
Subject-centered UMAP (top) and t-SNE (bottom) of per-(subject,emotion)
connectivity, per method, colored by emotion with centroids overlaid.
    python part3_embedding_topology/run_umap.py
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")
from lib.config import (CACHE, FIG_DIR, EMOS, EMO_COLORS, set_nature_style, save, W2, panel_letter)
from lib.embedding import umap_embed
from sklearn.manifold import TSNE

METHODS = {"spearman": "Spearman", "mtf": "MTF", "qg": "QG"}

def _subject_center(agg, subjects):
    out = agg.copy().astype(float)
    for s in np.unique(subjects): m = subjects == s; out[m] -= out[m].mean(0)
    return out

def _scatter(ax, Y, am, title, ylab=None):
    for e in EMOS:
        m = (am["emotion"] == e).values
        ax.scatter(Y[m,0], Y[m,1], s=3.5, c=EMO_COLORS[e], alpha=0.32, linewidths=0, rasterized=True)
    for e in EMOS:
        m = (am["emotion"] == e).values; c = Y[m].mean(0)
        ax.scatter(*c, s=42, c=EMO_COLORS[e], edgecolors="white", linewidths=0.8, zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, pad=5)
    if ylab: ax.set_ylabel(ylab, fontsize=6)

def main():
    set_nature_style()
    tags = [t for t in METHODS if (CACHE/f"agg_{t}.npy").exists()]
    if not tags: print("  [skip] no agg_*.npy."); return
    fig = plt.figure(figsize=(W2, W2*0.72))
    gs = GridSpec(2, len(tags), figure=fig, wspace=0.18, hspace=0.22, top=0.90, bottom=0.13)
    letters = iter("abcdef")
    for k, tag in enumerate(tags):
        agg = np.load(CACHE/f"agg_{tag}.npy"); am = pd.read_csv(CACHE/f"agg_meta_{tag}.csv")
        Xc = _subject_center(agg, am["subject"].values)
        U = umap_embed(Xc)
        T = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(Xc)
        axU = fig.add_subplot(gs[0, k]); _scatter(axU, U, am, METHODS[tag], "UMAP" if k==0 else None)
        panel_letter(axU, next(letters))
        axT = fig.add_subplot(gs[1, k]); _scatter(axT, T, am, None, "t-SNE" if k==0 else None)
        panel_letter(axT, next(letters))
    handles = [mp.Patch(facecolor=EMO_COLORS[e], edgecolor="k", lw=0.3, label=e) for e in EMOS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=9, fontsize=5.4, frameon=False, handlelength=1.0, columnspacing=0.9)
    save(fig, FIG_DIR/"part3"/"Figure_Part3_umap_tsne")
    plt.close(fig)

if __name__ == "__main__":
    main()
