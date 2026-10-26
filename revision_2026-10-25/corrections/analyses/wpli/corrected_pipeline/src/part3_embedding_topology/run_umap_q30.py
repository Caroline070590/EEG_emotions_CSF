"""
run_umap_q30.py — Fig. S5 (nonlinear embedding comparison) regenerated with
q=30 (node-count-matched) MTF/QG embeddings. Reuses run_umap.py's helper
functions; only the input agg_*.npy differ (agg_mtf_q30.npy / agg_qg_q30.npy
instead of agg_mtf.npy / agg_qg.npy, both now 435-dim like Spearman).
Saved separately -- does not overwrite the original q=13 Fig. S5.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, FIG_DIR, EMOS, EMO_COLORS, set_nature_style, save, W2, panel_letter
from lib.embedding import umap_embed
from sklearn.manifold import TSNE
from part3_embedding_topology.run_umap import _subject_center, _scatter

METHODS_Q30 = {"spearman": "Spearman", "mtf_q30": "MTF (q=30)", "qg_q30": "QG (q=30)"}


def main():
    set_nature_style()
    tags = [t for t in METHODS_Q30 if (CACHE / f"agg_{t}.npy").exists()]
    if len(tags) < 3:
        print(f"  [skip] missing agg_*.npy for: {set(METHODS_Q30) - set(tags)}")
        return
    fig = plt.figure(figsize=(W2, W2 * 0.72))
    gs = GridSpec(2, len(tags), figure=fig, wspace=0.18, hspace=0.22, top=0.88, bottom=0.13)
    letters = iter("abcdef")
    for k, tag in enumerate(tags):
        agg = np.load(CACHE / f"agg_{tag}.npy"); am = pd.read_csv(CACHE / f"agg_meta_{tag}.csv")
        Xc = _subject_center(agg, am["subject"].values)
        U = umap_embed(Xc)
        T = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(Xc)
        axU = fig.add_subplot(gs[0, k]); _scatter(axU, U, am, METHODS_Q30[tag], "UMAP" if k == 0 else None)
        panel_letter(axU, next(letters))
        axT = fig.add_subplot(gs[1, k]); _scatter(axT, T, am, None, "t-SNE" if k == 0 else None)
        panel_letter(axT, next(letters))
    handles = [mp.Patch(facecolor=EMO_COLORS[e], edgecolor="k", lw=0.3, label=e) for e in EMOS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=9, fontsize=5.4, frameon=False, handlelength=1.0, columnspacing=0.9)
    fig.suptitle("Fig. S5 (q=30 robustness version): MTF/QG node-count-matched to Spearman",
                 fontsize=7.5, y=0.97)
    save(fig, FIG_DIR / "part3" / "Figure_Part3_umap_tsne_q30")
    plt.close(fig)
    print("[Fig S5, q=30] saved Figure_Part3_umap_tsne_q30.pdf/.png")


if __name__ == "__main__":
    main()
