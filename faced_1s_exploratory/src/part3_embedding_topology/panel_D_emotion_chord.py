"""Panel D — emotion-similarity CHORD diagram.
Replaces the force-directed multigraph with a chord plot. Ribbon width between
two emotions ∝ representational similarity (1 − dissimilarity); when several
representations are cached the similarity is the across-layer consensus."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from _panels import standalone
from lib.config import CACHE, EMOS, EMO_COLORS
from lib.chord import chord_diagram

def _consensus_similarity(default_tag):
    rdms = []
    layers = []
    for t in ["spearman", "mtf", "qg"]:
        f = CACHE / f"rdm_{t}.csv"
        if f.exists():
            rdms.append(pd.read_csv(f, index_col=0).reindex(index=EMOS, columns=EMOS).values)
            layers.append(t)
    if not rdms:
        rdms = [pd.read_csv(CACHE / f"rdm_{default_tag}.csv", index_col=0).values]
        layers = [default_tag]
    D = np.nanmean(np.stack(rdms), 0)
    D = (D + D.T) / 2
    Dmax = np.nanmax(D[np.triu_indices(len(EMOS), 1)])
    S = (Dmax - D)                          # dissimilarity -> similarity
    np.fill_diagonal(S, 0)
    return S, layers

def draw(ax, D, tag):
    S, layers = _consensus_similarity(tag)
    chord_diagram(ax, S, EMOS, EMO_COLORS, keep_quantile=0.40)
    ax.set_title("Emotion-similarity chord")
    ax.text(0.5, -0.02, "layers: " + ", ".join(layers), transform=ax.transAxes,
            ha="center", va="top", fontsize=4.8, color="0.4")

def main(tag="spearman"): standalone(draw, "panelD_chord", tag, figsize=(2.4, 2.4))
if __name__ == "__main__": main()
