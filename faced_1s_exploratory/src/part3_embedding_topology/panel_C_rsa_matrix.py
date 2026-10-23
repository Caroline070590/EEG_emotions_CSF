"""Panel C — emotion representational dissimilarity matrix (RSA), clustered order."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from _panels import standalone
from lib.config import EMOS

def _order(D):
    M = D.values.copy(); M = (M+M.T)/2; np.fill_diagonal(M,0)
    Z = linkage(squareform(M, checks=False), method="average")
    return dendrogram(Z, labels=EMOS, no_plot=True)["leaves"], M, Z

def draw(ax, D, tag):
    order, M, _ = _order(D["rdm"])
    im = ax.imshow(M[np.ix_(order,order)], cmap="magma_r")
    ax.set_xticks(range(9)); ax.set_xticklabels([EMOS[i] for i in order], rotation=90, fontsize=5)
    ax.set_yticks(range(9)); ax.set_yticklabels([EMOS[i] for i in order], fontsize=5)
    ax.set_title("Emotion similarity (RSA)")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("1 − corr", fontsize=5)
    cb.ax.tick_params(labelsize=4.5)

def main(tag="spearman"): standalone(draw, "panelC_rsa", tag)
if __name__ == "__main__": main()
