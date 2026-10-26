"""Panel F — z-scored higher-order topological signature per emotion, LMM stars."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from _panels import standalone, stars
from lib.config import EMOS
from lib.topology import TCOLS

LBL = {"H0_total":"H0 total","H0_entropy":"H0 entropy","H1_n":"H1 count",
       "H1_total":"H1 total","H1_max":"H1 max","H1_entropy":"H1 entropy"}

def draw(ax, D, tag):
    tda = D["tda"]; lmm = D["lmm"].set_index("measure")
    M = np.vstack([[tda[tda.emotion==e][c].mean() for c in TCOLS] for e in EMOS])
    Mz = (M - M.mean(0)) / M.std(0)
    im = ax.imshow(Mz.T, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(len(TCOLS)))
    ax.set_yticklabels([f"{LBL[c]} {stars(lmm.loc[c,'p'])}" for c in TCOLS], fontsize=5)
    ax.set_xticks(range(9)); ax.set_xticklabels(EMOS, rotation=90, fontsize=5)
    ax.set_title("Higher-order signature (z)")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("z", fontsize=5)
    cb.ax.tick_params(labelsize=4.5)

def main(tag="spearman"): standalone(draw, "panelF_signature", tag)
if __name__ == "__main__": main()
