"""Panel A — emotion geometry map. 2D PCA of the 9 emotion centroids (the spatial
form of the RSA) with subject-bootstrap 95% confidence ellipses per emotion.
Clean and honest: shows which emotions cluster and which stand apart (sadness)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from matplotlib.patches import Ellipse
from _panels import standalone
from lib.config import EMOS, EI, EMO_COLORS
from lib.embedding import emotion_pca_map

_MAP = {}
def _map(D):
    if D["tag"] not in _MAP:
        _MAP[D["tag"]] = emotion_pca_map(
            D["agg"], D["emo_idx"], D["am"]["subject"].values, n_boot=500)
    return _MAP[D["tag"]]

def _ellipse(ax, pts, color, conf=2.448):
    mu = pts.mean(0); cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov); order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2*conf*np.sqrt(np.maximum(vals, 1e-12))
    ax.add_patch(Ellipse(mu, w, h, angle=ang, facecolor=color, edgecolor="none",
                         lw=0, alpha=0.10, zorder=2))

def draw(ax, D, tag):
    ref, boot, var = _map(D)
    dec = D["boot"]["decode"]
    for i, e in enumerate(EMOS):
        _ellipse(ax, boot[i], EMO_COLORS[e])
    for i, e in enumerate(EMOS):
        ax.scatter(*ref[i], s=34, c=EMO_COLORS[e], edgecolors="white", linewidths=0.6,
                   zorder=6, label=e)
    ax.axhline(0, color="0.9", lw=0.4, zorder=0); ax.axvline(0, color="0.9", lw=0.4, zorder=0)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Emotion geometry map")
    ax.legend(fontsize=4.3, ncol=2, framealpha=0.8, handletextpad=0.2,
              columnspacing=0.5, loc="upper right")
    ax.text(0.02, 0.02, f"LOSO decode {dec['obs']:.3f}\n95%CI[{dec['lo']:.3f},{dec['hi']:.3f}]  chance {dec['chance']:.3f}",
            transform=ax.transAxes, fontsize=4.4, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", lw=0.4, alpha=0.85))
    ax.margins(0.18)

def main(tag="spearman"): standalone(draw, "panelA_map", tag)
if __name__ == "__main__": main()
