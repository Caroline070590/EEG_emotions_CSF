"""Panel B — same emotion map, valence-resolved with convex hulls.
Overlapping hulls visualise why valence does not separate the geometry."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from _panels import standalone
from panel_A_umap_emotion import _map
from lib.config import EMOS, VAL, VAL_COLORS

def _hull(ax, pts, color):
    if len(pts) < 3:
        ax.scatter(pts[:,0], pts[:,1], s=10, c=color); return
    h = ConvexHull(pts)
    poly = Polygon(pts[h.vertices], closed=True, facecolor=color, edgecolor=color,
                   lw=0.8, alpha=0.16, zorder=2); ax.add_patch(poly)

def draw(ax, D, tag):
    ref, boot, var = _map(D); val = D["boot"]["valence"]
    groups = {"neg": [], "neu": [], "pos": []}
    for i, e in enumerate(EMOS): groups[VAL[e]].append(ref[i])
    for v, pts in groups.items():
        pts = np.array(pts); _hull(ax, pts, VAL_COLORS[v])
        ax.scatter(pts[:,0], pts[:,1], s=40, c=VAL_COLORS[v], edgecolors="k", linewidths=0.5,
                   zorder=6, label={"neg":"negative","neu":"neutral","pos":"positive"}[v])
    for i, e in enumerate(EMOS):
        ax.annotate(e[:3], ref[i], fontsize=4.5, color="0.25", ha="center", va="center",
                    xytext=(0, 7), textcoords="offset points", zorder=7)
    ax.axhline(0, color="0.85", lw=0.4); ax.axvline(0, color="0.85", lw=0.4)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Valence view  ·  {'n.s.' if val['p']>0.05 else 'p='+format(val['p'],'.3f')}")
    ax.legend(fontsize=5, loc="best", framealpha=0.85); ax.margins(0.18)

def main(tag="spearman"): standalone(draw, "panelB_valence", tag)
if __name__ == "__main__": main()
