"""
chord.py — filled chord diagram for emotion-similarity (Chaos-Solitons style).

chord_diagram(ax, S, labels, colors): S = symmetric similarity matrix (>=0).
Outer arc length per node is proportional to its total similarity; ribbon width
between i and j is proportional to S[i, j]. Ribbons curve through the centre.
"""
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Wedge

def _pt(r, a):
    return (r * np.cos(a), r * np.sin(a))

def _ribbon(a_i0, a_i1, a_j0, a_j1, R, color, alpha=0.55, n=24):
    V, C = [], []
    arc_i = np.linspace(a_i0, a_i1, n)
    arc_j = np.linspace(a_j0, a_j1, n)
    V.append(_pt(R, arc_i[0])); C.append(Path.MOVETO)
    for a in arc_i[1:]:
        V.append(_pt(R, a)); C.append(Path.LINETO)
    V.append((0, 0)); C.append(Path.CURVE3)            # through centre
    V.append(_pt(R, arc_j[0])); C.append(Path.CURVE3)
    for a in arc_j[1:]:
        V.append(_pt(R, a)); C.append(Path.LINETO)
    V.append((0, 0)); C.append(Path.CURVE3)
    V.append(_pt(R, arc_i[0])); C.append(Path.CURVE3)
    C.append(Path.CLOSEPOLY); V.append(_pt(R, arc_i[0]))
    return PathPatch(Path(V, C), facecolor=color, edgecolor="none", alpha=alpha, lw=0)

def chord_diagram(ax, S, labels, colors, gap_frac=0.012, keep_quantile=0.40, R=1.0, show_labels=True):
    S = np.array(S, float).copy(); np.fill_diagonal(S, 0)
    S[S < 0] = 0
    n = len(labels)
    # threshold weak links for readability
    off = S[np.triu_indices(n, 1)]
    thr = np.quantile(off, keep_quantile)
    Sd = S.copy(); Sd[Sd < thr] = 0

    deg = Sd.sum(1)
    deg[deg == 0] = S[deg == 0].sum(1) + 1e-9     # keep isolated nodes visible
    total = deg.sum()
    gap = gap_frac * 2 * np.pi
    span = (2 * np.pi - n * gap) * deg / total     # arc length per node

    # node arc start angles
    starts = np.zeros(n)
    ang = np.pi / 2                                # start at top
    for i in range(n):
        starts[i] = ang
        ang += span[i] + gap

    # within each node, sub-arcs for each partner j (cumulative)
    sub = {}
    for i in range(n):
        a = starts[i]
        order = np.argsort(-Sd[i])                 # strongest partner first
        wsum = Sd[i].sum()
        for j in order:
            if Sd[i, j] <= 0:
                continue
            frac = Sd[i, j] / wsum
            a0, a1 = a, a + frac * span[i]
            sub[(i, j)] = (a0, a1)
            a = a1

    # ribbons (each unordered pair once, coloured by stronger-degree node)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in sub and (j, i) in sub:
                src = i if deg[i] >= deg[j] else j
                ax.add_patch(_ribbon(*sub[(i, j)], *sub[(j, i)], R=R,
                                     color=colors[labels[src]], alpha=0.5))

    # outer arcs + labels
    for i in range(n):
        a0, a1 = starts[i], starts[i] + span[i]
        ax.add_patch(Wedge((0, 0), R * 1.08, np.degrees(a0), np.degrees(a1),
                           width=R * 0.08, facecolor=colors[labels[i]], edgecolor="white", lw=0.5))
        am = (a0 + a1) / 2
        x, y = _pt(R * 1.30, am)
        if show_labels:
            ax.text(x, y, labels[i], ha="center", va="center", fontsize=5.0,
                fontweight="bold", color="black")
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.75, 1.75)
    ax.set_aspect("equal"); ax.axis("off")
