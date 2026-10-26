"""10-20 scalp layout + topomap / scalp-edge drawing helpers for the 30 channels."""
import numpy as np
from scipy.interpolate import griddata
import matplotlib.patches as mp

# approximate top-down 10-20 coordinates (x: right+, y: anterior+), unit-ish circle
POS = {
 "FP1":(-0.31,0.95),"FP2":(0.31,0.95),
 "F7":(-0.81,0.59),"F3":(-0.39,0.61),"FZ":(0.0,0.63),"F4":(0.39,0.61),"F8":(0.81,0.59),
 "FC5":(-0.62,0.31),"FC1":(-0.22,0.32),"FC2":(0.22,0.32),"FC6":(0.62,0.31),
 "T7":(-1.0,0.0),"C3":(-0.5,0.0),"CZ":(0.0,0.0),"C4":(0.5,0.0),"T8":(1.0,0.0),
 "CP5":(-0.62,-0.31),"CP1":(-0.22,-0.32),"CP2":(0.22,-0.32),"CP6":(0.62,-0.31),
 "P7":(-0.81,-0.59),"P3":(-0.39,-0.61),"PZ":(0.0,-0.63),"P4":(0.39,-0.61),"P8":(0.81,-0.59),
 "PO3":(-0.33,-0.80),"PO4":(0.33,-0.80),
 "O1":(-0.31,-0.95),"OZ":(0.0,-0.97),"O2":(0.31,-0.95),
}

def head(ax, R=1.18):
    ax.add_patch(mp.Circle((0,0), R, fill=False, lw=1.0, color="0.25"))
    ax.add_patch(mp.Polygon([(-0.13*R,R*0.99),(0,R*1.16),(0.13*R,R*0.99)], closed=True,
                            fill=False, lw=1.0, color="0.25"))
    for s in (-1,1):
        ax.add_patch(mp.Ellipse((s*R,0),0.10*R,0.30*R, fill=False, lw=1.0, color="0.25"))
    ax.set_xlim(-R*1.25,R*1.25); ax.set_ylim(-R*1.3,R*1.3); ax.set_aspect("equal"); ax.axis("off")

def topomap(ax, ch_names, values, cmap="viridis", vmin=None, vmax=None, dots=True, R=1.18):
    xy = np.array([POS[c] for c in ch_names]); v = np.asarray(values, float)
    gx, gy = np.mgrid[-R:R:120j, -R:R:120j]
    gi = griddata(xy, v, (gx, gy), method="cubic")
    gi[gx**2+gy**2 > R**2] = np.nan
    im = ax.imshow(gi.T, origin="lower", extent=(-R,R,-R,R), cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="bilinear")
    head(ax, R)
    if dots: ax.scatter(xy[:,0], xy[:,1], s=2.5, c="k", zorder=5)
    return im

def scalp_edges(ax, ch_names, edges, signs, R=1.18, node_inv=None):
    head(ax, R)
    xy = {c: POS[c] for c in ch_names}
    for (i,j), sg in zip(edges, signs):
        a, b = ch_names[i], ch_names[j]
        ax.plot([xy[a][0],xy[b][0]],[xy[a][1],xy[b][1]],
                color=("#c0392b" if sg>0 else "#2c5fa8"), lw=0.9, alpha=0.8, zorder=3)
    deg = {}
    for (i,j) in edges: deg[i]=deg.get(i,0)+1; deg[j]=deg.get(j,0)+1
    for c in ch_names:
        k = deg.get(ch_names.index(c),0)
        ax.scatter(*xy[c], s=4+9*k, c="k", zorder=5)
