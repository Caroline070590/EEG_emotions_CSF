"""
config.py — shared constants, Nature-style matplotlib settings, colour palettes.

Nature figure specs implemented here:
  * single-column width  = 89 mm  (3.50 in)
  * 1.5-column width     = 120 mm (4.72 in)
  * double-column width  = 183 mm (7.20 in)
  * sans-serif typeface, 5–7 pt text, thin lines, no top/right spines
  * vector PDF + 600-dpi raster export
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------- paths
PKG_ROOT  = Path(__file__).resolve().parents[1]
DATA_ROOT = PKG_ROOT / "EEG-data-raw"          # <emotion>/<Patient###>/trial_*.csv
CACHE     = PKG_ROOT / "cache";            CACHE.mkdir(exist_ok=True)
FIG_DIR   = PKG_ROOT / "figures";          FIG_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------- data constants
CH = ['FP1','FP2','FZ','F3','F4','F7','F8','FC1','FC2','FC5','FC6','CZ','C3','C4',
      'T7','T8','CP1','CP2','CP5','CP6','PZ','P3','P4','P7','P8','PO3','PO4','OZ','O1','O2']
EMOS = ['anger','disgust','fear','sadness','neutral','amusement','inspiration','joy','tenderness']
VAL  = {'anger':'neg','disgust':'neg','fear':'neg','sadness':'neg','neutral':'neu',
        'amusement':'pos','inspiration':'pos','joy':'pos','tenderness':'pos'}
EI   = {e: i for i, e in enumerate(EMOS)}

# region map (10-20) used by regional analyses
REGION = {
    'frontal':  ['FP1','FP2','FZ','F3','F4','F7','F8'],
    'central':  ['FC1','FC2','FC5','FC6','CZ','C3','C4'],
    'temporal': ['T7','T8','CP5','CP6','P7','P8'],
    'parietal': ['CP1','CP2','PZ','P3','P4'],
    'occipital':['PO3','PO4','OZ','O1','O2'],
}

# ----------------------------------------------------------------- colours
EMO_COLORS = {'anger':'#c0392b','disgust':'#7d3c98','fear':'#34495e','sadness':'#2471a3',
              'neutral':'#7f8c8d','amusement':'#e67e22','inspiration':'#16a085',
              'joy':'#f1c40f','tenderness':'#e84393'}
VAL_COLORS = {'neg':'#2471a3','neu':'#7f8c8d','pos':'#e67e22'}

# Nature column widths (inches)
W1, W15, W2 = 3.50, 4.72, 7.20

# ----------------------------------------------------------------- style
def set_nature_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600, "figure.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "axes.titlepad": 4, "axes.labelpad": 2,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.6, "axes.edgecolor": "#222222",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.color": "#222222", "ytick.color": "#222222",
        "axes.labelcolor": "#222222", "text.color": "#222222",
        "lines.linewidth": 1.0, "lines.markersize": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })

def panel_letter(ax, letter, dx=-0.16, dy=1.08):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")

def despine(ax, left=True, bottom=True):
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_visible(left); ax.spines["bottom"].set_visible(bottom)

def nice_heatmap(ax, M, row_labels, col_labels, cmap="magma", vmin=None, vmax=None,
                 annot=False, fmt="{:.1f}", grid=True, fontsize=5.4, annot_size=4.6, aspect="equal"):
    """Publication-style heatmap: square cells, thin white gridlines, clean ticks."""
    M = np.asarray(M, float)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
    ax.set_xticks(np.arange(len(col_labels))); ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticklabels(row_labels, fontsize=fontsize)
    ax.set_xticks(np.arange(-.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    if grid:
        ax.grid(which="minor", color="white", linewidth=0.6); ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
    if annot:
        import numpy as _np
        rng = (_np.nanmax(M) - _np.nanmin(M)) or 1
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if _np.isfinite(v):
                    lum = (v - _np.nanmin(M)) / rng
                    ax.text(j, i, fmt.format(v), ha="center", va="center",
                            fontsize=annot_size, color="white" if lum < 0.45 else "#111111")
    return im

def slim_colorbar(fig, im, ax, label="", fontsize=5):
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=4.6, length=1.5, width=0.4)
    cb.set_label(label, fontsize=fontsize); return cb

def save(fig, path, also_png=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    if also_png:
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=600)
    print(f"  saved {path.with_suffix('.pdf').name} / .png")


# ---- representation / metric configuration (ported from Spearman-MTF-QG.ipynb)
# NOTE: the original load kept sample_index & time_index as channels (32 nodes);
# we exclude them and use the 30 real electrodes (correct, documented choice).
DROP_COLS_LIKE = ["subject","trial","emotion","label","timestamp","sample_index","time_index"]
# Primary manuscript design: fixed q=30 for both MTF and QG.  This is a
# node-matching choice (30 state nodes versus 30 retained EEG channels), not a
# function of the 250-sample segment length.
Q_FIXED = 30
Q_MIN, Q_MAX = Q_FIXED, Q_FIXED
ACF_THRESHOLD_FOR_K = 0.10
MAX_K_FRACTION = 0.25
TARGET_DENSITY = 0.10
N_RANDOMIZATIONS = 20          # paper default; lower (8-10) for quick runs
REWIRES_PER_EDGE = 5
Q_EMBED = Q_FIXED              # fixed q=30 for all primary MTF/QG representations
REPR_N = {"spearman": 30, "mtf": Q_EMBED, "qg": Q_EMBED}   # matrix node count per tag
METRICS = ["sigma_small_world","gamma_C_over_Crand","lambda_L_over_Lrand",
           "transitivity","char_path_len_gcc","global_efficiency"]
METRIC_LABELS = {"sigma_small_world":"σ","gamma_C_over_Crand":"γ",
                 "lambda_L_over_Lrand":"λ","transitivity":"Transitivity",
                 "char_path_len_gcc":"Path length","global_efficiency":"Efficiency"}
METHOD_ORDER = ["Spearman","MTF","QG"]
