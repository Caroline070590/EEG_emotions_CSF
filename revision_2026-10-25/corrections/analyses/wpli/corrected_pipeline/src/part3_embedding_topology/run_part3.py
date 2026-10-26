"""
run_part3.py — assemble the Part 3 main figure (Nature double-column).

Composes the six panels (A–F) onto one canvas and also re-renders each panel
standalone into figures/part3/. Run compute_features.py first.

Usage:
    python part3_embedding_topology/run_part3.py --tag spearman
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from _panels import load
from lib.config import set_nature_style, save, FIG_DIR, W2, panel_letter

import panel_A_umap_emotion as pA
import panel_B_valence_centroids as pB
import panel_C_rsa_matrix as pC
import panel_D_emotion_chord as pD
import panel_E_higherorder_bars as pE
import panel_F_topology_signature as pF


def assemble(tag="spearman"):
    set_nature_style()
    D = load(tag)
    fig = plt.figure(figsize=(W2, W2 * 0.62))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.45)
    panels = [(pA, gs[0,0], "a"), (pB, gs[0,1], "b"), (pC, gs[0,2], "c"),
              (pD, gs[1,0], "d"), (pE, gs[1,1], "e"), (pF, gs[1,2], "f")]
    for mod, cell, letter in panels:
        ax = fig.add_subplot(cell)
        mod.draw(ax, D, tag)
        panel_letter(ax, letter)
    save(fig, FIG_DIR / "part3" / f"Figure_Part3_embedding_topology_{tag}")
    plt.close(fig)


def standalone_all(tag="spearman"):
    for mod in (pA, pB, pC, pD, pE, pF):
        mod.main(tag)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="spearman")
    ap.add_argument("--standalone", action="store_true", help="also render each panel separately")
    ap.add_argument("--recompute", action="store_true", help="(re)run compute_features first")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    if a.recompute:
        from compute_features import run as compute
        compute(tag=a.tag, n_jobs=a.n_jobs, n_boot=a.n_boot)
    assemble(a.tag)
    if a.standalone:
        standalone_all(a.tag)
