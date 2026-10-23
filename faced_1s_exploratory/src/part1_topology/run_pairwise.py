"""
run_pairwise.py — pairwise emotion-vs-emotion comparison (subject-cluster bootstrap).
Lower-triangular matrices of the standardized difference (row - col) for a topology
metric (σ) and a synchrony metric (wPLI); cells whose 95% CI excludes 0 are dotted.
Also writes a supplementary CSV of all pairwise comparisons.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lib.config import (CACHE, FIG_DIR, EMOS, set_nature_style, save, W2,
                        panel_letter, nice_heatmap, slim_colorbar)
from lib.stats import boot_vs_neutral
from lib.synchrony import SYNC_COLS
from lib.config import METRICS

def pairwise_matrix(df, col, n_boot=1000):
    n = len(EMOS); D = np.full((n, n), np.nan); SIG = np.zeros((n, n), bool)
    for i, ei in enumerate(EMOS):
        for j, ej in enumerate(EMOS):
            if i <= j: continue
            d, lo, hi = boot_vs_neutral(df, col, ei, ref=ej, n_boot=n_boot)
            D[i, j] = d; SIG[i, j] = bool(lo > 0 or hi < 0)
    return D, SIG

def draw(ax, df, col, title):
    D, SIG = pairwise_matrix(df, col)
    im = nice_heatmap(ax, D, EMOS, EMOS, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                      aspect="equal", fontsize=5.2)
    for i in range(len(EMOS)):
        for j in range(len(EMOS)):
            if SIG[i, j]:
                ax.plot(j, i, "o", ms=2.2, color="white", mec="black", mew=0.4, zorder=6)
    ax.set_title(title, pad=6)
    return im

def main():
    set_nature_style()
    sp = pd.read_csv(CACHE/"trial_metrics_spearman.csv")
    sy = pd.read_csv(CACHE/"trial_synchrony_metrics.csv")
    fig = plt.figure(figsize=(W2*0.85, W2*0.42))
    gs = GridSpec(1, 2, figure=fig, wspace=0.55, top=0.86, bottom=0.24)
    axA = fig.add_subplot(gs[0,0]); im = draw(axA, sp, "sigma_small_world", "Topology σ — pairwise Δ")
    panel_letter(axA,"a",dx=-0.14); slim_colorbar(fig, im, axA, "standardized Δ (row−col)")
    axB = fig.add_subplot(gs[0,1]); im2 = draw(axB, sy, "wpli", "Synchrony wPLI — pairwise Δ")
    panel_letter(axB,"b",dx=-0.14); slim_colorbar(fig, im2, axB, "standardized Δ (row−col)")
    fig.text(0.5, 0.05, "dotted cell = subject-cluster bootstrap 95% CI excludes 0",
             ha="center", fontsize=5.2, color="0.35")
    save(fig, FIG_DIR/"part1"/"Figure_Part1c_pairwise"); plt.close(fig)

    # supplementary CSV: all pairwise comparisons, topology (3 methods) + synchrony
    recs = []
    for tag in ["spearman","mtf","qg"]:
        df = pd.read_csv(CACHE/f"trial_metrics_{tag}.csv")
        for m in METRICS:
            for i, ei in enumerate(EMOS):
                for j, ej in enumerate(EMOS):
                    if i <= j: continue
                    d, lo, hi = boot_vs_neutral(df, m, ei, ref=ej, n_boot=800)
                    recs.append(dict(domain="topology", method=tag, measure=m,
                                     emotion_a=ei, emotion_b=ej, d=d, ci_lo=lo, ci_hi=hi,
                                     ci_excludes_0=bool(lo>0 or hi<0)))
    for c in SYNC_COLS:
        for i, ei in enumerate(EMOS):
            for j, ej in enumerate(EMOS):
                if i <= j: continue
                d, lo, hi = boot_vs_neutral(sy, c, ei, ref=ej, n_boot=800)
                recs.append(dict(domain="synchrony", method="-", measure=c,
                                 emotion_a=ei, emotion_b=ej, d=d, ci_lo=lo, ci_hi=hi,
                                 ci_excludes_0=bool(lo>0 or hi<0)))
    pd.DataFrame(recs).to_csv(CACHE/"pairwise_emotion_comparisons.csv", index=False)
    print("  wrote cache/pairwise_emotion_comparisons.csv")

if __name__ == "__main__":
    main()
