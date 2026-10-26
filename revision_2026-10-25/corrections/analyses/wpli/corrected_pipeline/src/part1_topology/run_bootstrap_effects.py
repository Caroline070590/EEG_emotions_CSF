"""
run_bootstrap_effects.py — subject-cluster bootstrap effect sizes vs neutral.
Forest plots of standardized mean difference (emotion - neutral) with 95% CIs for
a topology metric (σ) and a synchrony metric (wPLI). Also writes a supplementary
CSV with bootstrap CIs for every topology metric (3 methods) and synchrony metric.
Resolves the 'anger' question: anger's topology CI spans 0; synchrony CI excludes 0.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lib.config import (CACHE, FIG_DIR, EMOS, EMO_COLORS, METRICS, METRIC_LABELS,
                        set_nature_style, save, W2, panel_letter)
from lib.stats import boot_vs_neutral
from lib.synchrony import SYNC_COLS

def forest(ax, df, col, title, xlabel):
    rows = [(e, *boot_vs_neutral(df, col, e)) for e in EMOS if e != "neutral"]
    rows.sort(key=lambda r: r[1])
    ax.axvline(0, color="0.6", lw=0.6, ls="--")
    for i, (e, d, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=EMO_COLORS[e], lw=1.4, solid_capstyle="round")
        ax.plot(d, i, "o", color=EMO_COLORS[e], ms=4, mec="white", mew=0.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    ax.set_title(title, pad=5); ax.set_xlabel(xlabel, fontsize=6)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    return {r[0]: r[1:] for r in rows}

def main():
    set_nature_style()
    sp = pd.read_csv(CACHE/"trial_metrics_spearman.csv")
    sy = pd.read_csv(CACHE/"trial_synchrony_metrics.csv")
    fig = plt.figure(figsize=(W2*0.72, W2*0.34))
    gs = GridSpec(1, 2, figure=fig, wspace=0.5, top=0.86, bottom=0.2)
    axA = fig.add_subplot(gs[0,0]); forest(axA, sp, "sigma_small_world",
            "Topology: σ vs neutral (Spearman)", "standardized Δ (95% CI)"); panel_letter(axA,"a",dx=-0.22)
    axB = fig.add_subplot(gs[0,1]); forest(axB, sy, "wpli",
            "Synchrony: wPLI vs neutral", "standardized Δ (95% CI)"); panel_letter(axB,"b",dx=-0.18)
    save(fig, FIG_DIR/"part1"/"Figure_Part1b_bootstrap_effects"); plt.close(fig)

    # supplementary CSV: bootstrap CI for every emotion x measure
    recs = []
    for tag in ["spearman","mtf","qg"]:
        df = pd.read_csv(CACHE/f"trial_metrics_{tag}.csv")
        for m in METRICS:
            for e in EMOS:
                if e == "neutral": continue
                d, lo, hi = boot_vs_neutral(df, m, e)
                recs.append(dict(domain="topology", method=tag, measure=m, emotion=e,
                                 d=d, ci_lo=lo, ci_hi=hi, ci_excludes_0=bool(lo>0 or hi<0)))
    for c in SYNC_COLS:
        for e in EMOS:
            if e == "neutral": continue
            d, lo, hi = boot_vs_neutral(sy, c, e)
            recs.append(dict(domain="synchrony", method="-", measure=c, emotion=e,
                             d=d, ci_lo=lo, ci_hi=hi, ci_excludes_0=bool(lo>0 or hi<0)))
    pd.DataFrame(recs).to_csv(CACHE/"bootstrap_effects_vs_neutral.csv", index=False)
    print("  wrote cache/bootstrap_effects_vs_neutral.csv")

if __name__ == "__main__":
    main()
