"""
run_part1_q30.py — Figure 1 regenerated with q=30 (node-count-matched) MTF/QG.

Reuses the EXACT plotting functions from part1_topology/run_part1.py
(draw_heatmap, metric_bars, chord) so the visual style is identical; only
the input data differs: MTF/QG now come from trial_metrics_mtf_q30.csv /
trial_metrics_qg_q30.csv (q=30 quantile states, node-count-matched to the
30-channel Spearman network) instead of the original q=13 data-driven
files. Saved under a separate filename -- this does NOT overwrite the
original q=13 Figure 1.

Panels a (Spearman) and e-g (Spearman bars) are identical to the original
Figure 1, since Spearman is untouched by this change. Panels b, c (MTF/QG
heatmaps), d (Delta-sigma), and h (consensus chord) are the ones that
actually differ from the original.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import (CACHE, FIG_DIR, EMOS, EMO_COLORS, METRICS, METRIC_LABELS,
                         set_nature_style, save, W2, panel_letter, nice_heatmap, slim_colorbar)
from lib.stats import lmm_emotion_contrasts
from lib.chord import chord_diagram

# import the plotting helpers straight from the original script so the
# visual style is byte-for-byte identical -- only the input data differs.
from part1_topology.run_part1 import draw_heatmap, metric_bars, chord, SP_METRICS

METHODS_Q30 = {"spearman": "Spearman", "mtf_q30": "MTF (q=30)", "qg_q30": "QG (q=30)"}


def _load(tag):
    f = CACHE / f"trial_metrics_{tag}.csv"
    return pd.read_csv(f) if f.exists() else None


def main():
    set_nature_style()
    dfs = {tag: _load(tag) for tag in METHODS_Q30}
    if dfs["spearman"] is None:
        print("  [skip] no spearman metrics."); return
    missing = [t for t in METHODS_Q30 if dfs[t] is None]
    if missing:
        print(f"  [skip] missing q30 metrics for: {missing}. Run compute_metrics_q30.py first.")
        return

    fig = plt.figure(figsize=(W2, W2 * 0.58))
    gs = GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.40, top=0.92, bottom=0.16,
                  height_ratios=[1.0, 0.98])
    corr = {}
    for k, (tag, name) in enumerate(METHODS_Q30.items()):
        ax = fig.add_subplot(gs[0, k])
        im, z = draw_heatmap(ax, dfs[tag], name, show_y=(k == 0))
        corr[tag] = np.corrcoef(z.values)
        panel_letter(ax, "abc"[k])
        if k == 2:
            slim_colorbar(fig, im, ax, "column z-score")

    axP = fig.add_subplot(gs[0, 3])
    emos_nn = [e for e in EMOS if e != "neutral"]
    P = np.full((len(emos_nn), len(METHODS_Q30)), np.nan)
    for j, tag in enumerate(METHODS_Q30):
        subj = dfs[tag].groupby(["subject", "emotion"])["sigma_small_world"].mean().reset_index()
        em = subj.groupby("emotion")["sigma_small_world"].mean(); neu = em.get("neutral", np.nan)
        for i, e in enumerate(emos_nn):
            P[i, j] = (em.get(e, np.nan) - neu) / abs(neu) * 100.0
    vmax = max(5.0, float(np.ceil(np.nanpercentile(np.abs(P), 90))))
    imP = nice_heatmap(axP, P, emos_nn, [METHODS_Q30[t] for t in METHODS_Q30], cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, annot=False, aspect="auto", fontsize=5.4)
    for i in range(len(emos_nn)):
        for j in range(len(METHODS_Q30)):
            val = P[i, j]
            if np.isnan(val):
                continue
            axP.text(j, i, f"{val:+.0f}", ha="center", va="center", fontsize=4.6,
                      color="white" if abs(val) / vmax > 0.5 else "black")
    axP.set_yticklabels([]); axP.set_title("Δσ vs neutral (%)", pad=6)
    panel_letter(axP, "d"); slim_colorbar(fig, imP, axP, "% vs neutral")

    for k, metric in enumerate(SP_METRICS):
        ax = fig.add_subplot(gs[1, k]); metric_bars(ax, dfs["spearman"], metric)
        panel_letter(ax, "efg"[k])

    axH = fig.add_subplot(gs[1, 3])
    consensus = np.mean([corr[t] for t in corr], axis=0)
    chord(axH, consensus, title="Consensus chord (q=30)"); panel_letter(axH, "h")

    handles = [mp.Patch(facecolor=EMO_COLORS[e], edgecolor="k", lw=0.3, label=e) for e in EMOS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=9, fontsize=5.4, frameon=False, handlelength=1.0, columnspacing=0.9)
    fig.suptitle("Figure 1 (q=30 robustness version): MTF/QG node-count-matched to Spearman",
                 fontsize=7.5, y=0.985)
    save(fig, FIG_DIR / "part1" / "Figure_Part1_global_topology_q30"); plt.close(fig)
    print("[Figure 1, q=30] saved Figure_Part1_global_topology_q30.pdf/.png")

    # console cross-representation correlations for quick reference
    pairs = [("spearman", "mtf_q30"), ("spearman", "qg_q30"), ("mtf_q30", "qg_q30")]
    for a, b in pairs:
        r = np.corrcoef(corr[a].flatten(), corr[b].flatten())[0, 1]
        print(f"  chord-similarity-matrix correlation {METHODS_Q30[a]} vs {METHODS_Q30[b]}: r={r:.3f}")


if __name__ == "__main__":
    main()
