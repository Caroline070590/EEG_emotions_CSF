"""
run_part1.py — Figure 1: global EEG network topology across emotions.
a-c  emotion x metric heatmaps (column z) for Spearman / MTF / QG; cells carry
     participant-clustered OLS contrasts vs neutral. Emotion labels shown once.
top-right  lower-triangular method-correlation heatmap.
d-f  emotion-correlation chord per method (untitled; aligned under a-c).
g    consensus emotion-correlation chord (mean across methods).
No baked-in figure title (goes in the caption).
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.gridspec import GridSpec
from lib.config import (CACHE, FIG_DIR, EMOS, EMO_COLORS, METRICS, METRIC_LABELS,
                        set_nature_style, save, W2, panel_letter, nice_heatmap, slim_colorbar)
from lib.stats import clustered_ols_emotion_contrasts
from lib.chord import chord_diagram

METHODS = {"spearman": "Spearman", "mtf": "MTF", "qg": "QG"}
def stars(p): return "***" if p<1e-3 else "**" if p<1e-2 else "*" if p<.05 else ""
def _load(tag):
    f = CACHE / f"trial_metrics_{tag}.csv"; return pd.read_csv(f) if f.exists() else None
def _profile(df):
    subj = df.groupby(["subject","emotion"])[METRICS].mean().reset_index()
    heat = subj.groupby("emotion")[METRICS].mean().reindex(EMOS)
    return (heat - heat.mean(0)) / heat.std(0)

def draw_heatmap(ax, df, title, show_y):
    z = _profile(df)
    im = nice_heatmap(ax, z.values, EMOS, [METRIC_LABELS[m] for m in METRICS],
                      cmap="viridis", vmin=-2, vmax=2, aspect="auto", fontsize=5.4)
    if not show_y: ax.set_yticklabels([])
    ax.set_title(title, pad=6)
    for j, m in enumerate(METRICS):
        pmap = clustered_ols_emotion_contrasts(df, m, ref="neutral")
        for i, e in enumerate(EMOS):
            p = pmap.get(e, np.nan); st = stars(p) if e != "neutral" else ""
            if st: ax.text(j, i, st, ha="center", va="center", fontsize=5,
                           color="white" if z.values[i, j] < 0.7 else "#111111")
    return im, z

def chord(ax, S, title=None):
    S = np.clip(S, 0, None); np.fill_diagonal(S, 0)
    chord_diagram(ax, S, EMOS, EMO_COLORS, keep_quantile=0.55, show_labels=False)
    ax.set_xlim(-1.18, 1.18); ax.set_ylim(-1.18, 1.18)
    if title: ax.set_title(title, pad=4)

def main():
    set_nature_style()
    dfs = {tag: _load(tag) for tag in METHODS}
    if all(v is None for v in dfs.values()):
        print("  [skip] no trial_metrics_*.csv."); return
    fig = plt.figure(figsize=(W2, W2*0.64))
    gs = GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.4, top=0.92, bottom=0.13,
                  height_ratios=[1.0, 1.05])
    zsig = {}; corr = {}; im = None
    for k, (tag, name) in enumerate(METHODS.items()):
        ax = fig.add_subplot(gs[0, k])
        if dfs[tag] is None: ax.axis("off"); continue
        im, z = draw_heatmap(ax, dfs[tag], name, show_y=(k == 0)); zsig[tag] = z
        corr[tag] = np.corrcoef(z.values)
        panel_letter(ax, "abc"[k])
        if k == 2: slim_colorbar(fig, im, ax, "column z-score")

    # top-right: lower-triangular method-correlation heatmap
    axM = fig.add_subplot(gs[0, 3])
    tags = list(METHODS.keys()); names = [METHODS[t] for t in tags]
    flats = np.vstack([zsig[t].values.flatten() for t in tags])
    Cm = np.corrcoef(flats)
    masked = np.where(np.tril(np.ones_like(Cm), 0) > 0, Cm, np.nan)
    im2 = nice_heatmap(axM, masked, names, names, cmap="PuOr_r", vmin=-1, vmax=1,
                       annot=True, fmt="{:.2f}", aspect="equal", fontsize=5.6, annot_size=5.4)
    axM.set_title("Method correlation", pad=6); slim_colorbar(fig, im2, axM, "r")

    # d-f: per-method emotion-correlation chords (no titles; aligned under a-c)
    for k, tag in enumerate(METHODS):
        ax = fig.add_subplot(gs[1, k])
        if dfs[tag] is None: ax.axis("off"); continue
        chord(ax, corr[tag]); panel_letter(ax, "def"[k])

    # g: consensus emotion-correlation chord (mean across methods)
    axG = fig.add_subplot(gs[1, 3])
    consensus = np.mean([corr[t] for t in METHODS], axis=0)
    chord(axG, consensus, title="Consensus (mean)"); panel_letter(axG, "g")

    # shared emotion colour key (bottom)
    handles = [mp.Patch(facecolor=EMO_COLORS[e], edgecolor="k", lw=0.3, label=e) for e in EMOS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015),
               ncol=9, fontsize=5.4, frameon=False, handlelength=1.0, columnspacing=0.9)
    save(fig, FIG_DIR/"part1"/"Figure_Part1_global_topology"); plt.close(fig)

if __name__ == "__main__":
    main()
