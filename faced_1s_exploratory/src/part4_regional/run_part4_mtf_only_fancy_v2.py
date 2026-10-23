"""
run_part4_mtf_only_fancy_v2.py — revision of the fancy MTF-only regional
figure: scalp-map row removed; the three small summary panels (positive-
minus-negative sigma per region, deviation-from-neutral count per
emotion, and emotion-modulation range per region) are merged into ONE
combined panel using a twin x-axis (bottom axis = sigma units for the
region rows; top axis = significant-region count for the emotion rows),
sharing a single categorical y-axis (regions, then emotions).

Final layout:
  a  regional sigma heatmap (MTF, q=30)
  b  combined summary: region rows (pos-neg sigma, emotion-modulation
     range) + emotion rows (# significant regions vs neutral)
  c  raw regional sigma trends by emotion
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patheffects import withStroke
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, FIG_DIR, EMOS, VAL, save, W2
from lib.regions import REGIONS
from lib.stats import lmm_emotion_contrasts

INK = "#1b1f2b"
PALETTE = {"frontal": "#b3552b", "central": "#c98a1f", "temporal": "#6a4c93",
           "parietal": "#1f6f8b", "occipital": "#1f8a70"}
EMO_PALETTE = {"anger": "#b3552b", "disgust": "#6a4c93", "fear": "#2b3a55",
               "sadness": "#1f6f8b", "neutral": "#8a8f9a", "amusement": "#c98a1f",
               "inspiration": "#1f8a70", "joy": "#d4a017", "tenderness": "#c2547e"}
POS_E = [e for e in EMOS if VAL[e] == "pos"]; NEG_E = [e for e in EMOS if VAL[e] == "neg"]
RLAB = [r.capitalize() for r in REGIONS]

SIGMA_CMAP = LinearSegmentedColormap.from_list(
    "fancy_sigma", ["#1b2a4a", "#2f5a7a", "#e8e3d3", "#c98a1f", "#7a2e2e"], N=256)


def _set_fancy_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 7.2, "text.color": INK,
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else ""


def fancy_heatmap(ax, M, row_labels, col_labels, cmap, vmin, vmax, fontsize=6.0):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels))); ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=42, ha="right", fontsize=fontsize)
    ax.set_yticklabels(row_labels, fontsize=fontsize, fontweight="medium")
    ax.set_xticks(np.arange(-.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1); ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def panel_tag(ax, letter):
    ax.text(-0.06, 1.12, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            color=INK, va="top", ha="left")


def rounded_card(fig, x0, y0, w, h, color="#f4f5f7", zorder=-1):
    import matplotlib.patches as mpatches
    fig.patches.append(mpatches.FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        transform=fig.transFigure, facecolor=color, edgecolor="none", zorder=zorder))


def main():
    _set_fancy_style()
    f = CACHE / "region_trial_metrics_q30.csv"
    if not f.exists():
        print("  [skip] region_trial_metrics_q30.csv missing"); return
    d = pd.read_csv(f); d = d[d.method == "MTF"]
    sm = d.groupby(["subject", "emotion", "region"])["sigma_small_world"].mean().reset_index()

    M = sm.groupby(["region", "emotion"])["sigma_small_world"].mean().reindex(
        pd.MultiIndex.from_product([REGIONS, EMOS], names=["region", "emotion"])).unstack()
    M = M.reindex(index=REGIONS, columns=EMOS)
    Zsc = M.sub(M.mean(1), axis=0).div(M.std(1).replace(0, np.nan), axis=0)

    # significance counts (region x emotion) for panel b's emotion rows
    sig = {e: 0 for e in EMOS}
    pmat = {}
    for region in REGIONS:
        sub = sm[sm.region == region].rename(columns={"sigma_small_world": "sigma"})
        if sub["sigma"].notna().sum() <= 20:
            continue
        pmap = lmm_emotion_contrasts(sub, "sigma", ref="neutral")
        pmat[region] = pmap
        for e in EMOS:
            p = pmap.get(e, np.nan)
            if e != "neutral" and np.isfinite(p) and p < 0.05:
                sig[e] += 1

    posneg = (M[POS_E].mean(1) - M[NEG_E].mean(1)).reindex(REGIONS)
    modrange = (M.max(1) - M.min(1)).reindex(REGIONS)
    emo_order = sorted([e for e in EMOS if e != "neutral"], key=lambda e: sig[e], reverse=True)

    fig = plt.figure(figsize=(W2 * 1.0, W2 * 0.85))
    rounded_card(fig, 0.015, 0.56, 0.97, 0.40)
    rounded_card(fig, 0.015, 0.02, 0.97, 0.50)

    gs = GridSpec(2, 2, figure=fig, hspace=0.85, wspace=0.45, top=0.90, bottom=0.07,
                  height_ratios=[1.0, 1.15], width_ratios=[1.35, 1.0])

    # ---- panel a: sigma heatmap (full width, top) -------------------------
    axA = fig.add_subplot(gs[0, :])
    im = fancy_heatmap(axA, Zsc.values, RLAB, EMOS, SIGMA_CMAP, -1.8, 1.8, fontsize=6.4)
    axA.set_title("MTF (q = 30) — regional small-world coefficient σ", pad=8, fontsize=8.4, fontweight="bold")
    panel_tag(axA, "a")
    for ri, region in enumerate(REGIONS):
        pmap = pmat.get(region, {})
        for ci, e in enumerate(EMOS):
            p = pmap.get(e, np.nan)
            if e != "neutral" and stars(p):
                txt = axA.text(ci, ri, stars(p), ha="center", va="center", fontsize=6.8,
                                color="white", fontweight="bold", zorder=10)
                txt.set_path_effects([withStroke(linewidth=1.6, foreground="#00000055")])
    cb = fig.colorbar(im, ax=axA, fraction=0.028, pad=0.015)
    cb.ax.tick_params(labelsize=5.6); cb.set_label("row z-score", fontsize=6.2, labelpad=3)
    cb.outline.set_visible(False)

    # ---- panel b: combined summary (regions + emotions, twin x-axis) -----
    axB = fig.add_subplot(gs[1, 0])
    n_reg, n_emo = len(REGIONS), len(emo_order)
    gap = 1.2
    y_reg = np.arange(n_reg)[::-1] + n_emo + gap          # regions on top
    y_emo = np.arange(n_emo)[::-1]                         # emotions below

    w = 0.36
    reg_colors = [PALETTE[r] for r in REGIONS]
    axB.barh(y_reg + w / 2, posneg.values, height=w, color=reg_colors, edgecolor="white",
             linewidth=0.5, label="pos − neg σ", alpha=0.95)
    axB.barh(y_reg - w / 2, modrange.values, height=w, color=reg_colors, edgecolor="white",
             linewidth=0.5, alpha=0.55, hatch="////", label="σ range (mod.)")
    for yi, v in zip(y_reg + w / 2, posneg.values):
        axB.text(v + (0.004 if v >= 0 else -0.004), yi, f"{v:+.2f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=5.0, color=INK)
    for yi, v in zip(y_reg - w / 2, modrange.values):
        axB.text(v + 0.004, yi, f"{v:.2f}", va="center", ha="left", fontsize=5.0, color="0.35")

    axB2 = axB.twiny()
    emo_colors = [EMO_PALETTE[e] for e in emo_order]
    axB2.barh(y_emo, [sig[e] for e in emo_order], height=0.62, color=emo_colors,
              edgecolor="white", linewidth=0.5, alpha=0.95)
    for yi, e in zip(y_emo, emo_order):
        axB2.text(sig[e] + 0.08, yi, str(sig[e]), va="center", fontsize=5.2, color=INK)

    axB.axhline(n_emo + gap / 2, color="0.75", lw=0.7, ls=":")
    yticks = list(y_reg) + list(y_emo)
    ylabels = RLAB + emo_order
    axB.set_yticks(yticks); axB.set_yticklabels(ylabels, fontsize=6.0)
    for lbl, region in zip(axB.get_yticklabels()[:n_reg], REGIONS):
        lbl.set_color(PALETTE[region]); lbl.set_fontweight("bold")
    for lbl, e in zip(axB.get_yticklabels()[n_reg:], emo_order):
        lbl.set_color(EMO_PALETTE[e])
    axB.set_ylim(-1, n_reg + n_emo + gap + 0.5)
    axB.set_xlabel("σ (regions: pos−neg  ·  modulation range)", fontsize=5.8)
    axB2.set_xlabel("# significant regions vs. neutral (of 5)", fontsize=5.8)
    axB2.set_xlim(0, 5.8); axB.tick_params(labelsize=5.6); axB2.tick_params(labelsize=5.6)
    axB.legend(fontsize=4.6, loc="lower right", frameon=False)
    for sp in ["top", "right", "left"]:
        axB.spines[sp].set_visible(False)
    axB2.spines["right"].set_visible(False); axB2.spines["left"].set_visible(False)
    axB.tick_params(left=False)
    axB.set_title("Regional σ summary + emotion deviation from neutral", pad=22, fontsize=7.6, fontweight="bold")
    panel_tag(axB, "b")

    # ---- panel c: raw regional trends -------------------------------------
    axC = fig.add_subplot(gs[1, 1])
    for e in EMOS:
        axC.plot(range(len(REGIONS)), M[e].values, marker="o", ms=3.6, lw=1.4,
                 color=EMO_PALETTE[e], label=e, alpha=0.9,
                 path_effects=[withStroke(linewidth=2.4, foreground="white")])
    axC.set_xticks(range(len(REGIONS))); axC.set_xticklabels(RLAB, fontsize=6, rotation=15, ha="right")
    axC.set_ylabel("mean σ", fontsize=6.4)
    axC.set_title("Raw regional σ by emotion  [ ~ Fig. 4f ]", pad=8, fontsize=7.6, fontweight="bold")
    axC.legend(fontsize=4.2, ncol=1, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    for sp in ["top", "right"]:
        axC.spines[sp].set_visible(False)
    panel_tag(axC, "c")

    fig.text(0.5, 0.975, "Regional MTF network topology (q = 30)", ha="center", fontsize=12.5,
              fontweight="bold", color=INK)
    fig.text(0.5, 0.955, "small-world modulation  ·  emotion-specific deviation from neutral",
              ha="center", fontsize=7, color="0.4")

    save(fig, FIG_DIR / "part4" / "Figure_Part4_regional_MTF_fancy_v2")
    plt.close(fig)
    print("[fancy MTF-only v2] saved Figure_Part4_regional_MTF_fancy_v2.pdf/.png")


if __name__ == "__main__":
    main()
