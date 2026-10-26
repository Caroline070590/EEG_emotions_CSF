"""
run_part4.py — Figure 4: REGIONAL small-world topology across emotions (MTF & QG).
a,b  MTF/QG regional σ (region x emotion, row z); cells carry clustered-OLS
     contrast vs neutral (within region).
c    positive-negative σ contrast (region x method) — distinct diverging map.
d    emotion modulation strength per region (σ range).
e    regional deviation from neutral per emotion (count of significant vs-neutral
     contrasts across regions x methods) — sorted summary.
No baked-in figure title.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")
from lib.config import (CACHE, FIG_DIR, EMOS, VAL, EMO_COLORS, set_nature_style, save, W2,
                        panel_letter, nice_heatmap, slim_colorbar)
from lib.regions import REGIONS
from lib.stats import clustered_ols_emotion_contrasts

METHODS = ["MTF", "QG"]
POS = [e for e in EMOS if VAL[e]=="pos"]; NEG = [e for e in EMOS if VAL[e]=="neg"]
RLAB = [r.capitalize() for r in REGIONS]
def stars(p): return "***" if p<1e-3 else "**" if p<1e-2 else "*" if p<.05 else ""

def main():
    set_nature_style()
    f = CACHE/"region_trial_metrics.csv"
    if not f.exists(): print("  [skip] region_trial_metrics.csv missing"); return
    d = pd.read_csv(f)
    sm = d.groupby(["subject","emotion","region","method"])["sigma_small_world"].mean().reset_index()

    def region_emotion(method):
        g = sm[sm.method==method]
        M = g.groupby(["region","emotion"])["sigma_small_world"].mean().reindex(
            pd.MultiIndex.from_product([REGIONS,EMOS],names=["region","emotion"])).unstack()
        return M.reindex(index=REGIONS, columns=EMOS)
    def rowz(M): return M.sub(M.mean(1),axis=0).div(M.std(1).replace(0,np.nan),axis=0)

    fig = plt.figure(figsize=(W2, W2*0.6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.95, wspace=0.6, top=0.93, bottom=0.12,
                  height_ratios=[1.1, 1.0])

    Ms = {}; sig = {e: 0 for e in EMOS}
    for k, method in enumerate(METHODS):
        ax = fig.add_subplot(gs[0, k]); M = region_emotion(method); Ms[method] = M; Z = rowz(M)
        im = nice_heatmap(ax, Z.values, RLAB, EMOS, cmap="magma", vmin=-1.8, vmax=1.8,
                          aspect="auto", fontsize=5.4)
        ax.set_title(f"{method} — regional σ (row z)", pad=6); panel_letter(ax, "ab"[k])
        for ri, region in enumerate(REGIONS):
            sub = sm[(sm.method==method)&(sm.region==region)].rename(columns={"sigma_small_world":"sigma"})
            if sub["sigma"].notna().sum() <= 20: continue
            pmap = clustered_ols_emotion_contrasts(sub, "sigma", ref="neutral")
            for ci, e in enumerate(EMOS):
                p = pmap.get(e, np.nan)
                if e != "neutral" and stars(p):
                    ax.text(ci, ri, stars(p), ha="center", va="center", fontsize=4.6,
                            color="white" if Z.values[ri, ci] < 0.4 else "#111111")
                if e != "neutral" and np.isfinite(p) and p < 0.05: sig[e] += 1
        if k == 1: slim_colorbar(fig, im, ax, "row z")

    # c: positive - negative contrast (distinct diverging map)
    axC = fig.add_subplot(gs[1, 0])
    C = np.zeros((len(REGIONS), len(METHODS)))
    for mi, method in enumerate(METHODS):
        for ri, region in enumerate(REGIONS):
            gg = sm[(sm.method==method)&(sm.region==region)]
            C[ri,mi] = gg[gg.emotion.isin(POS)]["sigma_small_world"].mean() - gg[gg.emotion.isin(NEG)]["sigma_small_world"].mean()
    vmax = np.nanmax(np.abs(C))
    im2 = nice_heatmap(axC, C, RLAB, METHODS, cmap="PuOr_r", vmin=-vmax, vmax=vmax,
                       annot=True, fmt="{:+.2f}", aspect="auto", fontsize=5.6, annot_size=5.0)
    axC.set_title("Positive − negative σ", pad=6); panel_letter(axC, "c")
    slim_colorbar(fig, im2, axC, "Δσ")

    # d: modulation strength per region
    axD = fig.add_subplot(gs[1, 1]); width = 0.38
    for mi, method in enumerate(METHODS):
        rng = (Ms[method].max(1) - Ms[method].min(1)).reindex(REGIONS)
        axD.barh(np.arange(len(REGIONS)) + (mi-0.5)*width, rng.values, height=width,
                 color="#34495e" if method=="MTF" else "#16a085", label=method, edgecolor="k", linewidth=0.3)
    axD.set_yticks(range(len(REGIONS))); axD.set_yticklabels(RLAB, fontsize=5.4)
    axD.set_xlabel("σ range across emotions"); axD.set_title("Emotion modulation", pad=6)
    axD.legend(fontsize=5, loc="lower right"); axD.invert_yaxis()
    for sp in ["top","right"]: axD.spines[sp].set_visible(False); 
    panel_letter(axD, "d")

    # e: regional deviation from neutral per emotion (summary)
    axE = fig.add_subplot(gs[1, 2])
    order = sorted([e for e in EMOS if e != "neutral"], key=lambda e: sig[e], reverse=True)
    axE.barh(np.arange(len(order)), [sig[e] for e in order],
             color=[EMO_COLORS[e] for e in order], edgecolor="k", linewidth=0.3)
    axE.set_yticks(range(len(order))); axE.set_yticklabels(order, fontsize=5.4); axE.invert_yaxis()
    axE.set_xlabel("# sig vs-neutral (of 10)"); axE.set_title("Regional deviation\nfrom neutral", pad=4)
    for sp in ["top","right"]: axE.spines[sp].set_visible(False)
    panel_letter(axE, "e")

    save(fig, FIG_DIR/"part4"/"Figure_Part4_regional_topology"); plt.close(fig)

if __name__ == "__main__":
    main()
