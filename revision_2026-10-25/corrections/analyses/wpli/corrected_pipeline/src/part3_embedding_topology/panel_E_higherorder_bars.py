"""Panel E — higher-order topology (H1 total persistence) by emotion.
Title shows the omnibus LMM; bars carry clustered-OLS stars versus neutral."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from _panels import standalone, stars
from lib.config import EMOS, EMO_COLORS
from lib.stats import clustered_ols_emotion_contrasts

def draw(ax, D, tag, metric="H1_total", ref="neutral"):
    tda = D["tda"]; lmm = D["lmm"].set_index("measure")
    means = [tda[tda.emotion==e][metric].mean() for e in EMOS]
    sems  = [tda[tda.emotion==e][metric].sem()  for e in EMOS]
    ax.bar(range(9), means, yerr=sems, color=[EMO_COLORS[e] for e in EMOS],
           edgecolor="k", linewidth=0.4, capsize=2, error_kw=dict(lw=0.6))
    # per-emotion stars vs reference
    pmap = clustered_ols_emotion_contrasts(tda, metric, ref=ref)
    ytop = max(m+s for m,s in zip(means,sems))
    for i,e in enumerate(EMOS):
        s = stars(pmap.get(e, np.nan)) if e != ref else ""
        if s and s != "n.s.":
            ax.text(i, means[i]+sems[i]+0.004, s, ha="center", va="bottom", fontsize=5)
    ax.set_xticks(range(9)); ax.set_xticklabels(EMOS, rotation=90, fontsize=5)
    ax.set_ylabel("H$_1$ total persistence")
    row = lmm.loc[metric]
    ax.set_title(f"Higher-order topology (H$_1$)\nLMM χ²({int(row['df'])})={row['stat']:.1f}, p={row['p']:.4f} {stars(row['p'])}")
    ax.text(0.02, 0.97, f"stars: vs {ref}", transform=ax.transAxes, fontsize=4.4, va="top", color="0.4")
    ax.set_ylim(min(means)-max(sems)*3, ytop+0.018)

def main(tag="spearman"): standalone(draw, "panelE_h1", tag)
if __name__ == "__main__": main()
