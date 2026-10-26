"""
run_fig4_panel_f_combined.py — single combined bar plot for Figure 4
panel F: nodal-strength range across emotions, mean wPLI, and their
synchrony-topology coupling strength, per region -- three grouped bars
per region, each a different color, with significance stars and p-values
annotated directly on the plot (Nature-style).

Statistics computed here (all using the same LMM / participant-clustered
machinery used throughout the rest of the pipeline, lib.stats):
  - nodal strength: omnibus emotion effect per region (lmm_emotion_lrt on
    region_trial_nodal_strength.csv)
  - wPLI: omnibus emotion effect per region (lmm_emotion_lrt on
    region_trial_synchrony.csv, filtered to that region)
  - coupling: pooled synchrony-topology coupling p-value per region,
    already computed in run_region_synchrony_coupling.py
    (TableS_region_synchrony_topology_coupling.csv, wpli row)

Values are min-max normalized across regions for visual comparability
(three different units/scales on one axis), but every bar is labeled
with its real, unnormalized value plus significance stars and the exact
p-value, so no information is lost to the normalization.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, FIG_DIR, EMOS, set_nature_style, save, W2
from lib.stats import lmm_emotion_lrt
from lib.regions import REGIONS

RLAB = [r.capitalize() for r in REGIONS]
INK = "#1b1f2b"
C_STRENGTH = "#4C72B0"   # nodal-strength range (deep blue)
C_WPLI = "#8172B2"       # mean wPLI (purple)
C_COUPLE = "#55A868"     # coupling strength (deep green)
C_MUTED = "#c9ccd1"      # non-significant bars (neutral gray, not a 4th hue)


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else "n.s."


def pfmt(p):
    if not np.isfinite(p):
        return "p = n/a"
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"


def main():
    set_nature_style()
    tab_dir = Path(CACHE).parent / "supplementary_tables"; tab_dir.mkdir(exist_ok=True, parents=True)

    # ---- 1. nodal-strength range + omnibus significance per region --------
    nodal = pd.read_csv(CACHE / "region_trial_nodal_strength.csv")
    strength_range, strength_p = {}, {}
    for region in REGIONS:
        col = f"strength_{region}"
        by_emo = nodal.groupby(["subject", "emotion"])[col].mean().reset_index()
        heat = by_emo.groupby("emotion")[col].mean().reindex(EMOS)
        strength_range[region] = float(heat.max() - heat.min())
        res = lmm_emotion_lrt(by_emo.rename(columns={col: "y"}), "y")
        strength_p[region] = res["p"]

    # ---- 2. mean wPLI + omnibus significance per region --------------------
    sync = pd.read_csv(CACHE / "region_trial_synchrony.csv")
    wpli_mean, wpli_p = {}, {}
    for region in REGIONS:
        sub = sync[sync.region == region][["subject", "emotion", "wpli"]].rename(columns={"wpli": "y"})
        wpli_mean[region] = float(sub["y"].mean())
        res = lmm_emotion_lrt(sub, "y")
        wpli_p[region] = res["p"]

    # ---- 3. coupling (wPLI ~ nodal strength) pooled p, already computed ---
    coupling_tab = pd.read_csv(tab_dir / "TableS_region_synchrony_topology_coupling.csv")
    coupling_tab = coupling_tab[coupling_tab.sync_metric == "wpli"].set_index("region")
    coupling_beta = coupling_tab["pooled_beta"].reindex(REGIONS)
    coupling_p = coupling_tab["pooled_p"].reindex(REGIONS)

    # ---- assemble table -----------------------------------------------------
    out_rows = []
    for region in REGIONS:
        out_rows.append(dict(
            region=region,
            nodal_strength_range=round(strength_range[region], 4), nodal_strength_p=strength_p[region],
            mean_wpli=round(wpli_mean[region], 4), wpli_p=wpli_p[region],
            coupling_beta=round(coupling_beta[region], 4), coupling_p=coupling_p[region],
        ))
    out_tab = pd.DataFrame(out_rows)
    out_tab.to_csv(tab_dir / "Table_fig4_panel_f_combined.csv", index=False)
    print(out_tab.to_string(index=False))

    # ---- figure: THREE small-multiple panels, one per metric ---------------
    # Rationale for this redesign (previous version normalized all three
    # metrics 0-1 ACROSS regions for a single shared axis; that forced
    # whichever region had the smallest raw value down to a literal
    # zero-length, invisible bar, and cluttered the plot with "n.s." text
    # labels). Each metric now gets its own small panel with its own
    # natural scale, so every bar has visible, honest length. Significance
    # is now shown by COLOR (full color = significant, muted gray = not)
    # rather than a text label, which is quieter and reads faster.
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(W2 * 1.05, W2 * 0.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.15, top=0.80, bottom=0.16, left=0.14, right=0.98)
    y = np.arange(len(REGIONS))[::-1]

    panels = [
        ("Nodal-strength range\n(Δ across emotions)", strength_range, strength_p, C_STRENGTH, "{:.2f}"),
        ("Mean wPLI", wpli_mean, wpli_p, C_WPLI, "{:.3f}"),
        ("|Coupling β|\n(wPLI ~ nodal strength)", {r: abs(coupling_beta[r]) for r in REGIONS}, coupling_p, C_COUPLE, "β={:.2f}"),
    ]
    for k, (title, vals, pvals, color, fmt) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        v = np.array([vals[r] for r in REGIONS])
        sig = np.array([np.isfinite(pvals[r]) and pvals[r] < 0.05 for r in REGIONS])
        bar_colors = [color if s else C_MUTED for s in sig]
        ax.barh(y, v, height=0.62, color=bar_colors, edgecolor="white", linewidth=0.6)
        xmax = v.max() * 1.42
        for yi, region, val in zip(y, REGIONS, v):
            p = pvals[region]
            label = f"{fmt.format(val)}  {stars(p)}" if np.isfinite(p) and p < 0.05 else f"{fmt.format(val)}  ({pfmt(p)})"
            ax.text(val + xmax * 0.03, yi, label, va="center", fontsize=5.6, color=INK)
        ax.set_xlim(0, xmax)
        if k == 0:
            ax.set_yticks(y); ax.set_yticklabels(RLAB, fontsize=7.4)
        else:
            ax.set_yticks(y); ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_title(title, fontsize=6.8, pad=8)
        for sp in ["top", "right", "bottom"]:
            ax.spines[sp].set_visible(False)

    # shared legend: significant vs. not, instead of per-bar "n.s." text
    sig_patch = mpatches.Patch(facecolor="0.4", edgecolor="white", label="p < 0.05")
    ns_patch = mpatches.Patch(facecolor=C_MUTED, edgecolor="white", label="not significant")
    fig.legend(handles=[sig_patch, ns_patch], loc="upper right", bbox_to_anchor=(0.99, 0.99),
               fontsize=5.6, frameon=False, ncol=2)

    fig.suptitle("Regional modulation (Spearman): nodal strength, synchrony (wPLI), and their coupling",
                 x=0.14, ha="left", fontsize=8.4, y=0.965)
    fig.text(-0.005, 0.99, "F", fontsize=22, fontweight="bold", va="top", ha="left", color=INK,
              transform=fig.transFigure)

    save(fig, FIG_DIR / "part4" / "Figure4_panel_f_combined")
    plt.close(fig)
    print("[Fig 4, panel F combined] saved Figure4_panel_f_combined.pdf/.png")


if __name__ == "__main__":
    main()
