"""
run_fig4_panel_f_spearman.py — replacement for main Figure 4 panel f
("Regional modulation"), using SPEARMAN networks instead of MTF/QG.

IMPORTANT METHODOLOGICAL NOTE: the original panel f's "regional
small-world sigma" is meaningful for MTF/QG because those are computed
PER CHANNEL (each channel gives its own q x q state-transition graph,
then channel-level metrics are averaged within a region) -- the small-world
computation itself never operates on a tiny subgraph.

A naive Spearman analog -- building a channel-correlation subgraph
restricted to just one region's channels (e.g. 2 channels for temporal,
7 for frontal) and computing small-world sigma on THAT small subgraph --
was tried first and is NOT usable: at 10% density, most regions have too
few nodes/edges to support a stable degree-preserving-swap null model,
and the resulting sigma is NaN for 4 of 5 regions (only parietal, the
largest region at 9 channels, gave any valid values, and even those were
47% missing).

This version instead reuses the region-averaging approach already used
successfully in Fig. S12 (part5_enrichment/run_network_enrichment.py):
nodal strength (sum of |Spearman correlation| to all other channels) is
computed from the FULL 30-channel network per emotion, and channels are
then grouped into regions afterward. This avoids the tiny-subgraph
problem entirely, because the underlying network computation always
uses all 30 channels -- only the region-averaging step for display is
restricted.

The plotted quantity is therefore "range of region-averaged nodal
strength across emotions," a valid and directly comparable substitute
for the MTF/QG panel's "range of regional sigma across emotions."
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, FIG_DIR, EMOS, set_nature_style, save, W1
from lib.regions import REGIONS, REGION_MAP
from lib.representations import vec_to_mat
from lib.config import CH

RLAB = [r.capitalize() for r in REGIONS]


def main():
    set_nature_style()
    if not (CACHE / "agg_spearman.npy").exists():
        print("  [skip] agg_spearman.npy missing"); return
    agg = np.load(CACHE / "agg_spearman.npy"); am = pd.read_csv(CACHE / "agg_meta_spearman.csv")

    # nodal strength per channel x emotion, from the FULL 30-channel network
    mats = {e: vec_to_mat(agg[(am["emotion"] == e).values].mean(0), len(CH)) for e in EMOS}
    strength = pd.DataFrame({e: np.abs(mats[e]).sum(1) for e in EMOS}, index=CH)  # 30 x 9

    # average within region, then range across emotions
    region_strength = pd.DataFrame({
        region: strength.loc[[c for c in chans if c in CH]].mean(0)
        for region, chans in REGION_MAP.items()
    }).T.reindex(REGIONS)  # regions x emotions
    rng = region_strength.max(axis=1) - region_strength.min(axis=1)

    fig, ax = plt.subplots(figsize=(W1 * 1.3, W1))
    ax.barh(np.arange(len(REGIONS)), rng.values, height=0.55, color="#2c5f7c",
            edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(REGIONS))); ax.set_yticklabels(RLAB)
    ax.set_xlabel("nodal-strength range across emotions")
    ax.set_title("Regional modulation (Spearman)", pad=8)
    ax.invert_yaxis()
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.text(-0.14, 1.08, "F", transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="top", ha="left")
    save(fig, FIG_DIR / "part4" / "Figure4_panel_f_spearman")
    plt.close(fig)

    rng_out = rng.round(4).reset_index(); rng_out.columns = ["region", "nodal_strength_range_spearman"]
    tab_dir = Path(CACHE).parent / "supplementary_tables"; tab_dir.mkdir(exist_ok=True, parents=True)
    rng_out.to_csv(tab_dir / "Table_fig4_panel_f_spearman.csv", index=False)
    print(rng_out.to_string(index=False))
    print("[Fig 4, panel f Spearman] saved Figure4_panel_f_spearman.pdf/.png")


if __name__ == "__main__":
    main()
