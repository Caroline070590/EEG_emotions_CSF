"""
run_chords_supp_q30.py — Fig. S2 (representation-similarity chords) regenerated
with q=30 (node-count-matched) MTF/QG. Reuses run_chords_supp.py's chord()
helper; only the input CSVs differ (mtf_q30/qg_q30 instead of mtf/qg).
Saved separately, does not overwrite the original q=13 Fig. S2.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, FIG_DIR, EMOS, EMO_COLORS, METRICS, set_nature_style, save, W2, panel_letter
from part1_topology.run_chords_supp import chord, _profile

METHODS_Q30 = {"spearman": "Spearman", "mtf_q30": "MTF (q=30)", "qg_q30": "QG (q=30)"}


def _load(tag):
    f = CACHE / f"trial_metrics_{tag}.csv"
    return pd.read_csv(f) if f.exists() else None


def main():
    set_nature_style()
    dfs = {tag: _load(tag) for tag in METHODS_Q30}
    if any(v is None for v in dfs.values()):
        print("  [skip] missing q30 metrics."); return
    corr = {tag: np.corrcoef(_profile(dfs[tag]).values) for tag in METHODS_Q30}
    fig = plt.figure(figsize=(W2, W2 * 0.32))
    gs = GridSpec(1, 4, figure=fig, wspace=0.12, top=0.82, bottom=0.16)
    for k, tag in enumerate(METHODS_Q30):
        ax = fig.add_subplot(gs[0, k])
        chord(ax, corr[tag], title=METHODS_Q30[tag]); panel_letter(ax, "abc"[k])
    axG = fig.add_subplot(gs[0, 3])
    consensus = np.mean([corr[t] for t in corr], axis=0)
    chord(axG, consensus, title="Consensus (mean)"); panel_letter(axG, "d")
    handles = [mp.Patch(facecolor=EMO_COLORS[e], edgecolor="k", lw=0.3, label=e) for e in EMOS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=9, fontsize=5.4, frameon=False, handlelength=1.0, columnspacing=0.9)
    fig.suptitle("Fig. S2 (q=30 robustness version)", fontsize=7.2, y=0.97)
    save(fig, FIG_DIR / "part1" / "Figure_S_emotion_similarity_chords_q30"); plt.close(fig)
    print("[Fig S2, q=30] saved Figure_S_emotion_similarity_chords_q30.pdf/.png")


if __name__ == "__main__":
    main()
