"""
run_fingerprint_splithalf_final.py
--------------------------------------------------------------------------
FINAL version of the Fig. 1a-c fingerprint split-half reliability check,
matching the adopted manuscript decision: Figure 1 and Fig. S2 (chords)
are reported using q=30 (node-count-matched) MTF/QG, while everything
else in the paper stays at the original q=13. This script therefore
reruns the split-half reliability + disattenuation analysis on
Spearman + MTF(q=30) + QG(q=30), so the reliability numbers reported in
the appendix match the data actually shown in Figure 1.

See part1_topology/run_fingerprint_splithalf.py for the full method
description (split-half definition, Spearman-Brown correction,
disattenuation formula) -- identical here, only the MTF/QG input data
differ (trial_metrics_mtf_q30.csv / trial_metrics_qg_q30.csv instead of
trial_metrics_mtf.csv / trial_metrics_qg.csv).

Outputs (kept as clearly-labeled "final" files, separate from the
earlier q=13-only exploratory version):
  figures/part1/FigS_fingerprint_splithalf_FINAL_q30.png/.pdf
  supplementary_tables/TableS_fingerprint_reliability_FINAL_q30.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, FIG_DIR, EMOS, METRICS, set_nature_style, save, W2, panel_letter

TAGS = {"spearman": "Spearman", "mtf_q30": "MTF (q=30)", "qg_q30": "QG (q=30)"}
N_SPLITS = 500
SEED = 0


def _load(tag):
    f = CACHE / f"trial_metrics_{tag}.csv"
    return pd.read_csv(f) if f.exists() else None


def _fingerprint(df, subs=None):
    d = df if subs is None else df[df["subject"].isin(subs)]
    subj = d.groupby(["subject", "emotion"])[METRICS].mean().reset_index()
    heat = subj.groupby("emotion")[METRICS].mean().reindex(EMOS)
    z = (heat - heat.mean(0)) / heat.std(0)
    return z


def splithalf_reliability(df, n_splits=N_SPLITS, seed=SEED):
    subs = np.array(sorted(df["subject"].unique()))
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_splits):
        rng.shuffle(subs)
        h = len(subs) // 2
        A = _fingerprint(df, subs[:h]).values.flatten()
        B = _fingerprint(df, subs[h:]).values.flatten()
        ok = np.isfinite(A) & np.isfinite(B)
        if ok.sum() > 3 and np.std(A[ok]) > 0 and np.std(B[ok]) > 0:
            rs.append(np.corrcoef(A[ok], B[ok])[0, 1])
    return np.array(rs)


def spearman_brown(r_half):
    return (2 * r_half) / (1 + r_half) if (1 + r_half) != 0 else np.nan


def disattenuate(r_obs, rel_x, rel_y):
    denom = np.sqrt(max(rel_x, 1e-9) * max(rel_y, 1e-9))
    return r_obs / denom if denom > 0 else np.nan


def main():
    dfs = {tag: _load(tag) for tag in TAGS}
    if any(v is None for v in dfs.values()):
        print(f"[fingerprint-splithalf-final] missing data for: {[t for t in TAGS if dfs[t] is None]}")
        return

    boot = {tag: splithalf_reliability(dfs[tag]) for tag in TAGS}
    med = {tag: float(np.median(boot[tag])) for tag in TAGS}
    ci = {tag: (float(np.percentile(boot[tag], 5)), float(np.percentile(boot[tag], 95))) for tag in TAGS}
    rel_full = {tag: spearman_brown(med[tag]) for tag in TAGS}

    full_fp = {tag: _fingerprint(dfs[tag]) for tag in TAGS}
    pairs = [("spearman", "mtf_q30"), ("spearman", "qg_q30"), ("mtf_q30", "qg_q30")]
    cross_obs, cross_disatt = {}, {}
    for a, b in pairs:
        A = full_fp[a].values.flatten(); B = full_fp[b].values.flatten()
        ok = np.isfinite(A) & np.isfinite(B)
        cross_obs[(a, b)] = float(np.corrcoef(A[ok], B[ok])[0, 1])
        cross_disatt[(a, b)] = disattenuate(cross_obs[(a, b)], rel_full[a], rel_full[b])

    tab_dir = Path(CACHE).parent / "supplementary_tables"; tab_dir.mkdir(exist_ok=True, parents=True)
    rel_rows = [dict(representation=TAGS[tag], splithalf_median_r=round(med[tag], 3),
                      splithalf_5th=round(ci[tag][0], 3), splithalf_95th=round(ci[tag][1], 3),
                      spearman_brown_full_sample_reliability=round(rel_full[tag], 3)) for tag in TAGS]
    cross_rows = [dict(pair=f"{TAGS[a]}-{TAGS[b]}", r_observed=round(cross_obs[(a, b)], 3),
                        r_disattenuated=round(cross_disatt[(a, b)], 3)) for a, b in pairs]
    rel_tab = pd.DataFrame(rel_rows); cross_tab = pd.DataFrame(cross_rows)
    with open(tab_dir / "TableS_fingerprint_reliability_FINAL_q30.csv", "w") as f:
        f.write("# FINAL (q=30, matching adopted Figure 1/S2): per-representation split-half reliability\n")
        rel_tab.to_csv(f, index=False)
        f.write("\n# Cross-representation correlation: observed vs. disattenuated (Spearman, 1904)\n")
        cross_tab.to_csv(f, index=False)
    print(rel_tab.to_string(index=False)); print(); print(cross_tab.to_string(index=False))

    set_nature_style()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(W2, W2 * 0.34))
    gs = GridSpec(1, 4, figure=fig, wspace=0.55, top=0.82, bottom=0.22, width_ratios=[1, 1, 1, 1.25])
    colors = {"spearman": "#2471a3", "mtf_q30": "#e67e22", "qg_q30": "#16a085"}
    for k, tag in enumerate(TAGS):
        ax = fig.add_subplot(gs[0, k])
        ax.hist(boot[tag], bins=24, color=colors[tag], edgecolor="white", linewidth=0.3)
        ax.axvline(med[tag], color="#c0392b", lw=1.2, ls="--")
        ax.set_title(f"{TAGS[tag]}\nmedian ρ={med[tag]:.2f}", fontsize=6.5, pad=4)
        ax.set_xlabel("split-half r", fontsize=6); ax.set_ylabel("count", fontsize=6)
        ax.tick_params(labelsize=5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        panel_letter(ax, "abc"[k], dx=-0.22)

    axD = fig.add_subplot(gs[0, 3])
    labels = [f"{TAGS[a]}\nvs {TAGS[b]}" for a, b in pairs]
    x = np.arange(len(pairs)); w = 0.35
    obs_vals = [cross_obs[p] for p in pairs]; dis_vals = [cross_disatt[p] for p in pairs]
    axD.bar(x - w / 2, obs_vals, width=w, color="#7f8c8d", edgecolor="k", linewidth=0.3, label="observed")
    axD.bar(x + w / 2, dis_vals, width=w, color="#c0392b", edgecolor="k", linewidth=0.3, label="disattenuated")
    axD.axhline(0, color="0.3", lw=0.6)
    axD.set_xticks(x); axD.set_xticklabels(labels, fontsize=4.8, rotation=15, ha="right")
    axD.set_ylabel("cross-representation r", fontsize=6); axD.tick_params(labelsize=5)
    axD.legend(fontsize=5, frameon=False, loc="best")
    axD.set_title("Observed vs. reliability-corrected", fontsize=6.5, pad=4)
    for sp in ["top", "right"]:
        axD.spines[sp].set_visible(False)
    panel_letter(axD, "d", dx=-0.20)

    fig.suptitle("FINAL: split-half reliability of Fig. 1a-c fingerprints (q=30 MTF/QG, matching adopted figure)",
                 fontsize=7.2, y=0.98)
    save(fig, FIG_DIR / "part1" / "FigS_fingerprint_splithalf_FINAL_q30"); plt.close(fig)
    print(f"\n[fingerprint-splithalf-final] wrote table + figure")


if __name__ == "__main__":
    main()
