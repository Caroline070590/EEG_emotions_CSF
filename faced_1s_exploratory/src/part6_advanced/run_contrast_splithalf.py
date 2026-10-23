"""Split-half directional stability of the nine FDR-significant 1-s synchronization contrasts
(regenerates Supplementary Figure S25; the original generator could not be recovered).

Estimator (as in Supplementary Table S1): each measure is z-scored within participant (ddof = 1) and the
emotion-versus-neutral effect is the difference of mean z between the emotion and neutral trials.
Procedure: 300 random partitions of the 123 participants into complementary halves (61 / 62). In each
partition the effect is re-estimated in both halves. Because the halves are complementary and the
estimator is a participant average, 61*b_A + 62*b_B = 123*b_full in every partition, so the two half
estimates are perfectly negatively related; their correlation carries no information about reliability
and is not reported. Reported instead: the spread of the half estimates and the percentage of partitions
in which BOTH halves keep the full-sample sign.

usage (from faced_1s_exploratory/src): python part6_advanced/run_contrast_splithalf.py ../derived/trial_scalar_synchronization.csv
Outputs: ../tables/contrast_splithalf.csv and ../figures/FigS25_contrast_splithalf.{pdf,png}
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; FIGS, TABS = ROOT / "figures", ROOT / "tables"
CONTRASTS = [("PLV", "disgust"), ("PLV", "anger"), ("PLV", "fear"), ("AEC", "disgust"), ("MaxXC", "disgust"),
             ("MaxXC", "anger"), ("Lag", "fear"), ("Lag", "disgust"), ("Lag", "joy")]
N_PART, SEED = 300, 2026

def participant_differences(d, metric, emo):
    z = d.groupby("subject_id")[metric].transform(lambda x: (x - x.mean()) / x.std(ddof=1))
    t = pd.DataFrame({"z": z, "s": d.subject_id, "e": d.emotion}).groupby(["s", "e"]).z.mean().unstack()
    return (t[emo] - t["neutral"]).dropna()

if __name__ == "__main__":
    d = pd.read_csv(sys.argv[1]); FIGS.mkdir(exist_ok=True); TABS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED); subjects = np.array(sorted(d.subject_id.unique()))
    parts = [rng.permutation(subjects) for _ in range(N_PART)]
    rows, halves = [], {}
    for metric, emo in CONTRASTS:
        pdiff = participant_differences(d, metric, emo); full = pdiff.mean()
        A = np.array([pdiff.loc[pdiff.index.isin(p[:len(p) // 2])].mean() for p in parts])
        B = np.array([pdiff.loc[pdiff.index.isin(p[len(p) // 2:])].mean() for p in parts])
        both = (np.sign(A) == np.sign(full)) & (np.sign(B) == np.sign(full))
        allh = np.r_[A, B]; halves[(metric, emo)] = allh
        rows.append({"metric": metric, "emotion": emo, "full_sample_effect": full,
                     "half_estimate_mean": allh.mean(), "half_estimate_sd": allh.std(ddof=1),
                     "half_estimate_2.5pct": np.percentile(allh, 2.5), "half_estimate_97.5pct": np.percentile(allh, 97.5),
                     "directional_replication_pct": 100 * both.mean(), "n_partitions": N_PART})
    T = pd.DataFrame(rows); T.to_csv(TABS / "contrast_splithalf.csv", index=False)
    print(T.round(3).to_string())

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif"})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.35})
    labels = [f"{m} {e}" for m, e in CONTRASTS]
    for i, (k, v) in enumerate(halves.items()):
        vp = ax.violinplot(v, positions=[i], widths=0.75, showextrema=False)
        for b in vp["bodies"]: b.set_facecolor("#9fb3c8"); b.set_alpha(0.9); b.set_edgecolor("none")
        ax.plot([i - 0.32, i + 0.32], [T.full_sample_effect[i]] * 2, color="#c0392b", lw=1.6)
    ax.axhline(0, color="#444", lw=0.7, ls="--")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Effect vs neutral in each half (z units)")
    ax.set_title("a  Half-sample estimates (300 complementary splits)", loc="left", fontsize=8)
    ax.text(0.01, 0.02, "red line: full-sample effect", transform=ax.transAxes, fontsize=7, color="#c0392b")
    ax2.barh(range(len(labels)), T.directional_replication_pct, color="#2b6cb0")
    for i, v in enumerate(T.directional_replication_pct): ax2.text(v - 1, i, f"{v:.1f}%", va="center", ha="right", color="white", fontsize=7)
    ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels); ax2.invert_yaxis(); ax2.set_xlim(0, 100)
    ax2.set_xlabel("Partitions with both halves\nkeeping the full-sample sign (%)")
    ax2.set_title("b  Directional replication", loc="left", fontsize=8)
    for ext in ("pdf", "png"): fig.savefig(FIGS / f"FigS25_contrast_splithalf.{ext}", dpi=300, bbox_inches="tight")
