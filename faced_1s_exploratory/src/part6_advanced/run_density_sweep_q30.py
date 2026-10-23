"""Density-sweep robustness of the 1-s topology profiles, recomputed with the primary settings.

Replaces the earlier rapid check (Supplementary Figure S9), which used the older q=13 MTF/QG networks and
density-matched Erdos-Renyi references, so that its 10% column did not reproduce Figure 4. Here every
density uses exactly the Figure 4 construction: q=30 MTF/QG and 30-channel Spearman networks from
lib.representations, fixed proportional thresholding, and degree-preserving randomized references
(lib.metrics.metrics_from_matrix; N_RANDOMIZATIONS=20, REWIRES_PER_EDGE=5). Aggregation as in Figure 4d:
participant x emotion means, then emotion means; percentage change relative to neutral.

usage (from faced_1s_exploratory/src):
  python part6_advanced/run_density_sweep_q30.py DATA_ROOT [N_JOBS]
DATA_ROOT holds <emotion>/<Patient###>/trial_*.csv (the 1-s export). Outputs: cache/density_sweep_q30_trial_metrics.csv
../figures/FigS9_density_sweep.{pdf,png} and ../tables/density_sweep_*.csv.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed

CODE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(CODE))
from lib.config import EMOS, METRICS, METRIC_LABELS, N_RANDOMIZATIONS, set_nature_style
from lib.metrics import metrics_from_matrix
from lib.representations import (load_raw_trial_csv, parse_subject_trial, choose_q_campanharo,
                                 choose_k_from_autocorrelation, matrix_spearman_channels, matrix_markov_transition)

DENSITIES = [0.05, 0.10, 0.15, 0.20]
FIGS, TABS = CODE.parent / "figures", CODE.parent / "tables"; CACHE = CODE / "cache"

def one(path, emo):
    X = load_raw_trial_csv(path); subj, _ = parse_subject_trial(path)
    q = choose_q_campanharo(X.shape[0]); k = choose_k_from_autocorrelation(X)
    mats = {"Spearman": matrix_spearman_channels(X), "MTF": matrix_markov_transition(X, q=q, lag=1),
            "QG": matrix_markov_transition(X, q=q, lag=k)}
    rows = []
    for rep, A in mats.items():
        for d in DENSITIES:
            rows.append({"representation": rep, "density": d, "subject": subj, "emotion": emo,
                         "trial": Path(path).stem, "q": q, **metrics_from_matrix(A, density=d, n_rand=N_RANDOMIZATIONS)})
    return rows

def z_profile(subj_df):
    heat = subj_df.groupby("emotion")[METRICS].mean().reindex(EMOS)
    return (heat - heat.mean(0)) / heat.std(0)

def summarize(df):
    subj = df.groupby(["representation", "density", "subject", "emotion"])[METRICS].mean().reset_index()
    rows = []
    for rep in ["Spearman", "MTF", "QG"]:
        zref = z_profile(subj[(subj.representation == rep) & np.isclose(subj.density, 0.10)])
        for d in DENSITIES:
            zd = z_profile(subj[(subj.representation == rep) & np.isclose(subj.density, d)])
            rows.append(dict(representation=rep, density=d, metric="all_metrics_profile",
                             r_vs_10pct=float(np.corrcoef(zref.values.flatten(), zd.values.flatten())[0, 1])))
            for m in METRICS:
                rows.append(dict(representation=rep, density=d, metric=m, r_vs_10pct=float(np.corrcoef(zref[m].values, zd[m].values)[0, 1])))
    summ = pd.DataFrame(rows)
    means = subj.groupby(["representation", "density", "emotion"])[METRICS].mean().reset_index()
    rows = []
    for rep in ["Spearman", "MTF", "QG"]:
        for d in DENSITIES:
            dd = means[(means.representation == rep) & np.isclose(means.density, d)].set_index("emotion")
            neu = dd.loc["neutral", "sigma_small_world"]
            for emo in EMOS:
                rows.append(dict(representation=rep, density=d, emotion=emo, sigma=dd.loc[emo, "sigma_small_world"],
                                 delta_sigma_vs_neutral_pct=100 * (dd.loc[emo, "sigma_small_world"] - neu) / abs(neu)))
    return summ, pd.DataFrame(rows)

def plot(summ, delta):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    set_nature_style(); fig = plt.figure(figsize=(9.2, 7.0))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.9, 1.05, 1.05], hspace=.58, wspace=.55)
    ax = fig.add_subplot(gs[0, :])
    for rep in ["Spearman", "MTF", "QG"]:
        s = summ[(summ.representation == rep) & (summ.metric == "all_metrics_profile")].sort_values("density")
        ax.plot(s.density * 100, s.r_vs_10pct, marker="o", label=rep)
    ax.axvline(10, color="0.5", lw=.8, ls="--"); ax.set_ylim(-.1, 1.05)
    ax.set_xlabel("Proportional density (%)"); ax.set_ylabel("Correlation vs 10%")
    ax.set_title("a  Stability of emotion-by-metric topology profiles")
    ax.legend(frameon=False, ncol=3, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    vmax = float(np.nanmax(np.abs(delta[delta.emotion != "neutral"].delta_sigma_vs_neutral_pct))); vmax = max(5, min(vmax, 80))
    for j, rep in enumerate(["Spearman", "MTF", "QG"]):
        ax = fig.add_subplot(gs[1, j])
        dd = delta[delta.representation == rep].pivot(index="emotion", columns="density", values="delta_sigma_vs_neutral_pct").reindex(EMOS)[DENSITIES]
        im = ax.imshow(dd.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{'bcd'[j]}  {rep}: $\\Delta\\sigma$ vs neutral (%)")
        ax.set_xticks(range(len(DENSITIES))); ax.set_xticklabels([f"{int(d*100)}%" for d in DENSITIES])
        ax.set_yticks(range(len(EMOS))); ax.set_yticklabels(EMOS if j == 0 else [])
        for ii in range(dd.shape[0]):
            for jj in range(dd.shape[1]):
                v = dd.values[ii, jj]
                if np.isfinite(v): ax.text(jj, ii, f"{v:+.0f}", ha="center", va="center", fontsize=6, color="white" if abs(v) > .55 * vmax else "#111111")
    cax = fig.add_axes([.92, .39, .015, .22]); cb = fig.colorbar(im, cax=cax); cb.set_label("$\\Delta\\sigma$ vs neutral (%)")
    for j, rep in enumerate(["Spearman", "MTF", "QG"]):
        ax = fig.add_subplot(gs[2, j]); s = summ[(summ.representation == rep) & (summ.metric != "all_metrics_profile")]
        mat = s.pivot(index="metric", columns="density", values="r_vs_10pct").reindex(METRICS)[DENSITIES]
        ax.imshow(mat.values, cmap="viridis", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(f"{'efg'[j]}  {rep}: metric r vs 10%")
        ax.set_xticks(range(len(DENSITIES))); ax.set_xticklabels([f"{int(d*100)}%" for d in DENSITIES])
        ax.set_yticks(range(len(METRICS))); ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in METRICS] if j == 0 else [])
        for ii in range(mat.shape[0]):
            for jj in range(mat.shape[1]):
                v = mat.values[ii, jj]
                if np.isfinite(v): ax.text(jj, ii, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if v < .3 else "#111111")
    FIGS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"): fig.savefig(FIGS / f"FigS9_density_sweep.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    root = Path(sys.argv[1]); n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    items = sorted((p, p.parts[-3]) for p in root.glob("*/Patient*/trial_*.csv"))
    assert len(items) == 3444, len(items)
    CACHE.mkdir(exist_ok=True); f = CACHE / "density_sweep_q30_trial_metrics.csv"
    if f.exists():
        df = pd.read_csv(f)
    else:
        out = Parallel(n_jobs=n_jobs, verbose=5, batch_size=2)(delayed(one)(p, e) for p, e in items)
        df = pd.DataFrame([r for rows in out for r in rows]); df.to_csv(f, index=False)
    summ, delta = summarize(df)
    TABS.mkdir(exist_ok=True)
    summ.to_csv(TABS / "density_sweep_profile_correlations.csv", index=False)
    delta.to_csv(TABS / "density_sweep_delta_sigma.csv", index=False)
    plot(summ, delta)
    print(delta[np.isclose(delta.density, 0.10)].pivot(index="emotion", columns="representation", values="delta_sigma_vs_neutral_pct").round(1).to_string())
    print(summ[summ.metric == "all_metrics_profile"].pivot(index="representation", columns="density", values="r_vs_10pct").round(3).to_string())
