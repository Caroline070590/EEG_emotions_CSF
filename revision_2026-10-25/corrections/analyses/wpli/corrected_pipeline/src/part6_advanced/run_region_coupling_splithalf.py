"""Split-half directional stability of the regional synchronization--nodal-strength coupling (1-s analysis;
regenerates Supplementary Figure S19, whose original generator could not be recovered).

Per trial: regional synchronization (lib.synchrony_regions, 1-45 Hz, six measures) and regional nodal
strength from the full 30-channel absolute-Spearman network averaged within region. Both are z-scored
within participant; the pooled coupling coefficient is the slope of nodal strength on synchronization
(OLS over trials), as in run_region_synchrony_coupling.py. Procedure: 300 random partitions of the 123
participants into complementary halves (61 / 62); the slope is re-estimated in each half. Reported: the
spread of the half estimates and the percentage of partitions in which BOTH halves keep the full-sample sign.
The correlation between complementary halves is not reported (it reflects their complementarity, not
reliability).

usage (from faced_1s_exploratory/src, with the 1-s export at ./EEG-data-raw): python part6_advanced/run_region_coupling_splithalf.py [N_JOBS]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
CODE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(CODE))
from lib.config import CACHE, CH
from lib.regions import REGIONS, REGION_MAP
from lib.representations import load_raw_trial_csv, parse_subject_trial, _list_trials, matrix_spearman_channels
from lib.synchrony_regions import compute_synchrony_regions

FIGS, TABS = CODE.parent / "figures", CODE.parent / "tables"; N_PART, SEED = 300, 2026
SYNC = ["plv", "pli", "wpli", "amp_env_corr", "max_cross_corr", "lag_ms"]
LAB = {"plv": "PLV", "pli": "PLI", "wpli": "wPLI", "amp_env_corr": "AEC", "max_cross_corr": "MaxXC", "lag_ms": "Lag"}
COLIDX = {c: i for i, c in enumerate(CH)}

def strength_one(path, emo):
    X = load_raw_trial_csv(path); subj, _ = parse_subject_trial(path)
    s = matrix_spearman_channels(X).sum(1)
    row = dict(subject=subj, emotion=emo, trial=Path(path).stem)
    for r, chans in REGION_MAP.items():
        idx = [COLIDX[c] for c in chans if c in COLIDX]; row[f"strength_{r}"] = float(np.mean(s[idx]))
    return row

def zw(x, g): return x.groupby(g).transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))

def slope(t, s):
    s = s - s.mean(); return float((s * (t - t.mean())).sum() / (s * s).sum())

if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    CACHE.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True); TABS.mkdir(exist_ok=True)
    sync = compute_synchrony_regions(n_jobs=n_jobs)
    f = CACHE / "region_trial_nodal_strength.csv"
    nodal = pd.read_csv(f) if f.exists() else pd.DataFrame(Parallel(n_jobs=n_jobs, verbose=2)(delayed(strength_one)(p, e) for p, _, e in _list_trials()))
    nodal.to_csv(f, index=False)
    rng = np.random.default_rng(SEED); subjects = np.array(sorted(nodal.subject.unique()))
    parts = [rng.permutation(subjects) for _ in range(N_PART)]
    rows, halves = [], {}
    for r in REGIONS:
        m = pd.merge(sync[sync.region == r][["subject", "emotion", "trial"] + SYNC],
                     nodal[["subject", "emotion", "trial", f"strength_{r}"]].rename(columns={f"strength_{r}": "topo"}),
                     on=["subject", "emotion", "trial"], validate="one_to_one")
        t = zw(m.topo, m.subject).to_numpy(); subj = m.subject.to_numpy()
        for sc in SYNC:
            s = zw(m[sc], m.subject).to_numpy(); full = slope(t, s)
            A = np.array([slope(t[np.isin(subj, p[:len(p) // 2])], s[np.isin(subj, p[:len(p) // 2])]) for p in parts])
            B = np.array([slope(t[np.isin(subj, p[len(p) // 2:])], s[np.isin(subj, p[len(p) // 2:])]) for p in parts])
            both = (np.sign(A) == np.sign(full)) & (np.sign(B) == np.sign(full)); allh = np.r_[A, B]
            halves[(r, sc)] = allh
            rows.append({"region": r, "metric": LAB[sc], "full_beta": full, "half_beta_sd": allh.std(ddof=1),
                         "half_beta_2.5pct": np.percentile(allh, 2.5), "half_beta_97.5pct": np.percentile(allh, 97.5),
                         "directional_replication_pct": 100 * both.mean(), "n_partitions": N_PART})
    T = pd.DataFrame(rows); T.to_csv(TABS / "regional_coupling_splithalf.csv", index=False)
    print(T.round(3).to_string()); print("directional replication: min %.1f%%" % T.directional_replication_pct.min())

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, len(REGIONS), figsize=(7.4, 2.6), sharey=True)
    for k, r in enumerate(REGIONS):
        ax = axs[k]
        for i, sc in enumerate(SYNC):
            v = halves[(r, sc)]; vp = ax.violinplot(v, positions=[i], widths=0.8, showextrema=False)
            for b in vp["bodies"]: b.set_facecolor("#9fb3c8"); b.set_alpha(0.9); b.set_edgecolor("none")
            fb = T[(T.region == r) & (T.metric == LAB[sc])].full_beta.item(); ax.plot([i - .35, i + .35], [fb, fb], color="#c0392b", lw=1.3)
        ax.axhline(0, color="#444", lw=0.6, ls="--"); ax.set_xticks(range(len(SYNC))); ax.set_xticklabels([LAB[s] for s in SYNC], rotation=60, ha="right")
        dr = T[T.region == r].directional_replication_pct.min()
        ax.set_title(f"{r.capitalize()}\nsign kept: {dr:.0f}-100%" if dr < 100 else f"{r.capitalize()}\nsign kept: 100%", fontsize=7)
    axs[0].set_ylabel("Coupling slope in each half\n(nodal strength on synchronization)")
    fig.suptitle("Half-sample coupling slopes, 300 complementary splits (red: full sample)", fontsize=7.5, y=1.04)
    for ext in ("pdf", "png"): fig.savefig(FIGS / f"FigS19_regional_coupling_splithalf.{ext}", dpi=300, bbox_inches="tight")
