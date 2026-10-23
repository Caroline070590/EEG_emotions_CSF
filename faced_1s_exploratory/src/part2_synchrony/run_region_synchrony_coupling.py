"""
run_region_synchrony_coupling.py — regional synchronization + regional
synchrony-topology coupling, with statistical tests.

Mirrors the main manuscript's Part 2 (Fig. 2) methodology exactly, but
computed PER ANATOMICAL REGION instead of pooled across all 30 channels:

  A) Regional synchrony fingerprint: for each region and each of the six
     synchrony measures (PLV/PLI/wPLI/AEC/MaxXC/Lag), emotion-vs-neutral
     contrasts via lib.stats.lmm_emotion_contrasts (participant-random-
     intercept LMM), FDR-corrected within region (family = 6 metrics x 8
     emotions = 48 tests per region, matching the main-text convention).

  B) Regional synchrony-topology coupling: region-averaged NODAL STRENGTH
     (NOT small-world sigma or transitivity -- see methodological note
     below) merged with region-level synchrony (region_trial_synchrony.csv)
     at the trial level, within-participant z-scored (matching
     run_part2.py's _zw), pooled coupling beta per region x
     synchrony-measure via participant-clustered OLS (identical model to
     run_part2.py's _pooled_beta), plus a Wald test
     (lib.stats.emotion_slope_lmm) for whether the coupling slope differs
     across emotions within each region.

METHODOLOGICAL NOTE (important, found empirically while building this):
  Region-restricted graph-theoretic metrics computed on a SUBGRAPH built
  from only that region's channels are NOT usable at these region sizes
  (2-9 channels). At the standard 10% edge density:
    - sigma_small_world requires degree-preserving double-edge-swap
      randomization, which needs far more edges than these subgraphs
      have (NaN for 77-100% of trials in every region but parietal).
    - transitivity requires triangles: with only 1-4 edges total in a
      2-9 node graph, triangles almost never form, so transitivity is
      mechanically ~0 (zero variance) for every region except parietal.
  Both failures are structural (not a bug): the region subgraphs are too
  small for standard graph-theoretic null models or triangle-counting.
  The fix used here (and in run_fig4_panel_f_spearman.py) is to compute
  NODAL STRENGTH from the FULL 30-channel network per trial (sum of
  |Spearman correlation| to all other channels), and only average within
  region as the final step. This never touches an unstable small
  subgraph, and is valid and well-behaved for every region.

Outputs:
  figures/part2_regional/Figure_region_synchrony_coupling.png/.pdf
  supplementary_tables/TableS_region_synchrony_contrasts.csv
  supplementary_tables/TableS_region_synchrony_topology_coupling.csv
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, FIG_DIR, EMOS, CH, set_nature_style, save, W2, panel_letter, nice_heatmap, slim_colorbar
from lib.stats import lmm_emotion_contrasts, emotion_slope_lmm
from lib.synchrony_regions import SYNC_COLS, compute_synchrony_regions
from lib.regions import REGIONS, REGION_MAP
from lib.representations import load_raw_trial_csv, matrix_spearman_channels, parse_subject_trial, _list_trials

SLAB = {"plv": "PLV", "pli": "PLI", "wpli": "wPLI", "amp_env_corr": "AEC",
        "max_cross_corr": "MaxXC", "lag_ms": "Lag"}
RLAB = [r.capitalize() for r in REGIONS]
COLIDX = {c: i for i, c in enumerate(CH)}


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else ""


def _zw(df, cols):
    o = df.copy()
    for c in cols:
        o[c + "_z"] = o.groupby("subject")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return o


def _pooled_beta(d, t, s):
    dd = d[[t, s, "subject"]].dropna()
    if dd[s].std() < 1e-9:
        return np.nan, np.nan
    try:
        m = smf.ols(f"{t} ~ {s}", dd).fit(cov_type="cluster", cov_kwds={"groups": dd["subject"]})
        return float(m.params[s]), float(m.pvalues[s])
    except Exception:
        return np.nan, np.nan


def _nodal_strength_one(path, emo):
    X = load_raw_trial_csv(path)
    subj, tr = parse_subject_trial(path)
    A = matrix_spearman_channels(X)
    strength = A.sum(1)  # per-channel nodal strength, full 30-channel network
    row = dict(subject=subj, emotion=emo, trial=Path(path).stem)
    for region, chans in REGION_MAP.items():
        idx = [COLIDX[c] for c in chans if c in COLIDX]
        row[f"strength_{region}"] = float(np.mean(strength[idx])) if idx else np.nan
    return row


def compute_regional_nodal_strength(n_jobs=1, force=False):
    f = CACHE / "region_trial_nodal_strength.csv"
    if f.exists() and not force:
        print("[nodal_strength] cached"); return pd.read_csv(f)
    items = _list_trials(); t0 = time.time()
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_nodal_strength_one)(p, emo) for p, _, emo in items)
    df = pd.DataFrame(out)
    df.to_csv(f, index=False)
    print(f"[nodal_strength] {len(df)} trials in {time.time()-t0:.1f}s")
    return df


def main():
    set_nature_style()
    from statsmodels.stats.multitest import multipletests
    tab_dir = Path(CACHE).parent / "supplementary_tables"; tab_dir.mkdir(exist_ok=True, parents=True)
    fig_dir = FIG_DIR / "part2_regional"; fig_dir.mkdir(exist_ok=True, parents=True)

    # ---- data ------------------------------------------------------------
    sync = compute_synchrony_regions()  # cached after first run (46s on real data)
    # (region_trial_metrics.csv is no longer needed here -- see module
    # docstring for why sigma_small_world/transitivity on region subgraphs
    # don't work; regional nodal strength is computed fresh below instead)

    # ================================================================ A
    # regional synchrony emotion-vs-neutral contrasts, all 6 measures
    rows = []
    for region in REGIONS:
        sub = sync[sync.region == region]
        for metric in SYNC_COLS:
            d = sub[["subject", "emotion", metric]].rename(columns={metric: "y"})
            contrasts = lmm_emotion_contrasts(d, "y")
            for e, p in contrasts.items():
                rows.append(dict(region=region, metric=metric, emotion=e, p_nominal=p))
    contrast_tab = pd.DataFrame(rows)
    contrast_tab["q_fdr_within_region"] = np.nan
    for region, g in contrast_tab.groupby("region"):
        valid = g["p_nominal"].notna()
        if valid.sum():
            q = multipletests(g.loc[valid, "p_nominal"].values, method="fdr_bh")[1]
            contrast_tab.loc[g.index[valid], "q_fdr_within_region"] = q
    contrast_tab.to_csv(tab_dir / "TableS_region_synchrony_contrasts.csv", index=False)
    n_sig_fdr = (contrast_tab.q_fdr_within_region < 0.05).sum()
    print(f"[region synchrony] {len(contrast_tab)} contrasts, {n_sig_fdr} survive FDR within-region")

    # ================================================================ B
    # regional synchrony-topology coupling, using per-trial regional
    # nodal strength (see methodological note in module docstring for
    # why sigma_small_world / transitivity on region subgraphs do not work)
    nodal = compute_regional_nodal_strength()  # cached after first run (~1 min on real data)
    merged = {}
    for region in REGIONS:
        s_r = sync[sync.region == region][["subject", "emotion", "trial"] + SYNC_COLS]
        t_r = nodal[["subject", "emotion", "trial", f"strength_{region}"]].rename(
            columns={f"strength_{region}": "topo"})
        m = pd.merge(s_r, t_r, on=["subject", "emotion", "trial"], how="inner", validate="one_to_one")
        m = _zw(m, ["topo"] + SYNC_COLS)
        merged[region] = m

    coupling_rows = []
    for region in REGIONS:
        m = merged[region]
        for s in SYNC_COLS:
            beta, p = _pooled_beta(m, "topo_z", s + "_z")
            slopes, inter_p = emotion_slope_lmm(m, "topo", s)
            coupling_rows.append(dict(region=region, sync_metric=s, pooled_beta=beta, pooled_p=p,
                                       emotion_interaction_p=inter_p))
    coupling_tab = pd.DataFrame(coupling_rows)
    coupling_tab.to_csv(tab_dir / "TableS_region_synchrony_topology_coupling.csv", index=False)
    print(coupling_tab.round(4).to_string(index=False))

    # ================================================================ FIGURE
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(W2, W2 * 0.68))
    gs = GridSpec(2, 2, figure=fig, wspace=0.55, hspace=0.75, top=0.90, bottom=0.10,
                  width_ratios=[1.6, 1.0])

    # panel A: regional synchrony fingerprint (all 6 metrics x region, wPLI-style
    # column z, averaged across emotions relative to neutral to keep it compact)
    axA = fig.add_subplot(gs[0, :])
    M = np.full((len(REGIONS), len(SYNC_COLS) * (len(EMOS) - 1)), np.nan)
    col_labels = []
    for j, s in enumerate(SYNC_COLS):
        for k, e in enumerate([e for e in EMOS if e != "neutral"]):
            col_labels.append(f"{SLAB[s]}\n{e[:3]}")
    for ri, region in enumerate(REGIONS):
        sub = sync[sync.region == region]
        heat = sub.groupby("emotion")[SYNC_COLS].mean().reindex(EMOS)
        z = (heat - heat.mean(0)) / heat.std(0)
        col = 0
        for s in SYNC_COLS:
            for e in [e for e in EMOS if e != "neutral"]:
                M[ri, col] = z.loc[e, s]; col += 1
    im = nice_heatmap(axA, M, RLAB, col_labels, cmap="PuBuGn", vmin=-2, vmax=2, aspect="auto", fontsize=4.2)
    for ri, region in enumerate(REGIONS):
        col = 0
        sub_reg = contrast_tab[contrast_tab.region == region]
        for s in SYNC_COLS:
            for e in [e for e in EMOS if e != "neutral"]:
                row = sub_reg[(sub_reg.metric == s) & (sub_reg.emotion == e)]
                if len(row) and stars(row.q_fdr_within_region.iloc[0]):
                    axA.text(col, ri, stars(row.q_fdr_within_region.iloc[0]), ha="center", va="center",
                              fontsize=4.2, color="white" if abs(np.nan_to_num(M[ri, col])) > 1 else "black")
                col += 1
    axA.set_title("Regional synchrony fingerprint (column z; stars = FDR q<.05 within region)", pad=8, fontsize=7.4)
    panel_letter(axA, "a")
    slim_colorbar(fig, im, axA, "column z")

    # panel B: regional coupling heatmap (region x sync metric), pooled beta
    axB = fig.add_subplot(gs[1, 0])
    B = coupling_tab.pivot(index="region", columns="sync_metric", values="pooled_beta").reindex(
        index=REGIONS, columns=SYNC_COLS)
    P = coupling_tab.pivot(index="region", columns="sync_metric", values="pooled_p").reindex(
        index=REGIONS, columns=SYNC_COLS)
    vmax = np.nanmax(np.abs(B.values))
    imB = nice_heatmap(axB, B.values, RLAB, [SLAB[s] for s in SYNC_COLS], cmap="PuOr_r",
                       vmin=-vmax, vmax=vmax, annot=False, aspect="auto", fontsize=5.4)
    for ri in range(len(REGIONS)):
        for ci in range(len(SYNC_COLS)):
            p = P.values[ri, ci]
            if stars(p):
                axB.text(ci, ri, stars(p), ha="center", va="center", fontsize=5.4,
                          color="white" if abs(B.values[ri, ci]) / vmax > 0.5 else "black")
    axB.set_title("Regional synchrony-topology\ncoupling (pooled β, nodal strength)", pad=8, fontsize=7.4)
    panel_letter(axB, "b"); slim_colorbar(fig, imB, axB, "β")

    # panel C: coupling strength per region (mean |beta| across sync metrics)
    axC = fig.add_subplot(gs[1, 1])
    strength = coupling_tab.groupby("region")["pooled_beta"].apply(lambda x: np.nanmean(np.abs(x))).reindex(REGIONS)
    n_sig = coupling_tab[coupling_tab.pooled_p < 0.05].groupby("region").size().reindex(REGIONS).fillna(0)
    order = strength.sort_values(ascending=False).index
    axC.barh(range(len(order)), strength.reindex(order).values, color="#2c5f7c", edgecolor="k", linewidth=0.3)
    for i, region in enumerate(order):
        axC.text(strength[region] + 0.003, i, f"{int(n_sig[region])} sig", va="center", fontsize=5.0)
    axC.set_yticks(range(len(order))); axC.set_yticklabels([r.capitalize() for r in order], fontsize=6)
    axC.invert_yaxis(); axC.set_xlabel("mean |β| coupling strength")
    axC.set_title("Coupling strength\nper region", pad=8, fontsize=7.4)
    for sp in ["top", "right"]:
        axC.spines[sp].set_visible(False)
    panel_letter(axC, "c")

    fig.suptitle("Regional synchronization and synchrony-topology coupling (Spearman nodal strength)",
                 fontsize=8.2, y=0.975)
    save(fig, fig_dir / "Figure_region_synchrony_coupling"); plt.close(fig)
    print(f"\n[region synchrony coupling] wrote table + figure to {fig_dir}")


if __name__ == "__main__":
    main()
