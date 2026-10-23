"""
run_band_robustness.py
--------------------------------------------------------------------------
Addresses the following caveat:

    "PLV/PLI/wPLI at the low end of the 1-45 Hz band. A 1 Hz oscillation
    completes one cycle in exactly 1 s, so phase estimates near the low
    end of the passband are built on less than one full cycle. This will
    bias phase-synchronization estimates and likely inflate variance more
    for some emotions than others if trial counts or spectral content
    differ."

Strategy
========
1. Recompute the six synchrony metrics separately for:
     delta (1-4 Hz, the culprit band), theta, alpha, beta, gamma,
     the original broadband (1-45 Hz), and a delta-excluded broadband
     (4-45 Hz) -- via lib.synchrony_bands.compute_synchrony_bands().

2. For every band and metric, refit the SAME omnibus LMM
   (lib.stats.lmm_emotion_lrt) and the SAME emotion-vs-neutral contrasts
   (lib.stats.lmm_emotion_contrasts) used in the main manuscript's Part 2
   / Table S1, so results are directly comparable at the level of
   p-values / q-values already reported in the paper.

3. Report, per metric:
     (a) whether the omnibus emotion effect survives in delta alone
         (if yes despite <3 cycles/trial, that specific result needs a
         cautious read);
     (b) whether the manuscript's headline broadband effects (e.g.
         reduced PLV in disgust/anger/fear, increased wPLI in sadness)
         are reproduced in theta/alpha/beta/gamma and in the delta-
         excluded broadband;
     (c) the correlation, across all emotion x metric cells, between the
         original 1-45 Hz standardized effect and the 4-45 Hz
         (delta-excluded) standardized effect -- a high correlation
         indicates the broadband conclusions do not depend on the
         unreliable low-frequency content.

4. Produces:
     figures/part2_band_reliability/FigS_band_robustness_heatmap.png
     figures/part2_band_reliability/FigS_broadband_vs_restricted.png
     supplementary_tables/TableS_band_robustness.csv

Run once raw trial CSVs are available under EEG-data-raw/ (see
DATA_INSTRUCTIONS.md). This script is a no-op (prints what it needs) if
that data is not present, so it is safe to keep in the repo.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, FIG_DIR, DATA_ROOT, EMOS, set_nature_style, save, W2, W1, panel_letter, nice_heatmap, slim_colorbar
from lib.stats import lmm_emotion_lrt, lmm_emotion_contrasts
from lib.synchrony_bands import BANDS, SYNC_COLS, compute_synchrony_bands

SLAB = {"plv": "PLV", "pli": "PLI", "wpli": "wPLI", "amp_env_corr": "AEC",
        "max_cross_corr": "MaxXC", "lag_ms": "Lag"}
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma", "broadband_1_45", "broadband_4_45"]
BAND_LABELS = {"delta": "δ  1–4 Hz", "theta": "θ  4–8 Hz", "alpha": "α  8–13 Hz",
               "beta": "β  13–30 Hz", "gamma": "γ  30–45 Hz",
               "broadband_1_45": "broadband 1–45 Hz (original)",
               "broadband_4_45": "broadband 4–45 Hz (δ-excluded)"}


def _stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else ""


def _standardized_effect(df, col, emotion, ref="neutral"):
    """Same standardized emotion-vs-neutral effect used for Fig. S3/S4:
    participant-level mean difference / pooled SD, no bootstrap here
    (bootstrap CIs can be added via lib.stats.boot_vs_neutral if desired)."""
    piv = df.groupby(["subject", "emotion"])[col].mean().unstack()
    if emotion not in piv.columns or ref not in piv.columns:
        return np.nan
    d = (piv[emotion] - piv[ref]).dropna()
    sd = pd.concat([piv[emotion], piv[ref]]).std()
    if sd == 0 or len(d) < 5:
        return np.nan
    return float(d.mean() / sd)


def main():
    if not DATA_ROOT.exists():
        print(f"[band-robustness] raw data folder not found at {DATA_ROOT}.")
        print("  This script needs the raw FACED trial CSVs (see DATA_INSTRUCTIONS.md).")
        print("  Run diagnostic_band_reliability.py in the meantime -- it needs no data.")
        return

    df = compute_synchrony_bands()  # cached after first run

    fig_dir = FIG_DIR / "part2_band_reliability"; fig_dir.mkdir(exist_ok=True, parents=True)
    tab_dir = Path(CACHE).parent / "supplementary_tables"; tab_dir.mkdir(exist_ok=True, parents=True)

    # ---- 1. omnibus LMM + emotion-vs-neutral contrasts, per band x metric
    rows = []
    for band in BAND_ORDER:
        for c in SYNC_COLS:
            col = f"{band}__{c}"
            if col not in df.columns:
                continue
            sub = df[["subject", "emotion", col]].rename(columns={col: "y"})
            omni = lmm_emotion_lrt(sub, "y")
            contrasts = lmm_emotion_contrasts(sub.rename(columns={"y": col}), col)
            for emo in EMOS:
                if emo == "neutral":
                    continue
                eff = _standardized_effect(df, col, emo)
                rows.append(dict(band=band, metric=c, emotion=emo,
                                  standardized_effect=eff,
                                  p_vs_neutral=contrasts.get(emo, np.nan),
                                  omnibus_p=omni["p"], omnibus_stat=omni["stat"]))
    res = pd.DataFrame(rows)
    res.to_csv(tab_dir / "TableS_band_robustness.csv", index=False)
    print(f"[band-robustness] wrote {tab_dir/'TableS_band_robustness.csv'} ({len(res)} rows)")

    # ---- 2. heatmap: standardized effect per band (rows) x emotion x metric (cols)
    set_nature_style()
    import matplotlib.pyplot as plt
    metrics_emos = [(m, e) for m in SYNC_COLS for e in EMOS if e != "neutral"]
    M = np.full((len(BAND_ORDER), len(metrics_emos)), np.nan)
    P = np.full_like(M, np.nan)
    for bi, band in enumerate(BAND_ORDER):
        for ci, (m, e) in enumerate(metrics_emos):
            r = res[(res.band == band) & (res.metric == m) & (res.emotion == e)]
            if len(r):
                M[bi, ci] = r.standardized_effect.iloc[0]
                P[bi, ci] = r.p_vs_neutral.iloc[0]
    fig, ax = plt.subplots(figsize=(W2, 3.6))
    im = nice_heatmap(ax, M, [BAND_LABELS[b] for b in BAND_ORDER],
                       [f"{SLAB[m]} · {e[:3]}" for m, e in metrics_emos],
                       cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto", fontsize=5.2)
    ax.tick_params(axis="y", labelrotation=0, pad=3)
    for lbl in ax.get_yticklabels():
        lbl.set_ha("right"); lbl.set_va("center")
    for bi in range(len(BAND_ORDER)):
        for ci in range(len(metrics_emos)):
            st = _stars(P[bi, ci])
            if st:
                ax.text(ci, bi, st, ha="center", va="center", fontsize=4.2,
                        color="white" if abs(np.nan_to_num(M[bi, ci])) > 0.2 else "#111111")
    ax.set_title("Emotion-vs-neutral synchrony effects across frequency bands\n"
                  "(rows: band, bottom two = original vs. delta-excluded broadband; "
                  "columns: metric × emotion; stars: nominal p<.05; δ band has <3 cycles/trial, interpret with caution)",
                  fontsize=6.2, pad=8)
    fig.subplots_adjust(left=0.20, right=0.90, top=0.82, bottom=0.30)
    slim_colorbar(fig, im, ax, "standardized Δ")
    save(fig, fig_dir / "FigS_band_robustness_heatmap"); plt.close(fig)

    # ---- 3. scatter: broadband(1-45) vs delta-excluded broadband(4-45) effects
    # NOTE: use pivot() not pivot_table() -- with pandas' string-backed dtype
    # (as read back from CSV) pivot_table's default aggfunc can silently
    # return an empty column index even though there is exactly one row per
    # (metric, emotion, band) cell here, so no aggregation is actually needed.
    piv = res.pivot(index=["metric", "emotion"], columns="band", values="standardized_effect")
    if "broadband_1_45" in piv.columns and "broadband_4_45" in piv.columns:
        x = piv["broadband_1_45"].values; y = piv["broadband_4_45"].values
        ok = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 2 else np.nan
        fig, ax = plt.subplots(figsize=(W1, W1))
        ax.scatter(x[ok], y[ok], s=10, color="#2471a3", edgecolor="white", linewidth=0.3)
        lim = np.nanmax(np.abs(np.concatenate([x[ok], y[ok]]))) * 1.15 if ok.sum() else 1
        ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.7, ls="--")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("standardized Δ, original broadband (1-45 Hz)")
        ax.set_ylabel("standardized Δ, delta-excluded (4-45 Hz)")
        ax.set_title(f"Original vs. delta-excluded broadband effects\n(r = {r:.2f}, n = {ok.sum()} emotion×metric cells)",
                     fontsize=6.5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        save(fig, fig_dir / "FigS_broadband_vs_restricted"); plt.close(fig)
        print(f"[band-robustness] broadband(1-45) vs delta-excluded(4-45) correlation: r = {r:.3f}")

    print(f"[band-robustness] done -> {fig_dir}")


if __name__ == "__main__":
    main()
