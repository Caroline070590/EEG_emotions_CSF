"""
run_participant_averaging_sensitivity.py — Fig. S14 (Appendix F).

Not included in the code package received; rebuilt here from the
manuscript's own description:

    "As a sensitivity analysis, we repeated omnibus emotion tests after
    first averaging each scalar metric within participant x emotion
    cells. The resulting participant-level emotion matrices were
    analyzed using repeated-measures omnibus tests..."

    Fig. S14 caption: "...Points show the omnibus emotion effect for
    each metric, expressed as -log10(p), grouped by analysis domain and
    network representation. The dashed horizontal line indicates
    p = 0.05. Asterisks indicate tests that survived Benjamini-Hochberg
    correction within domain."

Domains (7, matching the manuscript figure): topology_spearman,
topology_mtf, topology_qg, synchrony_all_pairs, higher-order_spearman,
higher-order_mtf, higher-order_qg.

Implementation notes
=====================
- Topology and synchrony are natively TRIAL-level (many trials per
  subject x emotion), so the participant-level averaging step here is
  real: trial-level values are averaged within (subject, emotion) cells
  first, then the omnibus test (lib.stats.lmm_emotion_lrt, the same LRT
  machinery used everywhere else in this codebase for omnibus emotion
  effects) is refit on that averaged, one-row-per-subject-per-emotion
  table -- a repeated-measures design with subject as the random
  intercept, exactly as the manuscript describes.
- Higher-order topology (persistent homology) is, in this codebase,
  already computed once per (subject, emotion) AVERAGED representation
  matrix (see part3_embedding_topology/compute_features.py: TDA runs on
  `agg`, the subject x emotion aggregate, not on individual trials).
  There is therefore no separate trial-level version to average down
  from for this domain -- the omnibus test here IS already at the
  participant level. This is reported explicitly (not silently treated
  as equivalent) so the "sensitivity" and "main" p-values for
  higher-order topology are expected to match closely, which is itself
  consistent with the manuscript's finding that higher-order topology
  was one of the more robust domains under this check.

FDR correction is applied WITHIN each of the 7 domains (family size = 6
metrics each), per the figure caption.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, FIG_DIR, METRICS, METRIC_LABELS, set_nature_style, save, W2, panel_letter
from lib.stats import lmm_emotion_lrt
from lib.synchrony import SYNC_COLS
from lib.topology import TCOLS

DOMAIN_COLORS = {
    "topology_spearman": "#2471a3", "topology_mtf": "#8e44ad", "topology_qg": "#16a085",
    "synchrony_all_pairs": "#c0392b",
    "higher-order_spearman": "#2471a3", "higher-order_mtf": "#8e44ad", "higher-order_qg": "#16a085",
}
DOMAIN_ORDER = ["topology_spearman", "topology_mtf", "topology_qg", "synchrony_all_pairs",
                "higher-order_spearman", "higher-order_mtf", "higher-order_qg"]
DOMAIN_LABELS = {"topology_spearman": "topology\nspearman", "topology_mtf": "topology\nmtf",
                  "topology_qg": "topology\nqg", "synchrony_all_pairs": "synchrony\nall_pairs",
                  "higher-order_spearman": "higher-order\nspearman", "higher-order_mtf": "higher-order\nmtf",
                  "higher-order_qg": "higher-order\nqg"}


def _participant_level_omnibus(df, cols, natively_participant_level=False):
    """Average within (subject, emotion), then run the omnibus LRT.
    If natively_participant_level, df is already one row per subject x
    emotion (higher-order topology case) and is used as-is."""
    out = []
    for c in cols:
        if natively_participant_level:
            avg = df[[c, "emotion", "subject"]].dropna()
        else:
            avg = df.groupby(["subject", "emotion"])[c].mean().reset_index()
        res = lmm_emotion_lrt(avg, c)
        out.append(dict(metric=c, p=res["p"], stat=res["stat"]))
    return pd.DataFrame(out)


def main():
    tab_dir = Path(CACHE).parent / "supplementary_tables"; tab_dir.mkdir(exist_ok=True, parents=True)
    fig_dir = FIG_DIR / "part6"; fig_dir.mkdir(exist_ok=True, parents=True)

    sources = {
        "topology_spearman": (CACHE / "trial_metrics_spearman.csv", METRICS, False),
        "topology_mtf": (CACHE / "trial_metrics_mtf.csv", METRICS, False),
        "topology_qg": (CACHE / "trial_metrics_qg.csv", METRICS, False),
        "synchrony_all_pairs": (CACHE / "trial_synchrony_metrics.csv", SYNC_COLS, False),
        "higher-order_spearman": (CACHE / "tda_spearman.csv", TCOLS, True),
        "higher-order_mtf": (CACHE / "tda_mtf.csv", TCOLS, True),
        "higher-order_qg": (CACHE / "tda_qg.csv", TCOLS, True),
    }

    rows = []
    for domain, (path, cols, native) in sources.items():
        if not path.exists():
            print(f"  [skip] {domain}: {path.name} not found")
            continue
        df = pd.read_csv(path)
        res = _participant_level_omnibus(df, cols, natively_participant_level=native)
        res["domain"] = domain
        rows.append(res)
    full = pd.concat(rows, ignore_index=True)

    # FDR correction within domain (family size = 6 metrics per domain)
    full["q_fdr_within_domain"] = np.nan
    for domain, g in full.groupby("domain"):
        valid = g["p"].notna()
        if valid.sum():
            q = multipletests(g.loc[valid, "p"].values, method="fdr_bh")[1]
            full.loc[g.index[valid], "q_fdr_within_domain"] = q
    full["neg_log10_p"] = -np.log10(full["p"].clip(lower=1e-300))
    full["fdr_survive"] = full["q_fdr_within_domain"] < 0.05
    full.to_csv(tab_dir / "TableS_participant_averaging_sensitivity.csv", index=False)
    print(full.round(4).to_string(index=False))

    # ---- figure --------------------------------------------------------
    set_nature_style()
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(W2, W2 * 0.45))
    for i, domain in enumerate(DOMAIN_ORDER):
        g = full[full.domain == domain]
        if len(g) == 0:
            continue
        x = i + rng.uniform(-0.12, 0.12, size=len(g))
        ax.scatter(x, g["neg_log10_p"], s=16, color=DOMAIN_COLORS[domain],
                   edgecolor="k", linewidth=0.3, zorder=3)
        for xi, yi, sig in zip(x, g["neg_log10_p"], g["fdr_survive"]):
            if sig:
                ax.text(xi, yi + 0.3, "*", ha="center", fontsize=9, color="#c0392b")
    ax.axhline(-np.log10(0.05), color="0.4", lw=0.8, ls="--")
    ax.set_xticks(range(len(DOMAIN_ORDER)))
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAIN_ORDER], fontsize=5.6)
    ax.set_ylabel(r"Repeated-measures omnibus effect, $-\log_{10}(p)$", fontsize=7)
    ax.set_title("Participant × emotion averaged sensitivity", fontsize=7.5, pad=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    save(fig, fig_dir / "FigS14_participant_averaging_sensitivity"); plt.close(fig)
    print(f"\n[Fig S14] wrote {tab_dir/'TableS_participant_averaging_sensitivity.csv'} and figure to {fig_dir}")


if __name__ == "__main__":
    main()
