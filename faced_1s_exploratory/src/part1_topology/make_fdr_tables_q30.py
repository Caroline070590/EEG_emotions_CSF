"""
make_fdr_tables_q30.py — Table S1 (topology half) regenerated with q=30
(node-count-matched) MTF/QG. Reuses make_fdr_tables.py's contrasts_beta_p /
add_fdr helpers; only the input CSVs differ. Saved separately -- does not
overwrite the original q=13 Table S1.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, EMOS, METRICS, METRIC_LABELS
from part1_topology.make_fdr_tables import contrasts_beta_p, add_fdr, star

METHODS_Q30 = {"spearman": "Spearman", "mtf_q30": "MTF (q=30)", "qg_q30": "QG (q=30)"}
OUT = Path(__file__).resolve().parents[1] / "supplementary_tables"; OUT.mkdir(exist_ok=True)


def _load(tag):
    f = CACHE / f"trial_metrics_{tag}.csv"
    return pd.read_csv(f) if f.exists() else None


def main():
    rows = []
    for tag, name in METHODS_Q30.items():
        df = _load(tag)
        if df is None:
            continue
        for metric in METRICS:
            for e, (b, p) in contrasts_beta_p(df, metric).items():
                rows.append(dict(representation=name, metric=METRIC_LABELS[metric],
                                  emotion=e, beta=b, p_nominal=p))
    t1 = pd.DataFrame(rows)
    add_fdr(t1, ["representation"], qcol="q_fdr_within_representation")
    add_fdr(t1, [], qcol="q_fdr_global")
    t1["nominal"] = t1["p_nominal"].map(star)
    t1["fdr_within_rep"] = t1["q_fdr_within_representation"].map(star)
    t1 = t1.round(4)
    t1.to_csv(OUT / "TableS_Fig1_topology_contrasts_FDR_q30.csv", index=False)

    surv1 = t1[t1["q_fdr_within_representation"] < 0.05][
        ["representation", "metric", "emotion", "beta", "p_nominal", "q_fdr_within_representation"]]
    surv1.to_csv(OUT / "TableS_FDR_surviving_topology_q30.csv", index=False)

    n1 = t1["p_nominal"].notna().sum()
    print(f"Figure 1 topology (q=30): {n1} contrasts | nominal p<.05: {(t1.p_nominal<0.05).sum()} "
          f"| survive FDR(within-rep): {(t1.q_fdr_within_representation<0.05).sum()} "
          f"| survive FDR(global): {(t1.q_fdr_global<0.05).sum()}")
    if len(surv1):
        print("\nTopology survivors (q=30):\n", surv1.to_string(index=False))
    print(f"\n[Table S1, q=30] wrote {OUT/'TableS_Fig1_topology_contrasts_FDR_q30.csv'}")


if __name__ == "__main__":
    main()
