"""All emotion-vs-neutral contrasts and omnibus tests, with ONE primary family.

Primary family (declared here, BH-FDR once across all 224 tests):
  * scalar topology: Spearman, MTF, QG (q = 30, density 10%) x 6 metrics x 8 contrasts = 144
  * frequency-specific synchronization: wPLI and orthogonalised AEC x 5 bands x 8 = 80
Exploratory (BH within the exploratory set, reported but not used for claims):
  PLV/PLI/AEC per band, the six submitted broadband 1-45 Hz measures, 4-45 Hz broadband,
  positive-only signed Spearman topology, and persistent-homology features.
Every test is run twice: participant-only (p1, the submitted inference) and with
participants and videos as random factors (p_minF, primary).
"""
import numpy as np, pandas as pd
from lib_rev import load, contrast_vs_neutral, omnibus, bh, NONNEU, TOPO, BANDS5, TAB

df, _ = load()

primary = [(f"{r}_q30_d10_{m}", "topology", r, m) for r in ("Spearman", "MTF", "QG") for m in TOPO]
primary += [(f"{s}_{b}", "synchronization", b, s) for s in ("wpli", "oaec") for b in BANDS5]
explor = [(f"{s}_{b}", "synchronization", b, s) for s in ("plv", "pli", "aec") for b in BANDS5]
explor += [(f"{s}_broad", "synchronization", "broad 1-45", s) for s in ("plv", "pli", "wpli", "aec", "oaec", "maxxc", "lag_ms")]
explor += [(f"{s}_broad4", "synchronization", "broad 4-45", s) for s in ("plv", "pli", "wpli", "aec", "oaec")]
explor += [(f"SpearmanPos_q30_d10_{m}", "topology", "SpearmanPos", m) for m in TOPO]
explor += [(c, "persistent_homology", c.split("_")[0], "_".join(c.split("_")[1:]))
           for c in df.columns if c.startswith("PHsigned_H") or c.startswith("PHabs_H")]

rows, omni = [], []
for fam, items in (("primary", primary), ("exploratory", explor)):
    for col, domain, sub, metric in items:
        y = df[col].to_numpy(float)
        o = omnibus(y, df); omni.append({"family": fam, "outcome": col, "domain": domain, "set": sub, "metric": metric, **o})
        for e in NONNEU:
            rows.append({"family": fam, "outcome": col, "domain": domain, "set": sub, "metric": metric,
                         **contrast_vs_neutral(y, df, e)})

res = pd.DataFrame(rows)
for fam in ("primary", "exploratory"):
    m = res.family == fam
    res.loc[m, "q_minF"] = bh(res.loc[m, "p_minF"])
    res.loc[m, "q_participant_only"] = bh(res.loc[m, "p1"])
res["significant_primary"] = (res.family == "primary") & (res.q_minF < 0.05)
res.to_csv(TAB / "all_contrasts.csv", index=False)

om = pd.DataFrame(omni)
for fam in ("primary", "exploratory"):
    m = om.family == fam
    om.loc[m, "q_minF"] = bh(om.loc[m, "p_minF"]); om.loc[m, "q_participant_only"] = bh(om.loc[m, "p1"])
om.to_csv(TAB / "omnibus_tests.csv", index=False)

p = res[res.family == "primary"]
print("primary tests:", len(p))
print("participant-only q<.05:", (p.q_participant_only < .05).sum(), "| participants+videos random q<.05:", (p.q_minF < .05).sum())
print(p.groupby("domain").apply(lambda g: pd.Series({"n": len(g), "q1<.05": (g.q_participant_only < .05).sum(),
                                                   "qminF<.05": (g.q_minF < .05).sum()})))
print(p[p.q_minF < .05][["outcome", "emotion", "dz", "t1", "t2", "p_minF", "q_minF"]].to_string())
print(om[om.family == "primary"][["outcome", "F1", "p1", "F2", "p2", "minF", "p_minF", "q_minF"]].round(4).to_string())
