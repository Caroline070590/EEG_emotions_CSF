"""SEED-VII replication: emotion-vs-neutral contrasts for the pre-specified primary family.

Primary family (BH-FDR once across all 168 tests):
  * scalar topology: Spearman, MTF, QG (q = 30, density 10%) x 6 metrics x 6 contrasts = 108
  * frequency-specific synchronization: wPLI and orthogonalised AEC x 5 bands x 6 = 60
Recurrence family (BH-FDR once across 36 tests): DET, LAM, L_mean, ENTR, TT and JRS x 6 contrasts.
Every test is run three ways: participant-only (clips fixed), participants and clips random
(min F', primary), and participants and source programmes random (sensitivity).
"""
import numpy as np, pandas as pd
from lib_rev import load, contrast_vs_neutral, omnibus, bh, NONNEU, TOPO, BANDS5, TAB, CACHE

df, _ = load()
rec = np.load(CACHE / "recurrence_features.npz")
for k, nm in enumerate(str(n) for n in rec["rqa_names"]):
    df[f"RQA_{nm}"] = rec["rqa"][:, :, k].mean(1)
df["JRS_mean"] = rec["jrs"].mean(1)

FAMILIES = {
    "primary": [(f"{r}_q30_d10_{m}", "topology") for r in ("Spearman", "MTF", "QG") for m in TOPO]
             + [(f"{s}_{b}", "synchronization") for s in ("wpli", "oaec") for b in BANDS5],
    "recurrence": [(f"RQA_{n}", "recurrence") for n in ("DET", "LAM", "Lmean", "ENTR", "TT")]
                + [("JRS_mean", "recurrence")],
}

rows, omni = [], []
for fam, items in FAMILIES.items():
    for col, domain in items:
        y = df[col].to_numpy(float)
        omni.append({"family": fam, "outcome": col, "domain": domain, **omnibus(y, df)})
        for e in NONNEU:
            r = contrast_vs_neutral(y, df, e)
            s = contrast_vs_neutral(y, df, e, unit="source")
            rows.append({"family": fam, "outcome": col, "domain": domain, **r,
                         "t2_source": s["t2"], "p_minF_source": s["p_minF"]})

res = pd.DataFrame(rows)
for fam in FAMILIES:
    m = res.family == fam
    res.loc[m, "q_minF"] = bh(res.loc[m, "p_minF"])
    res.loc[m, "q_minF_source"] = bh(res.loc[m, "p_minF_source"])
    res.loc[m, "q_participant_only"] = bh(res.loc[m, "p1"])
res.to_csv(TAB / "all_contrasts.csv", index=False)
om = pd.DataFrame(omni)
for fam in FAMILIES:
    m = om.family == fam
    om.loc[m, "q_minF"] = bh(om.loc[m, "p_minF"]); om.loc[m, "q_participant_only"] = bh(om.loc[m, "p1"])
om.to_csv(TAB / "omnibus_tests.csv", index=False)

summ = res.groupby(["family", "domain"]).apply(lambda g: pd.Series({
    "tests": len(g), "q<.05 clips fixed": int((g.q_participant_only < .05).sum()),
    "q<.05 clips random": int((g.q_minF < .05).sum()), "q<.05 sources random": int((g.q_minF_source < .05).sum())}))
summ.to_csv(TAB / "summary.csv")
print(summ.to_string())
print(res[res.q_minF < .05][["outcome", "emotion", "dz", "t1", "t2", "p_minF", "q_minF"]].round(4).to_string())
print(res.sort_values("p_minF")[["outcome", "emotion", "dz", "t2", "p_minF", "q_minF", "q_participant_only"]].head(12).round(4).to_string())
