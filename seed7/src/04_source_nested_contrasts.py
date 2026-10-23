"""SEED-VII sensitivity (added after an internal review): min F' with source programmes as the stimulus unit,
strictly nested within condition (the 4 clips of the 2 programmes spanning two conditions are excluded) and
with programmes weighted equally in both components (lib_rev.contrast_vs_neutral_source_nested).
Families and BH correction as in 03_stimulus_aware_contrasts.py. Output: tables/contrasts_source_nested.csv"""
import numpy as np, pandas as pd
from lib_rev import load, contrast_vs_neutral_source_nested, bh, NONNEU, TOPO, BANDS5, TAB, CACHE
df, _ = load()
rec = np.load(CACHE / "recurrence_features.npz")
for k, nm in enumerate(str(n) for n in rec["rqa_names"]):
    df[f"RQA_{nm}"] = rec["rqa"][:, :, k].mean(1)
df["JRS_mean"] = rec["jrs"].mean(1)
FAM = {"primary": [f"{r}_q30_d10_{m}" for r in ("Spearman", "MTF", "QG") for m in TOPO] + [f"{s}_{b}" for s in ("wpli", "oaec") for b in BANDS5],
       "recurrence": [f"RQA_{n}" for n in ("DET", "LAM", "Lmean", "ENTR", "TT")] + ["JRS_mean"]}
rows = [{"family": f, "outcome": o, **contrast_vs_neutral_source_nested(df[o].to_numpy(float), df, e)} for f, outs in FAM.items() for o in outs for e in NONNEU]
R = pd.DataFrame(rows)
for f in FAM:
    m = R.family == f; R.loc[m, "q_minF"] = bh(R.loc[m, "p_minF"])
R.to_csv(TAB / "contrasts_source_nested.csv", index=False)
print("programmes kept:", R.n_sources.iloc[0], "| df2:", R.df2.iloc[0], "| per condition:", R.groupby("emotion").n_sources_emo.first().to_dict(), "neutral", R.n_sources_neutral.iloc[0])
print(R.groupby("family").apply(lambda g: pd.Series({"tests": len(g), "q<.05": int((g.q_minF < .05).sum()), "p<.05": int((g.p_minF < .05).sum())})).to_string())
print(R[(R.family == "recurrence") & (R.q_minF < .05)][["outcome", "emotion", "dz_source_weighted", "t2", "p_minF", "q_minF"]].round(4).to_string())
