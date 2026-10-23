"""SEED-VII, exploratory (added after the pre-specified results were seen): do the recurrence effects
reflect slow (delta) activity?

Recurrence determinism is high for slow, smooth signals, and emotional clips differed from the neutral
documentaries in delta power. For each recurrence outcome, the trial values are standardized within
participant and residualized on within-participant standardized relative delta power (delta / total
1-45 Hz, averaged over channels); the residuals are tested exactly as in 03_stimulus_aware_contrasts.py (min F',
participants and clips random; BH-FDR across the 36 recurrence tests).
Also reports the delta-power contrasts themselves and the gamma (30-45 Hz) contrasts (EMG check).
Output: tables/recurrence_delta_adjustment.csv
"""
import numpy as np, pandas as pd
from lib_rev import load, contrast_vs_neutral, bh, NONNEU, BANDS5, TAB, CACHE

df, arr = load()
rec = np.load(CACHE / "recurrence_features.npz")
for k, nm in enumerate(str(n) for n in rec["rqa_names"]):
    df[f"RQA_{nm}"] = rec["rqa"][:, :, k].mean(1)
df["JRS_mean"] = rec["jrs"].mean(1)
# relative delta power = integrated 1-4 Hz power / integrated 1-45 Hz power. logpow_<band> is the log10 MEAN
# spectral density in the band, so each band mean is weighted by its width before summing (non-overlapping bands).
WIDTH = {"delta": 3, "theta": 4, "alpha": 5, "beta": 17, "gamma": 15}
tot = np.log10(sum(WIDTH[b] * 10 ** arr[f"logpow_{b}"] for b in BANDS5))
df["rel_delta"] = (np.log10(WIDTH["delta"]) + arr["logpow_delta"] - tot).mean(1)
df["log_delta"] = arr["logpow_delta"].mean(1)
df["log_gamma"] = arr["logpow_gamma"].mean(1)
zs = lambda c: df.groupby("subject")[c].transform(lambda x: (x - x.mean()) / x.std()).to_numpy()

OUT = ["RQA_DET", "RQA_LAM", "RQA_Lmean", "RQA_ENTR", "RQA_TT", "JRS_mean"]
x = zs("rel_delta")
rows = []
for o in OUT:
    y = zs(o); b = np.polyfit(x, y, 1); res = y - np.polyval(b, x)
    r_w = float(np.corrcoef(x, y)[0, 1])
    for e in NONNEU:
        raw = contrast_vs_neutral(df[o].to_numpy(float), df, e)
        adj = contrast_vs_neutral(res, df, e)
        rows.append({"outcome": o, "emotion": e, "r_within_rel_delta": r_w,
                     "dz_raw": raw["dz"], "p_minF_raw": raw["p_minF"], "dz_adjusted": adj["dz"], "p_minF_adjusted": adj["p_minF"]})
R = pd.DataFrame(rows)
R["q_minF_raw"] = bh(R.p_minF_raw); R["q_minF_adjusted"] = bh(R.p_minF_adjusted)
R.to_csv(TAB / "recurrence_delta_adjustment.csv", index=False)

print("within-participant r with relative delta:", R.groupby("outcome").r_within_rel_delta.first().round(2).to_dict())
print(f"q<.05 (clips random): raw {(R.q_minF_raw < .05).sum()}/36, adjusted for relative delta {(R.q_minF_adjusted < .05).sum()}/36")
print(R[R.q_minF_adjusted < .05][["outcome", "emotion", "dz_raw", "dz_adjusted", "q_minF_adjusted"]].round(3).to_string())
for c in ("rel_delta", "log_delta", "log_gamma"):
    print(c, {e: (round(r["dz"], 2), round(r["p_minF"], 4)) for e in NONNEU for r in [contrast_vs_neutral(df[c].to_numpy(float), df, e)]})
