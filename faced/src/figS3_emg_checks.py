"""Supplementary Figure S3: are effects concentrated where/when muscle (EMG) artifact would be expected?

EMG-prone sites: frontopolar and lateral frontotemporal electrodes over frontalis/temporalis
(FP1, FP2, F7, F8, T7, T8). Interior reference sites: FZ, FC1, FC2, CZ, C3, C4, CP1, CP2, PZ.
(a) affective-condition effects on 30-45 Hz log power (an EMG index) at both site groups;
(b) synchronization effects for edges within EMG-prone vs within interior sites, per band;
(c) primary synchronization effects before vs after adjusting each trial for the EMG index.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from lib_rev import (load, contrast_vs_neutral, bh, style, panel, save, TAB, NONNEU, CH, BANDS5,
                     EMO_COLORS, W2)

style()
df, arr = load()
EMGS = ["FP1", "FP2", "F7", "F8", "T7", "T8"]
INT = ["FZ", "FC1", "FC2", "CZ", "C3", "C4", "CP1", "CP2", "PZ"]
iu = np.triu_indices(30, 1)
def edge_mask(chs):
    s = {CH.index(c) for c in chs}
    return np.array([(i in s) and (j in s) for i, j in zip(*iu)])
m_emg, m_int = edge_mask(EMGS), edge_mask(INT)
ci = lambda chs: [CH.index(c) for c in chs]

df["emg_index"] = arr["logpow_gamma"][:, ci(EMGS)].mean(1)
df["gamma_interior"] = arr["logpow_gamma"][:, ci(INT)].mean(1)

rows = []
for e in NONNEU:
    for col, site in (("emg_index", "EMG-prone"), ("gamma_interior", "Interior")):
        rows.append({"analysis": "30-45 Hz power", "site": site, **contrast_vs_neutral(df[col].values, df, e)})
for s in ("wpli", "oaec"):
    for b in BANDS5:
        P = arr[f"pair_{s}_{b}"]
        for site, msk in (("EMG-prone", m_emg), ("Interior", m_int)):
            y = P[:, msk].mean(1)
            for e in NONNEU:
                rows.append({"analysis": f"{s}_{b}", "site": site, **contrast_vs_neutral(y, df, e)})

# (c) adjust primary synchronization for the EMG index, within participant
def zw(v):
    return pd.Series(v).groupby(df.subject.values).transform(lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-12)).values
emg_z = zw(df.emg_index.values)
for s in ("wpli", "oaec"):
    for b in BANDS5:
        y = zw(df[f"{s}_{b}"].values)
        beta = np.sum(y * emg_z) / np.sum(emg_z ** 2)
        yr = y - beta * emg_z
        for e in NONNEU:
            a = contrast_vs_neutral(y, df, e); r = contrast_vs_neutral(yr, df, e)
            rows.append({"analysis": f"{s}_{b}", "site": "adjusted_compare", "emotion": e,
                         "dz": a["dz"], "dz_adjusted": r["dz"], "p_minF": a["p_minF"], "p_minF_adjusted": r["p_minF"],
                         "emg_slope": beta})
E = pd.DataFrame(rows)
E.to_csv(TAB / "emg_checks.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(W2, 2.5), gridspec_kw={"wspace": 0.45})
ax = axes[0]
g = E[E.analysis == "30-45 Hz power"]
x = np.arange(len(NONNEU))
for k, (site, c) in enumerate((("EMG-prone", "#c0392b"), ("Interior", "#2b6cb0"))):
    d = g[g.site == site].set_index("emotion").loc[NONNEU]
    ax.bar(x + (k - 0.5) * 0.38, d.dz, 0.36, color=c, label=site)
    for xi, (_, r) in zip(x, d.iterrows()):
        if r.p_minF < 0.05: ax.text(xi + (k - 0.5) * 0.38, r.dz + np.sign(r.dz) * 0.02, "*", ha="center", va="center", fontsize=7)
ax.axhline(0, color="#999", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([e[:4].capitalize() for e in NONNEU], rotation=45, ha="right")
ax.set_ylabel("30–45 Hz power vs neutral ($d_z$)"); ax.legend(fontsize=5.5)
ax.set_title("EMG index by condition", loc="left"); panel(ax, "a")

ax = axes[1]
s = E[E.analysis.str.contains("_") & E.site.isin(["EMG-prone", "Interior"]) & (E.analysis != "30-45 Hz power")]
pv = s.pivot_table(index=["analysis", "emotion"], columns="site", values="dz").reset_index()
pv["band"] = pv.analysis.str.split("_").str[1]
cols = {"delta": "#c6dbef", "theta": "#9ecae1", "alpha": "#4292c6", "beta": "#2171b5", "gamma": "#c0392b"}
for b in BANDS5:
    d = pv[pv.band == b]; ax.scatter(d["Interior"], d["EMG-prone"], s=8, color=cols[b], lw=0, label=b.capitalize())
lim = np.nanmax(np.abs(pv[["Interior", "EMG-prone"]].values)) * 1.05
ax.plot([-lim, lim], [-lim, lim], color="#999", lw=0.6, ls="--")
ax.set_xlabel("Effect, interior edges ($d_z$)"); ax.set_ylabel("Effect, EMG-prone edges ($d_z$)")
ax.legend(fontsize=5.5, loc="upper left"); ax.set_title("wPLI / oAEC by site and band", loc="left"); panel(ax, "b")

ax = axes[2]
a = E[E.site == "adjusted_compare"]
ax.scatter(a.dz, a.dz_adjusted, s=8, color="#2b6cb0", lw=0)
lim = np.nanmax(np.abs(a[["dz", "dz_adjusted"]].values)) * 1.05
ax.plot([-lim, lim], [-lim, lim], color="#999", lw=0.6, ls="--")
ax.set_xlabel("Primary effect ($d_z$)"); ax.set_ylabel("After EMG-index adjustment ($d_z$)")
ax.text(0.04, 0.92, f"r = {np.corrcoef(a.dz, a.dz_adjusted)[0, 1]:.2f}", transform=ax.transAxes, fontsize=6)
ax.set_title("Robustness to EMG covariate", loc="left"); panel(ax, "c")
fig.text(0.06, -0.06, "* p < 0.05 with participants and videos random (min F′, uncorrected; exploratory check).", fontsize=6)
save(fig, "FigS3_emg_checks")
print(g[["emotion", "site", "dz", "p_minF"]].round(3).to_string())
print("adjusted vs primary r:", np.corrcoef(a.dz, a.dz_adjusted)[0, 1].round(3))
