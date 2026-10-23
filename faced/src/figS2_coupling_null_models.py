"""Supplementary Figure S2: how much synchronization-topology coupling is expected from shared signal
structure alone?

Same model as the manuscript (within-participant z-scores, linear mixed model with a
participant random intercept), fitted to four versions of the data:
  observed 30 s         : synchronization and Spearman topology from the same 30-s trial
  same half (15 s)      : both from the same 15-s half (length-matched reference)
  independent halves    : synchronization from one half, topology from the other half
                          (no shared samples; averages of both directions)
  phase-randomised      : multivariate phase-randomised surrogate of each trial, which keeps
                          every auto- and cross-spectrum (all linear inter-channel structure)
                          but removes everything else.
"""
import warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from lib_rev import load, style, panel, save, TAB, TOPO, TOPO_LABEL, W2

warnings.filterwarnings("ignore")
style()
df, _ = load()
SYNC = ["plv", "pli", "wpli", "aec", "maxxc", "lag_ms"]
SYNC_LABEL = dict(zip(SYNC, ["PLV", "PLI", "wPLI", "AEC", "MaxXC", "Lag"]))

def zw(s):
    return s.groupby(df.subject).transform(lambda v: (v - v.mean()) / (v.std(ddof=1) + 1e-12))

def beta(xcol, ycol):
    d = pd.DataFrame({"x": zw(df[xcol]), "y": zw(df[ycol]), "subject": df.subject}).dropna()
    f = smf.mixedlm("y ~ x", d, groups=d.subject).fit(reml=False)
    return f.params["x"], f.bse["x"]

rows = []
for s in SYNC:
    for m in TOPO:
        obs = beta(f"{s}_broad", f"Spearman_q30_d10_{m}")
        same = np.mean([beta(f"h1_{s}", f"h1_Spearman_{m}")[0], beta(f"h2_{s}", f"h2_Spearman_{m}")[0]])
        cross = np.mean([beta(f"h1_{s}", f"h2_Spearman_{m}")[0], beta(f"h2_{s}", f"h1_Spearman_{m}")[0]])
        surr = beta(f"surr_{s}", f"surr_Spearman_{m}")
        rows.append({"sync": s, "topology": m, "beta_observed": obs[0], "se_observed": obs[1],
                     "beta_same_half": same, "beta_independent_halves": cross,
                     "beta_surrogate": surr[0], "se_surrogate": surr[1]})
C = pd.DataFrame(rows)
C["observed_minus_surrogate"] = C.beta_observed - C.beta_surrogate
C["independent_over_same_half"] = C.beta_independent_halves / C.beta_same_half
C.to_csv(TAB / "coupling_null_models.csv", index=False)
print(C.round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [2.2, 1, 1], "wspace": 0.5})
ax = axes[0]
lab = [f"{SYNC_LABEL[r.sync]}–{TOPO_LABEL[r.topology]}" for _, r in C.iterrows()]
x = np.arange(len(C))
for col, c, mk, name in (("beta_observed", "#222222", "o", "Observed (30 s)"),
                         ("beta_same_half", "#7f8c8d", "^", "Same 15-s half"),
                         ("beta_independent_halves", "#2b6cb0", "D", "Independent halves"),
                         ("beta_surrogate", "#e67e22", "s", "Phase-randomised surrogate")):
    ax.scatter(x, C[col], s=9, marker=mk, color=c, label=name, lw=0, zorder=3)
ax.axhline(0, color="#bbb", lw=0.5)
for k in range(1, 6): ax.axvline(k * 6 - 0.5, color="#eee", lw=0.6, zorder=0)
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=90, fontsize=4.6)
ax.set_ylabel("Standardised coupling β"); ax.legend(ncol=2, loc="upper left", fontsize=5.5)
ax.set_title("Coupling β under three null models", loc="left"); panel(ax, "a")

ax = axes[1]
lim = np.nanmax(np.abs(C[["beta_observed", "beta_surrogate"]].values)) * 1.05
ax.scatter(C.beta_surrogate, C.beta_observed, s=10, color="#e67e22", lw=0)
ax.plot([-lim, lim], [-lim, lim], color="#999", lw=0.6, ls="--")
r = np.corrcoef(C.beta_surrogate, C.beta_observed)[0, 1]
ax.text(0.04, 0.94, f"r = {r:.2f}", transform=ax.transAxes, fontsize=6)
ax.set_xlabel("β, linear-structure surrogate"); ax.set_ylabel("β, observed")
ax.set_title("Linear-structure surrogate", loc="left"); panel(ax, "b")

ax = axes[2]
ax.scatter(C.beta_same_half, C.beta_independent_halves, s=10, color="#2b6cb0", lw=0)
lim2 = np.nanmax(np.abs(C[["beta_same_half", "beta_independent_halves"]].values)) * 1.05
ax.plot([-lim2, lim2], [-lim2, lim2], color="#999", lw=0.6, ls="--"); ax.axhline(0, color="#ddd", lw=0.4)
ax.set_xlabel("β, same 15-s half"); ax.set_ylabel("β, independent halves")
ax.set_title("No shared samples", loc="left"); panel(ax, "c")
save(fig, "FigS2_coupling_null_models")
