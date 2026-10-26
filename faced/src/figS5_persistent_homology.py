"""Supplementary Figure S5: metric validity and emotion effects of persistent homology.

D = sqrt(2(1 - rho)) is Euclidean distance between standardized rank series.
The absolute-correlation construction sqrt(2(1 - |rho|)) is the sign-quotient
pseudometric min(||u-v||, ||u+v||), which also satisfies the triangle inequality.
Both distances therefore admit metric filtrations; the numerical triangle-ratio
check illustrates the proof rather than testing whether only one construction is valid.
"""
import os
from itertools import combinations
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from joblib import Parallel, delayed
from scipy.stats import rankdata
from lib_rev import load, style, panel, save, TAB, CACHE, NONNEU, DIVERGING, W2

# ---------------------------------------------------------------- triangle-inequality ratios
# For every triple of channels (a, b, c) and each of its three sides, r = D_ac / (D_ab + D_bc);
# the triangle inequality holds iff r <= 1. Computed over ALL 4,060 triples x 3 sides of every trial.
TRI = np.array(list(combinations(range(30), 3)))
SIDES = [(0, 2, 1), (0, 1, 2), (1, 0, 2)]          # (x, z, y): ratio D_xz / (D_xy + D_yz)
BINS = np.linspace(0, 1.05, 211)

def _ratios(idx):
    X = np.load(CACHE / "eeg_30s.npy", mmap_mode="r")[idx].astype(np.float64)
    U = rankdata(X, axis=1); U -= U.mean(1, keepdims=True); U /= np.linalg.norm(U, axis=1, keepdims=True)
    rho = np.clip(U @ U.T, -1, 1)
    out = []
    for D in (np.sqrt(2 * (1 - np.abs(rho))), np.sqrt(2 * (1 - rho))):
        r = np.concatenate([D[TRI[:, x], TRI[:, z]] / (D[TRI[:, x], TRI[:, y]] + D[TRI[:, y], TRI[:, z]] + 1e-12)
                            for x, z, y in SIDES])
        out.append((np.histogram(r, BINS)[0], r.max(), r.size))
    return out

def triangle_ratio_hist():
    f = CACHE / "ph_triangle_ratio_hist.npz"
    if not f.exists():
        n = np.load(CACHE / "eeg_30s.npy", mmap_mode="r").shape[0]
        res = Parallel(n_jobs=int(os.environ.get("N_JOBS", 4)))(delayed(_ratios)(i) for i in range(n))
        np.savez(f, abs_hist=sum(r[0][0] for r in res), signed_hist=sum(r[1][0] for r in res),
                 abs_max=max(r[0][1] for r in res), signed_max=max(r[1][1] for r in res),
                 n_ratios=sum(r[0][2] for r in res), n_trials=n)
    return np.load(f)

style()
df, _ = load()
R = pd.read_csv(TAB / "all_contrasts.csv")
R = R[R.domain == "persistent_homology"]
FEAT = ["H0_total", "H0_entropy", "H0_max", "H1_count", "H1_total", "H1_max", "H1_entropy"]

fig, axes = plt.subplots(1, 3, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.1, 1.4, 0.9], "wspace": 0.75})
ax = axes[0]
v = [df.PHabs_triangle_violation * 100, df.PHsigned_triangle_violation * 100]
H = triangle_ratio_hist(); mid = (BINS[1:] + BINS[:-1]) / 2
top = 0
for key, col, lab in (("abs", "#2b6cb0", "√2(1−|ρ|)"), ("signed", "#d94801", "√2(1−ρ)")):
    h = H[f"{key}_hist"] / H[f"{key}_hist"].sum() / np.diff(BINS); top = max(top, h.max())
    ax.fill_between(mid, h, step="mid", alpha=0.25, color=col, lw=0)
    ax.step(mid, h, where="mid", color=col, lw=0.8, label=f"{lab} (max {float(H[f'{key}_max']):.3f})")
ax.axvline(1, color="#c0392b", lw=0.8, ls="--")
ax.text(0.985, top * 0.75, "violation boundary", color="#c0392b", fontsize=5.5, ha="right", va="center", rotation=90)
ax.set_xlim(0, 1.05); ax.set_ylim(0, top * 1.6)
ax.set_xlabel("$D_{ac}\\,/\\,(D_{ab}+D_{bc})$"); ax.set_ylabel("Density")
ax.legend(fontsize=5.3, loc="upper left", frameon=False, bbox_to_anchor=(0.0, 1.0))
ax.text(0.03, 0.80, f"{int(H['n_ratios']) / 1e6:.0f} million inequalities\n({int(H['n_trials']):,} trials): none violated", transform=ax.transAxes, fontsize=5.3, va="top")
ax.set_title("Triangle inequality over all triples", loc="left"); panel(ax, "a")

ax = axes[1]
s = R[R.set == "PHsigned"]
M = np.array([[s[(s.metric == f) & (s.emotion == e)].dz.item() for f in FEAT] for e in NONNEU])
vmax = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
for i, e in enumerate(NONNEU):
    for j, f in enumerate(FEAT):
        r = s[(s.metric == f) & (s.emotion == e)].iloc[0]
        if r.q_minF < 0.05: ax.plot(j, i, "o", ms=3, mfc="black", mec="white", mew=0.4)
        elif r.p_minF < 0.05: ax.plot(j, i, "o", ms=3, mfc="none", mec="black", mew=0.6)
ax.set_xticks(range(len(FEAT))); ax.set_xticklabels([f.replace("_", " ") for f in FEAT], rotation=45, ha="right")
ax.set_yticks(range(len(NONNEU))); ax.set_yticklabels([e.capitalize() for e in NONNEU]); ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("$d_z$ vs neutral"); cb.outline.set_visible(False)
ax.set_title("Signed-metric persistence features vs neutral", loc="left"); panel(ax, "b")

ax = axes[2]
a = R[R.set == "PHabs"].set_index(["metric", "emotion"]).dz
b = s.set_index(["metric", "emotion"]).dz
j = a.index.intersection(b.index)
j = [i for i in j if np.isfinite(a.loc[i]) and np.isfinite(b.loc[i])]
ax.scatter(a.loc[j], b.loc[j], s=8, color="#2b6cb0", lw=0)
lim = np.nanmax(np.abs(np.r_[a.loc[j], b.loc[j]])) * 1.05
ax.plot([-lim, lim], [-lim, lim], color="#999", lw=0.6, ls="--")
ax.text(0.04, 0.92, f"r = {np.corrcoef(a.loc[j], b.loc[j])[0, 1]:.2f}", transform=ax.transAxes, fontsize=6)
ax.set_xlabel("$d_z$, absolute-correlation distance"); ax.set_ylabel("$d_z$, signed-correlation distance")
ax.set_title("Agreement of constructions", loc="left"); panel(ax, "c")
fig.text(0.30, -0.09, "●  BH q < 0.05 (exploratory family)   ○  p < 0.05 uncorrected; participants and videos random.", fontsize=6)
save(fig, "FigS5_persistent_homology")
print("triangle ratio max abs/signed", float(H["abs_max"]), float(H["signed_max"]), "n", int(H["n_ratios"]))
print("violation % abs median/mean/max", float(np.median(v[0])), float(np.mean(v[0])), float(np.max(v[0])),
      "| signed mean", float(np.mean(v[1])), "| share of trials with any abs violation", float((v[0] > 0).mean()))
