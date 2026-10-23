"""Supplementary Figure S1: robustness of topology profiles to quantile resolution q and density.

For every (representation, q, density) cell we compute the 48-element effect profile
(8 emotion-vs-neutral contrasts x 6 metrics, Cohen's dz) and correlate it with the primary
profile (q = 30, 10%). We also count contrasts with BH q < 0.05 within the cell, with
participants and videos random. Each representation is analysed on its own scale; no metric
is compared across representations.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr
from lib_rev import load, contrast_vs_neutral, bh, style, panel, save, TAB, NONNEU, TOPO, W2

style()
df, _ = load()
QS, DENS = [10, 20, 25, 30, 35, 40], [5, 10, 15, 20, 25, 30]

def profile(rep, q, d):
    rows = []
    for m in TOPO:
        y = df[f"{rep}_q{q}_d{d:02d}_{m}"].to_numpy(float)
        for e in NONNEU:
            rows.append({"metric": m, **contrast_vs_neutral(y, df, e)})
    r = pd.DataFrame(rows); r["q_minF"] = bh(r.p_minF); r["q_p1"] = bh(r.p1)
    return r

out = []
for rep in ("Spearman", "MTF", "QG"):
    base = profile(rep, 30, 10)
    for q in ([30] if rep == "Spearman" else QS):
        for d in DENS:
            p = base if (q, d) == (30, 10) else profile(rep, q, d)
            ok = np.isfinite(p.dz.values) & np.isfinite(base.dz.values)
            out.append({"representation": rep, "q": q, "density": d,
                        "profile_r": spearmanr(p.dz.values[ok], base.dz.values[ok])[0],
                        "sign_agreement": np.mean(np.sign(p.dz.values[ok]) == np.sign(base.dz.values[ok])),
                        "n_q05_minF": int((p.q_minF < .05).sum()), "n_nominal_minF": int((p.p_minF < .05).sum()),
                        "n_q05_participants_only": int((p.q_p1 < .05).sum())})
S = pd.DataFrame(out); S.to_csv(TAB / "quantile_density_sweep.csv", index=False)
print(S.round(3).to_string())

fig = plt.figure(figsize=(W2, 2.5))
gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.2, 1.2], wspace=0.45)
ax = fig.add_subplot(gs[0])
s = S[S.representation == "Spearman"]
ax.plot(s.density, s.profile_r, "-o", color="#2b6cb0", ms=4, lw=1.2)
def mark(ax, x, y, n_random, n_fixed):
    """Same convention as the other figures: filled = q < .05 with participants and videos random;
    open = q < .05 only when videos are treated as fixed (at least one of the 48 contrasts)."""
    if n_random > 0: ax.plot(x, y, "o", ms=3.4, mfc="black", mec="white", mew=0.4, zorder=4)
    elif n_fixed > 0: ax.plot(x, y, "o", ms=3.4, mfc="none", mec="black", mew=0.6, zorder=4)

for _, r in s.iterrows():
    mark(ax, r.density, r.profile_r + 0.08, r.n_q05_minF, r.n_q05_participants_only)
ax.axvline(10, color="#999", lw=0.5, ls=":"); ax.set_ylim(-0.1, 1.17)
ax.set_xlabel("Density (%)"); ax.set_ylabel("Profile correlation with q = 30, 10%")
ax.set_title("Spearman channel network", loc="left"); panel(ax, "a")

for k, rep in enumerate(("MTF", "QG")):
    ax = fig.add_subplot(gs[1 + k])
    M = S[S.representation == rep].pivot(index="q", columns="density", values="profile_r").loc[QS, DENS]
    N = S[S.representation == rep].pivot(index="q", columns="density", values="n_q05_minF").loc[QS, DENS]
    F = S[S.representation == rep].pivot(index="q", columns="density", values="n_q05_participants_only").loc[QS, DENS]
    im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=1, aspect="auto", origin="lower")
    for i in range(len(QS)):
        for j in range(len(DENS)):
            v = M.values[i, j]
            ax.text(j, i - 0.12, f"{v:.2f}", ha="center", va="center", fontsize=4.8, color="white" if v > 0.6 else "#222")
            mark(ax, j, i + 0.25, int(N.values[i, j]), int(F.values[i, j]))
    ax.add_patch(plt.Rectangle((DENS.index(10) - .5, QS.index(30) - .5), 1, 1, fill=False, lw=1.2, ec="#c0392b"))
    ax.set_xticks(range(len(DENS))); ax.set_xticklabels(DENS); ax.set_yticks(range(len(QS))); ax.set_yticklabels(QS)
    ax.set_xlabel("Density (%)"); ax.set_ylabel("Quantile states q")
    ax.set_title(f"{rep} state-transition network", loc="left"); panel(ax, "bc"[k])
    for sp in ax.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=fig.axes[1:], fraction=0.03, pad=0.02); cb.set_label("Profile correlation (Spearman ρ)"); cb.outline.set_visible(False)
fig.text(0.08, -0.08, "Cell values: profile correlation with the primary setting (red box).   ●  ≥1 of 48 contrasts with q < 0.05, participants and videos random   ○  only with videos fixed", fontsize=6)
save(fig, "FigS1_quantile_density_sweep")
