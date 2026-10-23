"""Supplementary Figure S4: signed Spearman networks and NBS.

Edges are signed Spearman correlations computed on the complete 30-s trials. NBS is run
separately for increases and decreases relative to neutral (so a component never mixes
positive and negative differences), at primary thresholds |t| = 2.5, 3.0, 3.5, with 5,000
participant sign-flip permutations; BH across the 8 contrasts within each threshold and
direction. The submitted |rho| analysis is rerun on 30-s data for comparison.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
from joblib import Parallel, delayed
from lib_rev import load, bh, style, panel, save, TAB, NONNEU, EMOS, CH, EMO_COLORS, W2

style()
df, arr = load()
iu = np.triu_indices(30, 1)
N_PERM, THR = 5000, [2.5, 3.0, 3.5]

def subj_means(V):
    d = pd.DataFrame(V); d["subject"] = df.subject.values; d["emotion"] = df.emotion.values
    return d.groupby(["subject", "emotion"]).mean()

def max_comp(mask):
    if not mask.any(): return 0, None
    A = np.zeros((30, 30), bool); A[iu[0][mask], iu[1][mask]] = True; A |= A.T
    n, lab = connected_components(A, directed=False)
    sizes = [(mask & (lab[iu[0]] == c)).sum() for c in range(n)]
    c = int(np.argmax(sizes)); return sizes[c], (mask & (lab[iu[0]] == c))

def tstat(D):
    return D.mean(0) / (D.std(0, ddof=1) / np.sqrt(len(D)) + 1e-12)

def nbs(D, thr, sign, seed):
    t = tstat(D) * sign
    size, comp = max_comp(t > thr)
    rng = np.random.default_rng(seed); null = np.empty(N_PERM)
    for k in range(N_PERM):
        f = rng.choice([-1.0, 1.0], len(D))[:, None]
        null[k] = max_comp(tstat(D * f) * sign > thr)[0]
    p = (1 + (null >= size).sum()) / (1 + N_PERM) if size > 0 else 1.0
    return size, p, comp

results, comps = [], {}
jobs = []
for kind, key in (("signed", "rho_signed"), ("absolute", None)):
    V = arr["rho_signed"] if kind == "signed" else np.abs(arr["rho_signed"])
    M = subj_means(V)
    for e in NONNEU:
        D = (M.xs(e, level="emotion") - M.xs("neutral", level="emotion")).to_numpy()
        for thr in THR:
            for sign, direction in ((1, "increase"), (-1, "decrease")):
                jobs.append(((kind, e, thr, direction), D, thr, sign))
out = Parallel(n_jobs=12, verbose=2)(delayed(nbs)(D, thr, sign, 7) for _, D, thr, sign in jobs)
for (meta, *_), (size, p, comp) in zip(jobs, out):
    kind, e, thr, direction = meta
    results.append({"edges": kind, "emotion": e, "threshold": thr, "direction": direction,
                    "component_edges": int(size), "p_fwe": p})
    comps[meta] = comp
R = pd.DataFrame(results)
R["q_bh"] = R.groupby(["edges", "threshold", "direction"]).p_fwe.transform(bh)
R.to_csv(TAB / "signed_nbs.csv", index=False)
print(R[R.threshold == 3.0].round(4).to_string())

# figure
fig = plt.figure(figsize=(W2, 3.0))
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15], hspace=0.75, wspace=0.9)
ax = fig.add_subplot(gs[0, :2])
negf = df.groupby("emotion").rho_neg_fraction.mean().loc[EMOS]
ax.bar(range(9), negf.values * 100, color=[EMO_COLORS[e] for e in EMOS])
ax.set_xticks(range(9)); ax.set_xticklabels([e[:4].capitalize() for e in EMOS], rotation=45, ha="right")
ax.set_ylabel("Negative edges (%)"); ax.set_title("Share of negative Spearman edges", loc="left"); panel(ax, "a")

ax = fig.add_subplot(gs[0, 2:])
s = R[(R.threshold == 3.0) & (R.edges == "signed")]
x = np.arange(len(NONNEU))
for k, (d, c) in enumerate((("increase", "#c0392b"), ("decrease", "#2b6cb0"))):
    v = s[s.direction == d].set_index("emotion").loc[NONNEU]
    ax.bar(x + (k - .5) * .38, v.component_edges * (1 if d == "increase" else -1), .36, color=c, label=f"Signed ρ {d}")
    for xi, (_, r) in zip(x, v.iterrows()):
        if r.q_bh < .05:
            ax.text(xi + (k - .5) * .38, r.component_edges * (1 if d == "increase" else -1), "*", ha="center",
                    va="bottom" if d == "increase" else "top", fontsize=7)
ax.axhline(0, color="#999", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([e[:4].capitalize() for e in NONNEU], rotation=45, ha="right")
ax.set_ylabel("Component edges"); ax.legend(fontsize=5.5, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.0))
lim = s.component_edges.max() * 1.25; ax.set_ylim(-lim, lim * 1.35)
ax.set_title("Signed NBS vs neutral, |t| > 3.0 (* BH q < 0.05)", loc="left"); panel(ax, "b")

# scalp plots of the 4 largest significant signed components
pos = {c: (np.cos(a), np.sin(a)) for c, a in zip(CH, np.linspace(0, 2 * np.pi, 31)[:-1])}
try:
    import mne
    mont = mne.channels.make_standard_montage("standard_1020"); P = mont.get_positions()["ch_pos"]
    lut = {k.upper(): v for k, v in P.items()}
    pos = {c: (lut[c][0], lut[c][1]) for c in CH}
except Exception as err:
    print("WARNING: mne montage unavailable, using circular layout:", err)
sig = s[(s.q_bh < .05)].sort_values("component_edges", ascending=False).head(4)
for k, (_, r) in enumerate(sig.iterrows()):
    ax = fig.add_subplot(gs[1, k]); comp = comps[("signed", r.emotion, 3.0, r.direction)]
    xy = np.array([pos[c] for c in CH])
    for i, j in zip(iu[0][comp], iu[1][comp]):
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color="#c0392b" if r.direction == "increase" else "#2b6cb0", lw=0.5, alpha=0.7)
    ax.scatter(xy[:, 0], xy[:, 1], s=5, color="#222", zorder=3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{r.emotion.capitalize()} {r.direction}\n{int(r.component_edges)} edges, p = {r.p_fwe:.4f}", fontsize=6)
    if k == 0: ax.text(-0.18, 1.32, "c", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
save(fig, "FigS4_signed_nbs")
