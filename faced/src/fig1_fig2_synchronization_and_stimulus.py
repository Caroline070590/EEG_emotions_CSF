"""Figure 1: frequency-specific synchronization on 30-s trials.
Figure 2: participants AND videos as random factors."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from lib_rev import (load, style, panel, save, TAB, NONNEU, EMOS, EMO_COLORS, BANDS5, DIVERGING,
                     TOPO_LABEL, W2)

style()
df, _ = load()
R = pd.read_csv(TAB / "all_contrasts.csv")

def heat(ax, sub, cols, labels, title, vmax):
    M = np.array([[sub[(sub.outcome == c) & (sub.emotion == e)].dz.item() for c in cols] for e in NONNEU])
    im = ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for i, e in enumerate(NONNEU):
        for j, c in enumerate(cols):
            r = sub[(sub.outcome == c) & (sub.emotion == e)].iloc[0]
            q = r.q_minF
            if q < 0.05:
                ax.plot(j, i, "o", ms=3.2, mfc="black", mec="white", mew=0.4)
            elif r.q_participant_only < 0.05:
                ax.plot(j, i, "o", ms=3.2, mfc="none", mec="black", mew=0.6)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(NONNEU))); ax.set_yticklabels([e.capitalize() for e in NONNEU])
    ax.set_title(title, loc="left")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    return im

# ---------------------------------------------------------------- Figure R1
fig = plt.figure(figsize=(W2, 2.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.2, 1.3], wspace=0.55)

ax = fig.add_subplot(gs[0])
lo = {"delta": 1, "theta": 4, "alpha": 8, "beta": 13, "gamma": 30}
x = np.arange(5)
ax.bar(x - 0.2, [lo[b] * 1 for b in BANDS5], 0.38, color="#bdbdbd", label="1-s segment (submitted)")
ax.bar(x + 0.2, [lo[b] * 30 for b in BANDS5], 0.38, color="#2b6cb0", label="30-s trial (revision)")
ax.axhline(3, color="#c0392b", lw=0.7, ls="--"); ax.text(4.6, 3.6, "3 cycles", ha="right", fontsize=5.5, color="#c0392b")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([b.capitalize() for b in BANDS5], rotation=45, ha="right")
ax.set_ylabel("Cycles at lower band edge"); ax.legend(loc="upper left", fontsize=5.5)
ax.set_title("Cycles available per window", loc="left"); panel(ax, "a")

sub = R[R.family == "primary"]
cols = [f"wpli_{b}" for b in BANDS5] + [f"oaec_{b}" for b in BANDS5]
labs = [f"wPLI {b}" for b in BANDS5] + [f"oAEC {b}" for b in BANDS5]
vmax = np.nanpercentile(np.abs(R[R.domain == "synchronization"].dz), 99)
ax = fig.add_subplot(gs[1])
im = heat(ax, sub, cols, labs, "Primary: band-specific synchronization (30 s)", vmax)
ax.axvline(4.5, color="white", lw=2); panel(ax, "b")

ex = R[R.family == "exploratory"]
cols = ["plv_broad", "pli_broad", "wpli_broad", "aec_broad", "maxxc_broad", "lag_ms_broad"]
ax = fig.add_subplot(gs[2])
heat(ax, ex, cols, ["PLV", "PLI", "wPLI", "AEC", "MaxXC", "Lag"], "Exploratory: 1–45 Hz broadband", vmax)
panel(ax, "c")
cb = fig.colorbar(im, ax=fig.axes[1:], fraction=0.025, pad=0.02); cb.set_label("Effect vs neutral (Cohen's $d_z$)")
cb.outline.set_visible(False)
fig.text(0.40, -0.06, "●  FDR q < 0.05 with participants and videos random (primary family)     "
                      "○  q < 0.05 only when videos are treated as fixed", fontsize=6)
save(fig, "Fig1_band_specific_synchronization")

# ---------------------------------------------------------------- Figure R2
P = R[R.family == "primary"].copy()
fig = plt.figure(figsize=(W2, 3.0))
gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 0.8, 1.1, 1.1], wspace=0.75)

ax = fig.add_subplot(gs[0])
for dom, c, mk in (("topology", "#8c8c8c", "s"), ("synchronization", "#2b6cb0", "o")):
    d = P[P.domain == dom]
    ax.scatter(d.t1, d.t2, s=9, c=c, marker=mk, lw=0, alpha=0.8, label=dom.capitalize())
sig = P[P.q_minF < .05]
ax.scatter(sig.t1, sig.t2, s=16, facecolor="none", edgecolor="black", lw=0.6, label="q < 0.05 (min F′)")
from scipy import stats
for v, ls in ((stats.t.ppf(.975, 122), ":"),):
    ax.axvline(v, color="#999", lw=0.5, ls=ls); ax.axvline(-v, color="#999", lw=0.5, ls=ls)
v2 = stats.t.ppf(.975, 19); ax.axhline(v2, color="#999", lw=0.5, ls=":"); ax.axhline(-v2, color="#999", lw=0.5, ls=":")
ax.axhline(0, color="#ccc", lw=0.4); ax.axvline(0, color="#ccc", lw=0.4)
ax.set_xlabel("By-participant t (df = 122)"); ax.set_ylabel("By-video t (df = 19)")
ax.set_title("Participants vs videos", loc="left"); ax.legend(loc="upper left", fontsize=5.5)
panel(ax, "a")

ax = fig.add_subplot(gs[1])
cnt = [(P.q_participant_only < .05).sum(), (P.q_minF < .05).sum()]
ax.bar([0, 1], cnt, color=["#bdbdbd", "#2b6cb0"], width=0.6)
for i, c in enumerate(cnt): ax.text(i, c, str(c), ha="center", va="bottom", fontsize=6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Videos\nfixed", "Videos\nrandom"])
ax.set_ylabel(f"Primary tests with q < 0.05 (of {len(P)})"); ax.set_title("Surviving FDR", loc="left")
panel(ax, "b")

top = P.assign(a=P.t1.abs()).sort_values("a", ascending=False).drop_duplicates("outcome").head(2)
for k, (_, r) in enumerate(top.iterrows()):
    ax = fig.add_subplot(gs[2 + k])
    vm = df.groupby(["video", "emotion"])[r.outcome].mean().reset_index()
    for i, e in enumerate(EMOS):
        v = vm[vm.emotion == e][r.outcome].values
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=11, color=EMO_COLORS[e], lw=0.3, edgecolor="white", zorder=3)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color="#222", lw=0.8, zorder=2)
    ax.set_xticks(range(9)); ax.set_xticklabels([e[:4].capitalize() for e in EMOS], rotation=60, ha="right")
    name = r.outcome.replace("_q30_d10_", " ").replace("_", " ").replace("oaec", "oAEC").replace("wpli", "wPLI")
    for k2, v in TOPO_LABEL.items(): name = name.replace(k2.replace("_", " "), v)
    ax.set_ylabel(f"{name} (video mean)")
    ax.set_title(f"{r.emotion.capitalize()} vs neutral\np (videos random) = {r.p_minF:.3f}", loc="left", fontsize=6)
    panel(ax, "cd"[k])
save(fig, "Fig2_participants_and_videos_random")
