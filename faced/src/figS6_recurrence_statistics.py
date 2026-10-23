"""Stimulus-aware tests for the recurrence features (family declared before results were seen).

Family (48 tests, BH-FDR once): channel-averaged DET, LAM, L_mean, ENTR, TT and edge-averaged
joint-recurrence synchronization (JRS), each for the 8 emotion-vs-neutral contrasts, with
participants and videos as random factors (min F'). Supplementary Figure S6 + Table R10.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from lib_rev import (load, contrast_vs_neutral, omnibus, bh, style, panel, save, TAB, CACHE,
                     NONNEU, EMOS, EMO_COLORS, DIVERGING, W2)

style()
df, _ = load()
rec = np.load(CACHE / "recurrence_features.npz")
names = [str(n) for n in rec["rqa_names"]]
for k, nm in enumerate(names):
    df[f"RQA_{nm}"] = rec["rqa"][:, :, k].mean(1)
df["JRS_mean"] = rec["jrs"].mean(1)
OUT = [f"RQA_{n}" for n in names] + ["JRS_mean"]
LAB = {"RQA_DET": "Determinism", "RQA_LAM": "Laminarity", "RQA_Lmean": "Mean diagonal line",
       "RQA_ENTR": "Line entropy", "RQA_TT": "Trapping time", "JRS_mean": "Joint-recurrence sync."}

rows, om = [], []
for o in OUT:
    y = df[o].to_numpy(float)
    om.append({"outcome": o, **omnibus(y, df)})
    for e in NONNEU:
        rows.append({"outcome": o, **contrast_vs_neutral(y, df, e)})
R = pd.DataFrame(rows); R["q_minF"] = bh(R.p_minF); R["q_participant_only"] = bh(R.p1)
O = pd.DataFrame(om); O["q_minF"] = bh(O.p_minF)
R.to_csv(TAB / "recurrence_contrasts.csv", index=False); O.to_csv(TAB / "recurrence_omnibus.csv", index=False)
print("participant-only q<.05:", (R.q_participant_only < .05).sum(), "| videos random q<.05:", (R.q_minF < .05).sum(),
      "| nominal p_minF<.05:", (R.p_minF < .05).sum())
print(R.sort_values("p_minF")[["outcome", "emotion", "dz", "t1", "t2", "p_minF", "q_minF"]].head(10).round(4).to_string())
print(O[["outcome", "F1", "p1", "F2", "p2", "p_minF", "q_minF"]].round(4).to_string())

fig = plt.figure(figsize=(W2, 3.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.75)
ax = fig.add_subplot(gs[0])
M = np.array([[R[(R.outcome == o) & (R.emotion == e)].dz.item() for o in OUT] for e in NONNEU])
v = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(0, -v, v), aspect="auto")
for i, e in enumerate(NONNEU):
    for j, o in enumerate(OUT):
        r = R[(R.outcome == o) & (R.emotion == e)].iloc[0]
        if r.q_minF < .05: ax.plot(j, i, "o", ms=3.2, mfc="black", mec="white", mew=0.4)
        elif r.q_participant_only < .05: ax.plot(j, i, "o", ms=3.2, mfc="none", mec="black", mew=0.6)
ax.set_xticks(range(len(OUT))); ax.set_xticklabels([LAB[o] for o in OUT], rotation=45, ha="right")
ax.set_yticks(range(len(NONNEU))); ax.set_yticklabels([e.capitalize() for e in NONNEU]); ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("$d_z$ vs neutral"); cb.outline.set_visible(False)
ax.set_title("Recurrence measures vs neutral", loc="left"); panel(ax, "a")

top = R.assign(a=R.t1.abs()).sort_values("a", ascending=False).drop_duplicates("outcome").head(2)
for k, (_, r) in enumerate(top.iterrows()):
    ax = fig.add_subplot(gs[1 + k])
    vm = df.groupby(["video", "emotion"])[r.outcome].mean().reset_index()
    for i, e in enumerate(EMOS):
        vv = vm[vm.emotion == e][r.outcome].values
        ax.scatter(np.full(len(vv), i) + np.linspace(-0.12, 0.12, len(vv)), vv, s=11, color=EMO_COLORS[e], lw=0.3, edgecolor="white", zorder=3)
        ax.hlines(vv.mean(), i - 0.3, i + 0.3, color="#222", lw=0.8)
    ax.set_xticks(range(9)); ax.set_xticklabels([e[:4].capitalize() for e in EMOS], rotation=60, ha="right")
    ax.set_ylabel(f"{LAB[r.outcome]} (video mean)")
    ax.set_title(f"{r.emotion.capitalize()} vs neutral\np (videos random) = {r.p_minF:.4f}", loc="left", fontsize=6)
    panel(ax, "bc"[k])
fig.text(0.05, -0.20, "●  q < 0.05 (participants and videos random; 48-test recurrence family)   ○  q < 0.05 only with videos fixed", fontsize=6)
save(fig, "FigS6_recurrence_quantification")
