"""Overview figure for the README: analysis pipeline (top) and the two headline results (bottom).
Numbers are those reported in the manuscript (faced/tables, seed_vii/tables).
usage: python docs/make_overview_figure.py  ->  docs/overview.png"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
BLUE, ORANGE, NULL = "#2a78d6", "#eb6834", "#b9b8b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig = plt.figure(figsize=(12, 6.0), facecolor=SURF)

# ------------------------------------------------------------------ pipeline
ax = fig.add_axes([0.02, 0.60, 0.96, 0.38]); ax.set_xlim(0, 100); ax.set_ylim(0, 31); ax.axis("off")
steps = [("Data", "FACED\n123 participants × 28 videos\n9 conditions, 30-s epochs\n\nSEED-VII\n20 participants × 80 clips\n7 conditions"),
         ("EEG representations", "Band-specific synchronization\n(wPLI, orthogonalized AEC)\n\nNetwork topology\n(Spearman, MTF, QG)\n\nPersistence, recurrence (RQA)"),
         ("Stimulus-aware inference", "Participants and stimuli as\ncrossed random factors\n(min F')\n\nOne pre-declared family\nof 224 tests, BH-FDR"),
         ("Generalization", "Decoding of unseen stimuli\nvs stimulus-structured\npermutation nulls\n\nLinear models and\ngraph neural networks"),
         ("Replication", "Same code and\npre-specified plan\non SEED-VII\n\n4× more clips\nper emotion")]
w, gap, x0 = 17.6, 2.5, 0.6
for i, (title, body) in enumerate(steps):
    x = x0 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 1), w, 28.5, boxstyle="round,pad=0.3,rounding_size=1.6", fc="white", ec=GRID, lw=1.2))
    ax.add_patch(FancyBboxPatch((x, 24.2), w, 5.3, boxstyle="round,pad=0.3,rounding_size=1.6", fc=BLUE if i < 4 else ORANGE, ec="none"))
    ax.text(x + w / 2, 26.85, title, ha="center", va="center", color="white", fontsize=9.3, fontweight="bold")
    ax.text(x + w / 2, 12.6, body, ha="center", va="center", fontsize=8.3, color=INK, linespacing=1.35)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + w + gap - 0.2, 12.6), xytext=(x + w + 0.5, 12.6), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))

# ------------------------------------------------------------------ result a: tests surviving FDR
a = fig.add_axes([0.07, 0.08, 0.38, 0.40], facecolor=SURF)
data = {"FACED\n(224 tests)": (79, 0), "SEED-VII\n(168 tests)": (18, 0)}
for i, (k, (fixed, rand)) in enumerate(data.items()):
    a.bar(i - 0.19, fixed, 0.34, color=BLUE, edgecolor=SURF, lw=2)
    a.bar(i + 0.19, max(rand, 0.6), 0.34, color=ORANGE, edgecolor=SURF, lw=2)
    a.text(i - 0.19, fixed + 2, str(fixed), ha="center", fontsize=9, color=INK)
    a.text(i + 0.19, 2.6, str(rand), ha="center", fontsize=9, color=INK)
a.set_xticks([0, 1]); a.set_xticklabels(list(data)); a.set_ylim(0, 92)
a.set_ylabel("Emotion-vs-neutral tests with q < 0.05")
a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
         labels=["Stimuli treated as fixed", "Stimuli treated as random"], frameon=False, loc="upper right", fontsize=8.5)
a.set_title("a   Effects disappear once stimulus variability is modeled", loc="left", fontsize=10, color=INK)
for s in ("top", "right"): a.spines[s].set_visible(False)
a.yaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)

# ------------------------------------------------------------------ result b: decoding of unseen stimuli
b = fig.add_axes([0.57, 0.08, 0.38, 0.40], facecolor=SURF)
dec = [("FACED\nunseen videos", 13.7, 6.4, 100 / 9), ("SEED-VII\nunseen films", 29.5, 12.3, 100 / 7)]
for i, (k, obs, nul, chance) in enumerate(dec):
    b.bar(i - 0.19, nul, 0.34, color=NULL, edgecolor=SURF, lw=2)
    b.bar(i + 0.19, obs, 0.34, color=BLUE, edgecolor=SURF, lw=2)
    b.text(i - 0.19, nul - 1.6, f"{nul:.1f}%", ha="center", fontsize=9, color=INK)
    b.text(i + 0.19, obs + 0.8, f"{obs:.1f}%", ha="center", fontsize=9, color=INK)
    b.plot([i - 0.42, i + 0.42], [chance, chance], color=INK2, lw=1, ls=(0, (3, 2)))
    b.text(i - 0.43, chance + 0.5, f"chance {chance:.1f}%", fontsize=7.5, color=INK2, va="bottom", ha="left")
b.set_xticks([0, 1]); b.set_xticklabels([d[0] for d in dec]); b.set_ylim(0, 35); b.set_xlim(-0.6, 1.6)
b.set_ylabel("Balanced accuracy (%)")
b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=NULL), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
         labels=["Permutation null (mean)", "Synchronization + power, p = 0.001"], frameon=False, loc="upper left", fontsize=8.5)
b.set_title("b   Synchronization and power generalize to unseen stimuli", loc="left", fontsize=10, color=INK)
for s in ("top", "right"): b.spines[s].set_visible(False)
b.yaxis.grid(True, color=GRID, lw=0.6); b.set_axisbelow(True)

out = Path(__file__).resolve().parent / "overview.png"
fig.savefig(out, dpi=200, facecolor=SURF); print("saved", out)
