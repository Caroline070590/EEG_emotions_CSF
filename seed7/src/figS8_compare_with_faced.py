"""SEED-VII vs FACED: side-by-side summary of the three conclusions under test.

1. Primary family (topology + band-specific synchronization): tests with q < .05 when stimuli are
   fixed vs random.
2. Recurrence family: the same counts.
3. Decoding on unseen stimuli: balanced accuracy, its own permutation null and the gain over it.
Plus a direction check: for the four emotions present in both datasets (anger, disgust, fear,
sadness) and the same outcomes, the correlation of the effect sizes (d_z vs neutral).

Outputs: tables/comparison_with_faced_summary.csv, tables/comparison_with_faced_effect_sizes.csv,
figures/SEED7_vs_FACED.{pdf,png}, REPLICATION_REPORT.md
usage: figS8_compare_with_faced.py [FACED_TABLES_DIR]
"""
import sys, os
os.environ.setdefault("MPLBACKEND", "Agg")   # the default Qt backend blocks interpreter exit on this machine
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from lib_rev import TAB, FIG, HERE, style, panel, save, W2

FT = Path(sys.argv[1] if len(sys.argv) > 1 else HERE.parent / "faced" / "tables")
style()

# ---------------------------------------------------------------- FACED
f1 = pd.read_csv(FT / "all_contrasts.csv"); f1 = f1[f1.family == "primary"]
f10 = pd.read_csv(FT / "recurrence_contrasts.csv")
f8 = pd.read_csv(FT / "decoding_unseen_videos.csv")
fl = pd.read_csv(FT / "gnn_real_data.csv") if (FT / "gnn_real_data.csv").exists() else None
# ---------------------------------------------------------------- SEED-VII
s1 = pd.read_csv(TAB / "all_contrasts.csv")
sp, sr = s1[s1.family == "primary"], s1[s1.family == "recurrence"]
s3 = pd.read_csv(TAB / "decoding_unseen_sources.csv")

def counts(d, ds, fam):
    out = {"dataset": ds, "family": fam, "tests": len(d),
           "q<.05 stimuli fixed": int((d.q_participant_only < .05).sum()),
           "q<.05 stimuli random": int((d.q_minF < .05).sum()),
           "nominal p<.05 stimuli random": int((d.p_minF < .05).sum())}
    if "q_minF_source" in d: out["q<.05 sources random"] = int((d.q_minF_source < .05).sum())
    return out

S = pd.DataFrame([counts(f1, "FACED", "primary (topology + sync)"), counts(sp, "SEED-VII", "primary (topology + sync)"),
                  counts(f10, "FACED", "recurrence (RQA + JRS)"), counts(sr, "SEED-VII", "recurrence (RQA + JRS)")])
for dom in ("topology", "synchronization"):
    S = pd.concat([S, pd.DataFrame([counts(f1[f1.domain == dom], "FACED", f"  of which {dom}"),
                                    counts(sp[sp.domain == dom], "SEED-VII", f"  of which {dom}")])])
S.to_csv(TAB / "comparison_with_faced_summary.csv", index=False)

# decoding: FACED leave-one-video-out; SEED-VII leave-one-source-out (primary) and leave-one-clip-out
f8 = f8.assign(dataset="FACED", folds="leave-one-video-out", chance=1 / 9, gain_over_null=f8.balanced_accuracy - f8.null_mean)
D = pd.concat([f8[["dataset", "folds", "features", "balanced_accuracy", "chance", "null_mean", "gain_over_null", "p_perm"]],
               s3.assign(dataset="SEED-VII")[["dataset", "folds", "features", "balanced_accuracy", "chance", "null_mean", "gain_over_null", "p_perm"]]])
D.to_csv(TAB / "comparison_with_faced_decoding.csv", index=False)

# effect-size direction check on shared emotions and identical outcomes
SH = ["anger", "disgust", "fear", "sadness"]
cols = ["outcome", "emotion", "dz", "p_minF", "q_minF", "q_participant_only"]
E = pd.concat([f1, f10.assign(family="recurrence")])[cols].merge(s1[cols], on=["outcome", "emotion"], suffixes=("_FACED", "_SEED7"))
E = E[E.emotion.isin(SH)]
E.to_csv(TAB / "comparison_with_faced_effect_sizes.csv", index=False)
r, p = stats.spearmanr(E.dz_FACED, E.dz_SEED7)
sign = float((np.sign(E.dz_FACED) == np.sign(E.dz_SEED7)).mean())
sig_both = E[(E.q_participant_only_FACED < .05) & (E.q_participant_only_SEED7 < .05)]
sign_sig = float((np.sign(sig_both.dz_FACED) == np.sign(sig_both.dz_SEED7)).mean()) if len(sig_both) else np.nan

# ---------------------------------------------------------------- figure
fig, axs = plt.subplots(1, 3, figsize=(W2 * 1.1, 2.5), gridspec_kw={"wspace": 1.05, "width_ratios": [1.15, 0.95, 1.0]})
ax = axs[0]; x = np.arange(2); w = 0.2
for k, (fam, ds, col) in enumerate([("primary (topology + sync)", "FACED", "#9fb3c8"), ("primary (topology + sync)", "SEED-VII", "#2b6cb0"),
                                    ("recurrence (RQA + JRS)", "FACED", "#f2c6a0"), ("recurrence (RQA + JRS)", "SEED-VII", "#c05621")]):
    r_ = S[(S.family == fam) & (S.dataset == ds)].iloc[0]
    v = 100 * np.array([r_["q<.05 stimuli fixed"], r_["q<.05 stimuli random"]]) / r_.tests
    ax.bar(x + (k - 1.5) * w, v, w, color=col, label=f"{ds}, {fam.split()[0]} ({r_.tests} tests)")
    for xi, vi in zip(x + (k - 1.5) * w, v): ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=5)
ax.set_xticks(x); ax.set_xticklabels(["Stimuli fixed", "Stimuli random"]); ax.set_ylabel("Tests with q < .05 (%)")
ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=1)
ax.set_title("Effects vs neutral that survive FDR", loc="left"); panel(ax, "a")

ax = axs[1]
ax.axhline(0, color="#999", lw=0.5); ax.axvline(0, color="#999", lw=0.5)
for fam, col in (("primary", "#2b6cb0"), ("recurrence", "#c05621")):
    e = E[E.outcome.isin((sp if fam == "primary" else sr).outcome)]
    ax.scatter(e.dz_FACED, e.dz_SEED7, s=6, color=col, alpha=0.7, lw=0, label=fam)
ax.set_xlabel("$d_z$ vs neutral, FACED"); ax.set_ylabel("$d_z$ vs neutral, SEED-VII")
ax.set_title(f"Same outcome, same emotion\nSpearman r = {r:.2f}, same sign {100 * sign:.0f}%", loc="left", fontsize=6)
ax.legend(fontsize=5); panel(ax, "b")

ax = axs[2]
rows = [("FACED", "leave-one-video-out"), ("SEED-VII", "leave-one-source-out")]
feats = list(dict.fromkeys(D.features))
for k, (ds, fo) in enumerate(rows):
    d = D[(D.dataset == ds) & (D.folds == fo)].set_index("features").reindex(feats)
    y = np.arange(len(feats)) + (k - 0.5) * 0.35
    ax.barh(y, 100 * d.gain_over_null, 0.33, color=["#9fb3c8", "#2b6cb0"][k], label=f"{ds} ({fo})")
    for yi, (g, pp) in zip(y, zip(d.gain_over_null, d.p_perm)):
        if np.isfinite(g): ax.text(100 * max(g, 0) + 0.3, yi, f"p={pp:.3f}", va="center", fontsize=4.5)
ax.axvline(0, color="#222", lw=0.6); ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=5.5)
ax.invert_yaxis(); ax.set_xlabel("Accuracy above own null (points)")
ax.set_title("Decoding on unseen stimuli", loc="left"); ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(0, -0.22)); panel(ax, "c")
save(fig, "FigS8_seed7_replication")

# ---------------------------------------------------------------- report
def row(ds, fam):
    r_ = S[(S.dataset == ds) & (S.family == fam)].iloc[0]
    src = f" / {r_['q<.05 sources random']}" if pd.notna(r_.get("q<.05 sources random", np.nan)) else ""
    return f"| {ds} | {fam.strip()} | {r_.tests} | {r_['q<.05 stimuli fixed']} | {r_['q<.05 stimuli random']}{src} |"
lines = ["# SEED-VII replication vs FACED", "",
         "Generated automatically by `src/figS8_compare_with_faced.py`. Plan: `ANALYSIS_PLAN.md`.", "",
         "## 1-2. Emotion-vs-neutral tests that survive BH-FDR", "",
         "| Dataset | Family | Tests | q<.05, stimuli fixed | q<.05, stimuli random (SEED-VII: clips / sources) |", "|---|---|---|---|---|"]
for fam in S.family.unique():
    for ds in ("FACED", "SEED-VII"):
        if ((S.dataset == ds) & (S.family == fam)).any(): lines.append(row(ds, fam))
lines += ["", "## 3. Decoding on unseen stimuli", "", "| Dataset | Folds | Features | Accuracy | Chance | Null mean | Gain | p |", "|---|---|---|---|---|---|---|---|"]
for _, d in D.iterrows():
    lines.append(f"| {d.dataset} | {d.folds} | {d.features} | {100*d.balanced_accuracy:.1f}% | {100*d.chance:.1f}% | {100*d.null_mean:.1f}% | {100*d.gain_over_null:+.1f} | {d.p_perm:.3f} |")
lines += ["", "## Direction check (anger, disgust, fear, sadness; identical outcomes)", "",
          f"* {len(E)} outcome x emotion pairs; Spearman r of d_z = {r:.2f} (p = {p:.3g}); same sign in {100*sign:.0f}%.",
          f"* Pairs significant with stimuli fixed in both datasets: {len(sig_both)}; same sign in "
          + (f"{100*sign_sig:.0f}%." if len(sig_both) else "n/a."), "",
          "## Reading guide", "",
          "* FACED conclusion 1 replicates if SEED-VII, like FACED, has many fewer surviving tests with stimuli random than fixed.",
          "* Conclusion 3 replicates if band-specific synchronization + power decodes above its own null on held-out sources.",
          "* SEED-VII differs from FACED in culture, lab, montage, 7 vs 9 conditions, and 20 vs 123 participants (fewer",
          "  participants lowers power with stimuli fixed; 12 vs 3 clips per emotion raises it with stimuli random)."]
(HERE / "REPLICATION_REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
