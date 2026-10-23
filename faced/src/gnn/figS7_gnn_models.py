"""Supplementary Figure S7 + Tables R11a/b: every decoding model of the revision (baselines, SD-GNN, constrained
GNN, hybrid and summary-first GNNs) - which components it has, how much of a planted effect it
recovers in simulations with independent or real noise structure, and how it performs on the real
data against its own stimulus-structured permutation null (Amendments 1-4)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_rev import style, panel, save, TAB, W2

RES = Path(__file__).resolve().parents[2] / "results" / "gnn"
style()

def read(f):
    d = pd.DataFrame([json.loads(l) for l in open(RES / f)])
    for c, v in (("rex", 0.0), ("recurrence", False)):
        d[c] = d[c].fillna(v) if c in d else v
    d["method"] = d.apply(lambda r: r.model + (f" λ={r.lam}" if r.model in ("sdgnn", "cgnn", "hybrid", "sfgnn") else "")
                          + (" rex" if r.rex > 0 else ""), axis=1)
    d["acc"] = d.balanced_accuracy * 100
    return d

# rows: (method key, label, components, real-data variant uses recurrence features)
# components: message passing, view gates, global-summary path, video adversary, V-REx
ROWS = [("lda", "LDA, all features", "....."), ("mlp", "MLP, all features", "....."),
        ("lda_summary", "LDA, global summaries", "..S.."),
        ("sdgnn λ=0.0", "SD-GNN", "MG..."), ("sdgnn λ=0.3", "SD-GNN + adversary (λ=0.3)", "MG.A."),
        ("sdgnn λ=1.0", "SD-GNN + adversary (λ=1)", "MG.A."),
        ("cgnn λ=0.0", "Constrained GNN", "MC..."), ("cgnn λ=1.0", "Constrained GNN + adversary", "MC.A."),
        ("hybrid λ=0.0", "Hybrid GNN", "MCS.."), ("hybrid λ=1.0", "Hybrid GNN + adversary", "MCSA."),
        ("sfgnn λ=0.0", "Summary-first GNN", "MCs.."), ("sfgnn λ=1.0", "Summary-first + adversary", "MCsA."),
        ("sfgnn λ=0.0 rex", "Summary-first + V-REx", "MCs.V")]
KEYS = [r[0] for r in ROWS]
PRIMARY = {"cgnn λ=1.0": "A1–2", "hybrid λ=1.0": "A3", "sfgnn λ=0.0": "A4"}

# ---------------------------------------------------------------- simulations
s = pd.concat([read("simcal_results.jsonl").assign(data="simcal"), read("sfgnn_sim_results.jsonl").assign(data="simreal")])
s = s[s.method.isin(KEYS)].drop_duplicates(["data", "method", "scale", "kappa", "sim_seed"], keep="last")
key = ["data", "method", "kappa", "sim_seed"]
null = s[s.scale == 0].set_index(key).acc
g = s[s.scale > 0].copy(); g["gain"] = g.acc - g.set_index(key).index.map(null).values
G = g.groupby(["data", "method", "kappa", "scale"]).gain.agg(["mean", "std", "count"]).reset_index()
G.to_csv(TAB / "gnn_simulation_gain.csv", index=False)
sel = G[(G.data == "simreal") & G.method.str.startswith("sfgnn")].groupby("method")["mean"].mean()
print("Amendment-4 selection (mean simreal gain, scale > 0):\n", sel.round(2).to_string())

# ---------------------------------------------------------------- real data
r = pd.concat([read("real_results.jsonl"), read("hybrid_real_results.jsonl"), read("sfgnn_real_results.jsonl")])
r = r.drop_duplicates(["method", "recurrence", "perm"], keep="last")
rows = []
for k, lab, _ in ROWS:
    cand = r[r.method == k]
    # use the variant with a permutation null; prefer the one with recurrence features
    best = None
    for rec in (True, False):
        c = cand[cand.recurrence == rec]
        if len(c) and (best is None or (c.perm > 0).sum() > (best.perm > 0).sum()): best = c
    if best is None: rows.append(dict(key=k, model=lab)); continue
    o = best[best.perm == 0].acc; nl = best[best.perm > 0].acc.values
    rows.append(dict(key=k, model=lab, recurrence=bool(best.recurrence.iloc[0]), observed=o.iloc[0], n_perm=len(nl),
                     null_mean=nl.mean() if len(nl) else np.nan, null_95=np.percentile(nl, 95) if len(nl) else np.nan,
                     gain=o.iloc[0] - nl.mean() if len(nl) else np.nan,
                     p_perm=(1 + (nl >= o.iloc[0]).sum()) / (1 + len(nl)) if len(nl) else np.nan))
T = pd.DataFrame(rows)
T.drop(columns="key").to_csv(TAB / "gnn_real_data.csv", index=False)
print(T.drop(columns="key").round(3).to_string())

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(W2, 4.3))
gs = fig.add_gridspec(1, 3, width_ratios=[0.62, 1.0, 1.15], wspace=0.18)
y = np.arange(len(ROWS))

ax = fig.add_subplot(gs[0])
COLS = ["Message passing", "View gates", "Global summaries", "Video adversary", "V-REx penalty", "RQA inputs"]
for i, (k, lab, comp) in enumerate(ROWS):
    for j, ch in enumerate(comp):
        if ch == ".": ax.plot(j, i, "o", ms=2, color="#dddddd")
        else:
            mk = {"G": "s", "C": "D"}.get(ch, "o"); fc = "white" if ch == "s" else "#2b6cb0"
            ax.plot(j, i, mk, ms=4.2, mfc=fc, mec="#2b6cb0", mew=0.9)
    rec = T.loc[T.key == k, "recurrence"]; has = bool(len(rec) and rec.iloc[0] == True)
    ax.plot(5, i, "o", ms=4.2 if has else 2, color="#2b6cb0" if has else "#dddddd")
ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COLS, fontsize=5.2, rotation=60, ha="left", rotation_mode="anchor"); ax.xaxis.tick_top()
ax.set_yticks(y); ax.set_yticklabels([f"{lab}  [{PRIMARY[k]}]" if k in PRIMARY else lab for k, lab, _ in ROWS], fontsize=6)
ax.set_xlim(-0.6, 5.6); ax.set_ylim(len(ROWS) - 0.5, -0.5); ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
for b in (2.5, 5.5, 7.5, 9.5): ax.axhline(b, color="#eeeeee", lw=0.6, zorder=0)
ax.legend(handles=[Line2D([], [], marker="s", ls="", color="#2b6cb0", ms=4, label="softmax view gates"),
                   Line2D([], [], marker="D", ls="", color="#2b6cb0", ms=4, label="sigmoid gates + L1 (constrained)"),
                   Line2D([], [], marker="o", ls="", color="#2b6cb0", ms=4, label="summary path added to the GNN"),
                   Line2D([], [], marker="o", ls="", mfc="white", mec="#2b6cb0", ms=4, label="summary path first, GNN as residual")],
          fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False)
panel(ax, "a")

ax = fig.add_subplot(gs[1])
specs = [("simcal", 1.0, "#9ecae1", "independent noise, video var. ×1"), ("simcal", 3.0, "#2171b5", "independent noise, video var. ×3"),
         ("simreal", 1.0, "#fdae6b", "real noise structure, video var. ×1"), ("simreal", 3.0, "#d94801", "real noise structure, video var. ×3")]
for k_, (dd, kap, c, lab) in enumerate(specs):
    sub = G[(G.data == dd) & (G.kappa == kap) & (G.scale == 1.0)].set_index("method").reindex(KEYS)
    ax.errorbar(sub["mean"], y + (k_ - 1.5) * 0.17, xerr=sub["std"], fmt="o", ms=2.8, color=c, lw=0.7, label=lab)
ax.axvline(0, color="#999", lw=0.6)
for b in (2.5, 5.5, 7.5, 9.5): ax.axhline(b, color="#eeeeee", lw=0.6, zorder=0)
ax.set_yticks(y); ax.set_yticklabels([]); ax.set_ylim(len(ROWS) - 0.5, -0.5)
ax.set_xlabel("Gain over own null at the real effect size (pp)")
ax.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=1, frameon=False)
ax.set_title("Simulation calibrated to FACED", loc="left"); panel(ax, "b")

ax = fig.add_subplot(gs[2])
for i, row in T.iterrows():
    sub = r[(r.method == row.key) & (r.recurrence == row.get("recurrence", False)) & (r.perm > 0)].acc.values
    if len(sub):
        vp = ax.violinplot(sub, positions=[i], vert=False, widths=0.75, showextrema=False)
        for b in vp["bodies"]: b.set_facecolor("#d0d0d0"); b.set_alpha(1); b.set_edgecolor("none")
    if not np.isnan(row.get("observed", np.nan)):
        ax.plot(row.observed, i, "o", color="#2b6cb0", ms=4, zorder=3)
        txt = f"{row.gain:+.1f} pp, p = {row.p_perm:.3f}" if row.n_perm else "observed only"
        ax.text(33.5, i, txt, fontsize=5.3, va="center", ha="right", color="#222" if row.n_perm else "#888")
ax.axvline(100 / 9, color="#c0392b", lw=0.7, ls="--", zorder=0)
for b in (2.5, 5.5, 7.5, 9.5): ax.axhline(b, color="#eeeeee", lw=0.6, zorder=0)
ax.set_yticks(y); ax.set_yticklabels([]); ax.set_ylim(len(ROWS) - 0.5, -0.5); ax.set_xlim(0, 34)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#2b6cb0", label="observed"),
                   Patch(color="#d0d0d0", label="permutation null (200)"),
                   Line2D([], [], color="#c0392b", ls="--", label="nominal chance")],
          fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=1, frameon=False)
ax.set_xlabel("Balanced accuracy, unseen videos (%)")
ax.set_title("Real data: leave-one-video-out", loc="left"); panel(ax, "c")
save(fig, "FigS7_graph_neural_networks")
