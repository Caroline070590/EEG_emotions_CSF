"""Figure 3: formal test of affective-condition decoding that generalises to unseen videos.

Leave-one-video-out: 28 folds; each fold trains a shrinkage LDA classifier (Ledoit-Wolf) on the 27
remaining videos (all participants) and predicts the 123 trials of the held-out video. Features
are z-scored within participant (label-free). Balanced accuracy is compared with a permutation
null in which the video -> emotion assignment is shuffled (preserving 3 videos per emotion and
4 neutral videos), so the null respects the stimulus structure.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from lib_rev import load, style, panel, save, TAB, EMOS, BANDS5, W2

style()
df, arr = load()
N_PERM = 1000

def zw(X):
    X = pd.DataFrame(X); g = X.groupby(df.subject.values)
    return ((X - g.transform("mean")) / (g.transform("std") + 1e-12)).to_numpy()

SETS = {
    "Band-specific sync + power": np.c_[df[[f"{s}_{b}" for s in ("wpli", "oaec") for b in BANDS5]].values,
                                       np.hstack([arr[f"logpow_{b}"] for b in BANDS5])],
    "Spearman edges (signed)": arr["rho_signed"],
    "MTF q=30 edges": arr["mtf_q30"],
    "QG q=30 edges": arr["qg_q30"],
}
SETS = {k: zw(v) for k, v in SETS.items()}
video = df.video.values
vid_emo = df.groupby("video").emotion.first().reindex(range(28)).values
emo_idx = {e: i for i, e in enumerate(EMOS)}

def lovo(X, v2e):
    y = np.array([emo_idx[v2e[v]] for v in video]); pred = np.empty_like(y)
    for v in range(28):
        te = video == v; tr = ~te
        sc = StandardScaler().fit(X[tr])
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return balanced_accuracy_score(y, pred), y, pred

rows = []
for name, X in SETS.items():
    obs, y, pred = lovo(X, vid_emo)
    rng = np.random.default_rng(11)
    perms = [rng.permutation(vid_emo) for _ in range(N_PERM)]
    null = np.array(Parallel(n_jobs=12)(delayed(lambda p: lovo(X, p)[0])(p) for p in perms))
    # video-level bootstrap CI of balanced accuracy
    bs = []
    for _ in range(2000):
        vs = rng.choice(28, 28, replace=True); idx = np.concatenate([np.where(video == v)[0] for v in vs])
        bs.append(balanced_accuracy_score(y[idx], pred[idx]))
    rows.append({"features": name, "balanced_accuracy": obs, "chance": 1 / 9,
                 "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5),
                 "null_mean": null.mean(), "null_95": np.percentile(null, 95),
                 "p_perm": (1 + (null >= obs).sum()) / (1 + N_PERM),
                 "effect_vs_null_sd": (obs - null.mean()) / null.std(ddof=1)})
    np.save(TAB / f"decoding_null_{name.split()[0]}.npy", null)
    print(rows[-1])
D = pd.DataFrame(rows); D.to_csv(TAB / "decoding_unseen_videos.csv", index=False)

fig, ax = plt.subplots(figsize=(W2 * 0.6, 2.3))
for i, (_, r) in enumerate(D.iterrows()):
    null = np.load(TAB / f"decoding_null_{r.features.split()[0]}.npy")
    vp = ax.violinplot(null * 100, positions=[i], widths=0.7, showextrema=False)
    for b in vp["bodies"]: b.set_facecolor("#cfcfcf"); b.set_alpha(1); b.set_edgecolor("none")
    ax.errorbar(i, r.balanced_accuracy * 100, yerr=[[100 * (r.balanced_accuracy - r.ci_low)], [100 * (r.ci_high - r.balanced_accuracy)]],
                fmt="o", color="#2b6cb0", ms=4, lw=0.8, capsize=2)
    ax.text(i, max(r.ci_high, r.null_95) * 100 + 0.6, f"p = {r.p_perm:.3f}", ha="center", fontsize=5.5)
ax.axhline(100 / 9, color="#c0392b", lw=0.7, ls="--"); ax.text(len(D) - 0.5, 100 / 9 + 0.4, "chance 11.1%", fontsize=5.5, color="#c0392b", ha="right")
ax.set_xticks(range(len(D))); ax.set_xticklabels(D.features, rotation=20, ha="right")
ax.set_ylabel("Balanced accuracy, unseen videos (%)")
ax.set_ylim(top=D.ci_high.max() * 100 + 2.5)
ax.set_title("Leave-one-video-out decoding vs stimulus-structured null", loc="left", pad=8)
save(fig, "Fig3_decoding_unseen_videos")
