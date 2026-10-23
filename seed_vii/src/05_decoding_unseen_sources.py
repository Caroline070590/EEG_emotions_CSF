"""SEED-VII replication: condition decoding that generalises to unseen stimuli.

Shrinkage LDA (Ledoit-Wolf), features z-scored within participant (label-free), balanced accuracy
(chance 1/7). Primary folds: leave-one-SOURCE-out (all clips cut from the same film/programme are
held out together, so no excerpt of a test stimulus is ever in training; 54 folds). Secondary:
leave-one-clip-out (80 folds). Null: the clip -> condition map is permuted (class counts preserved:
12 clips per emotion, 8 neutral), the folds are kept, and the whole procedure is rerun.

usage: 05_decoding_unseen_sources.py [N_PERM] [FOLDS]   (default 1000; FOLDS = source | clip | both, default both)
The leave-one-clip-out comparison was run separately with 200 permutations (local run time).
"""
import sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from lib_rev import load, TAB, CACHE, EMOS, BANDS5

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
WHICH = sys.argv[2] if len(sys.argv) > 2 else "both"
df, arr = load()
rec = np.load(CACHE / "recurrence_features.npz")

def zw(X):
    X = pd.DataFrame(X); g = X.groupby(df.subject.values)
    return ((X - g.transform("mean")) / (g.transform("std") + 1e-12)).to_numpy()

SETS = {
    "Band-specific sync + power": np.c_[df[[f"{s}_{b}" for s in ("wpli", "oaec") for b in BANDS5]].values,
                                       np.hstack([arr[f"logpow_{b}"] for b in BANDS5])],
    "Spearman edges (signed)": arr["rho_signed"],
    "MTF q=30 edges": arr["mtf_q30"],
    "QG q=30 edges": arr["qg_q30"],
    "RQA + JRS": np.c_[rec["rqa"].reshape(len(df), -1), rec["jrs"]],
}
SETS = {k: zw(v) for k, v in SETS.items()}
video = df.video.values
clips = df.groupby("video").agg(emotion=("emotion", "first"), source=("source", "first")).reindex(range(80))
vid_emo = clips.emotion.values
emo_idx = {e: i for i, e in enumerate(EMOS)}
FOLDS = {"leave-one-source-out": clips.source.values, "leave-one-clip-out": np.arange(80)}
FOLDS = {k: v for k, v in FOLDS.items() if WHICH == "both" or k.split("-")[2] == WHICH}

def cv(X, v2e, groups):
    y = np.array([emo_idx[v2e[v]] for v in video]); pred = np.empty_like(y)
    g = groups[video]
    for k in np.unique(groups):
        te = g == k; tr = ~te
        sc = StandardScaler().fit(X[tr])
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return balanced_accuracy_score(y, pred)

rows = []
for fold_name, groups in FOLDS.items():
    for name, X in SETS.items():
        obs = cv(X, vid_emo, groups)
        rng = np.random.default_rng(11)
        perms = [rng.permutation(vid_emo) for _ in range(N_PERM)]
        null = np.array(Parallel(n_jobs=11)(delayed(cv)(X, p, groups) for p in perms))
        rows.append({"folds": fold_name, "features": name, "balanced_accuracy": obs, "chance": 1 / 7,
                     "null_mean": null.mean(), "null_95": np.percentile(null, 95),
                     "gain_over_null": obs - null.mean(),
                     "p_perm": (1 + (null >= obs).sum()) / (1 + N_PERM)})
        np.save(TAB / f"decoding_null_{fold_name}_{name.split()[0]}.npy", null)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
OUTNAME = {"both": "decoding_unseen_sources.csv", "source": "decoding_unseen_sources.csv", "clip": "decoding_leave_one_clip_out.csv"}[WHICH]
pd.DataFrame(rows).to_csv(TAB / OUTNAME, index=False)
