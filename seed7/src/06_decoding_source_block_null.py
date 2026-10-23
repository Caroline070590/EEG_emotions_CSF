"""SEED-VII decoding: stricter permutation null that preserves the dependence among excerpts of the
same source programme (added after review; complements 05_decoding_unseen_sources.py).

Units = (source programme, condition) blocks (56 blocks: 54 sources, two of which span two conditions).
Under the null, the condition labels of whole blocks are permuted, so all excerpts of a programme always
share a label, as in the real data (class sizes in clips may vary across permutations; balanced accuracy
is used). Folds: leave-one-source-out, as in the primary analysis. 1,000 permutations.
Output: tables/decoding_source_block_null.csv
usage: 06_decoding_source_block_null.py [N_PERM]
"""
import sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from lib_rev import load, TAB, CACHE, EMOS, BANDS5

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
df, arr = load(); rec = np.load(CACHE / "recurrence_features.npz")
def zw(X):
    X = pd.DataFrame(X); g = X.groupby(df.subject.values)
    return ((X - g.transform("mean")) / (g.transform("std") + 1e-12)).to_numpy()
SETS = {"Band-specific sync + power": np.c_[df[[f"{s}_{b}" for s in ("wpli", "oaec") for b in BANDS5]].values,
                                           np.hstack([arr[f"logpow_{b}"] for b in BANDS5])],
        "RQA + JRS": np.c_[rec["rqa"].reshape(len(df), -1), rec["jrs"]],
        "Spearman edges (signed)": arr["rho_signed"]}
SETS = {k: zw(v) for k, v in SETS.items()}
video = df.video.values
clips = df.groupby("video").agg(emotion=("emotion", "first"), source=("source", "first")).reindex(range(80))
groups = clips.source.values
blocks = (clips.source + "|" + clips.emotion).values
ublocks = np.unique(blocks); block_lab = np.array([b.split("|")[1] for b in ublocks])
emo_idx = {e: i for i, e in enumerate(EMOS)}

def cv(X, v2e):
    y = np.array([emo_idx[v2e[v]] for v in video]); pred = np.empty_like(y); g = groups[video]
    for k in np.unique(groups):
        te = g == k; tr = ~te
        if len(np.unique(y[tr])) < len(EMOS): pred[te] = -1; continue
        sc = StandardScaler().fit(X[tr])
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return balanced_accuracy_score(y, pred)

def perm_map(rng):
    lab = dict(zip(ublocks, rng.permutation(block_lab)))
    return np.array([lab[b] for b in blocks])

rows = []
for name, X in SETS.items():
    obs = cv(X, clips.emotion.values)
    rng = np.random.default_rng(23)
    perms = [perm_map(rng) for _ in range(N_PERM)]
    null = np.array(Parallel(n_jobs=11)(delayed(cv)(X, p) for p in perms))
    rows.append({"features": name, "balanced_accuracy": obs, "null": "source-block", "null_mean": null.mean(),
                 "null_95": np.percentile(null, 95), "gain_over_null": obs - null.mean(),
                 "p_perm": (1 + (null >= obs).sum()) / (1 + N_PERM)})
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
pd.DataFrame(rows).to_csv(TAB / "decoding_source_block_null.csv", index=False)
