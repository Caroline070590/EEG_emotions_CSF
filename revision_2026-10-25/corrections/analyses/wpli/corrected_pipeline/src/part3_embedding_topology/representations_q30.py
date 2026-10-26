"""
representations_q30.py — q=30 (node-count-matched) embedding representations
for MTF/QG, companion to lib.representations.build_representation.

The original Part 3 embedding pipeline uses Q_EMBED=13 (fixed, for equal
feature dimensionality across trials) for MTF/QG, giving 13x13=78
upper-triangular features versus Spearman's 30x30=435. This mismatch
means UMAP/t-SNE distances (Fig. S5), and anything downstream of the
aggregated feature vectors, are computed on very differently-sized
feature spaces for MTF/QG vs Spearman.

This module rebuilds MTF/QG representation vectors with q fixed at 30
(matching Spearman's 30x30=435-dim upper triangle exactly), saved under
separate cache filenames (repr_mtf_q30_triu.npy / repr_qg_q30_triu.npy,
agg_mtf_q30.npy / agg_qg_q30.npy) so the originals are untouched.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE
from lib.representations import (load_raw_trial_csv, choose_k_from_autocorrelation,
                                  matrix_markov_transition, triu_idx, _list_trials, aggregate)

Q30 = 30


def _repr_one_q30(path, tag):
    X = load_raw_trial_csv(path)
    if X.shape[0] < 20:
        return None
    if tag == "mtf_q30":
        A = matrix_markov_transition(X, q=Q30, lag=1)
    elif tag == "qg_q30":
        k = choose_k_from_autocorrelation(X)
        A = matrix_markov_transition(X, q=Q30, lag=k)
    else:
        raise ValueError(tag)
    n = A.shape[0]
    return A[triu_idx(n)].astype(np.float32)


def build_representation_q30(tag, n_jobs=1, force=False):
    f_feat = CACHE / f"repr_{tag}_triu.npy"
    f_meta = CACHE / f"repr_{tag}_meta.csv"
    if f_feat.exists() and f_meta.exists() and not force:
        return np.load(f_feat), pd.read_csv(f_meta)
    items = _list_trials(); t0 = time.time()
    out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_repr_one_q30)(p, tag) for p, _, _ in items)
    rows, mats = [], []
    for (p, subj, emo), v in zip(items, out):
        if v is None:
            continue
        rows.append({"subject": subj, "emotion": emo, "trial": Path(p).stem}); mats.append(v)
    feat = np.vstack(mats); meta = pd.DataFrame(rows)
    np.save(f_feat, feat); meta.to_csv(f_meta, index=False)
    print(f"[repr {tag}] {len(meta)} trials in {time.time()-t0:.1f}s -> {feat.shape}")
    return feat, meta


def main():
    for tag in ["mtf_q30", "qg_q30"]:
        feat, meta = build_representation_q30(tag, n_jobs=1)
        agg, am = aggregate(feat, meta)
        np.save(CACHE / f"agg_{tag}.npy", agg)
        am.to_csv(CACHE / f"agg_meta_{tag}.csv", index=False)
        print(f"[{tag}] aggregated -> agg_{tag}.npy {agg.shape}")


if __name__ == "__main__":
    main()
